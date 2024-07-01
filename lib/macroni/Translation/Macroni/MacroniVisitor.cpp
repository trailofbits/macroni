#include "macroni/Translation/Macroni/MacroniVisitor.hpp"

#include "macroni/Dialect/Macroni/MacroniOps.hpp"
#include "macroni/Translation/Macroni/PastaMetaGenerator.hpp"
#include "pasta/AST/Macro.h"
#include "pasta/AST/Stmt.h"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <optional>
#include <set>
#include <vector>

namespace macroni {
std::optional<pasta::MacroSubstitution>
lowest_unvisited_substitution(pasta::Stmt &stmt,
                              std::set<pasta::MacroSubstitution> &visited) {
  auto subs = stmt.AlignedSubstitutions();
  std::reverse(subs.begin(), subs.end());
  for (auto sub : subs) {
    // Don't visit macros more than once
    if (visited.contains(sub)) {
      continue;
    }

    // Only visit pre-expanded forms of function-like expansions.
    if (auto exp = pasta::MacroExpansion::From(sub)) {
      bool is_pre_expansion =
          (exp->Arguments().empty() || exp->IsArgumentPreExpansion());
      if (!is_pre_expansion) {
        continue;
      }
    }

    // Mark this substitution as visited so we don't visit it again.
    visited.insert(sub);
    return sub;
  }
  return std::nullopt;
}

bool is_function_like(pasta::MacroSubstitution &sub) {
  if (auto exp = pasta::MacroExpansion::From(sub)) {
    if (auto def = exp->Definition()) {
      return def->IsFunctionLike();
    }
  }
  return false;
}

std::vector<llvm::StringRef>
get_parameter_names(pasta::MacroSubstitution &sub) {
  std::vector<llvm::StringRef> param_names;
  auto exp = pasta::MacroExpansion::From(sub);
  if (!exp) {
    return param_names;
  }
  auto def = exp->Definition();
  if (!def) {
    return param_names;
  }
  for (auto macro_tok : def->Parameters()) {
    if (auto bt = macro_tok.BeginToken()) {
      param_names.push_back(bt->Data());
    } else {
      param_names.push_back("");
    }
  }
  return param_names;
}

macroni_visitor::macroni_visitor(vast::cg::visitor_base &head,
                                 vast::mcontext_t &mctx,
                                 vast::cg::codegen_builder &bld,
                                 vast::cg::meta_generator &mg,
                                 const pasta::AST &pctx)
    : vast::cg::fallthrough_list_node(), m_mctx(mctx), m_bld(bld), m_mg(mg),
      m_pctx(pctx), m_view(head) {}

vast::operation macroni_visitor::visit(const vast::cg::clang_stmt *stmt,
                                       vast::cg::scope_context &scope) {
  auto pasta_stmt = m_pctx.Adopt(stmt);

  if (clang::isa<clang::ImplicitValueInitExpr, clang::ImplicitCastExpr>(stmt)) {
    return next->visit(stmt, scope);
  }

  // Find the lowest macro that covers this statement, if any
  auto sub = lowest_unvisited_substitution(pasta_stmt, m_visited);
  if (!sub) {
    // If no substitution covers this statement, let a fallback visit it.
    return next->visit(stmt, scope);
  }

  // Get the substitution's location, name, parameter names, and whether it is
  // function-like.
  //
  // NOTE(Brent): We have to use a dynamic_cast here because
  // vast::cg::codegen_instance expects a vast::cg::meta_generator as its meta
  // generator, but we use static inheritance to pass it our own meta generator,
  // so simply calling location() directly won't work.
  auto meta = dynamic_cast<pasta_meta_generator *>(&m_mg);
  auto loc = meta ? meta->location(*sub) : mlir::UnknownLoc();
  auto name_tok = sub->NameOrOperator();
  auto macro_name = (name_tok ? name_tok->Data() : "<a nameless macro>");
  auto function_like = is_function_like(*sub);
  auto parameter_names = get_parameter_names(*sub);

  // Creates a region suitable for creating a macroni::MacroExpansion or
  // macroni::MacroParameter from the given stmt, which is expected to have been
  // expanded from a macro or macro parameter. This is based on vast's
  // default_stmt_visitor's last_effective_operation() and VisitStmtExpr()
  // methods. The differences are that the stmt may not be an expr at all, and
  // we are only concerned with the final return value, not the last effective
  // return value.
  auto make_expansion_region_builder = [&](const vast::cg::clang_stmt *stmt) {
    return [this, stmt, &scope](auto &state, auto) {
      // Let the fallbacks do the work for us.
      m_view.visit(stmt, scope);
      // Get the last operation in this block and check if it has a return
      // value.
      auto last_block = state.getBlock();
      auto last_op = std::prev(last_block->end());
      [[maybe_unused]] auto _ = m_bld.scoped_insertion_at_end(last_block);
      auto stmt_loc = m_view.location(stmt);
      auto loc = stmt_loc.value_or(mlir::UnknownLoc::get(&m_mctx));

      auto value = last_op->getNumResults() > 0 ? last_op->getResult(0)
                                                : m_bld.void_value(loc);

      m_bld.create<vast::hl::ValueYieldOp>(loc, value);
    };
  };

  // Check if the macro is an expansion or a parameter, and return the
  // appropriate operation.

  auto expr = clang::dyn_cast<vast::cg::clang_expr>(stmt);
  auto rty = expr ? m_view.visit(expr->getType(), scope)
                  : vast::hl::VoidType::get(&m_mctx);
  if (sub->Kind() == pasta::MacroKind::kExpansion) {
    auto name_attr = m_bld.getStringAttr(llvm::Twine(macro_name));
    auto parameter_names_attr =
        m_bld.getStrArrayAttr(llvm::ArrayRef(parameter_names));
    auto function_like_attr = m_bld.getBoolAttr(function_like);
    return m_bld.compose<macroni::MacroExpansion>()
        .bind(loc)
        .bind(name_attr)
        .bind(parameter_names_attr)
        .bind(function_like_attr)
        .bind(rty)
        .bind(make_expansion_region_builder(stmt))
        .freeze();
  }
  auto parameter_name_attr = m_bld.getStringAttr(llvm::Twine(macro_name));
  return m_bld.compose<macroni::MacroParameter>()
      .bind(loc)
      .bind(parameter_name_attr)
      .bind(rty)
      .bind(make_expansion_region_builder(stmt))
      .freeze();
}
} // namespace macroni
