#include "macroni/Translation/MacroniVisitor.hpp"
#include "macroni/Dialect/Macroni/MacroniOps.hpp"
#include "macroni/Translation/MacroniMetaGenerator.hpp"
#include "pasta/AST/Macro.h"
#include "pasta/AST/Stmt.h"
#include <llvm/ADT/StringRef.h>
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
      param_names.push_back("<a nameless macro parameter>");
    }
  }
  return param_names;
}

macroni_visitor::macroni_visitor(pasta::AST &pctx, vast::mcontext_t &mctx,
                                 vast::cg::codegen_builder &bld,
                                 vast::cg::meta_generator &mg,
                                 vast::cg::symbol_generator &sg,
                                 vast::cg::visitor_view view)
    : visitor_base(mctx, mg, sg, view.options()), m_pctx(pctx), m_bld(bld),
      m_view(view) {}

vast::operation macroni_visitor::visit(const vast::cg::clang_stmt *stmt,
                                       vast::cg::scope_context &scope) {
  auto pasta_stmt = m_pctx.Adopt(stmt);

  if (clang::isa<clang::ImplicitValueInitExpr, clang::ImplicitCastExpr>(stmt)) {
    return {};
  }

  // Find the lowest macro that covers this statement, if any
  auto sub = lowest_unvisited_substitution(pasta_stmt, m_visited);
  if (!sub) {
    // If no substitution covers this statement, let a fallback visit it.
    return {};
  }

  // Get the substitution's location, name, parameter names, and whether it is
  // function-like.
  //
  // NOTE(Brent): We have to use a dynamic_cast here because
  // vast::cg::codegen_instance expects a vast::cg::meta_generator as its meta
  // generator, but we use static inheritance to pass it our own meta generator,
  // so simply calling location() directly won't work.
  auto meta = dynamic_cast<macroni_meta_generator *>(&mg);
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
      (void)m_bld.scoped_insertion_at_end(last_block);
      auto stmt_loc = m_view.location(stmt);

      if (last_op->getNumResults() > 0) {
        m_bld.create<vast::hl::ValueYieldOp>(stmt_loc, last_op->getResult(0));
      } else {
        auto void_value = m_bld.void_value(m_view.location(stmt));
        m_bld.create<vast::hl::ValueYieldOp>(stmt_loc, void_value);
      }
    };
  };

  // Check if the macro is an expansion or a parameter, and return the
  // appropriate operation.

  auto expr = clang::dyn_cast<vast::cg::clang_expr>(stmt);
  auto rty = expr ? m_view.visit(expr->getType(), scope)
                  : vast::hl::VoidType::get(&mctx);
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

// We default these methods to return an empty op to force the fallback
// visitor to handle them instead.
//
// TODO(Brent): Maybe I should add an empty_visitor type to return empty
// options for all visitor methods, and inherit from that for all Macroni's
// visitors so they can just override the single methods they're interested
// in?

vast::operation macroni_visitor::visit(const vast::cg::clang_decl *decl,
                                       vast::cg::scope_context &scope) {
  return {};
}

vast::mlir_type macroni_visitor::visit(const vast::cg::clang_type *type,
                                       vast::cg::scope_context &scope) {
  return {};
}

vast::mlir_type macroni_visitor::visit(vast::cg::clang_qual_type type,
                                       vast::cg::scope_context &scope) {
  return {};
}

vast::mlir_attr macroni_visitor::visit(const vast::cg::clang_attr *attr,
                                       vast::cg::scope_context &scope) {
  return {};
}

vast::operation
macroni_visitor::visit_prototype(const vast::cg::clang_function *decl,
                                 vast::cg::scope_context &scope) {
  return {};
}
} // namespace macroni
