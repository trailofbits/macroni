#include "macroni/Translation/Safety/SafetyVisitor.hpp"
#include "macroni/Common/EmptyVisitor.hpp"
#include "macroni/Dialect/Safety/SafetyOps.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/DefaultStmtVisitor.hpp"
#include "vast/CodeGen/SymbolGenerator.hpp"
#include "vast/Util/Common.hpp"
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/LLVM.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

namespace macroni::safety {
safety_visitor::safety_visitor(safety_conditions &safe_block_conditions,
                               vast::mcontext_t &mctx,
                               vast::cg::codegen_builder &bld,
                               vast::cg::meta_generator &mg,
                               vast::cg::symbol_generator &sg,
                               vast::cg::visitor_view view)
    : ::macroni::empty_visitor(mctx, mg, sg, view),
      m_safe_block_conditions(safe_block_conditions), m_bld(bld), m_view(view) {
}

vast::operation safety_visitor::visit(const vast::cg::clang_stmt *stmt,
                                      vast::cg::scope_context &scope) {
  auto if_stmt = clang::dyn_cast<clang::IfStmt>(stmt);
  if (!if_stmt) {
    return {};
  }

  auto else_branch = if_stmt->getElse();
  if (!else_branch) {
    return {};
  }

  auto integer_literal =
      clang::dyn_cast<const clang::IntegerLiteral>(if_stmt->getCond());
  if (!integer_literal) {
    return {};
  }

  if (!m_safe_block_conditions.contains(integer_literal)) {
    return {};
  }

  auto mk_region_builder = [&](const vast::cg::clang_stmt *stmt) {
    return
        [this, stmt, &scope](auto &_bld, auto) { m_view.visit(stmt, scope); };
  };

  m_bld.compose<UnsafeRegion>()
      .bind(m_view.location(stmt))
      .bind(mk_region_builder(else_branch))
      .freeze();

  vast::cg::default_stmt_visitor visitor(m_bld, m_view, scope);
  auto op = visitor.visit(else_branch);
  return op;
}
} // namespace macroni::safety
