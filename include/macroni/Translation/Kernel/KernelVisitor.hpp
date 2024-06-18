#pragma once

#include "macroni/Common/EmptyVisitor.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/SymbolGenerator.hpp"
#include "vast/Util/Common.hpp"
#include <clang/AST/Attrs.inc>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <mlir/IR/Operation.h>
#include <optional>
#include <unordered_map>

namespace macroni::kernel {
using rcu_dereference_table =
    std::unordered_map<const clang::StmtExpr *, const clang::Expr *>;

struct rcu_assign_pointer_parameters {
  const clang::Expr *p;
  const clang::Expr *v;
};

struct rcu_replace_pointer_parameters {
  const clang::Expr *rcu_ptr;
  const clang::Expr *ptr;
  const clang::Expr *c;
};

using rcu_assign_pointer_table =
    std::unordered_map<const clang::DoStmt *, rcu_assign_pointer_parameters>;

using rcu_access_pointer_table =
    std::unordered_map<const clang::StmtExpr *, const clang::Expr *>;

using rcu_replace_pointer_table =
    std::unordered_map<const clang::StmtExpr *, rcu_replace_pointer_parameters>;

struct kernel_visitor : ::macroni::empty_visitor {
  [[nodiscard]] explicit kernel_visitor(
      rcu_dereference_table &rcu_dereference_to_p,
      rcu_assign_pointer_table &rcu_assign_pointer_params,
      rcu_access_pointer_table &rcu_access_pointer_to_p,
      rcu_replace_pointer_table &m_rcu_replace_pointer_to_params,
      vast::mcontext_t &mctx, vast::cg::codegen_builder &bld,
      vast::cg::meta_generator &mg, vast::cg::symbol_generator &sg,
      vast::cg::visitor_view view);

  [[nodiscard]] vast::operation visit(const vast::cg::clang_stmt *stmt,
                                      vast::cg::scope_context &scope) override;

  [[nodiscard]] std::optional<vast::operation>
  visit_rcu_dereference(const vast::cg::clang_stmt *stmt,
                        vast::cg::scope_context &scope);

  [[nodiscard]] std::optional<vast::operation>
  visit_rcu_read_lock_or_unlock(const vast::cg::clang_stmt *stmt,
                                vast::cg::scope_context &scope);

  [[nodiscard]] std::optional<vast::operation>
  visit_rcu_assign_pointer(const vast::cg::clang_stmt *stmt,
                           vast::cg::scope_context &scope);

  [[nodiscard]] std::optional<vast::operation>
  visit_rcu_access_pointer(const vast::cg::clang_stmt *stmt,
                           vast::cg::scope_context &scope);

  [[nodiscard]] std::optional<vast::operation>
  visit_rcu_replace_pointer(const vast::cg::clang_stmt *stmt,
                            vast::cg::scope_context &scope);

  void set_lock_level(mlir::Operation &op);

  void lock_op(mlir::Operation &op);

  void unlock_op(mlir::Operation &op);

  rcu_dereference_table &m_rcu_dereference_to_p;
  rcu_assign_pointer_table &m_rcu_assign_pointer_params;
  rcu_access_pointer_table &m_rcu_access_pointer_to_p;
  rcu_replace_pointer_table &m_rcu_replace_pointer_to_params;

  vast::cg::codegen_builder &m_bld;
  vast::cg::visitor_view m_view;
  std::int64_t lock_level = 0;
};
} // namespace macroni::kernel
