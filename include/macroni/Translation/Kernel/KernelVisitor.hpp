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
#include <map>
#include <mlir/IR/Operation.h>
#include <optional>

namespace macroni::kernel {
using rcu_dereference_table =
    std::map<const clang::StmtExpr *, const clang::DeclRefExpr *>;

struct kernel_visitor : ::macroni::empty_visitor {
  [[nodiscard]] kernel_visitor(rcu_dereference_table &rcu_dereference_to_p,
                               vast::mcontext_t &mctx,
                               vast::cg::codegen_builder &bld,
                               vast::cg::meta_generator &mg,
                               vast::cg::symbol_generator &sg,
                               vast::cg::visitor_view view);

  [[nodiscard]] vast::operation visit(const vast::cg::clang_stmt *stmt,
                                      vast::cg::scope_context &scope) override;

  [[nodiscard]] vast::mlir_type visit(vast::cg::clang_qual_type type,
                                      vast::cg::scope_context &scope) override;

  [[nodiscard]] vast::mlir_attr visit(const vast::cg::clang_attr *attr,
                                      vast::cg::scope_context &scope) override;

  [[nodiscard]] std::optional<vast::operation>
  visit_rcu_dereference(const vast::cg::clang_stmt *stmt,
                        vast::cg::scope_context &scope);

  [[nodiscard]] std::optional<vast::operation>
  visit_rcu_read_lock_or_unlock(const vast::cg::clang_stmt *stmt,
                                vast::cg::scope_context &scope);

  [[nodiscard]] bool is_context_attr(const clang::AnnotateAttr *attr);

  void set_lock_level(mlir::Operation &op);

  void lock_op(mlir::Operation &op);

  void unlock_op(mlir::Operation &op);

  rcu_dereference_table &m_rcu_dereference_to_p;
  vast::cg::codegen_builder &m_bld;
  vast::cg::visitor_view m_view;
  std::int64_t lock_level = 0;
};
} // namespace macroni::kernel
