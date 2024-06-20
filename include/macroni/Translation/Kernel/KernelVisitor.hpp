#pragma once

#include "macroni/Common/EmptyVisitor.hpp"
#include "macroni/Common/ExpansionTable.hpp"
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

namespace macroni::kernel {

struct kernel_visitor : ::macroni::empty_visitor {
  [[nodiscard]] explicit kernel_visitor(expansion_table &expansions,
                                        vast::acontext_t &actx,
                                        vast::mcontext_t &mctx,
                                        vast::cg::codegen_builder &bld,
                                        vast::cg::meta_generator &mg,
                                        vast::cg::symbol_generator &sg,
                                        vast::cg::visitor_view view);

  [[nodiscard]] vast::operation visit(const vast::cg::clang_stmt *stmt,
                                      vast::cg::scope_context &scope) override;

  [[nodiscard]] std::optional<vast::operation>
  visit_rcu_read_lock_or_unlock(const vast::cg::clang_stmt *stmt,
                                vast::cg::scope_context &scope);

  [[nodiscard]] vast::operation visit(const vast::cg::clang_decl *decl,
                                      vast::cg::scope_context &scope) override;

  void set_lock_level(mlir::Operation &op);

  void lock_op(mlir::Operation &op);

  void unlock_op(mlir::Operation &op);

  expansion_table &m_expansions;
  vast::acontext_t &m_actx;
  vast::cg::codegen_builder &m_bld;
  vast::cg::visitor_view m_view;
  std::int64_t lock_level = 0;
};
} // namespace macroni::kernel
