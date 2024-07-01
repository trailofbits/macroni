#pragma once

#include "macroni/Common/ExpansionTable.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenMetaGenerator.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/CodeGenVisitorList.hpp"
#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/SymbolGenerator.hpp"
#include "vast/Util/Common.hpp"
#include <clang/AST/Attrs.inc>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <mlir/IR/Operation.h>

namespace macroni::kernel {
struct kernel_visitor : vast::cg::fallthrough_list_node {
  [[nodiscard]] explicit kernel_visitor(vast::cg::visitor_base &head,
                                        vast::acontext_t &actx,
                                        vast::mcontext_t &mctx,
                                        vast::cg::codegen_builder &bld,
                                        vast::cg::meta_generator &mg,
                                        vast::cg::symbol_generator &sg,
                                        expansion_table &expansions);

  [[nodiscard]] vast::operation visit(const vast::cg::clang_stmt *stmt,
                                      vast::cg::scope_context &scope) override;

  [[nodiscard]] vast::operation visit(const vast::cg::clang_decl *decl,
                                      vast::cg::scope_context &scope) override;

protected:
  vast::acontext_t &m_actx;
  vast::mcontext_t &m_mctx;
  vast::cg::codegen_builder &m_bld;
  vast::cg::meta_generator &m_mg;
  expansion_table &m_expansions;
  vast::cg::visitor_view m_view;
};
} // namespace macroni::kernel
