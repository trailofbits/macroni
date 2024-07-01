#pragma once

#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/CodeGenVisitorList.hpp"
#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/Util/Common.hpp"
#include <clang/AST/Expr.h>
#include <unordered_set>

namespace macroni::safety {
using safety_conditions = std::unordered_set<const clang::IntegerLiteral *>;

struct safety_visitor : vast::cg::fallthrough_list_node {

  [[nodiscard]] safety_visitor(vast::cg::visitor_base &head,
                               vast::mcontext_t &mctx,
                               vast::cg::codegen_builder &bld,
                               safety_conditions &safe_block_conditions);

  [[nodiscard]] vast::operation visit(const vast::cg::clang_stmt *stmt,
                                      vast::cg::scope_context &scope) override;

protected:
  vast::mcontext_t &m_mctx;
  vast::cg::codegen_builder &m_bld;
  safety_conditions &m_safe_block_conditions;
  vast::cg::visitor_view m_view;
};
} // namespace macroni::safety
