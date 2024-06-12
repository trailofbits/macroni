#pragma once

#include "macroni/Common/EmptyVisitor.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/SymbolGenerator.hpp"
#include "vast/Util/Common.hpp"
#include <clang/AST/Expr.h>
#include <set>

namespace macroni::safety {
struct safety_visitor : macroni::empty_visitor {

  [[nodiscard]] safety_visitor(
      std::set<const clang::IntegerLiteral *> &safe_block_conditions,
      vast::mcontext_t &mctx, vast::cg::codegen_builder &bld,
      vast::cg::meta_generator &mg, vast::cg::symbol_generator &sg,
      vast::cg::visitor_view view);

  [[nodiscard]] vast::operation visit(const vast::cg::clang_stmt *stmt,
                                      vast::cg::scope_context &scope) override;

  std::set<const clang::IntegerLiteral *> &m_safe_block_conditions;
  vast::cg::codegen_builder &m_bld;
  vast::cg::visitor_view m_view;
};
} // namespace macroni::safety
