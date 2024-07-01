#pragma once

#include "pasta/AST/AST.h"
#include "pasta/AST/Macro.h"
#include "pasta/AST/Stmt.h"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenMetaGenerator.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/CodeGenVisitorList.hpp"
#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/Util/Common.hpp"
#include <optional>
#include <set>

namespace macroni {
// Given a set of visited substitutions, returns the lowest substitutions in
// this macros chain of aligned substitution that has not yet been visited, and
// marks it as visited.
[[nodiscard]] std::optional<pasta::MacroSubstitution>
lowest_unvisited_substitution(pasta::Stmt &stmt,
                              std::set<pasta::MacroSubstitution> &visited);

// Given a substitution, returns whether that substitution is an expansion of a
// function-like macro. Conservatively returns false if the substitution lacks
// the necessary information to determine whether it is function-like or not.
[[nodiscard]] bool is_function_like(pasta::MacroSubstitution &sub);

// Given a substitution, returns the names of the names of the substitution's
// macro parameters, if any.
std::vector<llvm::StringRef> get_parameter_names(pasta::MacroSubstitution &sub);

struct macroni_visitor : vast::cg::fallthrough_list_node {

  [[nodiscard]] macroni_visitor(vast::cg::visitor_base &head,
                                vast::mcontext_t &mctx,
                                vast::cg::codegen_builder &bld,
                                vast::cg::meta_generator &mg, const pasta::AST &pctx);

  [[nodiscard]] vast::operation visit(const vast::cg::clang_stmt *stmt,
                                      vast::cg::scope_context &scope) override;

protected:
  vast::mcontext_t &m_mctx;
  vast::cg::codegen_builder &m_bld;
  vast::cg::meta_generator &m_mg;
  const pasta::AST &m_pctx;
  std::set<pasta::MacroSubstitution> m_visited;
  vast::cg::visitor_view m_view;
};
} // namespace macroni
