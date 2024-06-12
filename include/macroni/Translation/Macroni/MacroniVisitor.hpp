#pragma once

#include "pasta/AST/AST.h"
#include "pasta/AST/Macro.h"
#include "pasta/AST/Stmt.h"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/SymbolGenerator.hpp"
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

struct macroni_visitor : vast::cg::visitor_base {

  [[nodiscard]] macroni_visitor(pasta::AST &pctx, vast::mcontext_t &mctx,
                                vast::cg::codegen_builder &bld,
                                vast::cg::meta_generator &mg,
                                vast::cg::symbol_generator &sg,
                                vast::cg::visitor_view view);

  [[nodiscard]] vast::operation visit(const vast::cg::clang_stmt *stmt,
                                      vast::cg::scope_context &scope) override;

  // We default these methods to return an empty op to force the fallback
  // visitor to handle them instead.
  //
  // TODO(Brent): Maybe I should add an empty_visitor type to return empty
  // options for all visitor methods, and inherit from that for all Macroni's
  // visitors so they can just override the single methods they're interested
  // in?

  vast::operation visit(const vast::cg::clang_decl *decl,
                        vast::cg::scope_context &scope) override;

  vast::mlir_type visit(const vast::cg::clang_type *type,
                        vast::cg::scope_context &scope) override;

  vast::mlir_type visit(vast::cg::clang_qual_type type,
                        vast::cg::scope_context &scope) override;

  vast::mlir_attr visit(const vast::cg::clang_attr *attr,
                        vast::cg::scope_context &scope) override;

  vast::operation visit_prototype(const vast::cg::clang_function *decl,
                                  vast::cg::scope_context &scope) override;

  pasta::AST &m_pctx;
  vast::cg::codegen_builder &m_bld;
  vast::cg::visitor_view m_view;
  std::set<pasta::MacroSubstitution> m_visited;
};
} // namespace macroni
