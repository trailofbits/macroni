#pragma once

#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/SymbolGenerator.hpp"
#include "vast/Util/Common.hpp"

namespace macroni {
struct empty_visitor : vast::cg::visitor_base {
  [[nodiscard]] explicit empty_visitor(vast::mcontext_t &mctx,
                                       vast::cg::meta_generator &mg,
                                       vast::cg::symbol_generator &sg,
                                       vast::cg::visitor_view view);

  vast::operation visit(const vast::cg::clang_stmt *stmt,
                        vast::cg::scope_context &scope) override;

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
};
} // namespace macroni
