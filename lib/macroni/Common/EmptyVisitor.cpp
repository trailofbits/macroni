#include "macroni/Common/EmptyVisitor.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/SymbolGenerator.hpp"
#include "vast/Util/Common.hpp"

namespace macroni {
empty_visitor::empty_visitor(vast::mcontext_t &mctx,
                             vast::cg::meta_generator &mg,
                             vast::cg::symbol_generator &sg,
                             vast::cg::visitor_view view)
    : vast::cg::visitor_base(mctx, mg, sg, view.options()) {}

vast::operation empty_visitor::visit(const vast::cg::clang_stmt *stmt,
                                     vast::cg::scope_context &scope) {
  return {};
}

vast::operation empty_visitor::visit(const vast::cg::clang_decl *decl,
                                     vast::cg::scope_context &scope) {
  return {};
}

vast::mlir_type empty_visitor::visit(const vast::cg::clang_type *type,
                                     vast::cg::scope_context &scope) {
  return {};
}

vast::mlir_type empty_visitor::visit(vast::cg::clang_qual_type type,
                                     vast::cg::scope_context &scope) {
  return {};
}

vast::mlir_attr empty_visitor::visit(const vast::cg::clang_attr *attr,
                                     vast::cg::scope_context &scope) {
  return {};
}

vast::operation
empty_visitor::visit_prototype(const vast::cg::clang_function *decl,
                               vast::cg::scope_context &scope) {
  return {};
}
} // namespace macroni
