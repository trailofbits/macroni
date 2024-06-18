#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include "macroni/Common/EmptyVisitor.hpp"
#include "macroni/Dialect/Kernel/KernelAttributes.hpp"
#include "macroni/Dialect/Kernel/KernelDialect.hpp"
#include "macroni/Dialect/Kernel/KernelOps.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/DefaultDeclVisitor.hpp"
#include "vast/CodeGen/DefaultStmtVisitor.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/SymbolGenerator.hpp"
#include "vast/Util/Common.hpp"
#include <clang/AST/Attrs.inc>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/LLVM.h>
#include <clang/Lex/Lexer.h>
#include <llvm/ADT/Twine.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <optional>
#include <string>

namespace macroni::kernel {
kernel_visitor::kernel_visitor(
    rcu_dereference_table &rcu_dereference_to_p,
    rcu_assign_pointer_table &rcu_assign_pointer_params,
    rcu_access_pointer_table &rcu_access_pointer_to_p,
    rcu_replace_pointer_table &rcu_replace_pointer_to_params,
    vast::acontext_t &actx, vast::mcontext_t &mctx,
    vast::cg::codegen_builder &bld, vast::cg::meta_generator &mg,
    vast::cg::symbol_generator &sg, vast::cg::visitor_view view)
    : ::macroni::empty_visitor(mctx, mg, sg, view),
      m_rcu_dereference_to_p(rcu_dereference_to_p),
      m_rcu_assign_pointer_params(rcu_assign_pointer_params),
      m_rcu_access_pointer_to_p(rcu_access_pointer_to_p),
      m_rcu_replace_pointer_to_params(rcu_replace_pointer_to_params),
      m_actx(actx), m_bld(bld), m_view(view) {}

vast::operation kernel_visitor::visit(const vast::cg::clang_stmt *stmt,
                                      vast::cg::scope_context &scope) {
  return visit_rcu_dereference(stmt, scope)
      .or_else([&] { return visit_rcu_read_lock_or_unlock(stmt, scope); })
      .or_else([&] { return visit_rcu_assign_pointer(stmt, scope); })
      .or_else([&] { return visit_rcu_access_pointer(stmt, scope); })
      .or_else([&] { return visit_rcu_replace_pointer(stmt, scope); })
      // If we can't match anything, return nullptr
      .value_or(nullptr);
}

std::optional<vast::operation>
kernel_visitor::visit_rcu_dereference(const vast::cg::clang_stmt *stmt,
                                      vast::cg::scope_context &scope) {
  auto rcu_dereference = clang::dyn_cast<clang::StmtExpr>(stmt);
  if (!rcu_dereference) {
    return std::nullopt;
  }

  if (!m_rcu_dereference_to_p.contains(rcu_dereference)) {
    return std::nullopt;
  }

  auto loc = m_view.location(rcu_dereference);
  auto p = m_rcu_dereference_to_p.at(rcu_dereference);
  vast::cg::default_stmt_visitor visitor(m_bld, m_view, scope);
  auto rty = m_view.visit(p->getType(), scope);
  auto p_op = m_view.visit(p, scope);

  return m_bld.create<RCUDereference>(loc, rty, p_op->getOpResult(0));
}

std::optional<vast::operation>
kernel_visitor::visit_rcu_read_lock_or_unlock(const vast::cg::clang_stmt *stmt,
                                              vast::cg::scope_context &scope) {
  auto call_expr = clang::dyn_cast<clang::CallExpr>(stmt);
  if (!call_expr) {
    return std::nullopt;
  }

  auto direct_callee = call_expr->getDirectCallee();
  if (!direct_callee) {
    return std::nullopt;
  }

  auto name = direct_callee->getName();
  if (KernelDialect::rcu_read_lock() != name &&
      KernelDialect::rcu_read_unlock() != name) {
    return std::nullopt;
  }

  vast::cg::default_stmt_visitor visitor(m_bld, m_view, scope);
  auto op = visitor.visit(stmt);
  if (KernelDialect::rcu_read_lock() == name) {
    lock_op(*op);
  } else {
    unlock_op(*op);
  }

  return op;
}

std::optional<vast::operation>
kernel_visitor::visit_rcu_assign_pointer(const vast::cg::clang_stmt *stmt,
                                         vast::cg::scope_context &scope) {
  auto rcu_assign_pointer = clang::dyn_cast<clang::DoStmt>(stmt);
  if (!rcu_assign_pointer) {
    return std::nullopt;
  }

  if (!m_rcu_assign_pointer_params.contains(rcu_assign_pointer)) {
    return std::nullopt;
  }

  auto loc = m_view.location(rcu_assign_pointer);
  auto rty = m_bld.void_type();
  auto params = m_rcu_assign_pointer_params.at(rcu_assign_pointer);
  auto p = params.p;
  auto v = params.v;
  auto visitor = vast::cg::default_stmt_visitor(m_bld, m_view, scope);
  auto p_op = m_view.visit(p, scope);
  auto v_op = m_view.visit(v, scope);
  auto p_result = p_op->getOpResult(0);
  auto v_result = v_op->getOpResult(0);

  return m_bld.create<RCUAssignPointer>(loc, rty, p_result, v_result);
}

std::optional<vast::operation>
kernel_visitor::visit_rcu_access_pointer(const vast::cg::clang_stmt *stmt,
                                         vast::cg::scope_context &scope) {
  auto rcu_access_pointer = clang::dyn_cast<clang::StmtExpr>(stmt);
  if (!rcu_access_pointer) {
    return std::nullopt;
  }

  if (!m_rcu_access_pointer_to_p.contains(rcu_access_pointer)) {
    return std::nullopt;
  }

  auto loc = m_view.location(rcu_access_pointer);
  auto p = m_rcu_access_pointer_to_p.at(rcu_access_pointer);
  vast::cg::default_stmt_visitor visitor(m_bld, m_view, scope);
  auto rty = m_view.visit(p->getType(), scope);
  auto p_op = m_view.visit(p, scope);

  return m_bld.create<RCUAccessPointer>(loc, rty, p_op->getOpResult(0));
}

std::optional<vast::operation>
kernel_visitor::visit_rcu_replace_pointer(const vast::cg::clang_stmt *stmt,
                                          vast::cg::scope_context &scope) {
  auto rcu_replace_pointer = clang::dyn_cast<clang::StmtExpr>(stmt);
  if (!rcu_replace_pointer) {
    return std::nullopt;
  }

  if (!m_rcu_replace_pointer_to_params.contains(rcu_replace_pointer)) {
    return std::nullopt;
  }

  auto loc = m_view.location(rcu_replace_pointer);
  auto params = m_rcu_replace_pointer_to_params.at(rcu_replace_pointer);
  vast::cg::default_stmt_visitor visitor(m_bld, m_view, scope);
  auto rty = m_view.visit(params.rcu_ptr->getType(), scope);
  auto rcu_ptr_op = m_view.visit(params.rcu_ptr, scope);
  auto ptr_op = m_view.visit(params.ptr, scope);
  auto c_op = m_view.visit(params.c, scope);

  return m_bld.create<RCUReplacePointer>(loc, rty, rcu_ptr_op->getOpResult(0),
                                         ptr_op->getOpResult(0),
                                         c_op->getOpResult(0));
}

vast::operation kernel_visitor::visit(const vast::cg::clang_decl *decl,
                                      vast::cg::scope_context &scope) {
  auto function_decl = clang::dyn_cast<clang::FunctionDecl>(decl);
  if (!function_decl || !function_decl->hasBody()) {
    return nullptr;
  }

  // Get the source text of this function declaration so we can check if it
  // contains an RCU annotation. The RCU annotations (__acquires(),
  // __releases(), and __must_hold()) are not standard so Clang will not embed
  // them in the AST, so we must check for their presence in the source text
  // instead.

  auto &sm = m_actx.getSourceManager();
  auto &lo = m_actx.getLangOpts();
  auto body = function_decl->getBody();
  auto begin = function_decl->getBeginLoc();
  auto end = body->getBeginLoc();
  auto s_range = clang::SourceRange(begin, end);
  auto cs_range = clang::CharSourceRange::getCharRange(s_range);
  auto source_text = clang::Lexer::getSourceText(cs_range, sm, lo);

  // Get the op for this function decl.

  vast::cg::default_decl_visitor visitor(m_bld, m_view, scope);
  auto op = visitor.visit(decl);

  // Create the lock level attribute.

  auto lock_level_twine = llvm::Twine(std::to_string(lock_level));
  auto lock_level_attr = mlir::StringAttr::get(&mctx, lock_level_twine);

  // Attach the present attributes to the operation. Because one function may be
  // annotaed with several RCU attributes (though I'm not sure if any actually
  // are), we name each annotation after its attribute so that the attributes
  // are unique.

  // TODO(Brent): Find out how to do this by iterating the types AcquiresAttr,
  // ReleasesAttr, and MustHoldAttr. This would make the code less repetitive
  // and error-prone.

  if (source_text.contains(AcquiresAttr::getMnemonic())) {
    op->setAttr(AcquiresAttr::getMnemonic(),
                AcquiresAttr::get(&mctx, lock_level_attr));
  }
  if (source_text.contains(ReleasesAttr::getMnemonic())) {
    op->setAttr(ReleasesAttr::getMnemonic(),
                ReleasesAttr::get(&mctx, lock_level_attr));
  }
  if (source_text.contains(MustHoldAttr::getMnemonic())) {
    op->setAttr(MustHoldAttr::getMnemonic(),
                MustHoldAttr::get(&mctx, lock_level_attr));
  }

  return op;
}

void kernel_visitor::set_lock_level(mlir::Operation &op) {
  op.setAttr("lock_level", m_bld.getI64IntegerAttr(lock_level));
}

void kernel_visitor::lock_op(mlir::Operation &op) {
  set_lock_level(op);
  lock_level++;
}

void kernel_visitor::unlock_op(mlir::Operation &op) {
  lock_level--;
  set_lock_level(op);
}
} // namespace macroni::kernel
