#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include "macroni/Common/EmptyVisitor.hpp"
#include "macroni/Common/ExpansionTable.hpp"
#include "macroni/Common/MacroniMetaGenerator.hpp"
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
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Lexer.h>
#include <llvm/ADT/Twine.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <string>
#include <string_view>

namespace macroni::kernel {
kernel_visitor::kernel_visitor(expansion_table &expansions,
                               vast::acontext_t &actx, vast::mcontext_t &mctx,
                               vast::cg::codegen_builder &bld,
                               vast::cg::meta_generator &mg,
                               vast::cg::symbol_generator &sg,
                               vast::cg::visitor_view view)
    : ::macroni::empty_visitor(mctx, mg, sg, view), m_expansions(expansions),
      m_actx(actx), m_bld(bld), m_view(view) {}

vast::operation kernel_visitor::visit(const vast::cg::clang_stmt *stmt,
                                      vast::cg::scope_context &scope) {
  // If we have started visiting a new function body, reset the current lock
  // level.
  if (m_function_bodies.contains(stmt)) {
    lock_level = 0;
  }
  if (!m_expansions.contains(stmt)) {
    return visit_rcu_read_lock_or_unlock(stmt, scope);
  }

  auto &meta = static_cast<macroni::macroni_meta_generator &>(mg);
  auto &sm = m_actx.getSourceManager();
  auto loc = meta.location(sm.getExpansionLoc(stmt->getBeginLoc()));
  auto expansion = m_expansions.at(stmt);
  auto rty = [&] -> vast::mlir_type {
    if (auto expr = clang::dyn_cast<clang::Expr>(stmt)) {
      return m_view.visit(expr->getType(), scope);
    }
    return m_bld.void_type();
  }();

  auto name = expansion.spelling.name;
  auto num_args = expansion.arguments.size();
  if (1 == num_args) {
    auto p = m_view.visit(expansion.arguments[0], scope)->getResult(0);

    if (KernelDialect::rcu_access_pointer.name == name) {
      return m_bld.create<RCUAccessPointer>(loc, rty, p);
    }
    if (KernelDialect::rcu_dereference.name == name) {
      return m_bld.create<RCUDereference>(loc, rty, p);
    }
    if (KernelDialect::rcu_dereference_bh.name == name) {
      return m_bld.create<RCUDereferenceBH>(loc, rty, p);
    }
    // if (KernelDialect::rcu_dereference_sched.name == name)
    return m_bld.create<RCUDereferenceSched>(loc, rty, p);
  }
  if (2 == num_args) {
    // KernelDialect::rcu_assign_pointer.name == name
    auto p = m_view.visit(expansion.arguments[0], scope)->getResult(0);
    auto v = m_view.visit(expansion.arguments[1], scope)->getResult(0);
    return m_bld.create<RCUAssignPointer>(loc, rty, p, v);
  }
  if (3 == num_args) {
    // KernelDialect::rcu_replace_pointer.name == name
    auto rcu_ptr = m_view.visit(expansion.arguments[0], scope)->getResult(0);
    auto ptr = m_view.visit(expansion.arguments[1], scope)->getResult(0);
    auto c = m_view.visit(expansion.arguments[2], scope)->getResult(0);
    return m_bld.create<RCUReplacePointer>(loc, rty, rcu_ptr, ptr, c);
  }
  return nullptr;
}

vast::operation
kernel_visitor::visit_rcu_read_lock_or_unlock(const vast::cg::clang_stmt *stmt,
                                              vast::cg::scope_context &scope) {
  auto call_expr = clang::dyn_cast<clang::CallExpr>(stmt);
  if (!call_expr) {
    return {};
  }

  auto direct_callee = call_expr->getDirectCallee();
  if (!direct_callee) {
    return {};
  }

  auto name = direct_callee->getName();
  if (KernelDialect::rcu_read_lock() != name &&
      KernelDialect::rcu_read_unlock() != name) {
    return {};
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

template <typename AttrT, typename... Rest>
void annotate_op_with_attrs_in_text(vast::operation op, std::string_view text,
                                    AttrT attr, Rest... rest) {
  if (text.contains(AttrT::getMnemonic())) {
    op->setAttr(AttrT::getMnemonic(), attr);
  }
  if constexpr (sizeof...(Rest) != 0) {
    annotate_op_with_attrs_in_text(op, text, rest...);
  }
}

vast::operation kernel_visitor::visit(const vast::cg::clang_decl *decl,
                                      vast::cg::scope_context &scope) {
  auto function_decl = clang::dyn_cast<clang::FunctionDecl>(decl);
  if (!function_decl || !function_decl->hasBody()) {
    return nullptr;
  }
  // Keep track of function decl bodies so that we can reset the lock level when
  // the stmt visitor visits them. We can't reset the lock level here when
  // visiting the function decl because of the way vast visitors visit the AST:
  // They visit the decls first, and then then statements. If vast visitors
  // visited the AST using plain DFS then this wouldn't be an issue and we could
  // just set the lock level here whenever we visit a function decl.
  auto body = function_decl->getBody();
  m_function_bodies.insert(body);

  // Get the source text of this function declaration so we can check if it
  // contains an RCU annotation. The RCU annotations (__acquires(),
  // __releases(), and __must_hold()) are not standard so Clang will not embed
  // them in the AST, so we must check for their presence in the source text
  // instead.

  auto &sm = m_actx.getSourceManager();
  auto &lo = m_actx.getLangOpts();
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

  annotate_op_with_attrs_in_text(op, source_text,
                                 AcquiresAttr::get(&mctx, lock_level_attr),
                                 ReleasesAttr::get(&mctx, lock_level_attr),
                                 MustHoldAttr::get(&mctx, lock_level_attr));

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
