#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include "macroni/Common/EmptyVisitor.hpp"
#include "macroni/Dialect/Kernel/KernelDialect.hpp"
#include "macroni/Dialect/Kernel/KernelOps.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/DefaultStmtVisitor.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/SymbolGenerator.hpp"
#include "vast/Util/Common.hpp"
#include <clang/AST/Attrs.inc>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/LLVM.h>
#include <mlir/IR/Operation.h>
#include <optional>

namespace macroni::kernel {
kernel_visitor::kernel_visitor(
    rcu_dereference_table &rcu_dereference_to_p,
    rcu_assign_pointer_table &rcu_assign_pointer_params,
    rcu_access_pointer_table &rcu_access_pointer_to_p,
    rcu_replace_pointer_table &rcu_replace_pointer_to_params,
    vast::mcontext_t &mctx, vast::cg::codegen_builder &bld,
    vast::cg::meta_generator &mg, vast::cg::symbol_generator &sg,
    vast::cg::visitor_view view)
    : ::macroni::empty_visitor(mctx, mg, sg, view),
      m_rcu_dereference_to_p(rcu_dereference_to_p),
      m_rcu_assign_pointer_params(rcu_assign_pointer_params),
      m_rcu_access_pointer_to_p(rcu_access_pointer_to_p),
      m_rcu_replace_pointer_to_params(rcu_replace_pointer_to_params),
      m_bld(bld), m_view(view) {}

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

vast::mlir_type kernel_visitor::visit(vast::cg::clang_qual_type type,
                                      vast::cg::scope_context &scope) {
  return {};
}

vast::mlir_attr kernel_visitor::visit(const vast::cg::clang_attr *attr,
                                      vast::cg::scope_context &scope) {
  return {};
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

bool kernel_visitor::is_context_attr(const clang::AnnotateAttr *attr) {
  return false;
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

//   bool is_context_attr(const clang::AnnotateAttr *attr) {
//     return attr->getAttrName()->getName() == "context" &&
//            attr->args_size() == 3;
//   }

//   mlir::Attribute Visit(const clang::Attr *attr) {
//     if (const auto *anno = clang::dyn_cast<clang::AnnotateAttr>(attr);
//         anno && is_context_attr(anno)) {
//       const auto *context_arg_1 = *anno->args_begin();
//       const auto *context_arg_2 =
//       clang::dyn_cast_or_null<clang::ConstantExpr>(
//           *std::next(anno->args_begin(), 1));
//       const auto *context_arg_3 =
//       clang::dyn_cast_or_null<clang::ConstantExpr>(
//           *std::next(anno->args_begin(), 2));
//       if (!(context_arg_1 && context_arg_2 && context_arg_3)) {
//         return parent_t::attr_visitor::Visit(attr);
//       }
//       std::string lock;
//       auto os = llvm::raw_string_ostream(lock);
//       context_arg_1->printPretty(os, nullptr,
//       acontext().getPrintingPolicy()); auto lock_name =
//       mlir::StringAttr::get(&mcontext(), llvm::Twine(lock));

//       auto starts_with_lock = context_arg_2->getResultAsAPSInt();
//       auto ends_with_lock = context_arg_3->getResultAsAPSInt();
//       if (starts_with_lock == 1 && ends_with_lock == 1) {
//         return macroni::kernel::MustHoldAttr::get(&mcontext(), lock_name);
//       } else if (starts_with_lock == 0 && ends_with_lock == 1) {
//         return macroni::kernel::AcquiresAttr::get(&mcontext(), lock_name);
//       } else if (starts_with_lock == 1 && ends_with_lock == 0) {
//         return macroni::kernel::ReleasesAttr::get(&mcontext(), lock_name);
//       }
//     }
//     return parent_t::attr_visitor::Visit(attr);
//   }

//   mlir::Type Visit(clang::QualType type) {
//     auto ty = parent_t::type_visitor::Visit(type);
//     auto attributed_type = clang::dyn_cast<clang::AttributedType>(type);
//     if (!attributed_type) {
//       return ty;
//     }
//     auto attr = attributed_type->getAttr();
//     auto addr_space = clang::dyn_cast_or_null<clang::AddressSpaceAttr>(attr);
//     if (!addr_space) {
//       return ty;
//     }
//     // NOTE(bpp): Clang does not record to address space passed to the
//     attribute
//     // in the source code. Instead, it records the value passed PLUS the
//     value
//     // of the last enumerator in Clang's LangAS enum. So to get the original
//     // value, we just subtract this enumerator's value from the value
//     attached
//     // to the AddressSpaceAttr.
//     using raw_t = std::underlying_type_t<clang::LangAS>;
//     int first = static_cast<raw_t>(clang::LangAS::FirstTargetAddressSpace);
//     int space = addr_space->getAddressSpace() - first;
//     return ::macroni::kernel::AddressSpaceType::get(&mcontext(), ty, space);
//   }
