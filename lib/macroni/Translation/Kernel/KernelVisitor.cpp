#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include "macroni/Common/EmptyVisitor.hpp"
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
#include <clang/Basic/LLVM.h>
#include <mlir/IR/Operation.h>
#include <optional>

namespace macroni::kernel {
kernel_visitor::kernel_visitor(rcu_dereference_table &rcu_dereference_to_p,
                               vast::mcontext_t &mctx,
                               vast::cg::codegen_builder &bld,
                               vast::cg::meta_generator &mg,
                               vast::cg::symbol_generator &sg,
                               vast::cg::visitor_view view)
    : ::macroni::empty_visitor(mctx, mg, sg, view),
      m_rcu_dereference_to_p(rcu_dereference_to_p), m_bld(bld), m_view(view) {}

vast::operation kernel_visitor::visit(const vast::cg::clang_stmt *stmt,
                                      vast::cg::scope_context &scope) {
  return visit_rcu_dereference(stmt, scope).value_or(nullptr);
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

// template <typename Derived>
// struct KernelCodeGenVisitorMixin
//     : macroni::MacroniCodeGenVisitorMixin<Derived> {

//   using parent_t = macroni::MacroniCodeGenVisitorMixin<Derived>;

//   using parent_t::mlir_builder;
//   using parent_t::stmt_visitor::lens::acontext;
//   using parent_t::stmt_visitor::lens::mcontext;

//   using parent_t::Visit;

//   std::int64_t lock_level = 0;

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

//   void UnalignedStmtVisited(pasta::Stmt &pasta_stmt, mlir::Operation *op) {
//     if (auto call_expr = pasta::CallExpr::From(pasta_stmt)) {
//       VisitCallExpr(*call_expr, op);
//     }
//   }

//   void set_lock_level(mlir::Operation *op) {
//     op->setAttr("lock_level", mlir_builder().getI64IntegerAttr(lock_level));
//   }

//   void lock_op(mlir::Operation *op) {
//     set_lock_level(op);
//     lock_level++;
//   }

//   void unlock_op(mlir::Operation *op) {
//     lock_level--;
//     set_lock_level(op);
//   }

//   void VisitCallExpr(pasta::CallExpr &call_expr, mlir::Operation *op) {
//     auto call_op = mlir::dyn_cast_or_null<vast::hl::CallOp>(op);
//     if (!call_op) {
//       return;
//     }

//     auto name = call_op.getCalleeAttr().getValue();
//     if (name == macroni::kernel::KernelDialect::rcu_read_lock()) {
//       lock_op(op);
//     } else if (name == macroni::kernel::KernelDialect::rcu_read_unlock()) {
//       unlock_op(op);
//     }
//   }
// };
