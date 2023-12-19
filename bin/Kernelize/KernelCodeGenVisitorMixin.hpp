#pragma once

#define GET_TYPEDEF_CLASSES 1

#include <macroni/Dialect/Kernel/KernelDialect.hpp>
#include <macroni/Dialect/Kernel/KernelTypes.hpp>
#include <macroni/Translation/MacroniCodeGenVisitorMixin.hpp>
#include <macroni/Translation/MacroniMetaGenerator.hpp>
#include <pasta/AST/Stmt.h>
#include <stdint.h>
#include <vast/CodeGen/CodeGen.hpp>
#include <vast/CodeGen/FallBackVisitor.hpp>
#include <vast/CodeGen/UnreachableVisitor.hpp>
#include <vast/CodeGen/UnsupportedVisitor.hpp>

template <typename Derived>
struct KernelCodeGenVisitorMixin
    : macroni::MacroniCodeGenVisitorMixin<Derived> {

    using parent_t = macroni::MacroniCodeGenVisitorMixin<Derived>;

    using parent_t::mlir_builder;
    using parent_t::stmt_visitor::lens::mcontext;

    using parent_t::Visit;

    std::int64_t lock_level = 0;

    mlir::Type Visit(clang::QualType type) {
        auto ty = parent_t::type_visitor_with_dl::Visit(type);
        auto attributed_type = clang::dyn_cast<clang::AttributedType>(type);
        if (!attributed_type) {
            return ty;
        }
        auto attr = attributed_type->getAttr();
        using ASA = clang::AddressSpaceAttr;
        auto addr_space = clang::dyn_cast_or_null<ASA>(attr);
        if (!addr_space) {
            return ty;
        }
        // NOTE(bpp): Clang does not record to address space passed to the
        // attribute in the source code. Instead, it records the value passed
        // PLUS the value of the last enumerator in Clang's LangAS enum. So to
        // get the original value, we just subtract this enumerator's value from
        // the value attached to the AddressSpaceAttr.
        using clang::LangAS;
        using std::underlying_type_t;
        auto FirstAddrSpace = LangAS::FirstTargetAddressSpace;
        int first = static_cast<underlying_type_t<LangAS>>(FirstAddrSpace);
        int space = addr_space->getAddressSpace() - first;
        return ::macroni::kernel::AddressSpaceType::get(&mcontext(), ty, space);
    }

    void UnalignedStmtVisited(pasta::Stmt &pasta_stmt,
                              mlir::Operation *op) {
        if (auto call_expr = pasta::CallExpr::From(pasta_stmt)) {
            VisitCallExpr(*call_expr, op);
        }
    }

    void set_lock_level(mlir::Operation *op) {
        op->setAttr("lock_level", mlir_builder().getI64IntegerAttr(lock_level));
    }

    void VisitCallExpr(pasta::CallExpr &call_expr, mlir::Operation *op) {
        auto call_op = mlir::dyn_cast<vast::hl::CallOp>(op);
        if (!call_op) {
            return;
        }
        auto name = call_op.getCalleeAttr().getValue();
        if (name == macroni::kernel::KernelDialect::rcu_read_lock()) {
            set_lock_level(op);
            lock_level++;
        } else if (name == macroni::kernel::KernelDialect::rcu_read_unlock()) {
            lock_level--;
            set_lock_level(op);
        }
    }
};

template<typename Derived>
using KernelVisitorConfig = vast::cg::fallback_visitor<Derived,
    KernelCodeGenVisitorMixin,
    vast::cg::unsup_visitor,
    vast::cg::unreach_visitor
>;

using KernelCodeGen = vast::cg::codegen_instance<KernelVisitorConfig>;