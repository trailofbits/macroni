#pragma once

#include <macroni/Translation/MacroniCodeGenVisitorMixin.hpp>
#include <macroni/Translation/MacroniMetaGenerator.hpp>
#include <pasta/AST/Stmt.h>
#include <stdint.h>

template <typename Derived>
struct KernelCodeGenVisitorMixin
    : macroni::MacroniCodeGenVisitorMixin<Derived> {

    using parent_t = macroni::MacroniCodeGenVisitorMixin<Derived>;

    using parent_t::builder;
    using parent_t::StmtVisitor::LensType::mcontext;

    using parent_t::Visit;

    std::int64_t lock_level = 0;

    mlir::Type Visit(clang::QualType type) {
        auto ty = parent_t::TypeVisitor::Visit(type);
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
        // attribute in the source code. Instead, it record the value passed
        // PLUS the value of the last enumerator in Clang's LangAS enum. So
        // to get the original value, we just subtract this enumerator's
        // value from the value attached to the AddressSpaceAttr.
        using clang::LangAS;
        using std::underlying_type_t;
        auto FirstAddrSpace = LangAS::FirstTargetAddressSpace;
        int first = static_cast<underlying_type_t<LangAS>>(FirstAddrSpace);
        int space = addr_space->getAddressSpace() - first;
        return macroni::kernel::AddressSpaceType::get(&mcontext(), ty, space);
    }

    void UnalignedStmtVisited(pasta::Stmt &pasta_stmt,
                              mlir::Operation *op) override {
        if (auto call_expr = pasta::CallExpr::From(pasta_stmt)) {
            VisitCallExpr(*call_expr, op);
        }
    }

    void set_lock_level(mlir::Operation *op) {
        op->setAttr("lock_level", builder().getI64IntegerAttr(lock_level));
    }

    void VisitCallExpr(pasta::CallExpr &call_expr, mlir::Operation *op) {
        auto call_op = mlir::dyn_cast<vast::hl::CallOp>(op);
        if (!call_op) {
            return;
        }
        auto name = call_op.getCalleeAttr().getValue();
        if ("rcu_read_lock" == name) {
            set_lock_level(op);
            lock_level++;
        } else if ("rcu_read_unlock" == name) {
            lock_level--;
            set_lock_level(op);
        }
    }
};

template<typename Derived>
using KernelVisitorConfig = vast::cg::CodeGenFallBackVisitorMixin<Derived,
    KernelCodeGenVisitorMixin, vast::cg::DefaultFallBackVisitorMixin>;

using KernelVisitor = vast::cg::CodeGenVisitor<KernelVisitorConfig,
    macroni::MacroniMetaGenerator>;