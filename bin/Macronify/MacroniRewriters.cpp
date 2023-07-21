#include "MacroniRewriters.hpp"

namespace macroni {
    bool is_get_user(macroni::MacroExpansion exp) {
        return exp.getNumResults() > 0 &&
            exp.getParameterNames().size() == 2 &&
            exp.getMacroName() == "get_user";
    }

    bool is_offsetof(macroni::MacroExpansion exp) {
        return exp.getNumResults() > 0 &&
            exp.getParameterNames().size() == 2 &&
            exp.getMacroName() == "offsetof";
    }

    bool is_container_of(macroni::MacroExpansion exp) {
        return exp.getNumResults() > 0 &&
            exp.getParameterNames().size() == 3 &&
            exp.getMacroName() == "container_of";
    }

    bool is_rcu_dereference(macroni::MacroExpansion exp) {
        return exp.getNumResults() > 0 &&
            exp.getParameterNames().size() == 1 &&
            exp.getMacroName() == "rcu_dereference";
    }

    bool is_smp_mb(macroni::MacroExpansion exp) {
        return exp.getNumResults() > 0 &&
            exp.getParameterNames().size() == 0 &&
            exp.getMacroName() == "smp_mb";
    }

    mlir::LogicalResult rewrite_get_user(
        macroni::MacroExpansion exp,
        mlir::PatternRewriter &rewriter) {
        if (!is_get_user(exp)) {
            return mlir::failure();
        }
        mlir::Operation *x = nullptr;
        mlir::Operation *ptr = nullptr;
        exp.getExpansion().walk([&](mlir::Operation *op) {
            if (auto mp = mlir::dyn_cast<macroni::MacroParameter>(op)) {
                auto name = mp.getParameterName();
                if (!x && name == "x") {
                    x = op;
                } else if (!ptr && name == "ptr") {
                    ptr = op;
                }
            }});
        auto is_legal = (x &&
                         ptr &&
                         x->getNumResults() == 1 &&
                         ptr->getNumResults() == 1);
        if (!is_legal) {
            return mlir::failure();
        }

        // The x and ptr macro parameters are expanded inside of the
        // expansion of get_user(), so if we simply replaced the original
        // macro expansion with the GetUser kernel operation, we would lose
        // the arguments. Therefore we clone the arguments to lift them out
        // of the macro expansion, and pass them to the GetUser operation.
        // NOTE(bpp): Maybe we should create an operation for lifted macro
        // parameters to make it possible to distinguish normal macro
        // parameters from ones that have been lifted?
        auto x_clone = rewriter.clone(*x);
        auto ptr_clone = rewriter.clone(*ptr);
        auto result_type = exp.getType(0);
        auto x_res = x_clone->getResult(0);
        auto ptr_res = ptr_clone->getResult(0);
        using GE = ::macroni::kernel::GetUser;
        rewriter.replaceOpWithNewOp<GE>(exp, result_type, x_res, ptr_res);
        return mlir::success();
    }

    mlir::LogicalResult rewrite_offsetof(
        macroni::MacroExpansion exp,
        mlir::PatternRewriter &rewriter) {
        if (!is_offsetof(exp)) {
            return mlir::failure();
        }
        std::optional<mlir::TypeAttr> type;
        std::optional<mlir::StringAttr> member;
        exp.getExpansion().walk([&](mlir::Operation *op) {
            if (auto member_op = mlir::dyn_cast<vast::hl::RecordMemberOp>(op)) {
                auto record = member_op.getRecord();
                auto record_ty = record.getType();
                using PT = vast::hl::PointerType;
                if (auto pty = mlir::dyn_cast<PT>(record_ty)) {
                    auto element_ty = pty.getElementType();
                    auto ty_attr = mlir::TypeAttr::get(element_ty);
                    type = ty_attr;
                } else {
                    auto ty_attr = mlir::TypeAttr::get(record_ty);
                    type = ty_attr;
                }
                member = member_op.getNameAttr();
            }});
        auto is_legal = type && member;
        if (!is_legal) {
            return mlir::failure();
        }

        auto op = exp.getOperation();
        auto result_type = exp.getType(0);
        using OO = ::macroni::kernel::OffsetOf;
        rewriter.replaceOpWithNewOp<OO>(op, result_type, *type, *member);
        return mlir::success();
    }

    mlir::LogicalResult rewrite_container_of(
        macroni::MacroExpansion exp,
        mlir::PatternRewriter &rewriter) {
        if (!is_container_of(exp)) {
            return mlir::failure();
        }
        mlir::Operation *ptr = nullptr;
        std::optional<mlir::TypeAttr> type;
        std::optional<mlir::StringAttr> member;
        using RMO = vast::hl::RecordMemberOp;
        exp.getExpansion().walk([&](mlir::Operation *op) {
            if (auto mp = mlir::dyn_cast<macroni::MacroParameter>(op)) {
                auto param_name = mp.getParameterName();
                if ("ptr" == param_name) {
                    ptr = op;
                }
            } else if (auto member_op = mlir::dyn_cast<RMO>(op)) {
                auto record = member_op.getRecord();
                auto record_ty = record.getType();
                using PT = vast::hl::PointerType;
                if (auto pty = mlir::dyn_cast<PT>(record_ty)) {
                    auto elem_ty = pty.getElementType();
                    auto ty_attr = mlir::TypeAttr::get(elem_ty);
                    type = ty_attr;
                } else {
                    auto ty_attr = mlir::TypeAttr::get(record_ty);
                    type = ty_attr;
                }
                member = member_op.getNameAttr();
            }});
        auto is_legal = ptr && type && member;
        if (!is_legal) {
            return mlir::failure();
        }

        auto op = exp.getOperation();
        auto ptr_clone = rewriter.clone(*ptr);
        auto exp_ty = exp.getType(0);
        auto ptr_res = ptr_clone->getResult(0);

        using CO = ::macroni::kernel::ContainerOf;
        rewriter.replaceOpWithNewOp<CO>(op, exp_ty, ptr_res, *type, *member);
        return mlir::success();
    }

    mlir::LogicalResult rewrite_rcu_dereference(
        macroni::MacroExpansion exp,
        mlir::PatternRewriter &rewriter) {
        if (!is_rcu_dereference(exp)) {
            return mlir::failure();
        }
        mlir::Operation *p = nullptr;
        exp.getExpansion().walk([&](mlir::Operation *op) {
            if (auto param = mlir::dyn_cast<macroni::MacroParameter>(op)) {
                auto param_name = param.getParameterName();
                if (param_name == "p") {
                    p = op;
                }
            }});
        auto is_legal = p != nullptr;
        if (!is_legal) {
            return mlir::failure();
        }

        auto op = exp.getOperation();
        auto p_clone = rewriter.clone(*p);
        auto res_ty = exp.getType(0);
        auto p_res = p_clone->getResult(0);
        using RCUD = ::macroni::kernel::RCUDereference;
        rewriter.replaceOpWithNewOp<RCUD>(op, res_ty, p_res);
        return mlir::success();
    }

    mlir::LogicalResult rewrite_smp_mb(
        macroni::MacroExpansion exp,
        mlir::PatternRewriter &rewriter) {
        if (!is_smp_mb(exp)) {
            return mlir::failure();
        }

        auto op = exp.getOperation();
        auto res_ty = exp.getType(0);
        rewriter.replaceOpWithNewOp<kernel::SMPMB>(op, res_ty);
        return mlir::success();
    }

    mlir::LogicalResult rewrite_list_for_each(
        vast::hl::ForOp for_op,
        mlir::PatternRewriter &rewriter) {
        mlir::Operation *pos = nullptr;
        mlir::Operation *head = nullptr;
        for_op.getCondRegion().walk(
            [&](mlir::Operation *op) {
                using MP = macroni::MacroParameter;
                if (auto param_op = mlir::dyn_cast<MP>(op)) {
                    if (param_op.getParameterName() == "pos") {
                        pos = op;
                    } else if (param_op.getParameterName() == "head") {
                        head = op;
                    }
                }
            }
        );
        auto is_legal = pos && head;
        if (!is_legal) {
            return mlir::failure();
        }

        auto op = for_op.getOperation();
        auto pos_clone = rewriter.clone(*pos);
        auto head_clone = rewriter.clone(*head);
        auto pos_res = pos_clone->getResult(0);
        auto head_res = head_clone->getResult(0);
        auto reg = std::make_unique<mlir::Region>();
        reg->takeBody(for_op.getBodyRegion());
        using LFE = ::macroni::kernel::ListForEach;
        rewriter.replaceOpWithNewOp<LFE>(op, pos_res, head_res, std::move(reg));
        return mlir::success();
    }

    llvm::APInt get_lock_level(mlir::Operation *op) {
        using IA = mlir::IntegerAttr;
        return op->getAttrOfType<IA>("lock_level").getValue();
    }

    mlir::LogicalResult rewrite_rcu_read_unlock(
        vast::hl::CallOp call_op,
        mlir::PatternRewriter &rewriter) {
        auto name = call_op.getCalleeAttr().getValue();
        if ("rcu_read_unlock" != name) {
            return mlir::failure();
        }
        auto unlock_op = call_op.getOperation();
        auto unlock_level = get_lock_level(unlock_op);
        mlir::Operation *lock_op = nullptr;
        for (auto op = unlock_op; op; op = op->getPrevNode()) {
            if (auto call_op = mlir::dyn_cast<vast::hl::CallOp>(op)) {
                name = call_op.getCalleeAttr().getValue();
                if ("rcu_read_lock" == name) {
                    auto lock_level = get_lock_level(op);
                    if (unlock_level == lock_level) {
                        lock_op = op;
                        break;
                    }
                }
            }
        }
        auto is_legal = lock_op != nullptr;
        if (!is_legal) {
            return mlir::failure();
        }

        rewriter.setInsertionPointAfter(lock_op);
        using CS = kernel::RCUCriticalSection;
        auto cs = rewriter.replaceOpWithNewOp<CS>(lock_op);
        auto cs_block = rewriter.createBlock(&cs.getBodyRegion());
        for (auto op = cs->getNextNode(); op != unlock_op;) {
            auto temp = op->getNextNode();
            op->moveBefore(cs_block, cs_block->end());
            op = temp;
        }
        rewriter.eraseOp(unlock_op);
        return mlir::success();
    }

} // namespace macroni
