#include <macroni/Conversion/Safety/SafetyRewriters.hpp>

namespace macroni::safety {

    mlir::LogicalResult rewrite_unsafe(
        vast::hl::IfOp if_op,
        mlir::PatternRewriter &rewriter) {
        if (if_op->hasAttr("unsafe")) {
            using UR = ::macroni::safety::UnsafeRegion;
            auto unsafe_op = rewriter.create<UR>(if_op.getLoc());
            auto &reg = unsafe_op.getBodyRegion();
            rewriter.inlineRegionBefore(if_op.getThenRegion(), reg, reg.end());
            rewriter.eraseOp(if_op);
            return mlir::success();
        }
        return mlir::failure();
    }

} // namespace macroni::safety
