#include "macroni/Dialect/Kernel/KernelDialect.hpp"
#include "macroni/Dialect/Kernel/KernelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Util/Common.hpp"
#include <llvm/ADT/APInt.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace macroni::kernel {
mlir::LogicalResult rewrite_label_stmt(vast::hl::LabelStmt label_stmt,
                                       mlir::PatternRewriter &rewriter) {
  // In the Linux Kernel, a common idiom is to call `rcu_read_unlock()` right
  // after declaring a label. This idiom prevents our `CallOp` pattern rewriter
  // from detecting such unlocks as the end of a critical section. To fix this,
  // we look for labels which are immediately followed by a call to
  // `rcu_read_unlock()`, and move the call before the label.

  auto ops = label_stmt.getOps();
  if (ops.empty()) {
    return mlir::failure();
  }

  auto call_op = mlir::dyn_cast<vast::hl::CallOp>(*ops.begin());
  if (!call_op ||
      call_op.getCalleeAttr().getValue() != KernelDialect::rcu_read_unlock()) {
    return mlir::failure();
  }

  // TODO(bpp): Keep track of which calls to `rcu_read_unlock()` were originally
  // nested under labels.
  rewriter.setInsertionPoint(label_stmt);
  rewriter.clone(*call_op.getOperation());
  rewriter.eraseOp(call_op);
  return mlir::success();
}

mlir::LogicalResult rewrite_rcu_read_unlock(vast::hl::CallOp call_op,
                                            mlir::PatternRewriter &rewriter) {
  auto name = call_op.getCalleeAttr().getValue();
  if (KernelDialect::rcu_read_unlock() != name) {
    return mlir::failure();
  }
  auto unlock_op = call_op.getOperation();
  mlir::Operation *lock_op = nullptr;
  for (auto op = unlock_op; op; op = op->getPrevNode()) {
    if (auto other_call_op = mlir::dyn_cast<vast::hl::CallOp>(op)) {
      name = other_call_op.getCalleeAttr().getValue();
      if (KernelDialect::rcu_read_lock() == name) {
        lock_op = op;
        break;
      }
    }
  }
  if (!lock_op) {
    return mlir::failure();
  }

  rewriter.setInsertionPointAfter(lock_op);
  auto cs = rewriter.replaceOpWithNewOp<RCUCriticalSection>(lock_op);
  auto cs_block = rewriter.createBlock(&cs.getBodyRegion());
  for (auto op = cs->getNextNode(); op != unlock_op;) {
    auto temp = op->getNextNode();
    op->moveBefore(cs_block, cs_block->end());
    op = temp;
  }
  rewriter.eraseOp(unlock_op);
  return mlir::success();
}

void rewrite_rcu(vast::mcontext_t *mctx, vast::vast_module &mod) {
  // Register conversions
  auto patterns = mlir::RewritePatternSet(mctx);
  patterns.add(rewrite_label_stmt).add(rewrite_rcu_read_unlock);

  // Apply the conversions
  auto frozen_pats = mlir::FrozenRewritePatternSet(std::move(patterns));
  mod->walk([&frozen_pats](mlir::Operation *op) {
    if (mlir::isa<vast::hl::ForOp, vast::hl::CallOp, vast::hl::LabelStmt>(op)) {
      std::ignore = mlir::applyOpPatternsAndFold(op, frozen_pats);
    }
  });
}
} // namespace macroni::kernel
