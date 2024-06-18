#pragma once

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace macroni::kernel {
mlir::LogicalResult rewrite_label_stmt(vast::hl::LabelStmt label_stmt,
                                       mlir::PatternRewriter &rewriter);

mlir::LogicalResult rewrite_rcu_read_unlock(vast::hl::CallOp call_op,
                                            mlir::PatternRewriter &rewriter);
} // namespace macroni::kernel
