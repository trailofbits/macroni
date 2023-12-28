#pragma once

#include <macroni/Dialect/Kernel/KernelDialect.hpp>
#include <macroni/Dialect/Kernel/KernelOps.hpp>
#include <macroni/Dialect/Macroni/MacroniDialect.hpp>
#include <macroni/Dialect/Macroni/MacroniOps.hpp>
#include <mlir/Transforms/DialectConversion.h>
#include <vast/Dialect/HighLevel/HighLevelOps.hpp>

namespace macroni::kernel {
mlir::LogicalResult rewrite_get_user(macroni::MacroExpansion exp,
                                     mlir::PatternRewriter &rewriter);

mlir::LogicalResult rewrite_offsetof(macroni::MacroExpansion exp,
                                     mlir::PatternRewriter &rewriter);

mlir::LogicalResult rewrite_container_of(macroni::MacroExpansion exp,
                                         mlir::PatternRewriter &rewriter);

mlir::LogicalResult rewrite_rcu_dereference(macroni::MacroExpansion exp,
                                            mlir::PatternRewriter &rewriter);

mlir::LogicalResult
rewrite_rcu_dereference_check(macroni::MacroExpansion exp,
                              mlir::PatternRewriter &rewriter);

mlir::LogicalResult rewrite_smp_mb(macroni::MacroExpansion exp,
                                   mlir::PatternRewriter &rewriter);

mlir::LogicalResult rewrite_rcu_access_pointer(macroni::MacroExpansion exp,
                                               mlir::PatternRewriter &rewriter);

mlir::LogicalResult rewrite_list_for_each(vast::hl::ForOp for_op,
                                          mlir::PatternRewriter &rewriter);

mlir::LogicalResult rewrite_rcu_read_unlock(vast::hl::CallOp call_op,
                                            mlir::PatternRewriter &rewriter);

mlir::LogicalResult rewrite_label_stmt(vast::hl::LabelStmt label_stmt,
                                       mlir::PatternRewriter &rewriter);
} // namespace macroni::kernel
