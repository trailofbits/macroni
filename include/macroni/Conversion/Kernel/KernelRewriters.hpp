#pragma once

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace macroni::kernel {
// mlir::LogicalResult rewrite_get_user(macroni::MacroExpansion exp,
//                                      mlir::PatternRewriter &rewriter);

// mlir::LogicalResult rewrite_offsetof(macroni::MacroExpansion exp,
//                                      mlir::PatternRewriter &rewriter);

// mlir::LogicalResult rewrite_container_of(macroni::MacroExpansion exp,
//                                          mlir::PatternRewriter &rewriter);

// mlir::LogicalResult rewrite_rcu_dereference(macroni::MacroExpansion exp,
//                                             mlir::PatternRewriter &rewriter);

// mlir::LogicalResult
// rewrite_rcu_dereference_check(macroni::MacroExpansion exp,
//                               mlir::PatternRewriter &rewriter);

// mlir::LogicalResult rewrite_smp_mb(macroni::MacroExpansion exp,
//                                    mlir::PatternRewriter &rewriter);

// mlir::LogicalResult rewrite_rcu_access_pointer(macroni::MacroExpansion exp,
//                                                mlir::PatternRewriter
//                                                &rewriter);

// mlir::LogicalResult rewrite_rcu_assign_pointer(macroni::MacroExpansion exp,
//                                                mlir::PatternRewriter
//                                                &rewriter);

// mlir::LogicalResult
// rewrite_rcu_replace_pointer(macroni::MacroExpansion exp,
//                             mlir::PatternRewriter &rewriter);

// mlir::LogicalResult rewrite_list_for_each(vast::hl::ForOp for_op,
//                                           mlir::PatternRewriter &rewriter);

mlir::LogicalResult rewrite_label_stmt(vast::hl::LabelStmt label_stmt,
                                       mlir::PatternRewriter &rewriter);

mlir::LogicalResult rewrite_rcu_read_unlock(vast::hl::CallOp call_op,
                                            mlir::PatternRewriter &rewriter);
} // namespace macroni::kernel
