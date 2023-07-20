#pragma once

#include <macroni/Dialect/Kernel/KernelDialect.hpp>
#include <macroni/Dialect/Kernel/KernelOps.hpp>
#include <macroni/Dialect/Macroni/MacroniDialect.hpp>
#include <macroni/Dialect/Macroni/MacroniOps.hpp>
#include <mlir/Transforms/DialectConversion.h>
#include <vast/Dialect/HighLevel/HighLevelOps.hpp>

namespace macroni {
    bool has_results(mlir::Operation *op);
    bool is_get_user(macroni::MacroExpansion &exp);
    bool is_offsetof(macroni::MacroExpansion &exp);
    bool is_container_of(macroni::MacroExpansion &exp);
    bool is_rcu_dereference(macroni::MacroExpansion &exp);
    bool is_smp_mb(macroni::MacroExpansion &exp);

    using ME = macroni::MacroExpansion;
    using PR = mlir::PatternRewriter;
    mlir::LogicalResult rewrite_get_user(ME exp, PR &rewriter);
    mlir::LogicalResult rewrite_offsetof(ME exp, PR &rewriter);
    mlir::LogicalResult rewrite_container_of(ME exp, PR &rewriter);
    mlir::LogicalResult rewrite_rcu_dereference(ME exp, PR &rewriter);
    mlir::LogicalResult rewrite_smp_mb(ME exp, PR &rewriter);

    using FO = vast::hl::ForOp;
    mlir::LogicalResult rewrite_list_for_each(FO for_op, PR &rewriter);

    llvm::APInt get_lock_level(mlir::Operation *op);
    using CO = vast::hl::CallOp;
    mlir::LogicalResult rewrite_rcu_read_unlock(CO call_op, PR &rewriter);

} // namespace macroni
