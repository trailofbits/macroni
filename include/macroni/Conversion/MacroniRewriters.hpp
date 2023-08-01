#pragma once

#include <macroni/Dialect/Kernel/KernelDialect.hpp>
#include <macroni/Dialect/Kernel/KernelOps.hpp>
#include <macroni/Dialect/Macroni/MacroniDialect.hpp>
#include <macroni/Dialect/Macroni/MacroniOps.hpp>
#include <mlir/Transforms/DialectConversion.h>
#include <vast/Dialect/HighLevel/HighLevelOps.hpp>

namespace macroni {
    using ME = macroni::MacroExpansion;
    using PR = mlir::PatternRewriter;
    mlir::LogicalResult rewrite_get_user(ME exp, PR &rewriter);
    mlir::LogicalResult rewrite_offsetof(ME exp, PR &rewriter);
    mlir::LogicalResult rewrite_container_of(ME exp, PR &rewriter);
    mlir::LogicalResult rewrite_rcu_dereference(ME exp, PR &rewriter);
    mlir::LogicalResult rewrite_smp_mb(ME exp, PR &rewriter);

    using FO = vast::hl::ForOp;
    mlir::LogicalResult rewrite_list_for_each(FO for_op, PR &rewriter);

    using CO = vast::hl::CallOp;
    mlir::LogicalResult rewrite_rcu_read_unlock(CO call_op, PR &rewriter);
    
    using IO = vast::hl::IfOp;
    mlir::LogicalResult rewrite_safe_unsafe(IO if_op, PR &rewriter);

} // namespace macroni
