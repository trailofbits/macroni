#pragma once

#include <macroni/Dialect/Macroni/MacroniDialect.hpp>
#include <macroni/Dialect/Macroni/MacroniOps.hpp>
#include <macroni/Dialect/Safety/SafetyDialect.hpp>
#include <macroni/Dialect/Safety/SafetyOps.hpp>
#include <mlir/Transforms/DialectConversion.h>
#include <vast/Dialect/HighLevel/HighLevelOps.hpp>

namespace macroni::safety {
    using PR = mlir::PatternRewriter;

    using IO = vast::hl::IfOp;
    mlir::LogicalResult rewrite_unsafe(IO if_op, PR &rewriter);
} // namespace macroni::safety
