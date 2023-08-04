#pragma once

#include <macroni/Dialect/Macroni/MacroniDialect.hpp>
#include <macroni/Dialect/Macroni/MacroniOps.hpp>
#include <macroni/Dialect/Safety/SafetyDialect.hpp>
#include <macroni/Dialect/Safety/SafetyOps.hpp>
#include <mlir/Transforms/DialectConversion.h>
#include <vast/Dialect/HighLevel/HighLevelOps.hpp>

namespace macroni::safety {
    mlir::LogicalResult rewrite_unsafe(
        vast::hl::IfOp if_op,
        mlir::PatternRewriter &rewriter);
} // namespace macroni::safety
