// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/IR/Dialect.h"
VAST_UNRELAX_WARNINGS

#include "macroni/Dialect/Macroni/MacroniDialect.hpp"

#include "vast/Util/Common.hpp"

namespace macroni {

    // inline void registerAllDialects(mlir::DialectRegistry &registry) {
    //     registry.insert<macroni::MacroniDialect>();
    // }

    // inline void registerAllDialects(mlir::MLIRContext &mctx) {
    //     mlir::DialectRegistry registry;
    //     registerAllDialects(registry);
    //     mctx.appendDialectRegistry(registry);
    // }

} // namespace macroni