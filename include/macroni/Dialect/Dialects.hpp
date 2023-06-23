// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "mlir/IR/Dialect.h"

#include "macroni/Dialect/Macroni/MacroniDialect.hpp"

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