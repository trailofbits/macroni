// Copyright (c) 2023-present, Trail of Bits, Inc.

#include <macroni/Dialect/Macroni/MacroniDialect.hpp>
#include <macroni/Dialect/Macroni/MacroniOps.hpp>

#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SMLoc.h>

namespace macroni::macroni
{
    void MacroniDialect::initialize()
    {
        addOperations<
            #define GET_OP_LIST
            #include "macroni/Dialect/Macroni/Macroni.cpp.inc"
        >();
    }

    using DialectParser = mlir::AsmParser;
    using DialectPrinter = mlir::AsmPrinter;

} // namespace macroni::macroni

#include "macroni/Dialect/Macroni/MacroniDialect.cpp.inc"