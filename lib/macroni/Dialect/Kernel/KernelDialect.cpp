// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "macroni/Dialect/Kernel/KernelDialect.hpp"
#include "macroni/Dialect/Kernel/KernelOps.hpp"

#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SMLoc.h>

namespace macroni::kernel
{
    void KernelDialect::initialize()
    {
        addOperations<
            #define GET_OP_LIST
            #include "macroni/Dialect/Kernel/Kernel.cpp.inc"
        >();
    }

    using DialectParser = mlir::AsmParser;
    using DialectPrinter = mlir::AsmPrinter;

} // namespace macroni::kernel

#include "macroni/Dialect/Kernel/KernelDialect.cpp.inc"