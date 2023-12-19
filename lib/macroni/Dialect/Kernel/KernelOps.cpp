// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "macroni/Dialect/Kernel/KernelOps.hpp"
#include "macroni/Dialect/Kernel/KernelDialect.hpp"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/FunctionImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace macroni::kernel {

void ListForEach::build(mlir::OpBuilder &odsBuilder,
                        mlir::OperationState &odsState, mlir::Value pos,
                        mlir::Value head,
                        std::unique_ptr<mlir::Region> &&region) {
  mlir::OpBuilder::InsertionGuard guard(odsBuilder);
  odsState.addOperands(pos);
  odsState.addOperands(head);
  odsState.addRegion(std::move(region));
}
} // namespace macroni::kernel

#define GET_OP_CLASSES
#include "macroni/Dialect/Kernel/Kernel.cpp.inc"