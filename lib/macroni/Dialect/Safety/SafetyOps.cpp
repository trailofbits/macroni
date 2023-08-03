// Copyright (c) 2023-present, Trail of Bits, Inc.


#include "macroni/Dialect/Safety/SafetyDialect.hpp"
#include "macroni/Dialect/Safety/SafetyOps.hpp"

#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/FunctionImplementation.h>
#include <llvm/Support/ErrorHandling.h>

namespace macroni::safety {

}

#define GET_OP_CLASSES
#include "macroni/Dialect/Safety/Safety.cpp.inc"