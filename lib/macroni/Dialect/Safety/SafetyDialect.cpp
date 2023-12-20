// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "macroni/Dialect/Safety/SafetyDialect.hpp"
#include "macroni/Dialect/Safety/SafetyOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/TypeSupport.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SMLoc.h>

namespace macroni::safety {
void SafetyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "macroni/Dialect/Safety/Safety.cpp.inc"
      >();
}

using DialectParser = mlir::AsmParser;
using DialectPrinter = mlir::AsmPrinter;

} // namespace macroni::safety

#include "macroni/Dialect/Safety/SafetyDialect.cpp.inc"