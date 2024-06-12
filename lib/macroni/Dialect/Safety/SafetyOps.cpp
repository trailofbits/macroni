// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "macroni/Dialect/Safety/SafetyOps.hpp"
#include "macroni/Dialect/Safety/SafetyDialect.hpp"
#include "vast/Util/Common.hpp"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace macroni::safety {
void build_region(vast::mlir_builder &bld, vast::op_state &st,
                  vast::builder_callback_ref callback) {
  bld.createBlock(st.addRegion());
  callback(bld, st.location);
}

void UnsafeRegion::build(mlir::OpBuilder &odsBuilder,
                         mlir::OperationState &odsState,
                         vast::builder_callback_ref callback) {
  mlir::OpBuilder::InsertionGuard guard(odsBuilder);
  build_region(odsBuilder, odsState, callback);
}
} // namespace macroni::safety

#define GET_OP_CLASSES
#include "macroni/Dialect/Safety/Safety.cpp.inc"