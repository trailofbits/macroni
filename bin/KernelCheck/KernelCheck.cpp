// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "macroni/Analysis/Kernel/KernelAnalysisPass.hpp"
#include "macroni/Dialect/Kernel/KernelDialect.hpp"
#include "vast/Dialect/Dialects.hpp"
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

// To run the kernel analysis, pass "--kernelcheck" to the kernelcheck binary.
int main(int argc, char **argv) {

  mlir::registerAllPasses();

  // Register dialects.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  vast::registerAllDialects(registry);
  registry.insert<macroni::kernel::KernelDialect>();

  // Register kernel check pass.
  macroni::kernel::register_kernel_analysis_pass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Kernel analyzer\n", registry));
}
