#pragma once

#include "vast/Util/Common.hpp"
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <mlir/Pass/Pass.h>

namespace macroni::kernel {
struct kernel_analysis_pass
    : public mlir::PassWrapper<kernel_analysis_pass,
                               mlir::OperationPass<vast::vast_module>> {
  llvm::StringRef getName() const override;

  std::unique_ptr<mlir::Pass> clonePass() const override;

  llvm::StringRef getArgument() const override;

  void runOnOperation() override;
};

void register_kernel_analysis_pass();
} // namespace macroni::kernel
