#include "macroni/Analysis/Kernel/KernelAnalysisPass.hpp"
#include "macroni/Analysis/Kernel/RCUAnalysis.hpp"
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace macroni::kernel {
llvm::StringRef kernel_analysis_pass::getName() const { return "KernelCheck"; }

std::unique_ptr<mlir::Pass> kernel_analysis_pass::clonePass() const {
  return std::make_unique<kernel_analysis_pass>(*this);
}

llvm::StringRef kernel_analysis_pass::getArgument() const {
  return "kernelcheck";
}

void kernel_analysis_pass::runOnOperation() {
  // Mark all analyses as preserved. We do this to allow for caching analysis
  // results. This is only valid because this operation pass does not perform
  // any transformations.
  markAllAnalysesPreserved();
  getAnalysis<rcu_analysis>();
}

void register_kernel_analysis_pass() {
  mlir::PassRegistration<kernel_analysis_pass>();
}
} // namespace macroni::kernel
