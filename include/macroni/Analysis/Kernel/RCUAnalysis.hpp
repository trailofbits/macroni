#pragma once

#include "macroni/Dialect/Kernel/KernelInterfaces.hpp"
#include "macroni/Dialect/Kernel/KernelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Operation.h>

namespace macroni::kernel {
struct rcu_analysis {
  rcu_analysis(mlir::Operation *op);

  virtual ~rcu_analysis(){};

protected:
  void warn_rcu_dereference(RCU_Dereference_Interface &op);
  void analyze_non_rcu_function(vast::hl::FuncOp &func);
  auto walk_until_call(llvm::StringRef name);
  void analyze_acquires_function(vast::hl::FuncOp &func);
  void analyze_releases_function(vast::hl::FuncOp &func);
  void analyze_critical_section(RCUCriticalSection cs);
  virtual void emit(std::string warning) { llvm::errs() << warning; }
};
} // namespace macroni::kernel
