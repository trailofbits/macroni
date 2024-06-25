#pragma once

#include "macroni/Dialect/Kernel/KernelInterfaces.hpp"
#include "macroni/Dialect/Kernel/KernelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include <format>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <string>

namespace macroni::kernel {
struct rcu_analysis {
  rcu_analysis(mlir::Operation *op);

  virtual ~rcu_analysis() = default;

protected:
  void analyze_non_rcu_function(vast::hl::FuncOp &func);
  auto walk_until_call(llvm::StringRef name);
  void analyze_acquires_function(vast::hl::FuncOp &func);
  void analyze_releases_function(vast::hl::FuncOp &func);
  void analyze_critical_section(RCUCriticalSection cs);

  // Format an operation's location as a string for diagnostics.
  virtual std::string format_location(mlir::Operation *op) {
    std::string s;
    auto os = llvm::raw_string_ostream(s);
    mlir::OpPrintingFlags flags;
    flags.enableDebugInfo(false, true);
    mlir::AsmState state(op, flags);
    op->getLoc()->print(os, state, true);
    return s;
  }

  // Emit the given warning.
  virtual void warn(std::string warning) { llvm::errs() << warning; }

  // Emit the given information.
  virtual void info(std::string information) { llvm::errs() << information; }

  // Inform about a call to RCU access pointer inside a critical section.
  virtual void
  info_rcu_access_pointer_in_critical_section(RCUAccessPointer &op) {
    auto information =
        std::format("{}: info: Use rcu_dereference_protected() instead of "
                    "rcu_access_pointer() in critical section\n",
                    format_location(op));
    info(information);
  }

  // Warn about an RCU dereference call outside a critical section.
  virtual void warn_rcu_dereference_outside_critical_section(
      RCU_Dereference_Interface &rcu_deref) {
    auto op = rcu_deref.getOperation();
    auto warning = std::format(
        "{}: warning: Invocation of {}() outside of RCU critical section\n",
        format_location(op), op->getName().stripDialect().str());
    warn(warning);
  }
};
} // namespace macroni::kernel
