#pragma once

#include "macroni/Dialect/Kernel/KernelInterfaces.hpp"
#include "macroni/Dialect/Kernel/KernelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Util/Common.hpp"
#include <llvm/ADT/StringRef.h>

namespace macroni::kernel {
class rcu_analyzer {
private:
  static void emit_warning_rcu_dereference_outside_critical_section(
      RCU_Dereference_Interface &op);

  // Check for invocations of RCU macros outside of RCU critical sections
  static void
  check_unannotated_function_for_rcu_invocations(vast::hl::FuncOp &func);

  static std::function<mlir::WalkResult(mlir::Operation *)>
  create_walker_until_call_with_name_found(llvm::StringRef name);

  // Check for invocations for RCU macros before first invocation of
  // rcu_read_lock()
  static void
  check_acquires_function_for_rcu_invocations(vast::hl::FuncOp &func);

  // Check for invocations for RCU macros after last invocation of
  // rcu_read_unlock()
  static void
  check_releases_function_for_rcu_invocations(vast::hl::FuncOp &func);
  // Check for invocations of certain RCU macros inside of RCU critical sections
  static void
  check_critical_section_section_for_rcu_invocations(RCUCriticalSection cs);

public:
  void analyze(vast::owning_module_ref &mod);
};
} // namespace macroni::kernel
