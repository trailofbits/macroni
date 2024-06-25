#include "macroni/Analysis/Kernel/RCUAnalysis.hpp"
#include "macroni/Dialect/Kernel/KernelInterfaces.hpp"
#include "macroni/Dialect/Kernel/KernelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Util/Common.hpp"
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Iterators.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>

namespace macroni::kernel {
// TODO(Brent): Use mlir::Operation::emitWarning() to emit warnings instead of
// using custom emit method.

// Check for invocations of RCU macros outside of RCU critical sections.
void rcu_analysis::analyze_non_rcu_function(vast::hl::FuncOp &func) {
  func.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (mlir::isa<RCUCriticalSection>(op)) {
      // NOTE(bpp): Skip checking for invocations of RCU macros inside RCU
      // critical sections because we only want to emit warnings for
      // invocations of RCU macros outside of critical sections. We walk the
      // tree using pre-order traversal instead of using post-order
      // traversal (the default) in order for this to work.
      return mlir::WalkResult::skip();
    }
    if (auto rcu_op = mlir::dyn_cast<RCU_Dereference_Interface>(op)) {
      warn_rcu_dereference_outside_critical_section(rcu_op);
    }
    return mlir::WalkResult::advance();
  });
}

// Create an MLIR operation walker that warns about RCU calls until finding a
// call to the function with the given name.
auto rcu_analysis::walk_until_call(llvm::StringRef name) {
  return [=, this](mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<vast::hl::CallOp>(op);
        call && name == call.getCalleeAttr().getValue()) {
      return mlir::WalkResult::interrupt();
    }
    if (auto rcu_op = mlir::dyn_cast<RCU_Dereference_Interface>(op)) {
      warn_rcu_dereference_outside_critical_section(rcu_op);
    }
    return mlir::WalkResult::advance();
  };
}

// Check for RCU macro invocations before first invocation of rcu_read_lock().
void rcu_analysis::analyze_acquires_function(vast::hl::FuncOp &func) {
  // Walk pre-order and warn until we find an invocation of rcu_read_lock().
  func.walk<mlir::WalkOrder::PreOrder>(
      walk_until_call(KernelDialect::rcu_read_lock()));
}

// Check for RCU macro invocations after last invocation of rcu_read_unlock().
void rcu_analysis::analyze_releases_function(vast::hl::FuncOp &func) {
  // Walk post-order in reverse and emit warnings until we encounter an
  // invocation of rcu_read_unlock()
  func.walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      walk_until_call(KernelDialect::rcu_read_unlock()));
}

// Check certain RCU macro invocations inside of an RCU critical section.
void rcu_analysis::analyze_critical_section(RCUCriticalSection cs) {
  cs.walk([&](RCUAccessPointer op) {
    info_rcu_access_pointer_in_critical_section(op);
  });
}

rcu_analysis::rcu_analysis(mlir::Operation *op) {
  auto mod = mlir::cast<vast::vast_module>(op);
  mod->walk([&](vast::hl::FuncOp func) {
    auto op = func.getOperation();
    if (op->getAttrOfType<MustHoldAttr>(MustHoldAttr::getMnemonic())) {
      // TODO(bpp): Investigate whether we should be checking __must_hold
      // functions for RCU macro invocations
    } else if (op->getAttrOfType<AcquiresAttr>(AcquiresAttr::getMnemonic())) {
      analyze_acquires_function(func);
    } else if (op->getAttrOfType<ReleasesAttr>(ReleasesAttr::getMnemonic())) {
      analyze_releases_function(func);
    } else {
      analyze_non_rcu_function(func);
    }
  });

  mod->walk([&](RCUCriticalSection cs) { analyze_critical_section(cs); });
}
} // namespace macroni::kernel
