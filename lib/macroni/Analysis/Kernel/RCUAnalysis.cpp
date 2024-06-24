#include "macroni/Analysis/Kernel/RCUAnalysis.hpp"
#include "macroni/Dialect/Kernel/KernelInterfaces.hpp"
#include "macroni/Dialect/Kernel/KernelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Util/Common.hpp"
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Iterators.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <string>

namespace macroni::kernel {
// TODO(Brent): Use mlir::Operation::emitWarning() to emit warnings instead of
// directly printing to stderr.

// Format an operation's location as a string for diagnostics
std::string format_location(mlir::Operation *op) {
  std::string s;
  auto os = llvm::raw_string_ostream(s);
  op->getLoc()->print(os);
  s.erase(s.find("loc("), 4);
  s.erase(s.find('"'), 1);
  s.erase(s.find('"'), 1);
  s.erase(s.rfind(')'), 1);
  return s;
}

// Warn about an RCU dereference call outside a critical section.
void warn_rcu_dereference(RCU_Dereference_Interface &op) {
  // Skip dialect namespace prefix when printing op name
  llvm::errs() << format_location(op.getOperation())
               << ": warning: Invocation of "
               << op.getOperation()->getName().stripDialect()
               << "() outside of RCU critical section\n";
}

// Check for invocations of RCU macros outside of RCU critical sections.
void analyze_non_rcu_function(vast::hl::FuncOp &func) {
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
      warn_rcu_dereference(rcu_op);
    }
    return mlir::WalkResult::advance();
  });
}

// Create an MLIR operation walker that warns about RCU calls until finding a
// call to the function with the given name.
auto walk_until_call(llvm::StringRef name) {
  return [=](mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<vast::hl::CallOp>(op);
        call && name == call.getCalleeAttr().getValue()) {
      return mlir::WalkResult::interrupt();
    }
    if (auto rcu_op = mlir::dyn_cast<RCU_Dereference_Interface>(op)) {
      warn_rcu_dereference(rcu_op);
    }
    return mlir::WalkResult::advance();
  };
}

// Check for RCU macro invocations before first invocation of rcu_read_lock().
void analyze_acquires_function(vast::hl::FuncOp &func) {
  // Walk pre-order and warn until we find an invocation of rcu_read_lock().
  func.walk<mlir::WalkOrder::PreOrder>(
      walk_until_call(KernelDialect::rcu_read_lock()));
}

// Check for RCU macro invocations after last invocation of rcu_read_unlock().
void analyze_releases_function(vast::hl::FuncOp &func) {
  // Walk post-order in reverse and emit warnings until we encounter an
  // invocation of rcu_read_unlock()
  func.walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      walk_until_call(KernelDialect::rcu_read_unlock()));
}

// Check certain RCU macro invocations inside of an RCU critical section.
void analyze_critical_section(RCUCriticalSection cs) {
  cs.walk([](RCUAccessPointer op) {
    llvm::errs() << format_location(op)
                 << ": info: Use rcu_dereference_protected() instead of"
                    " rcu_access_pointer() in critical section\n";
  });
}

kernel_analysis::kernel_analysis(mlir::Operation *op) {
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

  mod->walk(analyze_critical_section);
}
} // namespace macroni::kernel
