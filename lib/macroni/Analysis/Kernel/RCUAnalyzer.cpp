#include "macroni/Analysis/Kernel/RCUAnalyzer.hpp"
#include "macroni/Dialect/Kernel/KernelInterfaces.hpp"
#include "macroni/Dialect/Kernel/KernelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Util/Common.hpp"
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Iterators.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <string>

namespace macroni::kernel {

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

void rcu_analyzer::emit_warning_rcu_dereference_outside_critical_section(
    RCU_Dereference_Interface &op) {
  // Skip dialect namespace prefix when printing op name
  op.emitWarning() << format_location(op.getOperation())
                   << ": warning: Invocation of "
                   << op.getOperation()->getName().stripDialect()
                   << "() outside of RCU critical section\n";
}

// Check for invocations of RCU macros outside of RCU critical sections
void rcu_analyzer::check_unannotated_function_for_rcu_invocations(
    vast::hl::FuncOp &func) {
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
      emit_warning_rcu_dereference_outside_critical_section(rcu_op);
    }
    return mlir::WalkResult::advance();
  });
}

std::function<mlir::WalkResult(mlir::Operation *)>
rcu_analyzer::create_walker_until_call_with_name_found(llvm::StringRef name) {
  return [&](mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<vast::hl::CallOp>(op);
        call && name == call.getCalleeAttr().getValue()) {
      return mlir::WalkResult::interrupt();
    }
    if (auto rcu_op = mlir::dyn_cast<RCU_Dereference_Interface>(op)) {
      emit_warning_rcu_dereference_outside_critical_section(rcu_op);
    }
    return mlir::WalkResult::advance();
  };
}

// Check for invocations for RCU macros before first invocation of
// rcu_read_lock()
void rcu_analyzer::check_acquires_function_for_rcu_invocations(
    vast::hl::FuncOp &func) {
  // Walk pre-order and emit warnings until we encounter an invocation of
  // rcu_read_lock()
  func.walk<mlir::WalkOrder::PreOrder>(
      create_walker_until_call_with_name_found(KernelDialect::rcu_read_lock()));
}

// Check for invocations for RCU macros after last invocation of
// rcu_read_unlock()
void rcu_analyzer::check_releases_function_for_rcu_invocations(
    vast::hl::FuncOp &func) {
  // Walk post-order in reverse and emit warnings until we encounter an
  // invocation of rcu_read_unlock()
  func.walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      create_walker_until_call_with_name_found(
          KernelDialect::rcu_read_unlock()));
}

// Check for invocations of certain RCU macros inside of RCU critical
// sections
void rcu_analyzer::check_critical_section_section_for_rcu_invocations(
    RCUCriticalSection cs) {
  cs.walk([](RCUAccessPointer op) {
    op->emitWarning() << format_location(op)
                      << ": info: Use rcu_dereference_protected() instead of"
                         " rcu_access_pointer() in critical section\n";
  });
}

void rcu_analyzer::analyze(vast::owning_module_ref &mod) {
  mod->walk([&](vast::hl::FuncOp func) {
    auto op = func.getOperation();
    if (op->getAttrOfType<MustHoldAttr>(MustHoldAttr::getMnemonic())) {
      // TODO(bpp): Investigate whether we should be checking __must_hold
      // functions for RCU macro invocations
    } else if (op->getAttrOfType<AcquiresAttr>(AcquiresAttr::getMnemonic())) {
      check_acquires_function_for_rcu_invocations(func);
    } else if (op->getAttrOfType<ReleasesAttr>(ReleasesAttr::getMnemonic())) {
      check_releases_function_for_rcu_invocations(func);
    } else {
      check_unannotated_function_for_rcu_invocations(func);
    }
  });

  mod->walk(check_critical_section_section_for_rcu_invocations);
}
} // namespace macroni::kernel
