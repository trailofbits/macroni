// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in the
// LICENSE file found in the root directory of this source tree.

#include "KernelCodeGenVisitorMixin.hpp"
#include "macroni/Dialect/Kernel/KernelDialect.hpp"
#include "macroni/Dialect/Kernel/KernelInterfaces.hpp"
#include "mlir/IR/Visitors.h"
#include <functional>
#include <iostream>
#include <macroni/Common/GenerateMacroniModule.hpp>
#include <macroni/Common/ParseAST.hpp>
#include <macroni/Conversion/Kernel/KernelRewriters.hpp>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Iterators.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>
#include <pasta/AST/AST.h>
#include <string>
#include <vast/CodeGen/CodeGen.hpp>

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

void emit_warning_rcu_dereference_outside_critical_section(
    macroni::kernel::RCU_Dereference_Interface &op) {
  // Skip dialect namespace prefix when printing op name
  op.emitWarning() << format_location(op.getOperation())
                   << ": warning: Invocation of "
                   << op.getOperation()->getName().stripDialect()
                   << "() outside of RCU critical section\n";
}

// Check for invocations of RCU macros outside of RCU critical sections
void check_unannotated_function_for_rcu_invocations(vast::hl::FuncOp &func) {
  func.walk<mlir::WalkOrder::PreOrder>([](mlir::Operation *op) {
    if (mlir::isa<macroni::kernel::RCUCriticalSection>(op)) {
      // NOTE(bpp): Skip checking for invocations of RCU macros inside RCU
      // critical sections because we only want to emit warnings for
      // invocations of RCU macros outside of critical sections. We walk the
      // tree using pre-order traversal instead of using post-order
      // traversal (the default) in order for this to work.
      return mlir::WalkResult::skip();
    }
    if (auto rcu_op =
            mlir::dyn_cast<macroni::kernel::RCU_Dereference_Interface>(
                op)) {
      emit_warning_rcu_dereference_outside_critical_section(rcu_op);
    }
    return mlir::WalkResult::advance();
  });
}

std::function<mlir::WalkResult(mlir::Operation *)>
create_walker_until_call_with_name_found(llvm::StringRef name) {
  return [&](mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<vast::hl::CallOp>(op);
        call && name == call.getCalleeAttr().getValue()) {
      return mlir::WalkResult::interrupt();
    }
    if (auto rcu_op =
            mlir::dyn_cast<macroni::kernel::RCU_Dereference_Interface>(
                op)) {
      emit_warning_rcu_dereference_outside_critical_section(rcu_op);
    }
    return mlir::WalkResult::advance();
  };
}

// Check for invocations for RCU macros before first invocation of
// rcu_read_lock()
void check_acquires_function_for_rcu_invocations(vast::hl::FuncOp &func) {
  // Walk pre-order and emit warnings until we encounter an invocation of
  // rcu_read_lock()
  func.walk<mlir::WalkOrder::PreOrder>(create_walker_until_call_with_name_found(
      macroni::kernel::KernelDialect::rcu_read_lock()));
}

// Check for invocations for RCU macros after last invocation of
// rcu_read_unlock()
void check_releases_function_for_rcu_invocations(vast::hl::FuncOp &func) {
  // Walk post-order in reverse and emit warnings until we encounter an
  // invocation of rcu_read_unlock()
  func.walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      create_walker_until_call_with_name_found(
          macroni::kernel::KernelDialect::rcu_read_unlock()));
}

// Check for invocations of certain RCU macros inside of RCU critical
// sections
void check_critical_section_section_for_rcu_invocations(
    macroni::kernel::RCUCriticalSection cs) {
  cs.walk([](macroni::kernel::RCUAccessPointer op) {
    op->emitWarning() << format_location(op)
                      << ": info: Use rcu_dereference_protected() instead of "
                         "rcu_access_pointer() in critical section\n";
  });
}

int main(int argc, char **argv) {
  auto maybe_ast = pasta::parse_ast(argc, argv);
  if (!maybe_ast.Succeeded()) {
    std::cerr << maybe_ast.TakeError() << '\n';
    return EXIT_FAILURE;
  }
  auto pasta_ast = maybe_ast.TakeValue();

  // Register the MLIR dialects we will be lowering to
  mlir::DialectRegistry registry;
  registry.insert<vast::hl::HighLevelDialect, vast::unsup::UnsupportedDialect,
                  macroni::macroni::MacroniDialect,
                  macroni::kernel::KernelDialect>();
  auto mctx = mlir::MLIRContext(registry);

  // Generate the MLIR
  auto mod = macroni::generate_macroni_module<KernelCodeGen>(pasta_ast, mctx);

  // Register conversions
  auto patterns = mlir::RewritePatternSet(&mctx);
  patterns.add(macroni::kernel::rewrite_get_user)
      .add(macroni::kernel::rewrite_offsetof)
      .add(macroni::kernel::rewrite_container_of)
      .add(macroni::kernel::rewrite_rcu_dereference)
      .add(macroni::kernel::rewrite_rcu_dereference_check)
      .add(macroni::kernel::rewrite_rcu_access_pointer)
      .add(macroni::kernel::rewrite_rcu_assign_pointer)
      .add(macroni::kernel::rewrite_rcu_replace_pointer)
      .add(macroni::kernel::rewrite_smp_mb)
      .add(macroni::kernel::rewrite_list_for_each)
      .add(macroni::kernel::rewrite_label_stmt)
      .add(macroni::kernel::rewrite_rcu_read_unlock);

  // Apply the conversions
  auto frozen_pats = mlir::FrozenRewritePatternSet(std::move(patterns));
  mod->walk([&frozen_pats](mlir::Operation *op) {
    if (mlir::isa<macroni::macroni::MacroExpansion, vast::hl::ForOp,
                  vast::hl::CallOp, vast::hl::LabelStmt>(op)) {
      std::ignore = mlir::applyOpPatternsAndFold(op, frozen_pats);
    }
  });

  // Print the result
  mod->print(llvm::outs());

  mlir::DiagnosticEngine &engine = mctx.getDiagEngine();
  auto diagnostic_handler = engine.registerHandler([](mlir::Diagnostic &diag) {
    diag.print(llvm::errs());
    return;
  });

  mod->walk([](vast::hl::FuncOp func) {
    auto op = func.getOperation();
    if (op->getAttrOfType<macroni::kernel::MustHoldAttr>("annotate")) {
      // TODO(bpp): Investigate whether we should be checking __must_hold
      // functions for RCU macro invocations
    } else if (op->getAttrOfType<macroni::kernel::AcquiresAttr>("annotate")) {
      check_acquires_function_for_rcu_invocations(func);
    } else if (op->getAttrOfType<macroni::kernel::ReleasesAttr>("annotate")) {
      check_releases_function_for_rcu_invocations(func);
    } else {
      check_unannotated_function_for_rcu_invocations(func);
    }
  });

  mod->walk(check_critical_section_section_for_rcu_invocations);

  engine.eraseHandler(diagnostic_handler);

  return EXIT_SUCCESS;
}
