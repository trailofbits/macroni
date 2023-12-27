// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in the
// LICENSE file found in the root directory of this source tree.

#include "KernelCodeGenVisitorMixin.hpp"
#include <iostream>
#include <macroni/Common/GenerateMacroniModule.hpp>
#include <macroni/Common/ParseAST.hpp>
#include <macroni/Conversion/Kernel/KernelRewriters.hpp>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>
#include <optional>
#include <pasta/AST/AST.h>
#include <vast/CodeGen/CodeGen.hpp>

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

  mod->walk<mlir::WalkOrder::PreOrder>([](mlir::Operation *op) {
    if (mlir::isa<macroni::kernel::RCUCriticalSection>(op)) {
      // NOTE(bpp): Skip checking for invocations of RCU macros inside RCU
      // critical sections because we only want to emit warnings for invocations
      // of RCU macros outside of critical sections. We walk the tree using
      // pre-order traversal instead of using post-order traversal (the default)
      // in order for this to work.
      return mlir::WalkResult::skip();
    }
    if (mlir::isa<macroni::kernel::RCUDereference,
                  macroni::kernel::RCUDereferenceBH,
                  macroni::kernel::RCUDereferenceSched,
                  macroni::kernel::RCUDereferenceCheck,
                  macroni::kernel::RCUDereferenceBHCheck,
                  macroni::kernel::RCUDereferenceSchedCheck,
                  macroni::kernel::RCUDereferenceProtected>(op)) {
      auto name = "<error>";
      if (mlir::isa<macroni::kernel::RCUDereference>(op)) {
        name = "rcu_dereference";
      } else if (mlir::isa<macroni::kernel::RCUDereferenceBH>(op)) {
        name = "rcu_dereference_bh";
      } else if (mlir::isa<macroni::kernel::RCUDereferenceSched>(op)) {
        name = "rcu_dereference_sched";
      } else if (mlir::isa<macroni::kernel::RCUDereferenceCheck>(op)) {
        name = "rcu_dereference_check";
      } else if (mlir::isa<macroni::kernel::RCUDereferenceBHCheck>(op)) {
        name = "rcu_dereference_bh_check";
      } else if (mlir::isa<macroni::kernel::RCUDereferenceSchedCheck>(op)) {
        name = "rcu_dereference_sched_check";
      } else if (mlir::isa<macroni::kernel::RCUDereferenceProtected>(op)) {
        name = "rcu_dereference_protected";
      }
      std::string s;
      auto os = llvm::raw_string_ostream(s);
      op->getLoc()->print(os);
      s.erase(s.find("loc("), 4);
      s.erase(s.find('"'), 1);
      s.erase(s.find('"'), 1);
      s.erase(s.rfind(')'), 1);
      os << ": warning: Invocation of " << name
         << "() outside of RCU critical section\n";
      op->emitWarning() << s;
    }
    return mlir::WalkResult::advance();
  });

  engine.eraseHandler(diagnostic_handler);

  return EXIT_SUCCESS;
}
