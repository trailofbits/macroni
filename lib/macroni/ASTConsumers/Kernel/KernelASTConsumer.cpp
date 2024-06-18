#include "macroni/ASTConsumers/Kernel/KernelASTConsumer.hpp"
#include "macroni/ASTMatchers/Kernel/RCUMatchers.hpp"
#include "macroni/Common/CodeGenDriverSetup.hpp"
#include "macroni/Conversion/Kernel/KernelRewriters.hpp"
#include "macroni/Dialect/Kernel/KernelAttributes.hpp"
#include "macroni/Dialect/Kernel/KernelDialect.hpp"
#include "macroni/Dialect/Kernel/KernelInterfaces.hpp"
#include "macroni/Dialect/Kernel/KernelOps.hpp"
#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/IR/Iterators.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

// NOTE(Brent): Not sure where to put this file's local functions. It may be
// better to create some sort of "Analysis" module for them and place them
// there.

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

void emit_warning_rcu_dereference_outside_critical_section(
    RCU_Dereference_Interface &op) {
  // Skip dialect namespace prefix when printing op name
  op.emitWarning() << format_location(op.getOperation())
                   << ": warning: Invocation of "
                   << op.getOperation()->getName().stripDialect()
                   << "() outside of RCU critical section\n";
}

// Check for invocations of RCU macros outside of RCU critical sections
void check_unannotated_function_for_rcu_invocations(vast::hl::FuncOp &func) {
  func.walk<mlir::WalkOrder::PreOrder>([](mlir::Operation *op) {
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
create_walker_until_call_with_name_found(llvm::StringRef name) {
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
void check_acquires_function_for_rcu_invocations(vast::hl::FuncOp &func) {
  // Walk pre-order and emit warnings until we encounter an invocation of
  // rcu_read_lock()
  func.walk<mlir::WalkOrder::PreOrder>(
      create_walker_until_call_with_name_found(KernelDialect::rcu_read_lock()));
}

// Check for invocations for RCU macros after last invocation of
// rcu_read_unlock()
void check_releases_function_for_rcu_invocations(vast::hl::FuncOp &func) {
  // Walk post-order in reverse and emit warnings until we encounter an
  // invocation of rcu_read_unlock()
  func.walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      create_walker_until_call_with_name_found(
          KernelDialect::rcu_read_unlock()));
}

// Check for invocations of certain RCU macros inside of RCU critical
// sections
void check_critical_section_section_for_rcu_invocations(RCUCriticalSection cs) {
  cs.walk([](RCUAccessPointer op) {
    op->emitWarning() << format_location(op)
                      << ": info: Use rcu_dereference_protected() instead of"
                         "rcu_access_pointer() in critical section\n";
  });
}

void KernelASTConsumer::HandleTranslationUnit(clang::ASTContext &Ctx) {
  // Match safe blocks.

  clang::ast_matchers::MatchFinder finder;
  rcu_collector matcher;
  finder.addMatcher(rcu_deference_matcher, &matcher);
  finder.addMatcher(rcu_deference_bh_matcher, &matcher);
  finder.addMatcher(rcu_deference_sched_matcher, &matcher);
  finder.addMatcher(rcu_assign_pointer_matcher, &matcher);
  finder.addMatcher(rcu_access_pointer_matcher, &matcher);
  finder.addMatcher(rcu_replace_pointer_matcher, &matcher);
  finder.matchAST(Ctx);

  // Set up the driver.
  auto driver =
      ::macroni::generate_codegen_driver<KernelDialect,
                                         vast::cg::default_meta_gen>(Ctx);

  driver->push_visitor(std::make_unique<kernel_visitor>(
      matcher.m_rcu_dereference_to_p, matcher.m_rcu_dereference_bh_to_p,
      matcher.m_rcu_dereference_sched_to_p,
      matcher.m_rcu_assign_pointer_to_params, matcher.m_rcu_access_pointer_to_p,
      matcher.m_rcu_replace_pointer_to_params, Ctx, driver->mcontext(),
      driver->get_codegen_builder(), driver->get_meta_generator(),
      driver->get_symbol_generator(),
      vast::cg::visitor_view(driver->get_visitor_stack())));

  vast::cg::setup_default_visitor_stack(*driver);

  driver->emit(Ctx.getTranslationUnitDecl());
  driver->finalize();
  auto mod = driver->freeze();

  // Register conversions
  auto patterns = mlir::RewritePatternSet(&driver->mcontext());
  patterns.add(rewrite_label_stmt).add(rewrite_rcu_read_unlock);

  // Apply the conversions
  auto frozen_pats = mlir::FrozenRewritePatternSet(std::move(patterns));
  mod->walk([&frozen_pats](mlir::Operation *op) {
    if (mlir::isa<vast::hl::ForOp, vast::hl::CallOp, vast::hl::LabelStmt>(op)) {
      std::ignore = mlir::applyOpPatternsAndFold(op, frozen_pats);
    }
  });

  // Print the result

  mod->print(llvm::outs());

  // Run analyses. The type of analysis we do depends on the annotation (if any)
  // that a given function definition is annotated with.

  mlir::DiagnosticEngine &engine = driver->mcontext().getDiagEngine();
  auto diagnostic_handler = engine.registerHandler([](mlir::Diagnostic &diag) {
    diag.print(llvm::errs());
    return;
  });

  mod->walk([](vast::hl::FuncOp func) {
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

  engine.eraseHandler(diagnostic_handler);
}

std::unique_ptr<KernelASTConsumer>
KernelASTConsumerFactory::newASTConsumer(void) {
  return std::make_unique<KernelASTConsumer>();
}
} // namespace macroni::kernel