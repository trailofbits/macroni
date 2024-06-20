#include "macroni/ASTConsumers/Kernel/KernelASTConsumer.hpp"
#include "macroni/ASTMatchers/Kernel/RCUMatchers.hpp"
#include "macroni/Analysis/Kernel/RCUAnalyzer.hpp"
#include "macroni/Common/CodeGenDriverSetup.hpp"
#include "macroni/Conversion/Kernel/KernelRewriters.hpp"
#include "macroni/Dialect/Kernel/KernelDialect.hpp"
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
void KernelASTConsumer::HandleTranslationUnit(clang::ASTContext &Ctx) {
  // Match safe blocks.

  clang::ast_matchers::MatchFinder finder;
  rcu_collector matcher;
  matcher.attach_to(finder);
  finder.matchAST(Ctx);

  // Set up the driver.
  auto driver =
      ::macroni::generate_codegen_driver<KernelDialect,
                                         vast::cg::default_meta_gen>(Ctx);

  driver->push_visitor(std::make_unique<kernel_visitor>(
      matcher.m_expansions, Ctx, driver->mcontext(),
      driver->get_codegen_builder(), driver->get_meta_generator(),
      driver->get_symbol_generator(),
      vast::cg::visitor_view(driver->get_visitor_stack())));

  vast::cg::setup_default_visitor_stack(*driver);

  driver->emit(Ctx.getTranslationUnitDecl());
  driver->finalize();
  auto mod = driver->freeze();

  // Rewrite rcu_read_lock()/unlock() calls into critical sections.
  rewrite_rcu(&driver->mcontext(), mod);

  // Print the result

  mod->print(llvm::outs());

  // Run analyses. The type of analysis we do depends on the annotation (if any)
  // that a given function definition is annotated with.

  mlir::DiagnosticEngine &engine = driver->mcontext().getDiagEngine();
  auto diagnostic_handler = engine.registerHandler([](mlir::Diagnostic &diag) {
    diag.print(llvm::errs());
    return;
  });

  rcu_analyzer analyzer;
  analyzer.analyze(mod);

  engine.eraseHandler(diagnostic_handler);
}

std::unique_ptr<KernelASTConsumer>
KernelASTConsumerFactory::newASTConsumer(void) {
  return std::make_unique<KernelASTConsumer>();
}
} // namespace macroni::kernel