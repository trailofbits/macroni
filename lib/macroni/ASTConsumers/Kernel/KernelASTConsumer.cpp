#include "macroni/ASTConsumers/Kernel/KernelASTConsumer.hpp"
#include "macroni/ASTMatchers/Kernel/RCUMatchers.hpp"
#include "macroni/Common/CodeGenDriverSetup.hpp"
#include "macroni/Common/MacroniMetaGenerator.hpp"
#include "macroni/Conversion/Kernel/KernelRewriters.hpp"
#include "macroni/Dialect/Kernel/KernelDialect.hpp"
#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/IR/Iterators.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

// NOTE(Brent): Not sure where to put this file's local functions. It may be
// better to create some sort of "Analysis" module for them and place them
// there.

namespace macroni::kernel {
KernelASTConsumer::KernelASTConsumer(bool print_locations)
    : m_print_locations(print_locations) {}

void KernelASTConsumer::HandleTranslationUnit(clang::ASTContext &Ctx) {
  // Match safe blocks.

  clang::ast_matchers::MatchFinder finder;
  rcu_collector matcher;
  matcher.attach_to(finder);
  finder.matchAST(Ctx);

  // Set up the driver.
  auto driver =
      ::macroni::generate_codegen_driver<KernelDialect,
                                         macroni::macroni_meta_generator>(Ctx);

  driver->push_visitor(std::make_unique<kernel_visitor>(
      matcher.expansions, Ctx, driver->mcontext(),
      driver->get_codegen_builder(), driver->get_meta_generator(),
      driver->get_symbol_generator(),
      vast::cg::visitor_view(driver->get_visitor_stack())));

  vast::cg::setup_default_visitor_stack(*driver);

  driver->emit(Ctx.getTranslationUnitDecl());
  driver->finalize();
  auto mod = driver->freeze();

  // Rewrite rcu_read_lock()/unlock() calls into critical sections.
  rewrite_rcu(&driver->mcontext(), mod);

  // Print the result.

  mlir::OpPrintingFlags flags;
  // Only print original locations if the flag is enabled.
  flags.enableDebugInfo(m_print_locations, false);

  mod->print(llvm::outs(), flags);
}

KernelASTConsumerFactory::KernelASTConsumerFactory(bool print_locations)
    : m_print_locations(print_locations) {}

std::unique_ptr<KernelASTConsumer>
KernelASTConsumerFactory::newASTConsumer(void) {
  return std::make_unique<KernelASTConsumer>(m_print_locations);
}
} // namespace macroni::kernel