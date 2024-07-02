#include "macroni/ASTConsumers/Kernel/KernelASTConsumer.hpp"
#include "macroni/ASTMatchers/Kernel/RCUMatchers.hpp"
#include "macroni/Common/Common.hpp"
#include "macroni/Common/GenerateModule.hpp"
#include "macroni/Common/MacroniMetaGenerator.hpp"
#include "macroni/Conversion/Kernel/KernelRewriters.hpp"
#include "macroni/Dialect/Kernel/KernelDialect.hpp"
#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include "vast/Frontend/Action.hpp"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Iterators.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

// NOTE(Brent): Not sure where to put this file's local functions. It may be
// better to create some sort of "Analysis" module for them and place them
// there.

namespace macroni::kernel {
KernelASTConsumer::KernelASTConsumer(module_handler mod_handler)
    : m_module_handler(mod_handler) {}

void KernelASTConsumer::HandleTranslationUnit(clang::ASTContext &actx) {
  // Match safe blocks.

  clang::ast_matchers::MatchFinder finder;
  rcu_collector matcher;
  matcher.attach_to(finder);
  finder.matchAST(actx);

  auto maybe_mod_and_context =
      macroni::generate_module<KernelDialect, macroni::macroni_meta_generator,
                               kernel_visitor>(actx, actx, matcher.expansions);
  if (!maybe_mod_and_context) {
    llvm::errs() << "Could not make module\n";
    return;
  }

  auto mod = maybe_mod_and_context->first;
  auto mctx = std::move(maybe_mod_and_context->second);

  // Rewrite rcu_read_lock()/unlock() calls into critical sections.
  rewrite_rcu(&*mctx, mod);

  // Handle the result.
  m_module_handler(mod);
}

KernelASTConsumerFactory::KernelASTConsumerFactory(module_handler mod_handler)
    : m_module_handler(mod_handler) {}

std::unique_ptr<KernelASTConsumer>
KernelASTConsumerFactory::newASTConsumer(void) {
  return std::make_unique<KernelASTConsumer>(m_module_handler);
}
} // namespace macroni::kernel
