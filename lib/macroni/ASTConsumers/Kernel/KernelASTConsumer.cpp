#include "macroni/ASTConsumers/Kernel/KernelASTConsumer.hpp"
#include "macroni/ASTMatchers/Kernel/RCUMatchers.hpp"
#include "macroni/Common/Common.hpp"
#include "macroni/Common/MacroniMetaGenerator.hpp"
#include "macroni/Conversion/Kernel/KernelRewriters.hpp"
#include "macroni/Dialect/Kernel/KernelDialect.hpp"
#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include "vast/CodeGen/AttrVisitorProxy.hpp"
#include "vast/CodeGen/CodeGenDriver.hpp"
#include "vast/CodeGen/DefaultVisitor.hpp"
#include "vast/CodeGen/FallthroughVisitor.hpp"
#include "vast/CodeGen/TypeCachingProxy.hpp"
#include "vast/CodeGen/UnsupportedVisitor.hpp"
#include "vast/Dialect/Dialects.hpp"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Iterators.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

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

  // Load MLIR and VAST dialects into registry.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  vast::registerAllDialects(registry);

  // Insert our custom dialect.
  registry.insert<KernelDialect>();
  auto mctx = std::make_unique<vast::mcontext_t>();
  mctx->appendDialectRegistry(registry);
  mctx->loadAllAvailableDialects();

  // Set up variables for constructing the codegen driver.

  auto bld = vast::cg::mk_codegen_builder(*mctx);
  auto mg = std::make_shared<macroni::macroni_meta_generator>(&actx, &*mctx);
  auto sg = std::make_shared<vast::cg::default_symbol_generator>(
      actx.createMangleContext());

  using vast::cg::as_node;
  using vast::cg::as_node_with_list_ref;
  auto visitors = std::make_shared<vast::cg::visitor_list>() |
                  as_node_with_list_ref<kernel_visitor>(
                      actx, *mctx, *bld, *mg, *sg, matcher.expansions) |
                  as_node_with_list_ref<vast::cg::attr_visitor_proxy>() |
                  as_node<vast::cg::type_caching_proxy>() |
                  as_node_with_list_ref<vast::cg::default_visitor>(
                      *mctx, *bld, mg, sg,
                      /* strict return = */ false,
                      vast::cg::missing_return_policy::emit_trap) |
                  as_node_with_list_ref<vast::cg::unsup_visitor>(*mctx, *bld) |
                  as_node<vast::cg::fallthrough_visitor>();

  // Create the codegen driver.
  auto driver =
      std::make_unique<vast::cg::driver>(actx, *mctx, std::move(bld), visitors);
  driver->enable_verifier(true);

  // Emit the lowered MLIR.

  driver->emit(actx.getTranslationUnitDecl());
  driver->finalize();
  if (!driver->verify()) {
    llvm::errs() << "Failed to verify driver\n";
    return;
  }
  auto mod = driver->freeze();

  // Rewrite rcu_read_lock()/unlock() calls into critical sections.
  rewrite_rcu(&driver->mcontext(), mod);

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
