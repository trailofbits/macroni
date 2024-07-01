#include "macroni/ASTConsumers/Safety/SafetyASTConsumer.hpp"
#include "macroni/ASTMatchers/Safety/SafetyMatchers.hpp"
#include "macroni/Common/MacroniMetaGenerator.hpp"
#include "macroni/Dialect/Safety/SafetyDialect.hpp"
#include "macroni/Translation/Safety/SafetyVisitor.hpp"
#include "vast/CodeGen/AttrVisitorProxy.hpp"
#include "vast/CodeGen/CodeGenDriver.hpp"
#include "vast/CodeGen/CodeGenVisitorList.hpp"
#include "vast/CodeGen/DefaultSymbolGenerator.hpp"
#include "vast/CodeGen/DefaultVisitor.hpp"
#include "vast/CodeGen/FallthroughVisitor.hpp"
#include "vast/CodeGen/TypeCachingProxy.hpp"
#include "vast/CodeGen/UnsupportedVisitor.hpp"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace macroni::safety {
void SafetyASTConsumer::HandleTranslationUnit(clang::ASTContext &actx) {
  // Match safe blocks.

  clang::ast_matchers::MatchFinder finder;
  macroni::safety::safe_block_condition_collector matcher;
  finder.addMatcher(macroni::safety::safe_block_condition_matcher, &matcher);
  finder.matchAST(actx);

  // Load MLIR and VAST dialects into registry.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  vast::registerAllDialects(registry);

  // Insert our custom dialect.
  registry.insert<SafetyDialect>();
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
                  as_node_with_list_ref<safety_visitor>(
                      *mctx, *bld, matcher.m_safe_block_conditions) |
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
  mod->print(llvm::outs());
}

std::unique_ptr<SafetyASTConsumer>
SafetyASTConsumerFactory::newASTConsumer(void) {
  return std::make_unique<SafetyASTConsumer>();
}
} // namespace macroni::safety
