#include "SafetyASTConsumer.hpp"
#include "SafeBlockConditionCollector.hpp"
#include "macroni/Common/CodeGenDriverSetup.hpp"
#include "macroni/Dialect/Safety/SafetyDialect.hpp"
#include "macroni/Translation/Safety/SafetyVisitor.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <memory>

namespace macroni::safety {
void SafetyASTConsumer::HandleTranslationUnit(clang::ASTContext &Ctx) {
  // Match safe blocks.

  clang::ast_matchers::MatchFinder finder;
  macroni::safety::safe_block_condition_collector matcher;
  finder.addMatcher(macroni::safety::safe_block_condition_matcher, &matcher);
  auto driver =
      macroni::generate_codegen_driver<SafetyDialect,
                                       vast::cg::default_meta_gen>(Ctx);
  finder.matchAST(Ctx);

  // Generate the driver.

  driver->push_visitor(std::make_unique<safety_visitor>(
      matcher.m_safe_block_conditions, driver->mcontext(),
      driver->get_codegen_builder(), driver->get_meta_generator(),
      driver->get_symbol_generator(),
      vast::cg::visitor_view(driver->get_visitor_stack())));

  vast::cg::setup_default_visitor_stack(*driver);

  // Emit the lowered MLIR.

  driver->emit(Ctx.getTranslationUnitDecl());
  driver->finalize();
  auto mod = driver->freeze();
  mod->dump();
}

std::unique_ptr<SafetyASTConsumer>
SafetyASTConsumerFactory::newASTConsumer(void) {
  return std::make_unique<SafetyASTConsumer>();
}
} // namespace macroni::safety