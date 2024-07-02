#include "macroni/ASTConsumers/Safety/SafetyASTConsumer.hpp"
#include "macroni/ASTMatchers/Safety/SafetyMatchers.hpp"
#include "macroni/Common/GenerateModule.hpp"
#include "macroni/Common/MacroniMetaGenerator.hpp"
#include "macroni/Dialect/Safety/SafetyDialect.hpp"
#include "macroni/Translation/Safety/SafetyVisitor.hpp"
#include "vast/Frontend/Action.hpp"
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

  auto maybe_mod_and_context =
      macroni::generate_module<SafetyDialect, macroni::macroni_meta_generator,
                               safety_visitor>(actx,
                                               matcher.m_safe_block_conditions);
  if (!maybe_mod_and_context) {
    llvm::errs() << "Could not make module\n";
    return;
  }

  auto mod = maybe_mod_and_context->first;
  mod->print(llvm::outs());
}

std::unique_ptr<SafetyASTConsumer>
SafetyASTConsumerFactory::newASTConsumer(void) {
  return std::make_unique<SafetyASTConsumer>();
}
} // namespace macroni::safety
