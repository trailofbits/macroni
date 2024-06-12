#include "SafeBlockConditionCollector.hpp"
#include "macroni/Common/CodeGenDriverSetup.hpp"
#include "macroni/Dialect/Safety/SafetyDialect.hpp"
#include "macroni/Translation/Safety/SafetyVisitor.hpp"
#include "vast/CodeGen/CodeGenDriver.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <memory>
#include <set>

namespace macroni::safety {
using namespace clang::ast_matchers;

clang::ast_matchers::StatementMatcher safe_block_condition_matcher =
    integerLiteral(isExpandedFromMacro("unsafe")).bind("root");

void safe_block_condition_collector::run(
    const MatchFinder::MatchResult &Result) {
  m_actx = Result.Context;
  auto match = Result.Nodes.getNodeAs<clang::IntegerLiteral>("root");
  m_safe_block_conditions.insert(match);
}

void safe_block_condition_collector::onStartOfTranslationUnit() {
  m_actx = nullptr;
  m_safe_block_conditions.clear();
}

void safe_block_condition_collector::onEndOfTranslationUnit() {
  auto driver =
      macroni::generate_codegen_driver<macroni::safety::SafetyDialect,
                                       vast::cg::default_meta_gen>(*m_actx);

  driver->push_visitor(std::make_unique<macroni::safety::safety_visitor>(
      m_safe_block_conditions, driver->mcontext(),
      driver->get_codegen_builder(), driver->get_meta_generator(),
      driver->get_symbol_generator(),
      vast::cg::visitor_view(driver->get_visitor_stack())));

  vast::cg::setup_default_visitor_stack(*driver);

  driver->emit(m_actx->getTranslationUnitDecl());
  driver->finalize();
  auto mod = driver->freeze();
  mod->dump();
}
} // namespace macroni::safety