#include "macroni/Dialect/Safety/SafetyDialect.hpp"
#include "macroni/Translation/Safety/SafetyVisitor.hpp"
#include "vast/CodeGen/CodeGenDriver.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenOptions.hpp"
#include "vast/CodeGen/DefaultVisitor.hpp"
#include "vast/CodeGen/Mangler.hpp"
#include "vast/CodeGen/UnreachableVisitor.hpp"
#include "vast/CodeGen/UnsupportedVisitor.hpp"
#include "vast/Dialect/Dialects.hpp"
#include "vast/Frontend/Options.hpp"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/Frontend/FrontendAction.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <set>
#include <utility>

namespace macroni {
std::unique_ptr<vast::cg::driver> generate_safety_driver(
    clang::ASTContext &actx,
    std::set<const clang::IntegerLiteral *> &safe_block_conditions) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  vast::registerAllDialects(registry);
  registry.insert<safety::SafetyDialect>();
  auto mctx = std::make_unique<vast::mcontext_t>();
  mctx->appendDialectRegistry(registry);
  mctx->loadAllAvailableDialects();
  auto bld = vast::cg::mk_codegen_builder(mctx.get());
  auto mg = std::make_unique<vast::cg::default_meta_gen>(&actx, mctx.get());
  auto mangle_context = actx.createMangleContext();
  auto sg = std::make_unique<vast::cg::default_symbol_mangler>(mangle_context);
  vast::cg::options copts = {
      .lang = vast::cc::get_source_language(actx.getLangOpts()),
      .optimization_level = 0,
      .has_strict_return = false,
      .disable_unsupported = false,
      .disable_vast_verifier = true,
      .prepare_default_visitor_stack = false};

  auto driver = std::make_unique<vast::cg::driver>(
      actx, std::move(mctx), std::move(copts), std::move(bld), std::move(mg),
      std::move(sg));

  // Add the macroni visitor first. This ensures that our visitor will try to
  // lower stmts down to the Macroni dialect first before VAST's dialect.
  driver->push_visitor(std::make_unique<safety::safety_visitor>(
      safe_block_conditions, driver->mcontext(), driver->get_codegen_builder(),
      driver->get_meta_generator(), driver->get_symbol_generator(),
      vast::cg::visitor_view(driver->get_visitor_stack())));

  // Then add the default visitor stack.
  driver->push_visitor(std::make_unique<vast::cg::default_visitor>(
      driver->mcontext(), driver->get_codegen_builder(),
      driver->get_meta_generator(), driver->get_symbol_generator(),
      vast::cg::visitor_view(driver->get_visitor_stack())));

  driver->push_visitor(std::make_unique<vast::cg::unsup_visitor>(
      driver->mcontext(), driver->get_codegen_builder(),
      driver->get_meta_generator(), driver->get_symbol_generator(),
      vast::cg::visitor_view(driver->get_visitor_stack())));

  driver->push_visitor(std::make_unique<vast::cg::unreach_visitor>(
      driver->mcontext(), driver->get_meta_generator(),
      driver->get_symbol_generator(), driver->get_options()));

  return driver;
}
} // namespace macroni

static llvm::cl::OptionCategory MyToolCategory("safe-c options");

static llvm::cl::extrahelp
    CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static llvm::cl::extrahelp
    MoreHelp("\nUse the macro '#define unsafe if (0) ; else' "
             "to define unsafe blocks of C code \n");

using namespace clang::ast_matchers;

StatementMatcher safe_block_condition_matcher =
    integerLiteral(isExpandedFromMacro("unsafe")).bind("root");

class safe_block_condition_collector : public MatchFinder::MatchCallback {
  virtual void run(const MatchFinder::MatchResult &Result) override {
    m_actx = Result.Context;
    auto match = Result.Nodes.getNodeAs<clang::IntegerLiteral>("root");
    m_matches.insert(match);
  }

  virtual void onStartOfTranslationUnit() override {
    m_actx = nullptr;
    m_matches.clear();
  }

  virtual void onEndOfTranslationUnit() override {
    auto driver = macroni::generate_safety_driver(*m_actx, m_matches);
    driver->emit(m_actx->getTranslationUnitDecl());
    driver->finalize();
    auto mod = driver->freeze();
    mod->dump();
  }

private:
  clang::ASTContext *m_actx;
  std::set<const clang::IntegerLiteral *> m_matches;
};

int main(int argc, const char **argv) {
  auto ExpectedParser =
      clang::tooling::CommonOptionsParser::create(argc, argv, MyToolCategory);
  if (!ExpectedParser) {
    // Fail gracefully for unsupported options.
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  clang::tooling::CommonOptionsParser &OptionsParser = ExpectedParser.get();
  clang::tooling::ClangTool Tool(OptionsParser.getCompilations(),
                                 OptionsParser.getSourcePathList());

  MatchFinder finder;
  safe_block_condition_collector matcher;
  finder.addMatcher(safe_block_condition_matcher, &matcher);

  return Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
}
