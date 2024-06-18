#include "macroni/ASTConsumers/Safety/SafetyASTConsumer.hpp"
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <cstdlib>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

static llvm::cl::OptionCategory g_tool_category("safe-c options");

static llvm::cl::extrahelp
    g_common_help(clang::tooling::CommonOptionsParser::HelpMessage);

static llvm::cl::extrahelp
    g_more_help("\nUse the macro '#define unsafe if (0) ; else' "
                "to define unsafe blocks of C code \n");

int main(int argc, const char **argv) {
  auto ExpectedParser =
      clang::tooling::CommonOptionsParser::create(argc, argv, g_tool_category);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return EXIT_FAILURE;
  }

  clang::tooling::CommonOptionsParser &OptionsParser = ExpectedParser.get();
  clang::tooling::ClangTool Tool(OptionsParser.getCompilations(),
                                 OptionsParser.getSourcePathList());

  macroni::safety::SafetyASTConsumerFactory factory;

  return Tool.run(clang::tooling::newFrontendActionFactory(&factory).get());
}
