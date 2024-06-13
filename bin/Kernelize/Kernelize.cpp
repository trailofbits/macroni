// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in the
// LICENSE file found in the root directory of this source tree.

#include "RCUCollector.hpp"
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

static llvm::cl::OptionCategory g_tool_category("kernelize options");

static llvm::cl::extrahelp
    g_common_help(clang::tooling::CommonOptionsParser::HelpMessage);

static llvm::cl::extrahelp
    g_more_help("\nAnalyzes Linux kernel RCU macro usage\n");

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

  clang::ast_matchers::MatchFinder finder;
  macroni::kernel::rcu_collector matcher;
  finder.addMatcher(macroni::kernel::rcu_deference_matcher, &matcher);

  return Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
}
