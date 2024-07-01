// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in the
// LICENSE file found in the root directory of this source tree.

#include "macroni/ASTConsumers/Kernel/KernelASTConsumer.hpp"
#include "vast/Util/Common.hpp"
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

static llvm::cl::opt<bool> g_print_locations(
    "locations",
    llvm::cl::desc("Enable printing original source locations in MLIR output"),
    llvm::cl::cat(g_tool_category));

static llvm::cl::alias
    g_print_locations_alias("l", llvm::cl::desc("Alias for --locations"),
                            llvm::cl::aliasopt(g_print_locations),
                            llvm::cl::NotHidden,
                            llvm::cl::cat(g_tool_category));

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

  auto mod_handler = [=](vast::vast_module &mod) {
    mlir::OpPrintingFlags flags;
    // Only print original locations if the flag is enabled.
    flags.enableDebugInfo(g_print_locations, false);
    mod->print(llvm::outs(), flags);
  };

  macroni::kernel::KernelASTConsumerFactory factory(mod_handler);

  return Tool.run(clang::tooling::newFrontendActionFactory(&factory).get());
}
