// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in the
// LICENSE file found in the root directory of this source tree.

#include <iostream>
#include <macroni/Common/GenerateMacroniModule.hpp>
#include <macroni/Common/ParseAST.hpp>
#include <macroni/Dialect/Macroni/MacroniDialect.hpp>
#include <macroni/Translation/MacroniCodeGenVisitorMixin.hpp>
#include <mlir/IR/DialectRegistry.h>
#include <pasta/AST/AST.h>
#include <vast/Dialect/HighLevel/HighLevelDialect.hpp>

int main(int argc, char **argv) {
  auto maybe_ast = pasta::parse_ast(argc, argv);
  if (!maybe_ast.Succeeded()) {
    std::cerr << maybe_ast.TakeError() << '\n';
    return EXIT_FAILURE;
  }
  auto pasta_ast = maybe_ast.TakeValue();

  // Register the MLIR dialects we will be lowering to
  mlir::DialectRegistry registry;
  registry
      .insert<vast::hl::HighLevelDialect, macroni::macroni::MacroniDialect>();
  auto mctx = mlir::MLIRContext(registry);

  // Generate the MLIR
  auto mod = macroni::generate_macroni_module<macroni::MacroniCodeGenInstance>(
      pasta_ast, mctx);

  // Print the result
  mod->print(llvm::outs());

  return EXIT_SUCCESS;
}
