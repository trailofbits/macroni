// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in the
// LICENSE file found in the root directory of this source tree.

#include <iostream>
#include <macroni/Common/ParseAST.hpp>
#include <macroni/Translation/MacroniCodeGenContext.hpp>
#include <macroni/Translation/MacroniCodeGenVisitorMixin.hpp>
#include <macroni/Translation/MacroniMetaGenerator.hpp>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>
#include <optional>
#include <pasta/AST/AST.h>
#include <vast/CodeGen/CodeGen.hpp>
#include <vast/CodeGen/CodeGenDriver.hpp>
#include <vast/CodeGen/Passes.hpp>
#include <vast/Dialect/HighLevel/Passes.hpp>
#include <vast/Frontend/Consumer.hpp>
#include <vast/Frontend/Driver.hpp>

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
  auto &actx = pasta_ast.UnderlyingAST();
  auto mctx = mlir::MLIRContext(registry);
  auto cgctx = macroni::MacroniCodeGenContext(mctx, actx, pasta_ast);
  auto meta = macroni::MacroniMetaGenerator(&actx, &mctx);
  auto codegen_instance = macroni::MacroniCodeGenInstance(cgctx, meta);

  codegen_instance.emit_data_layout();
  codegen_instance.Visit(actx.getTranslationUnitDecl());
  codegen_instance.verify_module();
  cgctx.mod->print(llvm::outs());

  return EXIT_SUCCESS;
}
