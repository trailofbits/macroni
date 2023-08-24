// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in
// the LICENSE file found in the root directory of this source tree.

#include <iostream>
#include <macroni/Common/ParseAST.hpp>
#include <macroni/Translation/MacroniCodeGenContext.hpp>
#include <macroni/Translation/MacroniCodeGenVisitorMixin.hpp>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>
#include <optional>
#include <pasta/AST/AST.h>
#include <vast/CodeGen/CodeGen.hpp>

int main(int argc, char **argv) {
    auto maybe_ast = pasta::parse_ast(argc, argv);
    if (!maybe_ast.Succeeded()) {
        std::cerr << maybe_ast.TakeError() << '\n';
        return EXIT_FAILURE;
    }
    auto pasta_ast = maybe_ast.TakeValue();

    // Register the MLIR dialects we will be lowering to
    mlir::DialectRegistry registry;
    registry.insert<
        vast::hl::HighLevelDialect,
        macroni::macroni::MacroniDialect
    >();
    auto mctx = mlir::MLIRContext(registry);
    macroni::MacroniCodeGenContext cgctx(mctx, pasta_ast.UnderlyingAST(),
                                         pasta_ast);
    macroni::MacroniCodeGen codegen(cgctx);

    // Generate the MLIR
    auto tu_decl = pasta_ast.UnderlyingAST().getTranslationUnitDecl();
    auto mod = codegen.emit_module(tu_decl);

    // Print the result
    mod->print(llvm::outs());

    return EXIT_SUCCESS;
}
