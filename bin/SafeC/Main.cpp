// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in
// the LICENSE file found in the root directory of this source tree.

#include "SafeCCodeGenVisitorMixin.hpp"
#include <iostream>
#include <macroni/Common/ParseAST.hpp>
#include <macroni/Conversion/Safety/SafetyRewriters.hpp>
#include <macroni/Translation/MacroniCodeGenVisitorMixin.hpp>
#include <macroni/Translation/MacroniMetaGenerator.hpp>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>
#include <optional>
#include <pasta/AST/AST.h>
#include <vast/Translation/CodeGen.hpp>

int main(int argc, char **argv) {
    auto maybe_ast = pasta::parse_ast(argc, argv);
    if (!maybe_ast.Succeeded()) {
        std::cerr << maybe_ast.TakeError() << '\n';
        return EXIT_FAILURE;
    }
    auto ast = maybe_ast.TakeValue();

    // Register the MLIR dialects we will be lowering to
    mlir::DialectRegistry registry;
    registry.insert<
        vast::hl::HighLevelDialect,
        macroni::macroni::MacroniDialect,
        macroni::safety::SafetyDialect
    >();
    auto mctx = mlir::MLIRContext(registry);
    macroni::MacroniMetaGenerator meta(ast, &mctx);
    vast::cg::CodeGenContext cgctx(mctx, ast.UnderlyingAST());
    vast::cg::CodeGenBase<SafeCVisitor> codegen(cgctx, meta);

    // Generate the MLIR
    auto tu_decl = ast.UnderlyingAST().getTranslationUnitDecl();
    auto mod = codegen.emit_module(tu_decl);

    // Register conversions
    mlir::RewritePatternSet patterns(&mctx);
    patterns.add(macroni::safety::rewrite_unsafe);

    // Apply the conversions.
    mlir::FrozenRewritePatternSet frozen_pats(std::move(patterns));
    mod->walk([&frozen_pats](mlir::Operation *op) {
        if (mlir::isa<vast::hl::IfOp>(op)) {
            std::ignore = mlir::applyOpPatternsAndFold(op, frozen_pats);
        }}
    );
    // Print the result
    mod->print(llvm::outs());

    return EXIT_SUCCESS;
}
