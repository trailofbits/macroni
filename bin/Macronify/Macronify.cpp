// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in the
// LICENSE file found in the root directory of this source tree.

#include "ParseAST.hpp"
#include "macroni/Common/CodeGenDriverSetup.hpp"
#include "macroni/Dialect/Macroni/MacroniDialect.hpp"
#include "macroni/Translation/Macroni/MacroniMetaGenerator.hpp"
#include "macroni/Translation/Macroni/MacroniVisitor.hpp"
#include "pasta/AST/AST.h"
#include "vast/CodeGen/CodeGenDriver.hpp"
#include <clang/AST/ASTContext.h>
#include <iostream>
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>

int main(int argc, char **argv) {
  auto maybe_ast = pasta::parse_ast(argc, argv);
  if (!maybe_ast.Succeeded()) {
    std::cerr << maybe_ast.TakeError() << '\n';
    return EXIT_FAILURE;
  }

  auto pctx = maybe_ast.TakeValue();
  auto &actx = pctx.UnderlyingAST();
  auto driver =
      macroni::generate_codegen_driver<macroni::macroni::MacroniDialect,
                                       macroni::macroni_meta_generator>(actx);

  // Add the macroni visitor first. This ensures that our visitor will try to
  // lower stmts down to the Macroni dialect first before VAST's dialect.
  driver->push_visitor(std::make_unique<macroni::macroni_visitor>(
      pctx, driver->mcontext(), driver->get_codegen_builder(),
      driver->get_meta_generator(), driver->get_symbol_generator(),
      vast::cg::visitor_view(driver->get_visitor_stack())));

  vast::cg::setup_default_visitor_stack(*driver);

  driver->emit(actx.getTranslationUnitDecl());
  driver->finalize();
  auto mod = driver->freeze();
  mod->print(llvm::outs());

  return EXIT_SUCCESS;
}
