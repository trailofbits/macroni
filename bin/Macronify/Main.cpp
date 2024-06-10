// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in the
// LICENSE file found in the root directory of this source tree.

#include "macroni/Common/GenerateMacroniDriver.hpp"
#include "macroni/Common/ParseAST.hpp"
#include "vast/CodeGen/CodeGenDriver.hpp"
#include <clang/AST/ASTContext.h>
#include <iostream>

int main(int argc, char **argv) {
  auto maybe_ast = pasta::parse_ast(argc, argv);
  if (!maybe_ast.Succeeded()) {
    std::cerr << maybe_ast.TakeError() << '\n';
    return EXIT_FAILURE;
  }
  auto pasta_ast = maybe_ast.TakeValue();
  auto &actx = pasta_ast.UnderlyingAST();
  auto driver = macroni::generate_macroni_driver(actx);

  driver->emit(actx.getTranslationUnitDecl());
  driver->finalize();
  auto mod = driver->freeze();
  mod->print(llvm::outs());

  return EXIT_SUCCESS;
}
