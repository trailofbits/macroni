// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in the
// LICENSE file found in the root directory of this source tree.

#include "ParseAST.hpp"
#include "macroni/Common/GenerateModule.hpp"
#include "macroni/Dialect/Macroni/MacroniDialect.hpp"
#include "macroni/Translation/Macroni/MacroniVisitor.hpp"
#include "macroni/Translation/Macroni/PastaMetaGenerator.hpp"
#include "vast/Frontend/Action.hpp"
#include <cstdlib>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Verifier.h>

int main(int argc, char **argv) {
  auto maybe_ast = pasta::parse_ast(argc, argv);
  if (!maybe_ast.Succeeded()) {
    llvm::errs() << maybe_ast.TakeError() << '\n';
    return EXIT_FAILURE;
  }
  auto &pctx = maybe_ast.Value();
  auto &actx = pctx.UnderlyingAST();

  auto maybe_mod_and_context =
      macroni::generate_module<macroni::macroni::MacroniDialect,
                               macroni::pasta_meta_generator,
                               macroni::macroni_visitor>(actx, pctx);
  if (!maybe_mod_and_context) {
    llvm::errs() << "Could not make module\n";
    return EXIT_FAILURE;
  }

  auto mod = maybe_mod_and_context->first;
  mod->print(llvm::outs());

  return EXIT_SUCCESS;
}
