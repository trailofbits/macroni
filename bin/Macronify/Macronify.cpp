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
#include "vast/Util/Common.hpp"
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
  auto mctx = vast::mcontext_t();

  auto mod =
      macroni::generate_module<macroni::macroni::MacroniDialect,
                               macroni::pasta_meta_generator,
                               macroni::macroni_visitor>(actx, mctx, pctx);
  if (!mod) {
    llvm::errs() << "Could not make module\n";
    return EXIT_FAILURE;
  }
  mod->print(llvm::outs());

  return EXIT_SUCCESS;
}
