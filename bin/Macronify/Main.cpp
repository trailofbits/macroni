// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in the
// LICENSE file found in the root directory of this source tree.

#include "macroni/Common/ParseAST.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenDriver.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenOptions.hpp"
#include "vast/CodeGen/Mangler.hpp"
#include "vast/Frontend/Options.hpp"
#include <iostream>

int main(int argc, char **argv) {
  auto maybe_ast = pasta::parse_ast(argc, argv);
  if (!maybe_ast.Succeeded()) {
    std::cerr << maybe_ast.TakeError() << '\n';
    return EXIT_FAILURE;
  }
  auto pasta_ast = maybe_ast.TakeValue();
  auto &ast = pasta_ast.UnderlyingAST();
  auto mangle_ctx = ast.createMangleContext();

  auto mctx = vast::cg::mk_mcontext();
  auto codegen_builder = std::make_unique<vast::cg::codegen_builder>(&*mctx);
  auto mg = std::make_unique<vast::cg::default_meta_gen>(&ast, &*mctx);
  auto sg = std::make_unique<vast::cg::default_symbol_mangler>(mangle_ctx);
  vast::cg::options opts{
      .lang = vast::cg::source_language::C,
      .optimization_level = 0,
      .has_strict_return = false,
      .disable_unsupported = false,
      .disable_vast_verifier = true,
      .prepare_default_visitor_stack = true,
  };
  vast::cc::vast_args vast_args;

  vast::cg::driver driver(ast, std::move(mctx), opts,
                          std::move(codegen_builder), std::move(mg),
                          std::move(sg));

  driver.emit(ast.getTranslationUnitDecl());
  driver.finalize();
  driver.verify();

  auto mod = driver.freeze();

  mod->dump();

  return EXIT_SUCCESS;
}
