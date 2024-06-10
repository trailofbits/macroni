// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in the
// LICENSE file found in the root directory of this source tree.

#include "macroni/Common/ParseAST.hpp"
#include "vast/CodeGen/CodeGenDriver.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenOptions.hpp"
#include "vast/CodeGen/Mangler.hpp"
#include "vast/Frontend/Options.hpp"
#include "vast/Util/Common.hpp"
#include <clang/AST/ASTContext.h>
#include <iostream>

std::unique_ptr<vast::cg::driver>
generate_macroni_driver(vast::acontext_t &actx) {
  auto mctx = vast::cg::mk_mcontext();
  auto bld = vast::cg::mk_codegen_builder(mctx.get());
  auto mg = std::make_unique<vast::cg::default_meta_gen>(&actx, mctx.get());
  auto mangle_context = actx.createMangleContext();
  auto sg = std::make_unique<vast::cg::default_symbol_mangler>(mangle_context);
  vast::cg::options copts = {
      .lang = vast::cc::get_source_language(actx.getLangOpts()),
      .optimization_level = 0,
      .has_strict_return = false,
      .disable_unsupported = false,
      .disable_vast_verifier = true,
      .prepare_default_visitor_stack = true};

  return std::make_unique<vast::cg::driver>(actx, std::move(mctx),
                                            std::move(copts), std::move(bld),
                                            std::move(mg), std::move(sg));
}

int main(int argc, char **argv) {
  auto maybe_ast = pasta::parse_ast(argc, argv);
  if (!maybe_ast.Succeeded()) {
    std::cerr << maybe_ast.TakeError() << '\n';
    return EXIT_FAILURE;
  }
  auto pasta_ast = maybe_ast.TakeValue();
  auto &actx = pasta_ast.UnderlyingAST();
  auto driver = generate_macroni_driver(actx);

  driver->emit(actx.getTranslationUnitDecl());
  driver->finalize();
  auto mod = driver->freeze();
  mod->print(llvm::outs());

  return EXIT_SUCCESS;
}
