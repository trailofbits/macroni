// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in the
// LICENSE file found in the root directory of this source tree.

#include "macroni/Common/ParseAST.hpp"
#include "macroni/Dialect/Macroni/MacroniDialect.hpp"
#include "macroni/Translation/MacroniMetaGenerator.hpp"
#include "macroni/Translation/MacroniVisitor.hpp"
#include "pasta/AST/AST.h"
#include "vast/CodeGen/CodeGenDriver.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/DefaultVisitor.hpp"
#include "vast/CodeGen/UnreachableVisitor.hpp"
#include "vast/CodeGen/UnsupportedVisitor.hpp"
#include "vast/Dialect/Dialects.hpp"
#include "vast/Util/Common.hpp"
#include <clang/AST/ASTContext.h>
#include <iostream>
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>

// Creates a new base visitor suitable for pushing to the given driver's stack.
template <typename visitor_t>
vast::cg::visitor_base_ptr mk_visitor(vast::cg::driver &driver) {
  return std::make_unique<visitor_t>(
      driver.mcontext(), driver.get_codegen_builder(),
      driver.get_meta_generator(), driver.get_symbol_generator(),
      vast::cg::visitor_view(driver.get_visitor_stack()));
}

namespace macroni {
std::unique_ptr<vast::cg::driver> generate_macroni_driver(pasta::AST &pctx) {
  auto &actx = pctx.UnderlyingAST();

  // Load MLIR, VAST, and Macroni dialects into registry.

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  vast::registerAllDialects(registry);
  registry.insert<macroni::MacroniDialect>();
  auto mctx = std::make_unique<vast::mcontext_t>();
  mctx->appendDialectRegistry(registry);
  mctx->loadAllAvailableDialects();

  // Set up variables for constructing the codegen driver.

  auto bld = vast::cg::mk_codegen_builder(mctx.get());
  auto mg = std::make_unique<macroni_meta_generator>(&actx, mctx.get());
  auto mangle_context = actx.createMangleContext();
  auto sg = std::make_unique<vast::cg::default_symbol_mangler>(mangle_context);
  vast::cg::options copts = {
      .lang = vast::cc::get_source_language(actx.getLangOpts()),
      .optimization_level = 0,
      .has_strict_return = false,
      .disable_unsupported = false,
      .disable_vast_verifier = true,
      .prepare_default_visitor_stack = false};

  // Create the codegen driver and set up visitor stack.

  auto driver = std::make_unique<vast::cg::driver>(
      actx, std::move(mctx), std::move(copts), std::move(bld), std::move(mg),
      std::move(sg));

  // Add the macroni visitor first. This ensures that our visitor will try to
  // lower stmts down to the Macroni dialect first before VAST's dialect.
  driver->push_visitor(std::make_unique<macroni_visitor>(
      pctx, driver->mcontext(), driver->get_codegen_builder(),
      driver->get_meta_generator(), driver->get_symbol_generator(),
      vast::cg::visitor_view(driver->get_visitor_stack())));

  // Then add the default visitor stack.
  driver->push_visitor(mk_visitor<vast::cg::default_visitor>(*driver));

  driver->push_visitor(mk_visitor<vast::cg::unsup_visitor>(*driver));

  driver->push_visitor(std::make_unique<vast::cg::unreach_visitor>(
      driver->mcontext(), driver->get_meta_generator(),
      driver->get_symbol_generator(), driver->get_options()));

  return driver;
}
} // namespace macroni

int main(int argc, char **argv) {
  auto maybe_ast = pasta::parse_ast(argc, argv);
  if (!maybe_ast.Succeeded()) {
    std::cerr << maybe_ast.TakeError() << '\n';
    return EXIT_FAILURE;
  }

  auto pctx = maybe_ast.TakeValue();
  auto &actx = pctx.UnderlyingAST();
  auto driver = macroni::generate_macroni_driver(pctx);

  driver->emit(actx.getTranslationUnitDecl());
  driver->finalize();
  auto mod = driver->freeze();
  mod->print(llvm::outs());

  return EXIT_SUCCESS;
}
