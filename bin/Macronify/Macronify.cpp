// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in the
// LICENSE file found in the root directory of this source tree.

#include "ParseAST.hpp"
#include "macroni/Dialect/Macroni/MacroniDialect.hpp"
#include "macroni/Translation/Macroni/MacroniVisitor.hpp"
#include "macroni/Translation/Macroni/PastaMetaGenerator.hpp"
#include "pasta/AST/AST.h"
#include "vast/CodeGen/AttrVisitorProxy.hpp"
#include "vast/CodeGen/CodeGenDriver.hpp"
#include "vast/CodeGen/CodeGenVisitorList.hpp"
#include "vast/CodeGen/DefaultVisitor.hpp"
#include "vast/CodeGen/FallthroughVisitor.hpp"
#include "vast/CodeGen/TypeCachingProxy.hpp"
#include "vast/CodeGen/UnsupportedVisitor.hpp"
#include <clang/AST/ASTContext.h>
#include <cstdlib>
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

  // Load MLIR and VAST dialects into registry.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  vast::registerAllDialects(registry);

  // Insert our custom dialect.
  registry.insert<macroni::macroni::MacroniDialect>();
  auto mctx = std::make_unique<vast::mcontext_t>();
  mctx->appendDialectRegistry(registry);
  mctx->loadAllAvailableDialects();

  // Set up variables for constructing the codegen driver.

  auto &actx = maybe_ast->UnderlyingAST();
  auto bld = vast::cg::mk_codegen_builder(*mctx);
  auto mg = std::make_shared<macroni::pasta_meta_generator>(&actx, &*mctx);
  auto sg = std::make_shared<vast::cg::default_symbol_generator>(
      actx.createMangleContext());

  using vast::cg::as_node;
  using vast::cg::as_node_with_list_ref;
  auto visitors = std::make_shared<vast::cg::visitor_list>() |
                  as_node_with_list_ref<macroni::macroni_visitor>(
                      *mctx, *bld, *mg, maybe_ast.Value()) |
                  as_node_with_list_ref<vast::cg::attr_visitor_proxy>() |
                  as_node<vast::cg::type_caching_proxy>() |
                  as_node_with_list_ref<vast::cg::default_visitor>(
                      *mctx, *bld, mg, sg,
                      /* strict return = */ false,
                      vast::cg::missing_return_policy::emit_trap) |
                  as_node_with_list_ref<vast::cg::unsup_visitor>(*mctx, *bld) |
                  as_node<vast::cg::fallthrough_visitor>();

  // Create the codegen driver.
  auto driver =
      std::make_unique<vast::cg::driver>(actx, *mctx, std::move(bld), visitors);
  driver->enable_verifier(true);

  // Emit the lowered MLIR.

  driver->emit(actx.getTranslationUnitDecl());
  driver->finalize();
  if (!driver->verify()) {
    llvm::errs() << "Failed to verify driver\n";
    return EXIT_FAILURE;
  }
  auto mod = driver->freeze();
  mod->print(llvm::outs());

  return EXIT_SUCCESS;
}
