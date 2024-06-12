#pragma once

#include "vast/CodeGen/CodeGenDriver.hpp"
#include "vast/Util/Common.hpp"
#include <clang/AST/ASTContext.h>
#include <memory>

namespace macroni {
// Creates a vast::cg::driver type with all MLIR and VAST dialects preloaded,
// along with the given dialect and metagenerator, from the given AST. The
// driver does not have any visitors loaded into it, so callers will have to set
// up the visitor stack themselves.
template <typename dialect_t, typename metagen_t>
std::unique_ptr<vast::cg::driver>
generate_codegen_driver(vast::acontext_t &actx) {
  // Load MLIR and VAST dialects into registry.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  vast::registerAllDialects(registry);

  // Insert our custom dialect.
  registry.insert<dialect_t>();
  auto mctx = std::make_unique<vast::mcontext_t>();
  mctx->appendDialectRegistry(registry);
  mctx->loadAllAvailableDialects();

  // Set up variables for constructing the codegen driver.
  auto bld = vast::cg::mk_codegen_builder(mctx.get());
  auto mg = std::make_unique<metagen_t>(&actx, mctx.get());
  auto mangle_context = actx.createMangleContext();
  auto sg = std::make_unique<vast::cg::default_symbol_mangler>(mangle_context);
  vast::cg::options copts = {
      .lang = vast::cc::get_source_language(actx.getLangOpts()),
      .optimization_level = 0,
      .has_strict_return = false,
      .disable_unsupported = false,
      .disable_vast_verifier = true,
      .prepare_default_visitor_stack = false};

  // Create the codegen driver.
  return std::make_unique<vast::cg::driver>(actx, std::move(mctx),
                                            std::move(copts), std::move(bld),
                                            std::move(mg), std::move(sg));
}
} // namespace macroni
