#include "macroni/Common/GenerateMacroniDriver.hpp"
#include "vast/CodeGen/CodeGenDriver.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/Mangler.hpp"
#include "vast/Frontend/Options.hpp"
#include "vast/Util/Common.hpp"
#include <memory>
#include <utility>

namespace macroni {
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
} // namespace macroni