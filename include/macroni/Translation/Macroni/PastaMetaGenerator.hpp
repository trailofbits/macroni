#pragma once

#include "macroni/Common/MacroniMetaGenerator.hpp"
#include "pasta/AST/Macro.h"
#include "vast/Util/Common.hpp"
#include <clang/AST/DeclBase.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/SourceLocation.h>

namespace macroni {
// The pasta meta generator is mostly the same as macroni's base meta generator,
// with an additional method for returning the locations of pasta macro
// substitutions.
struct pasta_meta_generator final : public macroni_meta_generator {

public:
  [[nodiscard]] pasta_meta_generator(vast::acontext_t *actx,
                                     vast::mcontext_t *mctx);

  using macroni_meta_generator::m_actx;
  using macroni_meta_generator::m_mctx;

  [[nodiscard]] vast::loc_t location(pasta::MacroSubstitution sub) const;
};
} // namespace macroni
