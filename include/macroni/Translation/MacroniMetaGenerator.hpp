#pragma once

#include "pasta/AST/Macro.h"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/Util/Common.hpp"
#include <clang/AST/DeclBase.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/SourceLocation.h>

namespace macroni {
// The macroni meta generator is mostly the same as vast's default meta
// generator, with an additional method for returning the locations of
// pasta macro substitutions.
struct macroni_meta_generator final : public vast::cg::meta_generator {

public:
  [[nodiscard]] macroni_meta_generator(vast::acontext_t *actx,
                                       vast::mcontext_t *mctx);

  [[nodiscard]] vast::loc_t location(const clang::Decl *decl) const override;

  [[nodiscard]] vast::loc_t location(const clang::Stmt *stmt) const override;

  [[nodiscard]] vast::loc_t location(const clang::Expr *expr) const override;

  [[nodiscard]] vast::loc_t location(pasta::MacroSubstitution sub) const;

private:
  [[nodiscard]] vast::loc_t location(const clang::FullSourceLoc &loc) const;

  [[nodiscard]] vast::loc_t location(const clang::SourceLocation &loc) const;

  vast::acontext_t *m_actx;
  vast::mcontext_t *m_mctx;
};
} // namespace macroni
