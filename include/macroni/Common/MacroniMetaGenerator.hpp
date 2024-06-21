#pragma once

#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/Util/Common.hpp"
#include <clang/AST/DeclBase.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/SourceLocation.h>

namespace macroni {
// The macroni meta generator is mostly the same as vast's default meta
// generator, with the exception that its method for visiting source locations
// is public.
struct macroni_meta_generator : public vast::cg::meta_generator {

public:
  [[nodiscard]] macroni_meta_generator(vast::acontext_t *actx,
                                       vast::mcontext_t *mctx);

  [[nodiscard]] vast::loc_t location(const clang::Decl *decl) const override;

  [[nodiscard]] vast::loc_t location(const clang::Stmt *stmt) const override;

  [[nodiscard]] vast::loc_t location(const clang::Expr *expr) const override;

  [[nodiscard]] vast::loc_t location(const clang::SourceLocation &loc) const;

private:
  [[nodiscard]] vast::loc_t location(const clang::FullSourceLoc &loc) const;

protected:
  vast::acontext_t *m_actx;
  vast::mcontext_t *m_mctx;
};
} // namespace macroni
