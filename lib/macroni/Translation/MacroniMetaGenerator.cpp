#include "macroni/Translation/MacroniMetaGenerator.hpp"
#include "pasta/AST/Macro.h"
#include "pasta/Util/File.h"
#include "vast/Util/Common.hpp"
#include <clang/AST/DeclBase.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/SourceLocation.h>
#include <llvm/ADT/Twine.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <optional>

namespace macroni {
macroni_meta_generator::macroni_meta_generator(vast::acontext_t *actx,
                                               vast::mcontext_t *mctx)
    : m_actx(actx), m_mctx(mctx) {}

vast::loc_t macroni_meta_generator::location(const clang::Decl *decl) const {
  return location(decl->getLocation());
}

vast::loc_t macroni_meta_generator::location(const clang::Stmt *stmt) const {
  return location(stmt->getBeginLoc());
}

vast::loc_t macroni_meta_generator::location(const clang::Expr *expr) const {
  return location(expr->getExprLoc());
}

vast::loc_t
macroni_meta_generator::location(pasta::MacroSubstitution sub) const {
  auto begin_token = sub.BeginToken();
  auto file_loc = begin_token ? begin_token->FileLocation() : std::nullopt;
  auto file = file_loc ? pasta::File::Containing(file_loc) : std::nullopt;
  auto line = file_loc ? file_loc->Line() : 0;
  auto column = file_loc ? file_loc->Column() : 0;
  auto filepath = file ? file->Path().string() : "<unknown>";
  auto filename = mlir::StringAttr::get(m_mctx, llvm::Twine(filepath));
  return mlir::FileLineColLoc::get(filename, line, column);
}

vast::loc_t
macroni_meta_generator::location(const clang::FullSourceLoc &loc) const {
  auto file_entry_ref = loc.getFileEntryRef();
  auto file = file_entry_ref ? file_entry_ref->getName() : "unknown";
  auto line = loc.getLineNumber();
  auto col = loc.getColumnNumber();
  return {mlir::FileLineColLoc::get(m_mctx, file, line, col)};
}

vast::loc_t
macroni_meta_generator::location(const clang::SourceLocation &loc) const {
  if (loc.isValid()) {
    auto &SM = m_actx->getSourceManager();
    auto full_source_loc = clang::FullSourceLoc(loc, SM);
    return location(full_source_loc);
  }
  return mlir::UnknownLoc::get(m_mctx);
}
} // namespace macroni
