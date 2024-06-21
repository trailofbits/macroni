#include "macroni/Translation/Macroni/PastaMetaGenerator.hpp"
#include "macroni/Common/MacroniMetaGenerator.hpp"
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
pasta_meta_generator::pasta_meta_generator(vast::acontext_t *actx,
                                           vast::mcontext_t *mctx)
    : macroni_meta_generator(actx, mctx) {}

vast::loc_t pasta_meta_generator::location(pasta::MacroSubstitution sub) const {
  auto begin_token = sub.BeginToken();
  auto file_loc = begin_token ? begin_token->FileLocation() : std::nullopt;
  auto file = file_loc ? pasta::File::Containing(file_loc) : std::nullopt;
  auto line = file_loc ? file_loc->Line() : 0;
  auto column = file_loc ? file_loc->Column() : 0;
  auto filepath = file ? file->Path().string() : "<unknown>";
  auto filename = mlir::StringAttr::get(m_mctx, llvm::Twine(filepath));
  return mlir::FileLineColLoc::get(m_mctx, filename, line, column);
}

} // namespace macroni
