#pragma once

#include <pasta/AST/AST.h>
#include <pasta/AST/Macro.h>
#include <vast/CodeGen/CodeGenMeta.hpp>

namespace macroni {
struct MacroniMetaGenerator final : public vast::cg::meta_generator {

public:
  MacroniMetaGenerator(clang::ASTContext *ast, mlir::MLIRContext *mctx)
      : ast(ast), mctx(mctx), unknown_location(mlir::UnknownLoc::get(mctx)) {}

  mlir::Location location(const clang::FullSourceLoc &loc) const {
    auto file_entry = loc.getFileEntry();
    auto file = file_entry ? file_entry->getName() : "unknown";
    auto line = loc.getLineNumber();
    auto col = loc.getColumnNumber();
    return {mlir::FileLineColLoc::get(mctx, file, line, col)};
  }

  mlir::Location location(const clang::SourceLocation &loc) const {
    return location(clang::FullSourceLoc(loc, ast->getSourceManager()));
  }

  mlir::Location location(const clang::Decl *decl) const final {
    return location(decl->getLocation());
  }

  mlir::Location location(const clang::Stmt *stmt) const final {
    return location(stmt->getBeginLoc());
  }

  mlir::Location location(const clang::Expr *expr) const final {
    return location(expr->getExprLoc());
  }

  mlir::Location location(const clang::Type *type) const {
    return unknown_location;
  }

  mlir::Location location(const clang::Attr *type) const {
    return unknown_location;
  }

  mlir::Location location(const clang::QualType type) const {
    return unknown_location;
  }

  mlir::Location location(pasta::MacroSubstitution sub) const {
    auto begin_token = sub.BeginToken();
    auto file_loc = begin_token ? begin_token->FileLocation() : std::nullopt;
    auto file = file_loc ? pasta::File::Containing(file_loc) : std::nullopt;
    auto filepath = file ? file->Path().string() : "<unknown>";
    return mlir::FileLineColLoc::get(
        mlir::StringAttr::get(mctx, llvm::Twine(filepath)),
        file_loc ? file_loc->Line() : 0, file_loc ? file_loc->Column() : 0);
  }

  clang::ASTContext *ast;
  mlir::MLIRContext *mctx;
  const mlir::Location unknown_location;
};
} // namespace macroni
