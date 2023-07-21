#pragma once

#include <pasta/AST/AST.h>
#include <pasta/AST/Macro.h>
#include <vast/Translation/CodeGenMeta.hpp>

namespace macroni {
    struct MacroniMetaGenerator {
        MacroniMetaGenerator(const pasta::AST &ast, mlir::MLIRContext *mctx)
            : ast(ast), mctx(mctx) {}

        vast::cg::DefaultMeta get(const clang::FullSourceLoc &loc) const {
            if (!loc.isValid()) {
                return { mlir::FileLineColLoc::get(mctx, "<invalid>", 0, 0) };
            }
            auto file_entry = loc.getFileEntry();
            auto file = file_entry ? file_entry->getName() : "unknown";
            auto line = loc.getLineNumber();
            auto col = loc.getColumnNumber();
            return { mlir::FileLineColLoc::get(mctx, file, line, col) };
        }

        vast::cg::DefaultMeta get(const clang::SourceLocation &loc) const {
            clang::SourceManager &sm = ast.UnderlyingAST().getSourceManager();
            return get(clang::FullSourceLoc(loc, sm));
        }

        vast::cg::DefaultMeta get(const clang::Decl *decl) const {
            return get(decl->getLocation());
        }

        vast::cg::DefaultMeta get(const clang::Stmt *stmt) const {
            // TODO: use SourceRange
            return get(stmt->getBeginLoc());
        }

        vast::cg::DefaultMeta get(const clang::Expr *expr) const {
            // TODO: use SourceRange
            return get(expr->getExprLoc());
        }

        vast::cg::DefaultMeta get(const clang::TypeLoc &loc) const {
            // TODO: use SourceRange
            return get(loc.getBeginLoc());
        }

        vast::cg::DefaultMeta get(const clang::Type *type) const {
            return get(clang::TypeLoc(type, nullptr));
        }

        vast::cg::DefaultMeta get(clang::QualType type) const {
            return get(clang::TypeLoc(type, nullptr));
        }

        vast::cg::DefaultMeta get(pasta::MacroSubstitution &sub) const {
            // TODO(bpp): Define this to something that makes sense. Right now
            // this just returns an invalid source location.
            return get(clang::SourceLocation());
        }

        const pasta::AST &ast;
        mlir::MLIRContext *mctx;
    };
} // namespace macroni
