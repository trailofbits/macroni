#pragma once

#include <macroni/Dialect/Kernel/KernelTypes.hpp>
#include <macroni/Dialect/Macroni/MacroniDialect.hpp>
#include <macroni/Dialect/Macroni/MacroniOps.hpp>
#include <macroni/Translation/MacroniMetaGenerator.hpp>
#include <pasta/AST/AST.h>
#include <pasta/AST/Macro.h>
#include <vast/Translation/CodeGenVisitor.hpp>

namespace macroni {
    // Given a set of visited substitutions, returns the lowest substitutions in
    // this macros chain of aligned substitution that has not yet been visited,
    // and marks it as visited.
    std::optional<pasta::MacroSubstitution> lowest_unvisited_substitution(
        pasta::Stmt &stmt,
        std::set<pasta::MacroSubstitution> &visited);

    // Given a substitution, returns whether that substitution is an expansion
    // of a function-like macro. Returns the given default value if the
    // substitution lacks the necessary information to determine this.
    bool is_function_like(pasta::MacroSubstitution &sub, bool default_ = false);

    // Given a substitution, returns the names of the names of the
    // substitution's macro parameters, if any.
    std::vector<llvm::StringRef>
        get_parameter_names(pasta::MacroSubstitution &sub);

    template<typename Derived>
    struct MacroniCodeGenVisitorMixin
        : vast::cg::CodeGenDeclVisitorMixin<Derived>
        , vast::cg::CodeGenStmtVisitorMixin<Derived>
        , vast::cg::CodeGenTypeVisitorWithDataLayoutMixin<Derived> {
        using DeclVisitor = vast::cg::CodeGenDeclVisitorMixin<Derived>;
        using StmtVisitor = vast::cg::CodeGenStmtVisitorMixin<Derived>;
        using TypeVisitor =
            vast::cg::CodeGenTypeVisitorWithDataLayoutMixin<Derived>;

        using StmtVisitor::make_maybe_value_yield_region;
        using StmtVisitor::builder;
        using StmtVisitor::LensType::meta_gen;

        std::set<pasta::MacroSubstitution> visited;

        virtual ~MacroniCodeGenVisitorMixin() {}

        mlir::Type Visit(const clang::Type *type) {
            return TypeVisitor::Visit(type);
        }

        mlir::Type Visit(clang::QualType type) {
            return TypeVisitor::Visit(type);
        }

        mlir::Operation *Visit(const clang::Decl *decl) {
            return DeclVisitor::Visit(decl);
        }

        mlir::Operation *Visit(const clang::Stmt *stmt) {
            // FIXME(bpp): Retrieve the pasta::AST from context() once VAST
            // allows visitors to retrieve code gen contexts inherited from
            // vast::cg::CodeGenContext
            auto pasta_stmt = meta_gen().ast.Adopt(stmt);

            if (clang::isa<clang::ImplicitValueInitExpr,
                clang::ImplicitCastExpr>(stmt)) {
                // Don't visit implicit expressions
                auto op = StmtVisitor::Visit(stmt);
                UnalignedStmtVisited(pasta_stmt, op);
                return op;
            }

            // Find the lowest macro that covers this statement, if any
            auto sub = lowest_unvisited_substitution(pasta_stmt, visited);
            if (!sub) {
                // If no substitution covers this statement, visit it normally.
                auto op = StmtVisitor::Visit(stmt);
                UnalignedStmtVisited(pasta_stmt, op);
                return op;
            }

            // Get the substitution's location, name, parameter names, and
            // whether it is function-like
            auto loc = StmtVisitor::meta_location(*sub);
            auto name_tok = sub->NameOrOperator();
            auto macro_name = (name_tok
                               ? name_tok->Data()
                               : "<a nameless macro>");
            auto function_like = is_function_like(*sub);
            auto parameter_names = get_parameter_names(*sub);

            // We call `make_maybe_value_yield_region` here because a macro may
            // not expand to an expression
            auto [region, return_type] = make_maybe_value_yield_region(stmt);

            // Check if the macro is an expansion or a parameter, and return the
            // appropriate operation
            mlir::Operation *op = nullptr;
            if (sub->Kind() == pasta::MacroKind::kExpansion) {
                op = StmtVisitor::template make<macroni::MacroExpansion>(
                    loc,
                    builder().getStringAttr(llvm::Twine(macro_name)),
                    builder().getStrArrayAttr(llvm::ArrayRef(parameter_names)),
                    builder().getBoolAttr(function_like),
                    return_type,
                    std::move(region)
                );
            } else {
                op = StmtVisitor::template make<macroni::MacroParameter>(
                    loc,
                    builder().getStringAttr(llvm::Twine(macro_name)),
                    return_type,
                    std::move(region)
                );
            }
            AlignedStmtVisited(pasta_stmt, *sub, op);
            return op;
        }

        // Hook called whenever Macroni finishes visiting a Stmt that does not
        // align with a macro substitution.
        // \param pasta_stmt The `pasta::Stmt` that does not align with a macro
        // substitution.
        // \param op The `mlir::Operation` obtained from visiting the `Stmt`.
        virtual void UnalignedStmtVisited(pasta::Stmt &pasta_stmt,
                                          mlir::Operation *op) {}

        // Hook called whenever Macroni finishes visiting a Stmt that align with
        // a macro substitution.
        // \param pasta_stmt The `pasta::Stmt` that aligns with a macro
        // substitution.
        // \param sub The `pasta::MacroSubstitution` that `pasta_stmt` aligns
        // with.
        // \param op The `mlir::Operation` obtained from visiting the `Stmt`.
        virtual void AlignedStmtVisited(pasta::Stmt &pasta_stmt,
                                        pasta::MacroSubstitution &sub,
                                        mlir::Operation *op) {}

    };

    template<typename Derived>
    using MacroniVisitorConfig = vast::cg::CodeGenFallBackVisitorMixin<Derived,
        MacroniCodeGenVisitorMixin, vast::cg::DefaultFallBackVisitorMixin>;

    using MacroniVisitor = vast::cg::CodeGenVisitor<MacroniVisitorConfig,
        MacroniMetaGenerator>;

} // namespace macroni
