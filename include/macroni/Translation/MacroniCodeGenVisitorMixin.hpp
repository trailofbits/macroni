#pragma once

#include <macroni/Dialect/Macroni/MacroniDialect.hpp>
#include <macroni/Dialect/Macroni/MacroniOps.hpp>
#include <macroni/Translation/MacroniMetaGenerator.hpp>
#include <vast/Translation/CodeGenVisitor.hpp>

#include <pasta/AST/AST.h>
#include <pasta/AST/Macro.h>

// TODO(bpp): Instead of using a global variable for the PASTA AST and MLIR
// context, find out how to pass these to a CodeGen object.
extern std::optional<pasta::AST> ast;

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

        using DeclVisitor::Visit;
        using TypeVisitor::Visit;
        using StmtVisitor::make_maybe_value_yield_region;
        using StmtVisitor::builder;

        std::set<pasta::MacroSubstitution> visited;
        std::int64_t lock_level = 0;

        mlir::Operation *Visit(const clang::Stmt *stmt) {
            if (clang::isa<clang::ImplicitValueInitExpr,
                clang::ImplicitCastExpr>(stmt)) {
                // Don't visit implicit expressions
                return VisitNonMacro(stmt);
            }

            // Find the lowest macro that covers this statement, if any
            auto pasta_stmt = ast->Adopt(stmt);
            auto sub = lowest_unvisited_substitution(pasta_stmt, visited);
            if (!sub) {
                // If no substitution covers this statement, visit it normally.
                return VisitNonMacro(stmt);
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
            if (sub->Kind() == pasta::MacroKind::kExpansion) {
                return StmtVisitor::template make<macroni::MacroExpansion>(
                    loc,
                    builder().getStringAttr(llvm::Twine(macro_name)),
                    builder().getStrArrayAttr(llvm::ArrayRef(parameter_names)),
                    builder().getBoolAttr(function_like),
                    return_type,
                    std::move(region)
                );
            } else {
                return StmtVisitor::template make<macroni::MacroParameter>(
                    loc,
                    builder().getStringAttr(llvm::Twine(macro_name)),
                    return_type,
                    std::move(region)
                );
            }
        }

        void set_lock_level(mlir::Operation *op) {
            op->setAttr("lock_level", builder().getI64IntegerAttr(lock_level));
        }

        mlir::Operation *VisitNonMacro(const clang::Stmt *stmt) {
            auto op = StmtVisitor::Visit(stmt);
            auto call_op = mlir::dyn_cast<vast::hl::CallOp>(op);
            if (!call_op) {
                return op;
            }

            auto name = call_op.getCalleeAttr().getValue();
            if ("rcu_read_lock" == name) {
                set_lock_level(call_op);
                lock_level++;
            } else if ("rcu_read_unlock" == name) {
                lock_level--;
                set_lock_level(call_op);
            }
            return op;
        }
    };

    template<typename Derived>
    using MacroniVisitorConfig = vast::cg::CodeGenFallBackVisitorMixin<Derived,
        MacroniCodeGenVisitorMixin, vast::cg::DefaultFallBackVisitorMixin>;

    using MacroniVisitor = vast::cg::CodeGenVisitor<MacroniVisitorConfig,
        MacroniMetaGenerator>;

} // namespace macroni
