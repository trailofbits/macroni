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

        using DeclVisitor::Visit;
        using StmtVisitor::make_maybe_value_yield_region;
        using StmtVisitor::builder;
        using StmtVisitor::LensType::mcontext;
        using StmtVisitor::LensType::meta_gen;

        std::set<pasta::MacroSubstitution> visited;
        std::int64_t lock_level = 0;

        mlir::Type Visit(const clang::Type *type) {
            return TypeVisitor::Visit(type);
        }

        // Overload the TypeVisitor's visit method for QualTypes to convert
        // special VAST types to Kernel types.
        mlir::Type Visit(clang::QualType type) {
            auto ty = TypeVisitor::Visit(type);
            // Return early if we are not converting.
            if (!meta_gen().convert) {
                return ty;
            }
            auto attributed_type = clang::dyn_cast<clang::AttributedType>(type);
            if (!attributed_type) {
                return ty;
            }
            auto attr = attributed_type->getAttr();
            using ASA = clang::AddressSpaceAttr;
            auto addr_space = clang::dyn_cast_or_null<ASA>(attr);
            if (!addr_space) {
                return ty;
            }
            // NOTE(bpp): Clang does not record to address space passed to the
            // attribute in the source code. Instead, it record the value passed
            // PLUS the value of the last enumerator in Clang's LangAS enum. So
            // to get the original value, we just subtract this enumerator's
            // value from the value attached to the AddressSpaceAttr.
            using clang::LangAS;
            using std::underlying_type_t;
            auto FirstAddrSpace = LangAS::FirstTargetAddressSpace;
            int first = static_cast<underlying_type_t<LangAS>>(FirstAddrSpace);
            int space = addr_space->getAddressSpace() - first;
            return kernel::AddressSpaceType::get(&mcontext(), ty, space);
        }

        mlir::Operation *Visit(const clang::Stmt *stmt) {
            if (clang::isa<clang::ImplicitValueInitExpr,
                clang::ImplicitCastExpr>(stmt)) {
                // Don't visit implicit expressions
                return VisitNonMacro(stmt);
            }

            // Find the lowest macro that covers this statement, if any
            auto pasta_stmt = meta_gen().ast.Adopt(stmt);
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

        mlir::Operation *VisitNonMacro(const clang::Stmt *stmt) {
            if (auto call_expr = clang::dyn_cast<clang::CallExpr>(stmt)) {
                return VisitCallExpr(call_expr);
            } else if (auto if_stmt = clang::dyn_cast<clang::IfStmt>(stmt)) {
                return VisitIfStmt(if_stmt);
            }
            return StmtVisitor::Visit(stmt);
        }

        void set_lock_level(mlir::Operation *op) {
            op->setAttr("lock_level", builder().getI64IntegerAttr(lock_level));
        }

        mlir::Operation *VisitCallExpr(const clang::CallExpr *call_expr) {
            auto op = StmtVisitor::Visit(call_expr);
            auto call_op = mlir::dyn_cast<vast::hl::CallOp>(op);
            if (!call_op) {
                return op;
            }
            auto name = call_op.getCalleeAttr().getValue();
            if ("rcu_read_lock" == name) {
                set_lock_level(op);
                lock_level++;
            } else if ("rcu_read_unlock" == name) {
                lock_level--;
                set_lock_level(op);
            }
            return op;
        }

        mlir::Operation *VisitIfStmt(const clang::IfStmt *if_stmt) {
            auto op = StmtVisitor::Visit(if_stmt);
            pasta::Stmt pasta_stmt = meta_gen().ast.Adopt(if_stmt);
            auto pasta_if = pasta::IfStmt::From(pasta_stmt);
            if (!pasta_if) {
                return op;
            }
            // The `unsafe` macro expands to `if (0) ; else`. Therefore, any if
            // statement expanded from `unsafe` should be a NullStmt.
            if (!pasta::NullStmt::From(pasta_if->Then())) {
                return op;
            }
            // Find the macro that aligns with the statement's `if` and `else`
            // tokens.
            auto token_range = pasta::TokenRange::From(pasta_if->IfToken(),
                                                       pasta_if->ElseToken());
            if (!token_range) {
                return op;
            }
            auto aligned_subs = token_range->AlignedSubstitutions(false);
            if (std::any_of(
                aligned_subs.begin(),
                aligned_subs.end(),
                [](pasta::MacroSubstitution &sub) {
                    auto exp = pasta::MacroSubstitution::From(sub);
                    if (!exp) {
                        return false;
                    }
                    auto name = exp->NameOrOperator();
                    if (!name) {
                        return false;
                    }
                    return "unsafe" == name->Data();
                })) {
                op->setAttr("unsafe", builder().getBoolAttr(true));
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
