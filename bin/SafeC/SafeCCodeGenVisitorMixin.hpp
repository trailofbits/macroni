#pragma once

#include <macroni/Dialect/Safety/SafetyDialect.hpp>
#include <macroni/Translation/MacroniCodeGenVisitorMixin.hpp>
#include <macroni/Translation/MacroniMetaGenerator.hpp>
#include <pasta/AST/Macro.h>
#include <pasta/AST/Stmt.h>
#include <vast/CodeGen/CodeGen.hpp>

template <typename Derived>
struct SafeCCodeGenVisitorMixin
    : macroni::MacroniCodeGenVisitorMixin<Derived> {

    using parent_t = macroni::MacroniCodeGenVisitorMixin<Derived>;

    using parent_t::builder;
    using parent_t::StmtVisitor::LensType::mcontext;

    using parent_t::Visit;

    void UnalignedStmtVisited(pasta::Stmt &pasta_stmt,
                              mlir::Operation *op) {
        if (auto if_stmt = pasta::IfStmt::From(pasta_stmt)) {
            VisitIfStmt(*if_stmt, op);
        }
    }

    void VisitIfStmt(pasta::IfStmt &pasta_if, mlir::Operation *op) {
        // The `unsafe` macro expands to `if (0) ; else`. Therefore, any if
        // statement expanded from `unsafe` should be a NullStmt.
        if (!pasta::NullStmt::From(pasta_if.Then())) {
            return;
        }
        // Find the macro that aligns with the statement's `if` and `else`
        // tokens.
        auto token_range = pasta::TokenRange::From(pasta_if.IfToken(),
                                                   pasta_if.ElseToken());
        if (!token_range) {
            return;
        }
        auto aligned_subs = token_range->AlignedSubstitutions(false);
        auto is_unsafe_exp = [](pasta::MacroSubstitution &sub) -> bool {
            auto exp = pasta::MacroSubstitution::From(sub);
            if (!exp) {
                return false;
            }
            auto name = exp->NameOrOperator();
            if (!name) {
                return false;
            }
            // Use .equals() instead of == to avoid operator ambiguity.
            return macroni::safety::SafetyDialect::unsafe().
                equals(name->Data());
            };
        if (std::any_of(aligned_subs.begin(), aligned_subs.end(),
                        is_unsafe_exp)) {
            op->setAttr(macroni::safety::SafetyDialect::unsafe(),
                        builder().getBoolAttr(true));
        }
    }
};

template<typename Derived>
using SafeCVisitorConfig = vast::cg::FallBackVisitor<Derived,
    SafeCCodeGenVisitorMixin,
    vast::cg::UnsupportedVisitor,
    vast::cg::UnreachableVisitor
>;

using SafeCCodeGen = vast::cg::CodeGen<
    macroni::MacroniCodeGenContext,
    SafeCVisitorConfig,
    macroni::MacroniMetaGenerator
>;