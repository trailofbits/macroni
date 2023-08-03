#pragma once

#include <macroni/Translation/MacroniCodeGenVisitorMixin.hpp>
#include <macroni/Translation/MacroniMetaGenerator.hpp>
#include <pasta/AST/Macro.h>
#include <pasta/AST/Stmt.h>

template <typename Derived>
struct SafeCCodeGenVisitorMixin
    : macroni::MacroniCodeGenVisitorMixin<Derived> {

    using parent_t = macroni::MacroniCodeGenVisitorMixin<Derived>;

    using parent_t::builder;
    using parent_t::StmtVisitor::LensType::mcontext;

    using parent_t::Visit;

    void UnalignedStmtVisited(pasta::Stmt &pasta_stmt,
                              mlir::Operation *op) override {
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
    }
};

template<typename Derived>
using SafeCVisitorConfig = vast::cg::CodeGenFallBackVisitorMixin<Derived,
    SafeCCodeGenVisitorMixin, vast::cg::DefaultFallBackVisitorMixin>;

using SafeCVisitor = vast::cg::CodeGenVisitor<SafeCVisitorConfig,
    macroni::MacroniMetaGenerator>;