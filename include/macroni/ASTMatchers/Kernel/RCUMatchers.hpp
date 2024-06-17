#pragma once

#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>

namespace macroni::kernel {
static const clang::ast_matchers::StatementMatcher rcu_deference_matcher =
    clang::ast_matchers::stmtExpr(
        clang::ast_matchers::allOf(
            clang::ast_matchers::isExpandedFromMacro("rcu_dereference"),
            clang::ast_matchers::hasDescendant(
                clang::ast_matchers::parenExpr(clang::ast_matchers::has(
                    clang::ast_matchers::parenExpr(clang::ast_matchers::has(
                        clang::ast_matchers::expr().bind("p"))))))))
        .bind("rcu_dereference");

static const clang::ast_matchers::StatementMatcher rcu_assign_pointer_matcher =
    clang::ast_matchers::doStmt(
        clang::ast_matchers::allOf(
            clang::ast_matchers::isExpandedFromMacro("rcu_assign_pointer"),
            clang::ast_matchers::hasDescendant(
                clang::ast_matchers::callExpr(clang::ast_matchers::allOf(
                    clang::ast_matchers::callee(
                        clang::ast_matchers::functionDecl(
                            clang::ast_matchers::hasName("__write_once_size"))),
                    clang::ast_matchers::hasArgument(
                        0, clang::ast_matchers::hasDescendant(
                               clang::ast_matchers::parenExpr(
                                   clang::ast_matchers::has(
                                       clang::ast_matchers::parenExpr(
                                           clang::ast_matchers::has(
                                               clang::ast_matchers::expr().bind(
                                                   "p")))))))))),
            clang::ast_matchers::hasDescendant(
                clang::ast_matchers::callExpr(clang::ast_matchers::allOf(
                    clang::ast_matchers::callee(
                        clang::ast_matchers::functionDecl(
                            clang::ast_matchers::hasName(
                                "__builtin_constant_p"))),
                    clang::ast_matchers::hasArgument(
                        0, clang::ast_matchers::expr().bind("v")))))

                ))
        .bind("rcu_assign_pointer");

static const clang::ast_matchers::StatementMatcher rcu_access_pointer_matcher =
    clang::ast_matchers::stmtExpr(
        clang::ast_matchers::allOf(
            clang::ast_matchers::isExpandedFromMacro("rcu_access_pointer"),
            clang::ast_matchers::hasDescendant(
                clang::ast_matchers::parenExpr(clang::ast_matchers::has(
                    clang::ast_matchers::parenExpr(clang::ast_matchers::has(
                        clang::ast_matchers::expr().bind("p"))))))))
        .bind("rcu_access_pointer");

static const clang::ast_matchers::StatementMatcher rcu_replace_pointer_matcher =
    clang::ast_matchers::stmtExpr(
        clang::ast_matchers::allOf(
            clang::ast_matchers::isExpandedFromMacro("rcu_replace_pointer"),

            clang::ast_matchers::hasDescendant(clang::ast_matchers::callExpr(
                clang::ast_matchers::callee(clang::ast_matchers::functionDecl(
                    clang::ast_matchers::hasName("__write_once_size"))),
                // NOTE(Brent): We have to use hasAnyArgument() instead of
                // hasArgument() because for some reason Clang implicitly
                // ignores parentheseses for the latter matcher.
                clang::ast_matchers::hasAnyArgument(
                    clang::ast_matchers::ignoringImplicit(
                        clang::ast_matchers::unaryOperator(
                            clang::ast_matchers::hasOperatorName("&"),
                            clang::ast_matchers::has(
                                clang::ast_matchers::parenExpr(
                                    clang::ast_matchers::has(
                                        clang::ast_matchers::parenExpr(
                                            clang::ast_matchers::has(
                                                clang::ast_matchers::parenExpr(
                                                    clang::ast_matchers::has(
                                                        clang::ast_matchers::
                                                            expr()
                                                                .bind("rcu_ptr")

                                                            ))))))))))

                    )),

            clang::ast_matchers::hasDescendant(clang::ast_matchers::callExpr(
                clang::ast_matchers::callee(clang::ast_matchers::functionDecl(
                    clang::ast_matchers::hasName("__builtin_constant_p"))),
                clang::ast_matchers::hasAnyArgument(
                    clang::ast_matchers::ignoringImplicit(
                        clang::ast_matchers::parenExpr(clang::ast_matchers::has(
                            clang::ast_matchers::expr().bind("ptr"))))))),

            clang::ast_matchers::hasDescendant(clang::ast_matchers::binaryOperator(
                clang::ast_matchers::hasOperatorName("&&"),
                clang::ast_matchers::hasLHS(
                    clang::ast_matchers::ignoringImplicit(
                        clang::ast_matchers::integerLiteral(
                            clang::ast_matchers::equals(0)))),
                clang::ast_matchers::hasRHS(clang::ast_matchers::ignoringImplicit(
                    clang::ast_matchers::parenExpr(clang::ast_matchers::has(
                        clang::ast_matchers::ignoringImplicit(clang::ast_matchers::unaryOperator(
                            clang::ast_matchers::hasOperatorName("!"),
                            clang::ast_matchers::hasUnaryOperand(
                                clang::ast_matchers::ignoringImplicit(
                                    clang::ast_matchers::parenExpr(clang::ast_matchers::has(
                                        clang::ast_matchers::parenExpr(
                                            clang::ast_matchers::has(
                                                clang::ast_matchers::parenExpr(
                                                    clang::ast_matchers::has(
                                                        clang::ast_matchers::expr()
                                                            .bind(
                                                                "c")))))))))))))))))

                ))

        .bind("rcu_replace_pointer");

class rcu_collector : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  virtual void
  run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  rcu_dereference_table m_rcu_dereference_to_p;
  rcu_assign_pointer_table m_rcu_assign_pointer_to_params;
  rcu_access_pointer_table m_rcu_access_pointer_to_p;
  rcu_replace_pointer_table m_rcu_replace_pointer_to_params;
};
} // namespace macroni::kernel
