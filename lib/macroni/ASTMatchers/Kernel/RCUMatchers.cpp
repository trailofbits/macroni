#include "macroni/ASTMatchers/Kernel/RCUMatchers.hpp"
#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>

namespace macroni::kernel {
using namespace clang::ast_matchers;

const StatementMatcher
rcu_collector::make_rcu_dereference_matcher(std::string suffix) {
  auto name = "rcu_dereference" + suffix;
  return stmtExpr(allOf(isExpandedFromMacro(name),
                        hasDescendant(
                            parenExpr(has(parenExpr(has(expr().bind("p"))))))))
      .bind(name);
}

const StatementMatcher rcu_collector::rcu_deference_matcher =
    make_rcu_dereference_matcher("");

const StatementMatcher rcu_collector::rcu_deference_bh_matcher =
    make_rcu_dereference_matcher("_bh");

const StatementMatcher rcu_collector::rcu_deference_sched_matcher =
    make_rcu_dereference_matcher("_sched");

const StatementMatcher rcu_collector::rcu_assign_pointer_matcher =
    doStmt(allOf(isExpandedFromMacro("rcu_assign_pointer"),
                 hasDescendant(callExpr(
                     allOf(callee(functionDecl(hasName("__write_once_size"))),
                           hasArgument(0, hasDescendant(parenExpr(has(parenExpr(
                                              has(expr().bind("p")))))))))),
                 hasDescendant(callExpr(allOf(
                     callee(functionDecl(hasName("__builtin_constant_p"))),
                     hasArgument(0, expr().bind("v")))))

                     ))
        .bind("rcu_assign_pointer");

const StatementMatcher rcu_collector::rcu_access_pointer_matcher =
    stmtExpr(
        allOf(isExpandedFromMacro("rcu_access_pointer"),
              hasDescendant(parenExpr(has(parenExpr(has(expr().bind("p"))))))))
        .bind("rcu_access_pointer");

const StatementMatcher rcu_collector::rcu_replace_pointer_matcher =
    stmtExpr(
        allOf(isExpandedFromMacro("rcu_replace_pointer"),

              hasDescendant(
                  callExpr(callee(functionDecl(hasName("__write_once_size"))),
                           // NOTE(Brent): We have to use hasAnyArgument()
                           // instead of hasArgument() because for some reason
                           // Clang implicitly ignores parentheseses for the
                           // latter matcher.
                           hasAnyArgument(ignoringImplicit(unaryOperator(
                               hasOperatorName("&"),
                               has(parenExpr(has(parenExpr(has(parenExpr(has(

                                   expr().bind("rc"
                                               "u_"
                                               "pt"
                                               "r")

                                       ))))))))))

                               )),

              hasDescendant(callExpr(
                  callee(functionDecl(hasName("__builtin_constant_p"))),
                  hasAnyArgument(
                      ignoringImplicit(parenExpr(has(expr().bind("ptr"))))))),

              hasDescendant(binaryOperator(
                  hasOperatorName("&&"),
                  hasLHS(ignoringImplicit(integerLiteral(equals(0)))),
                  hasRHS(ignoringImplicit(parenExpr(has(ignoringImplicit(
                      unaryOperator(hasOperatorName("!"),
                                    hasUnaryOperand(ignoringImplicit(
                                        parenExpr(has(parenExpr(has(parenExpr(
                                            has(expr().bind("c")))))))))))))))))

                  ))

        .bind("rcu_replace_pointer");

void rcu_collector::run(const MatchFinder::MatchResult &Result) {
  if (auto rcu_dereference =
          Result.Nodes.getNodeAs<clang::StmtExpr>("rcu_dereference")) {
    auto p = Result.Nodes.getNodeAs<clang::Expr>("p");
    m_rcu_dereference_to_p.insert({rcu_dereference, p});
  } else if (auto rcu_dereference_bh = Result.Nodes.getNodeAs<clang::StmtExpr>(
                 "rcu_dereference_bh")) {
    auto p = Result.Nodes.getNodeAs<clang::Expr>("p");
    m_rcu_dereference_bh_to_p.insert({rcu_dereference_bh, p});
  } else if (auto rcu_dereference_sched =
                 Result.Nodes.getNodeAs<clang::StmtExpr>(
                     "rcu_dereference_sched")) {
    auto p = Result.Nodes.getNodeAs<clang::Expr>("p");
    m_rcu_dereference_sched_to_p.insert({rcu_dereference_sched, p});
  } else if (auto rcu_assign_pointer =
                 Result.Nodes.getNodeAs<clang::DoStmt>("rcu_assign_pointer")) {
    auto p = Result.Nodes.getNodeAs<clang::Expr>("p");
    auto v = Result.Nodes.getNodeAs<clang::Expr>("v");
    m_rcu_assign_pointer_to_params.insert({rcu_assign_pointer, {p, v}});
  } else if (auto rcu_access_pointer = Result.Nodes.getNodeAs<clang::StmtExpr>(
                 "rcu_access_pointer")) {
    auto p = Result.Nodes.getNodeAs<clang::Expr>("p");
    m_rcu_access_pointer_to_p.insert({rcu_access_pointer, p});
  } else if (auto rcu_replace_pointer = Result.Nodes.getNodeAs<clang::StmtExpr>(
                 "rcu_replace_pointer")) {
    auto rcu_ptr = Result.Nodes.getNodeAs<clang::Expr>("rcu_ptr");
    auto ptr = Result.Nodes.getNodeAs<clang::Expr>("ptr");
    auto c = Result.Nodes.getNodeAs<clang::Expr>("c");
    m_rcu_replace_pointer_to_params.insert(
        {rcu_replace_pointer, {rcu_ptr, ptr, c}});
  }
}
} // namespace macroni::kernel
