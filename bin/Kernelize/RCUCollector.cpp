#include "RCUCollector.hpp"
#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>

namespace macroni::kernel {
using namespace clang::ast_matchers;

StatementMatcher rcu_deference_matcher =
    stmtExpr(
        allOf(isExpandedFromMacro("rcu_dereference"),
              hasDescendant(parenExpr(has(parenExpr(has(expr().bind("p"))))))))
        .bind("rcu_dereference");

StatementMatcher rcu_assign_pointer_matcher =
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

void rcu_collector::run(const MatchFinder::MatchResult &Result) {
  if (auto rcu_dereference =
          Result.Nodes.getNodeAs<clang::StmtExpr>("rcu_dereference")) {
    auto p = Result.Nodes.getNodeAs<clang::Expr>("p");
    m_rcu_dereference_to_p.insert({rcu_dereference, p});
  } else if (auto rcu_assign_pointer =
                 Result.Nodes.getNodeAs<clang::DoStmt>("rcu_assign_pointer")) {
    auto p = Result.Nodes.getNodeAs<clang::Expr>("p");
    auto v = Result.Nodes.getNodeAs<clang::Expr>("v");
    m_rcu_assign_pointer_to_params.insert({rcu_assign_pointer, {p, v}});
  }
}
} // namespace macroni::kernel
