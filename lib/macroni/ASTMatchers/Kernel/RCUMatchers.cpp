#include "macroni/ASTMatchers/Kernel/RCUMatchers.hpp"
#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>

namespace macroni::kernel {
void rcu_collector::run(
    const clang::ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto rcu_dereference =
          Result.Nodes.getNodeAs<clang::StmtExpr>("rcu_dereference")) {
    auto p = Result.Nodes.getNodeAs<clang::Expr>("p");
    m_rcu_dereference_to_p.insert({rcu_dereference, p});
  } else if (auto rcu_assign_pointer =
                 Result.Nodes.getNodeAs<clang::DoStmt>("rcu_assign_pointer")) {
    auto p = Result.Nodes.getNodeAs<clang::Expr>("p");
    auto v = Result.Nodes.getNodeAs<clang::Expr>("v");
    m_rcu_assign_pointer_to_params.insert({rcu_assign_pointer, {p, v}});
  } else if (auto rcu_access_pointer = Result.Nodes.getNodeAs<clang::StmtExpr>(
                 "rcu_access_pointer")) {
    auto p = Result.Nodes.getNodeAs<clang::Expr>("p");
    m_rcu_access_pointer_to_p.insert({rcu_access_pointer, p});
  }
}
} // namespace macroni::kernel
