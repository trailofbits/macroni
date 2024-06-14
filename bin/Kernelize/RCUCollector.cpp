#include "RCUCollector.hpp"
#include <clang/AST/Expr.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>

namespace macroni::kernel {
using namespace clang::ast_matchers;

StatementMatcher rcu_deference_matcher =
    stmtExpr(allOf(isExpandedFromMacro("rcu_dereference"),
                   hasDescendant(declRefExpr(hasType(pointerType())).bind("p")),
                   hasDescendant(stmtExpr())))
        .bind("rcu_dereference");

void rcu_collector::run(const MatchFinder::MatchResult &Result) {
  auto rcu_dereference =
      Result.Nodes.getNodeAs<clang::StmtExpr>("rcu_dereference");
  auto p = Result.Nodes.getNodeAs<clang::DeclRefExpr>("p");
  m_rcu_dereference_to_p.insert({rcu_dereference, p});
}
} // namespace macroni::kernel
