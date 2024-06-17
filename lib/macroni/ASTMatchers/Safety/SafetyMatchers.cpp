#include "macroni/ASTMatchers/Safety/SafetyMatchers.hpp"
#include <clang/AST/Expr.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>

namespace macroni::safety {
using namespace clang::ast_matchers;
clang::ast_matchers::StatementMatcher safe_block_condition_matcher =
    integerLiteral(isExpandedFromMacro("unsafe")).bind("root");

void safe_block_condition_collector::run(
    const MatchFinder::MatchResult &Result) {
  auto match = Result.Nodes.getNodeAs<clang::IntegerLiteral>("root");
  m_safe_block_conditions.insert(match);
}
} // namespace macroni::safety