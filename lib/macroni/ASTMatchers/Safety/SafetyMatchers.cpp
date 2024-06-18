#include "macroni/ASTMatchers/Safety/SafetyMatchers.hpp"
#include <clang/AST/Expr.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>

namespace macroni::safety {
void safe_block_condition_collector::run(
    const clang::ast_matchers::MatchFinder::MatchResult &Result) {
  auto match = Result.Nodes.getNodeAs<clang::IntegerLiteral>("root");
  m_safe_block_conditions.insert(match);
}
} // namespace macroni::safety