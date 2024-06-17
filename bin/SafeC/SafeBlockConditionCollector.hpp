#pragma once

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <unordered_set>

namespace macroni::safety {

extern clang::ast_matchers::StatementMatcher safe_block_condition_matcher;

class safe_block_condition_collector
    : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  virtual void
  run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  std::unordered_set<const clang::IntegerLiteral *> m_safe_block_conditions;
};
} // namespace macroni::safety
