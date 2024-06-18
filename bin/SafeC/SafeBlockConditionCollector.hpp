#pragma once

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <set>

namespace macroni::safety {
using namespace clang::ast_matchers;

extern StatementMatcher safe_block_condition_matcher;

class safe_block_condition_collector : public MatchFinder::MatchCallback {
public:
  virtual void run(const MatchFinder::MatchResult &Result) override;

  std::set<const clang::IntegerLiteral *> m_safe_block_conditions;
};
} // namespace macroni::safety
