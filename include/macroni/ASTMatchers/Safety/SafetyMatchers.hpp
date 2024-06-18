#pragma once

#include "macroni/Translation/Safety/SafetyVisitor.hpp"
#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>

namespace macroni::safety {
static const clang::ast_matchers::StatementMatcher
    safe_block_condition_matcher =
        clang::ast_matchers::integerLiteral(
            clang::ast_matchers::isExpandedFromMacro("unsafe"))
            .bind("root");

class safe_block_condition_collector
    : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  virtual void
  run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  safety_conditions m_safe_block_conditions;
};
} // namespace macroni::safety
