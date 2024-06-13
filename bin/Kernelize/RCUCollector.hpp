#pragma once

#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>

namespace macroni::kernel {
using namespace clang::ast_matchers;

extern StatementMatcher rcu_deference_matcher;

class rcu_collector : public MatchFinder::MatchCallback {
  virtual void run(const MatchFinder::MatchResult &Result) override;

  virtual void onStartOfTranslationUnit() override;

  virtual void onEndOfTranslationUnit() override;

private:
  clang::ASTContext *m_actx;
  rcu_dereference_table m_rcu_dereference_to_p;
};
} // namespace macroni::kernel
