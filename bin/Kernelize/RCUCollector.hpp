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
public:
  virtual void run(const MatchFinder::MatchResult &Result) override;

  rcu_dereference_table m_rcu_dereference_to_p;
};
} // namespace macroni::kernel
