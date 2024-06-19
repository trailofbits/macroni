#pragma once

#include "macroni/Translation/Kernel/KernelVisitor.hpp"
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>

namespace macroni::kernel {

class rcu_collector : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  static const clang::ast_matchers::StatementMatcher
  make_rcu_dereference_matcher(std::string suffix);

public:
  static const clang::ast_matchers::StatementMatcher rcu_deference_matcher;
  static const clang::ast_matchers::StatementMatcher rcu_deference_bh_matcher;
  static const clang::ast_matchers::StatementMatcher
      rcu_deference_sched_matcher;
  static const clang::ast_matchers::StatementMatcher rcu_assign_pointer_matcher;
  static const clang::ast_matchers::StatementMatcher rcu_access_pointer_matcher;
  static const clang::ast_matchers::StatementMatcher
      rcu_replace_pointer_matcher;

  virtual void
  run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  rcu_dereference_table m_rcu_dereference_to_p;
  rcu_dereference_table m_rcu_dereference_bh_to_p;
  rcu_dereference_table m_rcu_dereference_sched_to_p;
  rcu_assign_pointer_table m_rcu_assign_pointer_to_params;
  rcu_access_pointer_table m_rcu_access_pointer_to_p;
  rcu_replace_pointer_table m_rcu_replace_pointer_to_params;
};
} // namespace macroni::kernel
