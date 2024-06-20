#pragma once

#include "macroni/Common/ExpansionTable.hpp"
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

  expansion_table m_expansions;
};
} // namespace macroni::kernel
