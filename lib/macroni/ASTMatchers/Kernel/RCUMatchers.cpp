#include "macroni/ASTMatchers/Kernel/RCUMatchers.hpp"
#include "macroni/Common/ExpansionTable.hpp"
#include "macroni/Common/MacroSpelling.hpp"
#include "macroni/Dialect/Kernel/KernelDialect.hpp"
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <vector>

namespace macroni::kernel {
using namespace clang::ast_matchers;

const StatementMatcher
rcu_collector::make_rcu_dereference_matcher(std::string suffix) {
  auto name = "rcu_dereference" + suffix;
  return stmtExpr(allOf(isExpandedFromMacro(name),
                        hasDescendant(
                            parenExpr(has(parenExpr(has(expr().bind("p"))))))))
      .bind(name);
}

const StatementMatcher rcu_collector::rcu_deference_matcher =
    make_rcu_dereference_matcher("");

const StatementMatcher rcu_collector::rcu_deference_bh_matcher =
    make_rcu_dereference_matcher("_bh");

const StatementMatcher rcu_collector::rcu_deference_sched_matcher =
    make_rcu_dereference_matcher("_sched");

const StatementMatcher rcu_collector::rcu_assign_pointer_matcher =
    doStmt(allOf(isExpandedFromMacro("rcu_assign_pointer"),
                 hasDescendant(callExpr(
                     allOf(callee(functionDecl(hasName("__write_once_size"))),
                           hasArgument(0, hasDescendant(parenExpr(has(parenExpr(
                                              has(expr().bind("p")))))))))),
                 hasDescendant(callExpr(allOf(
                     callee(functionDecl(hasName("__builtin_constant_p"))),
                     hasArgument(0, expr().bind("v")))))

                     ))
        .bind("rcu_assign_pointer");

const StatementMatcher rcu_collector::rcu_access_pointer_matcher =
    stmtExpr(
        allOf(isExpandedFromMacro("rcu_access_pointer"),
              hasDescendant(parenExpr(has(parenExpr(has(expr().bind("p"))))))))
        .bind("rcu_access_pointer");

const StatementMatcher rcu_collector::rcu_replace_pointer_matcher =
    stmtExpr(
        allOf(isExpandedFromMacro("rcu_replace_pointer"),

              hasDescendant(
                  callExpr(callee(functionDecl(hasName("__write_once_size"))),
                           // NOTE(Brent): We have to use hasAnyArgument()
                           // instead of hasArgument() because for some reason
                           // Clang implicitly ignores parentheseses for the
                           // latter matcher.
                           hasAnyArgument(ignoringImplicit(unaryOperator(
                               hasOperatorName("&"),
                               has(parenExpr(has(parenExpr(has(parenExpr(has(

                                   expr().bind("rc"
                                               "u_"
                                               "pt"
                                               "r")

                                       ))))))))))

                               )),

              hasDescendant(callExpr(
                  callee(functionDecl(hasName("__builtin_constant_p"))),
                  hasAnyArgument(
                      ignoringImplicit(parenExpr(has(expr().bind("ptr"))))))),

              hasDescendant(binaryOperator(
                  hasOperatorName("&&"),
                  hasLHS(ignoringImplicit(integerLiteral(equals(0)))),
                  hasRHS(ignoringImplicit(parenExpr(has(ignoringImplicit(
                      unaryOperator(hasOperatorName("!"),
                                    hasUnaryOperand(ignoringImplicit(
                                        parenExpr(has(parenExpr(has(parenExpr(
                                            has(expr().bind("c")))))))))))))))))

                  ))

        .bind("rcu_replace_pointer");

void rcu_collector::attach_to(clang::ast_matchers::MatchFinder &finder) {
  finder.addMatcher(rcu_collector::rcu_deference_matcher, this);
  finder.addMatcher(rcu_collector::rcu_deference_bh_matcher, this);
  finder.addMatcher(rcu_collector::rcu_deference_sched_matcher, this);
  finder.addMatcher(rcu_collector::rcu_assign_pointer_matcher, this);
  finder.addMatcher(rcu_collector::rcu_access_pointer_matcher, this);
  finder.addMatcher(rcu_collector::rcu_replace_pointer_matcher, this);
}

void rcu_collector::run(const MatchFinder::MatchResult &Result) {
  for (auto rcu_macro : KernelDialect::rcu_macro_spellings) {
    if (auto expansion = Result.Nodes.getNodeAs<clang::Stmt>(rcu_macro.name)) {
      std::vector<const clang::Expr *> args;
      for (auto param_name : rcu_macro.parameter_names) {
        args.push_back(Result.Nodes.getNodeAs<clang::Expr>(param_name));
      }
      expansions.emplace(expansion, expansion_info{rcu_macro, std::move(args)});
    }
  }
}
} // namespace macroni::kernel
