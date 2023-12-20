#include <macroni/Translation/MacroniCodeGenVisitorMixin.hpp>
#include <pasta/AST/Macro.h>

namespace macroni {
std::optional<pasta::MacroSubstitution>
lowest_unvisited_substitution(pasta::Stmt &stmt,
                              std::set<pasta::MacroSubstitution> &visited) {
  auto subs = stmt.AlignedSubstitutions();
  for (auto sub : subs) {
    // Don't visit macros more than once
    if (visited.contains(sub)) {
      continue;
    }

    // Only visit pre-expanded forms of function-like expansions.
    if (auto exp = pasta::MacroExpansion::From(sub)) {
      bool is_pre_expansion =
          (exp->Arguments().empty() || exp->IsArgumentPreExpansion());
      if (!is_pre_expansion) {
        continue;
      }
    }

    // Mark this substitution as visited so we don't visit it again.
    visited.insert(sub);
    return sub;
  }
  return std::nullopt;
}

bool is_sub_function_like(pasta::MacroSubstitution &sub) {
  if (auto exp = pasta::MacroExpansion::From(sub)) {
    if (auto def = exp->Definition()) {
      return def->IsFunctionLike();
    }
  }
  return false;
}

std::vector<llvm::StringRef>
get_parameter_names(pasta::MacroSubstitution &sub) {
  std::vector<llvm::StringRef> param_names;
  auto exp = pasta::MacroExpansion::From(sub);
  if (!exp) {
    return param_names;
  }
  auto def = exp->Definition();
  if (!def) {
    return param_names;
  }
  for (auto macro_tok : def->Parameters()) {
    if (auto bt = macro_tok.BeginToken()) {
      param_names.push_back(bt->Data());
    } else {
      param_names.push_back("<a nameless macro parameter>");
    }
  }
  return param_names;
}
} // namespace macroni
