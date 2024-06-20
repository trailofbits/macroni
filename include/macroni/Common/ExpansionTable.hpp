#pragma once

#include "macroni/Common/MacroSpelling.hpp"
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <unordered_map>
#include <vector>
namespace macroni {

struct expansion_info {
  // The name of the macro this is an expansion of and the names of its
  // arguments.
  macro_spelling spelling;
  // The expressions that align with this macro's arguments.
  std::vector<const clang::Expr *> arguments;
};

// Maps a stmt that aligns with a macro expansion to information about the
// expansion.
using expansion_table = std::unordered_map<const clang::Stmt *, expansion_info>;
} // namespace macroni