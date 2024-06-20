#pragma once

#include <string>
#include <vector>

// Contains a macro's name and the names of its parameters.
namespace macroni {
struct macro_spelling {
  std::string name;
  std::vector<std::string> parameter_names;
};
} // namespace macroni