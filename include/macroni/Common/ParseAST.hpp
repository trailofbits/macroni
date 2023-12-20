#pragma once

#include <pasta/AST/AST.h>

namespace pasta {
Result<AST, std::string> parse_ast(int argc, char **argv);
} // namespace pasta
