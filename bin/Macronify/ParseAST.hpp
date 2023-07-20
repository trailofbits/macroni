#pragma once

#include <pasta/AST/AST.h>
#include <pasta/Compile/Command.h>
#include <pasta/Compile/Compiler.h>
#include <pasta/Util/ArgumentVector.h>
#include <pasta/Util/FileSystem.h>
#include <pasta/Util/Init.h>

namespace pasta {
    Result<AST, std::string> parse_ast(int argc, char **argv);
} // namespace pasta
