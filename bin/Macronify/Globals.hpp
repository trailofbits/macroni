#pragma once

#include <mlir/IR/Builders.h>
#include <optional>
#include <pasta/AST/AST.h>

// TODO(bpp): Instead of using a global variable for the PASTA AST and MLIR
// context, find out how to pass these to a CodeGen object.
extern std::optional<pasta::AST> ast;
extern std::optional<mlir::Builder> builder;
