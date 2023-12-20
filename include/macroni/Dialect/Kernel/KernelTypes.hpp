#pragma once

#include <vast/Util/Warnings.hpp>

VAST_RELAX_WARNINGS
#include <clang/AST/Type.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
VAST_UNRELAX_WARNINGS

#define GET_TYPEDEF_CLASSES
#include "macroni/Dialect/Kernel/KernelTypes.h.inc"