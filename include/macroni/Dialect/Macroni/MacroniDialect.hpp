// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/TypeID.h>

// Pull in the dialect definition.
#include "macroni/Dialect/Macroni/MacroniDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "macroni/Dialect/Macroni/MacroniEnums.h.inc"