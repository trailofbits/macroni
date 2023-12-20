// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include <vast/Util/Warnings.hpp>

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/TypeID.h>
VAST_UNRELAX_WARNINGS

// Pull in the dialect definition.
#include "macroni/Dialect/Kernel/KernelDialect.h.inc"
