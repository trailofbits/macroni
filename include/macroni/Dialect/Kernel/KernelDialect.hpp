// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "macroni/Common/MacroSpelling.hpp"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/TypeID.h>

// Pull in the dialect definition.
#include "macroni/Dialect/Kernel/KernelDialect.h.inc"
