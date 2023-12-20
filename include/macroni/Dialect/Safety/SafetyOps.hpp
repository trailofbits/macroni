// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include <vast/Util/Warnings.hpp>

#include "macroni/Dialect/Safety/SafetyDialect.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Support/TypeID.h>
VAST_UNRELAX_WARNINGS

#define GET_OP_CLASSES
#include "macroni/Dialect/Safety/Safety.h.inc"