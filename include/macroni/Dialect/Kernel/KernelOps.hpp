// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "macroni/Dialect/Kernel/KernelDialect.hpp"

#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Support/TypeID.h>

#include <macroni/Dialect/Kernel/KernelAttributes.hpp>
#include <macroni/Dialect/Kernel/KernelTypes.hpp>

#define GET_OP_CLASSES
#include "macroni/Dialect/Kernel/Kernel.h.inc"