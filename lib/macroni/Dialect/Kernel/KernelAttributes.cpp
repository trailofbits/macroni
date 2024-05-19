// Copyright (c) 2023-present, Trail of Bits, Inc.

#include <macroni/Dialect/Kernel/KernelAttributes.hpp>
#include <macroni/Dialect/Kernel/KernelTypes.hpp>

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#define GET_ATTRDEF_CLASSES
#include <macroni/Dialect/Kernel/KernelAttributes.cpp.inc>

namespace macroni::kernel {
void KernelDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include <macroni/Dialect/Kernel/KernelAttributes.cpp.inc>
      >();
}
} // namespace macroni::kernel
