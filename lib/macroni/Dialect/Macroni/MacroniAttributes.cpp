// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "macroni/Dialect/Macroni/MacroniDialect.hpp"
#include "macroni/Dialect/Macroni/MacroniOps.hpp"
#include "macroni/Dialect/Macroni/MacroniTypes.hpp"
#include "macroni/Dialect/Macroni/MacroniAttributes.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

#define GET_ATTRDEF_CLASSES
#include "macroni/Dialect/Macroni/MacroniAttributes.cpp.inc"

namespace macroni::macroni
{
    void MacroniDialect::registerAttributes() {
        addAttributes<
            #define GET_ATTRDEF_LIST
            #include "macroni/Dialect/Macroni/MacroniAttributes.cpp.inc"
        >();
    }

} // namespace macroni::macroni
