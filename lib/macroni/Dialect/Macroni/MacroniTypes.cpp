// Copyright (c) 20223-present, Trail of Bits, Inc.

#include "macroni/Dialect/Macroni/MacroniDialect.hpp"
#include "macroni/Dialect/Macroni/MacroniOps.hpp"
#include "macroni/Dialect/Macroni/MacroniTypes.hpp"
#include "macroni/Dialect/Macroni/MacroniAttributes.hpp"

using StringRef = llvm::StringRef; // to fix missing namespace in generated file

namespace macroni::macroni {
    void MacroniDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "macroni/Dialect/Macroni/MacroniTypes.cpp.inc"
        >();
    }
} // namespace macroni::macroni

#define  GET_TYPEDEF_CLASSES
#include "macroni/Dialect/Macroni/MacroniTypes.cpp.inc"
