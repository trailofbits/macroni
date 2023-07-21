#include <llvm/ADT/TypeSwitch.h>
#include <macroni/Dialect/Kernel/KernelDialect.hpp>
#include <macroni/Dialect/Kernel/KernelOps.hpp>
#include <macroni/Dialect/Kernel/KernelTypes.hpp>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

namespace macroni::kernel {
    void KernelDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "macroni/Dialect/Kernel/KernelTypes.cpp.inc"
        >();
    }
} // namespace macroni::kernel

#define  GET_TYPEDEF_CLASSES
#include "macroni/Dialect/Kernel/KernelTypes.cpp.inc"