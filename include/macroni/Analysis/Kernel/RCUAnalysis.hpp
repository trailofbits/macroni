#pragma once

#include <mlir/IR/Operation.h>

namespace macroni::kernel {
struct kernel_analysis {
  kernel_analysis(mlir::Operation *op);
};
} // namespace macroni::kernel
