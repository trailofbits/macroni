#pragma once

#include "vast/Util/Common.hpp"
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace macroni::kernel {
void rewrite_rcu(vast::mcontext_t *mctx, vast::vast_module &mod);
} // namespace macroni::kernel
