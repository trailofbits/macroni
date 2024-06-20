#pragma once

#include "vast/Util/Common.hpp"
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace macroni::kernel {
void rewrite_rcu(vast::mcontext_t *mctx, vast::owning_module_ref &mod);
} // namespace macroni::kernel
