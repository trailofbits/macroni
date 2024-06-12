#pragma once

#include "vast/Util/Common.hpp"

namespace macroni {
void build_region(vast::mlir_builder &bld, vast::op_state &st,
                  vast::builder_callback_ref callback);
} // namespace macroni