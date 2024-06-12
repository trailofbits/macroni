#include "macroni/Common/MLIRUtils.hpp"
#include "vast/Util/Common.hpp"

namespace macroni {
void build_region(vast::mlir_builder &bld, vast::op_state &st,
                  vast::builder_callback_ref callback) {
  bld.createBlock(st.addRegion());
  callback(bld, st.location);
}
} // namespace macroni