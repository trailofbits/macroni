#pragma once

#include "vast/CodeGen/CodeGenDriver.hpp"
#include "vast/Util/Common.hpp"
#include <memory>

namespace macroni {

std::unique_ptr<vast::cg::driver>
generate_macroni_driver(vast::acontext_t &actx);
} // namespace macroni
