#pragma once

#include "vast/Util/Common.hpp"
#include <functional>

namespace macroni {
using module_handler = std::function<void(vast::vast_module &)>;
}
