//
// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in
// the LICENSE file found in the root directory of this source tree.
//

/* This generated file is for internal use. Do not include it from headers. */

#include <string_view>

#ifdef MACRONI_CONFIG_H
#error config.h can only be included once
#else
#define MACRONI_CONFIG_H

#define CLANG_RESOURCE_DIR "${CLANG_RESOURCE_DIR}"

namespace macroni {

    constexpr std::string_view version = "${MACRONI_VERSION}";

    constexpr std::string_view bug_report_url = "${BUG_REPORT_URL}";
} // namespace macroni

#endif


