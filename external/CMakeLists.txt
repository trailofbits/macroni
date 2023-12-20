#
# Copyright (c) 2023-present, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

option(MACRONI_USE_VENDORED_GAP "Enable build with VAST's gap submodule" ON)
option(MACRONI_USE_VENDORED_PASTA "Enable build with pasta submodule" ON)
option(MACRONI_USE_VENDORED_VAST "Enable build with VAST submodule" ON)

if (MACRONI_USE_VENDORED_PASTA)
    set(PASTA_BOOTSTRAP_MACROS OFF)
    set(PASTA_BOOTSTRAP_TYPES OFF)
    set(PASTA_ENABLE_TESTING OFF)
    set(PASTA_ENABLE_INSTALL OFF)
    add_subdirectory(pasta)
endif()

if (MACRONI_USE_VENDORED_VAST)
    add_subdirectory(vast)
endif()
