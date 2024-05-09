#
# Copyright (c) 2023-present, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

################################################################################
# Macroni version info
################################################################################

# Configure macroni's Version.cpp file. We configure a source file and not a
# header so that if we change macroni's version number, we only have to relink
# targets that reference the version number, not recompile them.
configure_file(Version.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/Version.cpp @ONLY)

add_library(macroni_version STATIC ${CMAKE_CURRENT_BINARY_DIR}/Version.cpp)

target_link_libraries(macroni_version PUBLIC macroni_settings)

################################################################################
# Macroni libraries
################################################################################

add_subdirectory(macroni)

add_library(macroni_api INTERFACE)

target_link_libraries(macroni_api
    INTERFACE
        macroni_settings
        macroni_common
        macroni_translation_api
)

add_library(macroni::api ALIAS macroni_api)

