#
# Copyright (c) 2023-present, Trail of Bits, Inc. All rights reserved.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
#

add_subdirectory(macroni)

add_library(macroni_api INTERFACE)

target_link_libraries(macroni_api INTERFACE macroni_settings macroni_common
                                            macroni_translation_api)

add_library(macroni::api ALIAS macroni_api)
