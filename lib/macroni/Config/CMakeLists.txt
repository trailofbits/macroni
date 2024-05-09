# Copyright (c) 2024-present, Trail of Bits, Inc.

# Configure a source file containing macroni configuration info. We configure a
# source file and not a header so that if we change anny of this information, we
# only have to relink targets that reference it, not recompile them.

configure_file(Config.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/Config.cpp @ONLY)

add_library(macroni_config STATIC ${CMAKE_CURRENT_BINARY_DIR}/Config.cpp)

target_link_libraries(macroni_config PUBLIC macroni_settings)
