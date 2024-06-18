# Copyright (c) 2024-present, Trail of Bits, Inc.

add_library(macroni_common STATIC EmptyVisitor.cpp)
target_link_libraries(macroni_common PUBLIC macroni_settings)
