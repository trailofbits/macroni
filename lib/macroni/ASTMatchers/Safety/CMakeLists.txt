# Copyright (c) 2024-present, Trail of Bits, Inc.

add_library(safety_matchers SafetyMatchers.cpp)
target_link_libraries(safety_matchers PUBLIC macroni_settings)
