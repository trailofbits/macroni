# Copyright (c) 2024-present, Trail of Bits, Inc.

add_library(rcu_matchers RCUMatchers.cpp)
target_link_libraries(rcu_matchers PUBLIC macroni_settings)
