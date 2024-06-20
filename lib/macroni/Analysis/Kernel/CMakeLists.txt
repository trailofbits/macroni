# Copyright (c) 2024-present, Trail of Bits, Inc.

add_library(rcu_analyzer RCUAnalyzer.cpp)
target_link_libraries(rcu_analyzer PUBLIC macroni_settings)
