# Copyright (c) 2024-present, Trail of Bits, Inc.

add_library(safety_consumer SafetyASTConsumer.cpp)
target_link_libraries(safety_consumer PUBLIC macroni_settings)
