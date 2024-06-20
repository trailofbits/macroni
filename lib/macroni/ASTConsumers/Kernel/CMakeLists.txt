# Copyright (c) 2024-present, Trail of Bits, Inc.

add_library(kernel_consumer KernelASTConsumer.cpp)
target_link_libraries(kernel_consumer PUBLIC macroni_settings)
