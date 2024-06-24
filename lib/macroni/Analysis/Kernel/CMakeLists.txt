# Copyright (c) 2024-present, Trail of Bits, Inc.

add_library(kernel_analysis KernelAnalysisPass.cpp RCUAnalysis.cpp)
target_link_libraries(kernel_analysis PUBLIC macroni_settings MLIRKernel)
