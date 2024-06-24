add_executable(kernelcheck KernelCheck.cpp)

llvm_update_compile_flags(kernelcheck)

target_link_libraries(
  kernelcheck
  PRIVATE macroni_settings
          macroni_common
          kernel_analysis
          MLIRKernel
          MLIROptLib)

mlir_check_link_libraries(kernelcheck)
