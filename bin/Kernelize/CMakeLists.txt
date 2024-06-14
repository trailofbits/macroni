set(LLVM_LINK_COMPONENTS Core Support AsmParser)

set(MACRONI_ORIGINAL_LLVM_RUNTIME_OUTPUT_INTDIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
add_clang_executable(kernelize Kernelize.cpp RCUCollector.cpp
                     KernelASTConsumer.cpp)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${MACRONI_ORIGINAL_LLVM_RUNTIME_OUTPUT_INTDIR})

target_link_libraries(
  kernelize PRIVATE macroni_settings macroni_common kernel_translation_api
                    MLIRKernel kernel_conversion)
mlir_check_link_libraries(kernelize)
