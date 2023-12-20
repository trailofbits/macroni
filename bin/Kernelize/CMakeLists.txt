set(LLVM_LINK_COMPONENTS Core Support AsmParser)

add_executable(kernelize
    Main.cpp
)
llvm_update_compile_flags(kernelize)

target_link_libraries(kernelize
    PRIVATE
        macroni_api
        MLIRMacroni
        MLIRKernel
        kernel_conversion
        VAST::VASTCodeGen
)

mlir_check_link_libraries(kernelize)