set(LLVM_LINK_COMPONENTS Core Support AsmParser)

add_executable(kernelize
    Main.cpp
)
llvm_update_compile_flags(kernelize)

target_link_libraries(kernelize
    macroni
    macroni_common
    macroni_translation_api
    kernel_conversion

    MLIROptLib
    MLIRHighLevel

    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport

    MLIRKernel
    MLIRMacroni

    vast_high_level_transforms
    vast_conversion_from_hl

    FromSourceParser
)

mlir_check_link_libraries(kernelize)