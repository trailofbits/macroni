set(LLVM_LINK_COMPONENTS Core Support AsmParser)

add_executable(safe-c
    Main.cpp
)
llvm_update_compile_flags(safe-c)

target_link_libraries(safe-c
    macroni
    macroni_common
    macroni_translation_api
    safety_conversion

    MLIROptLib
    VASTHighLevel

    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport

    MLIRMacroni
    MLIRSafety

    VASTFromSourceParser
)

mlir_check_link_libraries(safe-c)