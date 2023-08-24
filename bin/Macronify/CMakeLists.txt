set(LLVM_LINK_COMPONENTS Core Support AsmParser)

add_executable(macronify
    Main.cpp
)
llvm_update_compile_flags(macronify)

target_link_libraries(macronify
    macroni
    macroni_common
    macroni_translation_api

    MLIROptLib
    VASTHighLevel

    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport

    MLIRMacroni

    VASTFromSourceParser
)

mlir_check_link_libraries(macronify)