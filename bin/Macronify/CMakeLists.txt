set(LLVM_LINK_COMPONENTS Core Support AsmParser)

add_executable(macronify
    MacroniCodeGenVisitorMixin.cpp
    MacroniRewriters.cpp
    Main.cpp
    ParseAST.cpp
)
llvm_update_compile_flags(macronify)

target_link_libraries(macronify
    macroni

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

mlir_check_link_libraries(macronify)