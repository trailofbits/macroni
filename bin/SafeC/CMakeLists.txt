set(LLVM_LINK_COMPONENTS Core Support AsmParser)

add_executable(safe-c
    Main.cpp
)
llvm_update_compile_flags(safe-c)

target_link_libraries(safe-c
    PRIVATE
        macroni_api
        MLIRMacroni
        MLIRSafety
        safety_conversion
        VAST::VASTCodeGen
)

mlir_check_link_libraries(safe-c)