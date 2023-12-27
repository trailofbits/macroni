set(LLVM_LINK_COMPONENTS Core Support AsmParser)

add_executable(macronify
    Main.cpp
)
llvm_update_compile_flags(macronify)

target_link_libraries(macronify
    PRIVATE
        macroni_api
        MLIRMacroni
        VAST::VASTCodeGen
        VAST::VASTFrontend
)

mlir_check_link_libraries(macronify)