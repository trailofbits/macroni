set(LLVM_LINK_COMPONENTS Core Support AsmParser)

add_executable(macronify
    Main.cpp
)
llvm_update_compile_flags(macronify)

target_link_libraries(macronify
    PRIVATE
        macroni_settings
        macroni_common
        VASTCodeGen
        VASTFrontend
)

mlir_check_link_libraries(macronify)
