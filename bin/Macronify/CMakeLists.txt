set(LLVM_LINK_COMPONENTS Core Support AsmParser)

add_executable(macronify Macronify.cpp ParseAST.cpp)

llvm_update_compile_flags(macronify)

target_link_libraries(
  macronify PRIVATE macroni_settings pasta pasta_cxx_settings macroni_common
                    macroni_translation_api MLIRMacroni)

mlir_check_link_libraries(macronify)
