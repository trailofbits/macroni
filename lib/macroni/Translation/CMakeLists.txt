add_library(macroni_translation_api STATIC
    MacroniCodeGenVisitorMixin.cpp
)

target_link_libraries(macroni_translation_api
  PRIVATE
    clangAST
    clangASTMatchers
    clangBasic

    MLIRMeta
    MLIRHighLevel
    MLIRSupport

    macroni::settings
)

add_library(macroni::translation_api ALIAS macroni_translation_api)

