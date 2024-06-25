add_library(macroni_translation_api STATIC PastaMetaGenerator.cpp
                                           MacroniVisitor.cpp)
target_link_libraries(
  macroni_translation_api PRIVATE macroni_settings pasta pasta_cxx_settings
                                  macroni_common)
add_library(macroni::translation_api ALIAS macroni_translation_api)
