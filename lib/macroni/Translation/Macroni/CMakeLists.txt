add_library(macroni_translation_api STATIC MacroniMetaGenerator.cpp
                                           MacroniVisitor.cpp)
target_link_libraries(macroni_translation_api PRIVATE macroni_settings
                                                      macroni_common)
add_library(macroni::translation_api ALIAS macroni_translation_api)
