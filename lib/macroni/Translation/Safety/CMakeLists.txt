add_library(safety_translation_api STATIC SafetyVisitor.cpp)
target_link_libraries(safety_translation_api PRIVATE macroni_settings
                                                     macroni_common)
add_library(macroni::safety_translation_api ALIAS safety_translation_api)
