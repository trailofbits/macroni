add_library(kernel_translation_api STATIC KernelVisitor.cpp)
target_link_libraries(kernel_translation_api PRIVATE macroni_settings
                                                     macroni_common)
add_library(macroni::kernel_translation_api ALIAS kernel_translation_api)
