add_library(kernel_conversion KernelRewriters.cpp)

target_link_libraries(kernel_conversion PUBLIC macroni_settings)

add_library(macroni::kernel_conversion ALIAS kernel_conversion)
