add_library(macroni_common ParseAST.cpp)

target_link_libraries(macroni_common PRIVATE pasta_cxx_settings)

add_library(macroni::common ALIAS macroni_common)
