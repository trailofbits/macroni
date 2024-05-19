add_library(safety_conversion SafetyRewriters.cpp)

target_link_libraries(safety_conversion PRIVATE ${MLIR_LIBS} vast_settings)

add_library(macroni::safety_conversion ALIAS safety_conversion)
