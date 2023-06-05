add_executable(macronify Main.cpp)

include_directories(
  /external/vast/include
  /external/pasta/include
  ${PROJECT_SOURCE_DIR}/build/external/vast/include
)

add_dependencies(macronify pasta)
add_dependencies(macronify vast)

target_link_libraries(macronify
  pasta
  gap::gap
  vast
  FromSourceParser
  macroni
)
