add_subdirectory(Common)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Translation)

configure_file(Version.hpp.in Version.hpp @ONLY)