# See
# https://github.com/trailofbits/vast/blob/64cbb58644e62686f7107b531e4d7de2f902bdb7/CMakeLists.txt

#
# Copyright (c) 2023-present, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.25)

# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

if (POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(Macroni
    LANGUAGES C CXX
    VERSION 0.0.0
    DESCRIPTION "Intermediate representation of macros for program analysis"
    HOMEPAGE_URL "https://github.com/trailofbits/macroni.git"
)

set(BUG_REPORT_URL "https://github.com/trailofbits/macroni/issues" CACHE STRING "")

# include(prevent_in_source_builds)

# check if macroni is being used directly or via add_subdirectory,
# but allow overriding
if(NOT DEFINED MACRONI_MASTER_PROJECT)
    if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(MACRONI_MASTER_PROJECT ON)
    else()
        set(MACRONI_MASTER_PROJECT OFF)
    endif()
endif()

include(project_settings)
include(project_utils)

#
# Setup package version
#
# setup_package_version_variables(macroni)

set(MACRONI_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MACRONI_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MACRONI_INCLUDE_DIR ${MACRONI_BINARY_DIR}/include)

# Configure Macroni Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/macroni/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/macroni/Version.inc
)

# Configure Macroni version info header file.

#
# Clang
#
find_package(Clang REQUIRED CONFIG)

set(CLANG_RESOURCE_DIR  ${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR})

configure_file(
  ${MACRONI_SOURCE_DIR}/include/macroni/Config/config.h.cmake
  ${MACRONI_BINARY_DIR}/include/macroni/Config/config.h
)

#
# CCACHE
#
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

# FindAndSelectClangCompiler()

#
# LLVM & MLIR & Clang
#
find_package(LLVM 16 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
#
# MLIR
#
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

#
# LLVM Libraries
#
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# TODO(Heno): clean up & modernize
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

include_directories(${PROJECT_SOURCE_DIR}/external/pasta/include)
include_directories(${PROJECT_BINARY_DIR}/external/pasta/include)

include_directories(${PROJECT_SOURCE_DIR}/external/vast/include)
include_directories(${PROJECT_BINARY_DIR}/external/vast/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(${CLANG_DEFINITIONS})

if (LLVM_LINK_LLVM_DYLIB)
    set(LLVM_LIBS LLVM)
else()
    llvm_map_components_to_libnames(LLVM_LIBS
      ${LLVM_TARGETS_TO_BUILD} support target option
    )
endif()

# TODO(cmake): With newer cmake there should be a better way to do this.
function(mk_clang_libs out)
    if (CLANG_LINK_CLANG_DYLIB)
        set(${out} clang-cpp PARENT_SCOPE)
    else()
        set(${out} ${ARGN} PARENT_SCOPE)
    endif()
endfunction()

#
# Macroni build settings
#

add_library(macroni_settings INTERFACE)

include(cmake/compiler_warnings.cmake)
set_project_warnings(macroni_settings)

target_include_directories(macroni_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if (NOT LLVM_ENABLE_RTTI)
  message( FATAL_ERROR
    "Macroni does not support build without RTTI, yet. Please build LLVM with LLVM_ENABLE_RTTI."
  )
endif()

# sanitizer options if supported by compiler
include(cmake/sanitizers.cmake)
enable_sanitizers(macroni_settings)

# allow for static analysis options
include(cmake/static_analyzers.cmake)

option(ENABLE_PDLL_CONVERSIONS "Enable PDLL conversions" OFF)
if (ENABLE_PDLL_CONVERSIONS)
  message(STATUS "ENABLE_PDLL_CONVERSIONS")
  target_compile_definitions(macroni_settings
    INTERFACE
      -DENABLE_PDLL_CONVERSIONS
  )
endif()

add_library(macroni::settings ALIAS macroni_settings)

#
# Add external libraries
#
add_subdirectory(external)

get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)



target_link_libraries(macroni_settings
  INTERFACE
    ${DIALECT_LIBS}
    ${CONVERSION_LIBS}

    gap::gap

    pasta
)

#
# Macroni libraries
#
add_subdirectory(include/macroni)
add_subdirectory(lib)

#
# Macroni executables
#
add_subdirectory(bin)

add_library(macroni INTERFACE)

target_link_libraries(macroni
  INTERFACE
    macroni::settings
)

target_include_directories(macroni
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# test options
option(MACRONI_ENABLE_TESTING "Enable Test Builds" ON)

if (MACRONI_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

#
# install settings
#

option(MACRONI_INSTALL "Generate the install target." ${MACRONI_MASTER_PROJECT})

if (MACRONI_INSTALL)
  set(MACRONI_CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  set(MACRONI_INSTALL_TARGETS
    gap-core
    gap-settings
    gap

    vast_settings
    VASTCodeGen
    VASTUtil

    VASTABI
    VASTCore
    VASTHighLevel
    VASTLowLevel
    VASTMeta
    VASTUnsupported
    
    VASTABIConversionPasses
    VASTCommonConversionPasses
    VASTCoreConversionPasses
    VASTHighLevelConversionPasses
    VASTHighLevelTransforms
    VASTLowLevelTransforms
    VASTTargetLLVMIR

    VASTSymbolInterface
    VASTTypeQualifiersInterfaces

    pasta_cxx_settings
    pasta_thirdparty_llvm
    pasta_util
    pasta_compiler
    pasta

    macroni
    macroni_settings

    MLIRMacroni
    MLIRKernel
    MLIRSafety
  )


  install(DIRECTORY ${CMAKE_BINARY_DIR}/include/macroni
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT macroni-headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.h.inc"
    PATTERN "*.hpp.inc"
    PATTERN "CMakeFiles" EXCLUDE
  )

  install(DIRECTORY include/macroni
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT macroni-headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "CMakeFiles" EXCLUDE
  )

  set(MACRONI_EXPORT_NAME MacroniTargets)

  install(TARGETS ${MACRONI_INSTALL_TARGETS}
    EXPORT ${MACRONI_EXPORT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      NAMELINK_SKIP
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(EXPORT ${MACRONI_EXPORT_NAME}
    FILE ${MACRONI_EXPORT_NAME}.cmake
    NAMESPACE Macroni::
    DESTINATION ${MACRONI_CMAKE_INSTALL_DIR}
  )

  #
  # packaging support
  #

  set(CPACK_PACKAGE_VENDOR "Trail of Bits")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "Macroni: a VAST dialect extension with macros"
  )
  set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH})
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
  set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
  set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package")
  set(CPACK_PACKAGE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set(SYSTEM ${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_VERSION})
  set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}-${SYSTEM}")
  set(CPACK_SOURCE_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}-src")

  include(CPack)

  set(MACRONI_CONFIG_NAME "${PROJECT_NAME}Config")
  set(MACRONI_PACKAGE_CONFIG_FILE "${MACRONI_CONFIG_NAME}.cmake")
  set(MACRONI_PACKAGE_CONFIG_VERSION_FILE "${MACRONI_CONFIG_NAME}Version.cmake")

  include(CMakePackageConfigHelpers)

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${MACRONI_CONFIG_NAME}.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${MACRONI_PACKAGE_CONFIG_FILE}"
    INSTALL_DESTINATION ${MACRONI_CMAKE_INSTALL_DIR}
  )

  write_basic_package_version_file(
    ${MACRONI_PACKAGE_CONFIG_VERSION_FILE}
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${MACRONI_PACKAGE_CONFIG_FILE}"
    "${CMAKE_CURRENT_BINARY_DIR}/${MACRONI_PACKAGE_CONFIG_VERSION_FILE}"
    DESTINATION ${MACRONI_CMAKE_INSTALL_DIR}
  )

endif()
