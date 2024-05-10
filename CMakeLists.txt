#
# Copyright (c) 2023-present, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.29)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(project_version_details)

project(macroni
    LANGUAGES C CXX
    VERSION ${macroni_PROJECT_VERSION_INTERNAL}
    DESCRIPTION "Intermediate representation of macros for program analysis"
    HOMEPAGE_URL "https://github.com/trailofbits/macroni.git"
)

set(BUG_REPORT_URL "https://github.com/trailofbits/macroni/issues" CACHE STRING "")

include(prevent_in_source_builds)

# Check if macroni is being used directly or via add_subdirectory, but allow
# overriding
if(NOT DEFINED MACRONI_MASTER_PROJECT)
  set(MACRONI_MASTER_PROJECT "$<STREQUAL:${CMAKE_CURRENT_SOURCE_DIR},${CMAKE_SOURCE_DIR}>")
endif()

include(project_settings)
include(project_utils)

set(MACRONI_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MACRONI_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MACRONI_INCLUDE_DIR ${MACRONI_BINARY_DIR}/include)

#
# Git
#

find_package(Git REQUIRED)
include(get_git_hash)
get_git_hash(macroni_GIT_HASH)
include(count_commits_since_last_version_change)
count_commits_since_last_version_change(macroni_COMMITS_SINCE_LAST_VERSION_CHANGE)

#
# Clang
#
find_package(Clang REQUIRED CONFIG)

#
# CCACHE
#
find_program(CCACHE ccache)
if (CCACHE)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE}")
endif()

find_and_select_clang_compiler()

#
# LLVM & MLIR & Clang
#
find_package(LLVM 17 REQUIRED CONFIG)
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

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(${CLANG_DEFINITIONS})

if (LLVM_LINK_LLVM_DYLIB)
    set(LLVM_LIBS LLVM)
else()
    llvm_map_components_to_libnames(LLVM_LIBS
      ${LLVM_TARGETS_TO_BUILD} target option
    )
endif()

if (MLIR_LINK_MLIR_DYLIB)
    set(MLIR_LIBS MLIR)
else()
    get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
    get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
    get_property(MLIR_EXTENSION_LIBS GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
    get_property(MLIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

    set(MLIR_LIBS
      MLIRAnalysis
      MLIRDialect
      MLIRExecutionEngine
      MLIRIR
      MLIRParser
      MLIRPass
      MLIRSupport
      MLIRTransforms
      MLIRTransformUtils

      ${MLIR_DIALECT_LIBS}
      ${MLIR_CONVERSION_LIBS}
      ${MLIR_EXTENSION_LIBS}
      ${MLIR_TRANSLATION_LIBS}
    )
endif()

if (CLANG_LINK_CLANG_DYLIB)
    set(CLANG_LIBS clang-cpp)
else()
    set(CLANG_LIBS
        clangAST
        clangASTMatchers
        clangBasic
        clangCodeGen
        clangDriver
        clangFrontend
        clangSerialization
        clangTooling
    )
endif()

#
# macroni build settings
#

set(MACRONI_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MACRONI_MAIN_INCLUDE_DIR ${MACRONI_MAIN_SRC_DIR}/include)

set(MACRONI_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)
set(MACRONI_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

# They are used as destination of target generators.
set(MACRONI_RUNTIME_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin)
set(MACRONI_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${MACRONI_LIBDIR_SUFFIX})
if(WIN32 OR CYGWIN)
  # DLL platform -- put DLLs into bin.
  set(MACRONI_SHLIB_OUTPUT_INTDIR ${MACRONI_RUNTIME_OUTPUT_INTDIR})
else()
  set(MACRONI_SHLIB_OUTPUT_INTDIR ${MACRONI_LIBRARY_OUTPUT_INTDIR})
endif()

set(MACRONI_TOOLS_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
  "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')"
)

set(MACRONI_TABLEGEN_EXE "${MLIR_TABLEGEN_EXE}" CACHE INTERNAL "")
set(MACRONI_TABLEGEN_TARGET "${MLIR_TABLEGEN_TARGET}" CACHE INTERNAL "")
set(MACRONI_PDLL_TABLEGEN_EXE "${MLIR_PDLL_TABLEGEN_EXE}" CACHE INTERNAL "")
set(MACRONI_PDLL_TABLEGEN_TARGET "${MLIR_PDLL_TABLEGEN_TARGET}" CACHE INTERNAL "")

add_custom_target(macroni-generic-headers)
set_target_properties(macroni-generic-headers PROPERTIES FOLDER "Misc")
add_custom_target(macroni-headers)
set_target_properties(macroni-headers PROPERTIES FOLDER "Misc")
add_dependencies(macroni-headers macroni-generic-headers)
add_custom_target(macroni-tools)
set_target_properties(macroni-tools PROPERTIES FOLDER "Misc")
add_custom_target(macroni-doc)

define_property(GLOBAL PROPERTY MACRONI_INSTALL_TARGETS)

if(CMAKE_GENERATOR MATCHES "Ninja" AND
    NOT "${NINJA_VERSION}" VERSION_LESS "1.9.0" AND
    CMAKE_HOST_APPLE AND CMAKE_HOST_SYSTEM_VERSION VERSION_GREATER "15.6.0")
  set(MACRONI_TOUCH_STATIC_LIBRARIES ON)
endif()

add_library(macroni_settings INTERFACE)

include(cmake/compiler_warnings.cmake)
set_project_warnings(macroni_settings)

target_include_directories(macroni_settings INTERFACE
  $<BUILD_INTERFACE:${MACRONI_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${MACRONI_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# enable exception on failure
option(MACRONI_ENABLE_EXCEPTIONS "Enable exception throw on macroni failure" OFF)
if (MACRONI_ENABLE_EXCEPTIONS)
  target_compile_definitions(macroni_settings
    INTERFACE
      -DMACRONI_ENABLE_EXCEPTIONS
  )

  target_compile_options(macroni_settings
    INTERFACE
      -fexceptions
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

if (NOT MACRONI_USE_VENDORED_GAP)
  find_package(gap CONFIG REQUIRED)
endif()

if (NOT MACRONI_USE_VENDORED_PASTA)
  find_package(pasta CONFIG REQUIRED)
endif()

if (NOT MACRONI_USE_VENDORED_VAST)
  # TODO(bpp): Allow using vendored VAST without using vendored gap
  if (MACRONI_USE_VENDORED_GAP)
    message(FATAL_ERROR "Using non-vendored VAST requires using non-vendored gap")
  endif()
  find_package(VAST CONFIG REQUIRED)
endif()

target_link_libraries(macroni_settings
  INTERFACE
    gap::gap
    pasta::pasta
    VAST::vast_settings
)

get_target_property(GAP_CORE_INCLUDE_DIR
  gap::gap-core INTERFACE_INCLUDE_DIRECTORIES)

get_target_property(GAP_SETTINGS_INCLUDE_DIR
gap::gap-settings INTERFACE_INCLUDE_DIRECTORIES)

get_target_property(VAST_SETTINGS_INCLUDE_DIR
  VAST::vast_settings INTERFACE_INCLUDE_DIRECTORIES)

get_target_property(PASTA_CXX_SETTINGS_INCLUDE_DIR
  pasta::pasta_cxx_settings INTERFACE_INCLUDE_DIRECTORIES)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${GAP_CORE_INCLUDE_DIR})
include_directories(${GAP_SETTINGS_INCLUDE_DIR})
include_directories(${PASTA_CXX_SETTINGS_INCLUDE_DIR})
include_directories(${VAST_SETTINGS_INCLUDE_DIR})

#
# macroni libraries
#
add_subdirectory(include/macroni)
add_subdirectory(lib)

#
# macroni executables
#
add_subdirectory(bin)

#
# macroni tests
#
enable_testing()
add_subdirectory(test)

#
# Install settings
#

option(MACRONI_INSTALL "Generate the install target." ${MACRONI_MASTER_PROJECT})

if (MACRONI_INSTALL)
  set(MACRONI_CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  get_property(MACRONI_INSTALL_TARGETS GLOBAL PROPERTY MACRONI_INSTALL_TARGETS)

  install(DIRECTORY ${MACRONI_INCLUDE_DIR}/macroni
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT macroni-headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.h.inc"
    PATTERN "*.hpp.inc"
    PATTERN "CMakeFiles" EXCLUDE
  )

  install(DIRECTORY ${MACRONI_SOURCE_DIR}/include/macroni
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT macroni-headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "CMakeFiles" EXCLUDE
  )

  install(TARGETS macroni_settings EXPORT macroniTargets)

  set(MACRONI_EXPORT_NAME macroniTargets)

  install(EXPORT macroniTargets
    FILE ${MACRONI_EXPORT_NAME}.cmake
    NAMESPACE macroni::
    DESTINATION ${MACRONI_CMAKE_INSTALL_DIR}
  )

  #
  # packaging support
  #

  set(CPACK_PACKAGE_VENDOR "Trail of Bits")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "macroni: A C/C++ frontend for lowering macros down to MLIR"
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
