#
# Copyright (c) 2023-present, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.25)

include("cmake/ccache.cmake")

project("macroni" CXX)

include(CMakeDependentOption)
include(GNUInstallDirs)
include("cmake/options.cmake")

find_package(Clang CONFIG REQUIRED)
find_package(gap CONFIG REQUIRED)
find_package(LLVM CONFIG REQUIRED)
find_package(MLIR CONFIG REQUIRED)
find_package(PASTA CONFIG REQUIRED)
find_package(VAST CONFIG REQUIRED)
find_package(Z3 CONFIG REQUIRED)

add_subdirectory("lib")
add_subdirectory("bin")