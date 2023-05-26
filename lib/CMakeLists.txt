#
# Copyright (c) 2023-present, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

set(PROJECT_NAME "macroni")

set(source_include_dir "${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}")

# https://github.com/JonathanSalwan/Triton/issues/1082#issuecomment-1030826696
if(LLVM_LINK_LLVM_DYLIB)
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs
    support core
    irreader
    bitreader bitwriter
    passes asmprinter
    aarch64desc aarch64codegen aarch64asmparser
    armdesc armcodegen armasmparser
    interpreter mcjit
    nvptxdesc nvptxcodegen
    x86desc x86codegen x86asmparser
    sparcdesc sparccodegen sparcasmparser
    webassemblydesc webassemblycodegen webassemblyasmparser)
endif()


add_library("${PROJECT_NAME}" STATIC
  
)

target_link_libraries("${PROJECT_NAME}"
  PUBLIC
    MLIRDialect
    MLIRMLProgramDialect
    MLIROpenMPDialect
    MLIRTosaDialect

    MLIRBufferizationTransformOps
    MLIRSparseTensorTransforms
    MLIROpenACCDialect
    MLIRLLVMIRTransforms
    MLIRROCDLDialect
    MLIRAsyncDialect
    MLIRSCFTransformOps
    MLIREmitCDialect
    MLIRNVGPUDialect
    MLIRShapeDialect
    MLIRAMDGPUDialect
    MLIRShapeOpsTransforms
    MLIRLinalgTransformOps
    MLIRSPIRVDialect
    MLIRTensorInferTypeOpInterfaceImpl
    
    ${llvm_libs}
    
    clang
    clangFrontend
    clangDriver
    clangSerialization
    clangParse
    clangAnalysis
    clangEdit
    clangAST
    clangLex
    clangBasic
    clangTooling
    clangIndex
)

target_include_directories("${PROJECT_NAME}"
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features("${PROJECT_NAME}"
  PRIVATE
    cxx_std_20
)