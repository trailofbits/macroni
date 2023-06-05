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

get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries("${PROJECT_NAME}"
  PUBLIC
    ${DIALECT_LIBS}
    ${CONVERSION_LIBS}
    MLIRAffineAnalysis
    MLIRAffineDialect
    MLIRAffineTransformOps
    MLIRAffineTransforms
    MLIRAffineUtils
    MLIRAMDGPUDialect
    MLIRAsyncDialect
    MLIRBufferizationTransformOps
    MLIRCore
    MLIRDialect
    MLIREmitCDialect
    MLIRExecutionEngine
    MLIRGPUOps
    MLIRGPUTransformOps
    MLIRGPUTransforms
    MLIRHighLevel
    MLIRHighLevelTransforms
    MLIRIR
    MLIRLinalgTransformOps
    MLIRLLVMCommonConversion
    MLIRLLVMIRTransforms
    MLIRLLVMToLLVMIRTranslation
    MLIRLowLevel
    MLIRMemRefDialect
    MLIRMemRefTransformOps
    MLIRMemRefTransforms
    MLIRMemRefUtils
    MLIRMeta
    MLIRMLProgramDialect
    MLIRNVGPUDialect
    MLIROpenACCDialect
    MLIROpenMPDialect
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRROCDLDialect
    MLIRSCFTransformOps
    MLIRShapeDialect
    MLIRShapeOpsTransforms
    MLIRSparseTensorTransforms
    MLIRSPIRVDialect
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRToLLVMIRTranslationRegistration
    MLIRTosaDialect
    MLIRTransforms
    MLIRTransformUtils
    MLIRVectorDialect
    MLIRVectorTransformOps
    MLIRVectorTransforms
    MLIRVectorUtils
    
    ${llvm_libs}
    
    clang
    clangAnalysis
    clangAST
    clangBasic
    clangDriver
    clangEdit
    clangFrontend
    clangIndex
    clangLex
    clangParse
    clangSerialization
    clangTooling
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