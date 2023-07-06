# Copyright (c) 2023-present, Trail of Bits, Inc.

add_mlir_dialect_library(MLIRKernel
    KernelDialect.cpp
    KernelOps.cpp

    ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/macroni

    DEPENDS
        MLIRKernelIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRSideEffectInterfaces

        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRControlFlowInterfaces
        MLIRDataLayoutInterfaces
        MLIRInferTypeOpInterface

        VASTSymbolInterface
        VASTTypeQualifiersInterfaces
)

target_link_libraries(MLIRKernel PRIVATE macroni)