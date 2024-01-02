# Copyright (c) 2023-present, Trail of Bits, Inc.

add_mlir_dialect_library(MLIRKernel
    KernelDialect.cpp
    KernelInterfaces.cpp
    KernelOps.cpp
    KernelAttributes.cpp
    KernelTypes.cpp

    ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

    DEPENDS
        MLIRKernelAttributesIncGen
        MLIRKernelIncGen
        MLIRKernelInterfacesIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRSideEffectInterfaces

        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRControlFlowInterfaces
        MLIRDataLayoutInterfaces
        MLIRInferTypeOpInterface
)
