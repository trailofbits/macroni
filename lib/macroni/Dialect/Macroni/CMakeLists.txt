# Copyright (c) 2023-present, Trail of Bits, Inc.

add_mlir_dialect_library(MLIRMacroni
    MacroniDialect.cpp
    MacroniOps.cpp

    ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/macroni

    DEPENDS
        MLIRMacroniIncGen

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

target_link_libraries(MLIRMacroni PRIVATE macroni)