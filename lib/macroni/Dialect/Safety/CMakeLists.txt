# Copyright (c) 2023-present, Trail of Bits, Inc.

add_mlir_dialect_library(MLIRSafety
  SafetyDialect.cpp
  SafetyOps.cpp

  DEPENDS MLIRSafetyIncGen

  LINK_LIBS PUBLIC ${MLIR_LIBS})
