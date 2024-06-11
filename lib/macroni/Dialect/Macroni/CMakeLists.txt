# Copyright (c) 2023-present, Trail of Bits, Inc.

add_mlir_dialect_library(MLIRMacroni
  MacroniDialect.cpp
  MacroniOps.cpp

  DEPENDS
    MLIRMacroniIncGen

  LINK_LIBS PUBLIC macroni_settings)
