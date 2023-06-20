# Copyright (c) 2023-present, Trail of Bits, Inc.
add_mlir_dialect(Macroni macroni)
add_mlir_doc(Macroni Macroni Macroni/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Macroni.td)

mlir_tablegen(MacroniEnums.h.inc -gen-enum-decls)
mlir_tablegen(MacroniEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMacroniEnumsIncGen)

mlir_tablegen(MacroniAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(MacroniAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRMacroniAttributesIncGen)

