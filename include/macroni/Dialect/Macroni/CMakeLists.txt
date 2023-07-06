# Copyright (c) 2023-present, Trail of Bits, Inc.
add_mlir_dialect(Macroni macroni)
add_mlir_doc(Macroni Macroni Macroni/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Macroni.td)
