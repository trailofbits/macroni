# Copyright (c) 2023-present, Trail of Bits, Inc.
add_mlir_dialect(Safety safety)
add_mlir_doc(Safety Safety Safety/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Safety.td)
