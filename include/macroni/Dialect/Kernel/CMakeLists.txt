# Copyright (c) 2023-present, Trail of Bits, Inc.
add_mlir_dialect(Kernel kernel)
add_mlir_doc(Kernel Kernel Kernel/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Kernel.td)
