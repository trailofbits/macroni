# Copyright (c) 2023-present, Trail of Bits, Inc.

add_mlir_dialect(Kernel kernel)
add_mlir_doc(Kernel Kernel Kernel/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS KernelAttributes.td)
mlir_tablegen(KernelAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(KernelAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRKernelAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS KernelInterfaces.td)
mlir_tablegen(KernelInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(KernelInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRKernelInterfacesIncGen)
