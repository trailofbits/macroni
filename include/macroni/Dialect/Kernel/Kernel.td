// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef MACRONI_DIALECT_IR_KERNEL
#define MACRONI_DIALECT_IR_KERNEL

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Kernel_Dialect : Dialect {
    let name = "kernel";
    let cppNamespace = "::macroni::kernel";

    let summary = "A dialect for analyzing Linux kernel macro invocations";

    let description = [{
        macroni::kernel provides special Macroni-like operations for common
        macros in the Linux kernel, e.g., offsetof(), container_of(), and
        list_for_each().
    }];

    let extraClassDeclaration = [{
        void registerTypes();
        static llvm::StringRef rcu_read_lock() { return "rcu_read_lock"; }
        static llvm::StringRef rcu_read_unlock() { return "rcu_read_unlock"; }
    }];

    let useDefaultTypePrinterParser = 1;
}

class Kernel_Op<string mnemonic, list<Trait> traits = []>
    : Op<Kernel_Dialect, mnemonic, traits>;

include "KernelOps.td"
include "KernelTypes.td"

#endif // MACRONI_DIALECT_IR_KERNEL
