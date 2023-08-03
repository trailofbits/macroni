// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef MACRONI_DIALECT_IR_SAFETY
#define MACRONI_DIALECT_IR_SAFETY

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Safety_Dialect : Dialect {
    let name = "safety";
    let cppNamespace = "::macroni::safety";

    let summary = "A dialect providing Rust-like unsafe regions to C";

    let description = [{
        macroni::Safety provides C support Rust-like safety regions.
    }];

    let useFoldAPI = kEmitFoldAdaptorFolder;
}

class Safety_Op<string mnemonic, list<Trait> traits = []>
    : Op<Safety_Dialect, mnemonic, traits>;

include "SafetyOps.td"

#endif // MACRONI_DIALECT_IR_SAFETY
