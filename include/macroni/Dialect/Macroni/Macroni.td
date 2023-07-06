// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef MACRONI_DIALECT_IR_MACRONI
#define MACRONI_DIALECT_IR_MACRONI

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Macroni_Dialect : Dialect {
    let name = "macroni";
    let cppNamespace = "::macroni::macroni";

    let summary = "A dialect for analyzing C/C++ macro invocations";

    let description = [{
        Macroni is a language that allows you to represent macro invocations
        that align with a C/C++ program's AST in a high-level format. This
        dialect provides a representation of macros that is amenable to
        analysis and optimization.
    }];

    let useFoldAPI = kEmitFoldAdaptorFolder;
}

class Macroni_Op<string mnemonic, list<Trait> traits = []>
    : Op<Macroni_Dialect, mnemonic, traits>;

include "MacroniOps.td"

#endif // MACRONI_DIALECT_IR_MACRONI
