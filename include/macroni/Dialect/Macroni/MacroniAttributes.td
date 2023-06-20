// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef MACRONI_DIALECT_MACRONI_IR_MACRONIATTRIBUTES
#define MACRONI_DIALECT_MACRONI_IR_MACRONIATTRIBUTES

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class Macroni_Attr<string name, string attr_mnemonic, list<Trait> traits = []>
    : AttrDef< Macroni_Dialect, name, traits >
{
  let mnemonic = attr_mnemonic;
}

#endif // MACRONI_DIALECT_MACRONI_IR_MACRONIATTRIBUTES
