// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef MACRONI_DIALECT_IR_MACRONIOPS
#define MACRONI_DIALECT_IR_MACRONIOPS

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

/* Expansion and parameter expansions */

def MacroExpansion :
  Macroni_Op< "expansion", [SingleBlock, NoTerminator] >
  , Arguments<(ins
    StrAttr:$macroName,
    ArrayAttr:$parameterNames,
    DefaultValuedAttr<BoolAttr, "false">:$functionLike
  )>
  , Results<(outs Optional<AnyType>:$result)> {
  let summary = "A macro expansion";
  let description = "A macro expansion which may result in a a value";
  let regions = (region SizedRegion<1>:$expansion);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "mlir::StringAttr":$macroName,
      "mlir::ArrayAttr":$parameterNames,
      "mlir::BoolAttr":$functionLike,
      "mlir::Type":$rty,
      "std::unique_ptr< mlir::Region > &&":$region
    )>
  ];
  let assemblyFormat = [{
    custom<MacroParameters>($macroName, $functionLike, $parameterNames)
    attr-dict `:` type($result) $expansion
  }];
}

def MacroParameter :
  Macroni_Op< "parameter", [SingleBlock, NoTerminator] >
  , Arguments<(ins StrAttr:$parameterName )>
  , Results<(outs Optional<AnyType>:$result)> {
  let summary = "A macro parameter substitution";
  let description = [{
    A macro parameter substitution which may result in a a value
  }];
  let regions = (region SizedRegion<1>:$expansion);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "mlir::StringAttr":$parameterName,
      "mlir::Type":$rty,
      "std::unique_ptr< mlir::Region > &&":$region
    )>
  ];
  let assemblyFormat = "$parameterName attr-dict `:` type($result) $expansion";
}

#endif // MACRONI_DIALECT_IR_MACRONIOPS
