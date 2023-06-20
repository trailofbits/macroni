// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef MACRONI_DIALECT_IR_MACRONIOPS
#define MACRONI_DIALECT_IR_MACRONIOPS

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def MacroExpansionExpr :
  Macroni_Op< "macro.expr", [SameOperandsAndResultType] >
  , Arguments<(ins
    AnyType:$expansion,
    I64Attr:$macroID,
    StrAttr:$macroName,
    ArrayAttr:$parameterNames,
    DefaultValuedAttr<BoolAttr, "false">:$functionLike
  )>
  , Results<(outs AnyType:$result)> {
  let summary = "Macro expansion to an expression";
  let description = "A macro that expands to an expression";
  let assemblyFormat = " $macroID $macroName ($parameterNames $functionLike^)? attr-dict $expansion `:` type($result)";
}

def MacroExpansionStmt :
  Macroni_Op< "macro.stmt", [NoTerminator] >
  , Arguments<(ins
    I64Attr:$macroID,
    StrAttr:$macroName,
    ArrayAttr:$parameterNames,
    DefaultValuedAttr<BoolAttr, "false">:$functionLike
  )>
{
  let summary = "Macro expansion to a non-expression statement";
  let description = "A macro that expands to a non-expression statement";
  let regions = (region SizedRegion<1>:$expansion);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "std::optional<llvm::function_ref< void(mlir::OpBuilder &, mlir::Location) >>":$expansionBuilder,
      "mlir::IntegerAttr":$macroID,
      "mlir::StringAttr":$macroName,
      "mlir::ArrayAttr":$parameterNames,
      "mlir::BoolAttr":$functionLike
    )>
  ];
  let assemblyFormat = " $macroID $macroName ($parameterNames $functionLike^)? $expansion attr-dict";
}

#endif // MACRONI_DIALECT_IR_MACRONIOPS