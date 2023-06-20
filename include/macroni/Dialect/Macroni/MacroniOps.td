// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef MACRONI_DIALECT_IR_MACRONIOPS
#define MACRONI_DIALECT_IR_MACRONIOPS

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def MacroExpansionExpr :
  Macroni_Op< "expansion.expr", [SameOperandsAndResultType] >
  , Arguments<(ins
    AnyType:$expansion,
    StrAttr:$macroName,
    ArrayAttr:$parameterNames,
    DefaultValuedAttr<BoolAttr, "false">:$functionLike
  )>
  , Results<(outs AnyType:$result)> {
  let summary = "Macro expansion to an expression";
  let description = "A macro that expands to an expression";
  let assemblyFormat = " $macroName ($parameterNames $functionLike^)? attr-dict $expansion `:` type($result)";
}

def MacroParameterExpr :
  Macroni_Op< "parameter.expr", [SameOperandsAndResultType] >
  , Arguments<(ins
    AnyType:$expansion,
    StrAttr:$parameterName
  )>
  , Results<(outs AnyType:$result)> {
  let summary = "Macro parameter expansion to an expression";
  let description = "A macro parameter that expands to an expression";
  let assemblyFormat = " $parameterName attr-dict $expansion `:` type($result)";
}

// TODO: Add operations for stringifications token-pastings? We will only need
// these if we add operations for non-aligned macro expansions

def MacroExpansionStmt :
  Macroni_Op< "expansion.stmt", [NoTerminator] >
  , Arguments<(ins
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
      "mlir::StringAttr":$macroName,
      "mlir::ArrayAttr":$parameterNames,
      "mlir::BoolAttr":$functionLike
    )>
  ];
  let assemblyFormat = " $macroName ($parameterNames $functionLike^)? $expansion attr-dict";
}

def MacroParameterStmt :
  Macroni_Op< "parameter.stmt", [NoTerminator] >
  , Arguments<(ins
    StrAttr:$parameterName
  )>
{
  let summary = "Macro parameter expansion to a non-expression statement";
  let description = "A macro parameter that expands to a non-expression statement";
  let regions = (region SizedRegion<1>:$expansion);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "std::optional<llvm::function_ref< void(mlir::OpBuilder &, mlir::Location) >>":$expansionBuilder,
      "mlir::StringAttr":$parameterName
    )>
  ];
  let assemblyFormat = " $parameterName $expansion attr-dict";
}

#endif // MACRONI_DIALECT_IR_MACRONIOPS