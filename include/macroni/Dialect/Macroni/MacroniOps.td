// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef MACRONI_DIALECT_IR_MACRONIOPS
#define MACRONI_DIALECT_IR_MACRONIOPS

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

/* Expression macros */

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
  let assemblyFormat = "custom<MacroParameters>($macroName, $functionLike, $parameterNames) attr-dict $expansion `->` type($result)";
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
  let assemblyFormat = "$parameterName attr-dict $expansion `->` type($result)";
}

// TODO: Add operations for stringifications token-pastings? We will only need
// these if we add operations for non-aligned macro expansions

/* Statement macros */

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
  let assemblyFormat = "custom<MacroParameters>($macroName, $functionLike, $parameterNames) $expansion attr-dict";
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
  let assemblyFormat = "$parameterName $expansion attr-dict";
}

/* Special macros */

def GetUser :
  Macroni_Op< "get_user", [] >
  , Arguments<(ins
    AnyType:$x,
    AnyType:$ptr
  )>
  , Results<(outs AnyType:$result)> {
  let summary = "An expansion of the Linux kernel macro, get_user";
  let description = [{
    The get_user macro copies a single simple variable from user space to kernel
    space. It accepts two arguments:
    - x: A variable in which to store the result.
    - ptr: A source address in user space.
    get_user returns zero on success, or -EFAULT on error.

    More information: https://www.cs.bham.ac.uk/~exr/lectures/opsys/12_13/docs/kernelAPI/r3776.html
  }];
  
  // GOAL: Rewrite certain MacroExpansionExpr operations with GetUser
  // operations. To improve dataflow analysis, GetUser should return two values,
  // the modified value and whether the operation was successful
  
  // Problem: In order to use the MLIR rewriter to rewrite one operation into
  // another, the results of the two operations must be the same, but GetUser
  // returns two values, whereas MacroExpansionExpr returns one.
  
  let assemblyFormat = "`get_user` `(` $x `,` $ptr `)` attr-dict `:` functional-type( operands, results )";
}

#endif // MACRONI_DIALECT_IR_MACRONIOPS