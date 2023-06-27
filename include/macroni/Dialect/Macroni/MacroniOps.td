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