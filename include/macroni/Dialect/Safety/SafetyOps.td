// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef MACRONI_DIALECT_IR_SAFETYOPS
#define MACRONI_DIALECT_IR_SAFETYOPS

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def UnsafeRegion :
  Safety_Op< "unsafe", [SingleBlock, NoTerminator] > {
  let summary = "A superficially unsafe region of code.";
  let description = [{
    A region of code wrapped in an expansion of the "unsafe" macro.
  }];
  let regions = (region SizedRegion<1>:$bodyRegion);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "vast::builder_callback_ref":$bodyBuilder)>
  ];
  let assemblyFormat = [{ $bodyRegion attr-dict }];
}

#endif // MACRONI_DIALECT_IR_SAFETYOPS
