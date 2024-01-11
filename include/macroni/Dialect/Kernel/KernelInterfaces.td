#ifndef KERNEL_IR_KERNEL_INTERFACES
#define KERNEL_IR_KERNEL_INTERFACES

include "mlir/IR/OpBase.td"

def RCU_Macro_Interface :
  OpInterface<"RCU_Macro_Interface"> {
  let cppNamespace = "::macroni::kernel";
}

def RCU_Dereference_Interface :
  OpInterface<"RCU_Dereference_Interface"> {
  let cppNamespace = "::macroni::kernel";
}

#endif // KERNEL_IR_KERNEL_INTERFACES
