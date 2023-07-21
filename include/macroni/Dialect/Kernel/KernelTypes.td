#ifndef MACRONI_DIALECT_KERNEL_IR_KERNELTYPES
#define MACRONI_DIALECT_KERNEL_IR_KERNELTYPES

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//
// Generic kernel type
//
class KernelType<string name, list<Trait> traits = []>
    : TypeDef<Kernel_Dialect, name, traits> { }

def AddressSpaceType : KernelType< "AddressSpace" > {
  let mnemonic = "address_space";
  let parameters = (ins "mlir::Type":$elementType, "int":$space);
  let assemblyFormat = " `(` $space `)` `<` $elementType `>`";
}

#endif
