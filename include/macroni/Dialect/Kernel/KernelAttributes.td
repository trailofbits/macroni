#ifndef MACRONI_DIALECT_KERNEL_IR_KERNELATTRIBUTES
#define MACRONI_DIALECT_KERNEL_IR_KERNELATTRIBUTES

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "macroni/Dialect/Kernel/Kernel.td"

//
// Generic kernel attribute
//
class KernelAttr<string name, string attr_mnemonic, list<Trait> traits = []>
    : AttrDef<Kernel_Dialect, name, traits>
{
  let mnemonic = attr_mnemonic;
}

class ContextAttr<string name, string attr_mnemonic>
  : KernelAttr< name, attr_mnemonic >
{
  let parameters = (ins "::mlir::StringAttr":$lock);
  let assemblyFormat = "`<` $lock `>`";
}

def MustHoldAttr : ContextAttr < "MustHold", "__must_hold" >;
def AcquiresAttr : ContextAttr < "Acquires", "__acquires" >;
def ReleasesAttr : ContextAttr < "Releases", "__releases" >;

#endif // MACRONI_DIALECT_KERNEL_IR_KERNELATTRIBUTES
