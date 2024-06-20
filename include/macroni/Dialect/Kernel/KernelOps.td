// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef MACRONI_DIALECT_IR_KERNELOPS
#define MACRONI_DIALECT_IR_KERNELOPS

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

include "macroni/Dialect/Kernel/KernelInterfaces.td"

/* Special macros */

def GetUser :
  Kernel_Op<"get_user">
  , Arguments<(ins
    AnyType:$x,
    AnyType:$ptr
  )>
  , Results<(outs AnyType:$result)> {
  let summary = "An expansion of the Linux kernel macro, get_user()";
  let description = [{
    The get_user macro copies a single simple variable from user space to kernel
    space. It accepts two arguments:
    - x: A variable in which to store the result.
    - ptr: A source address in user space.
    et_user returns zero on success, or -EFAULT on error.

    More information: https://www.cs.bham.ac.uk/~exr/lectures/opsys/12_13/docs/kernelAPI/r3776.html
  }];
  let assemblyFormat = [{
    `(` $x `,` $ptr `)` attr-dict `:` functional-type( operands, results )
  }];
}

def OffsetOf :
  Kernel_Op<"offsetof">
  , Arguments<(ins
    TypeAttr:$type,
    StrAttr:$member
  )>
  , Results<(outs AnyType:$result)> {
  let summary = "An expansion of the Linux kernel macro, offsetof()";
  let description = [{
     The macro offsetof() returns the offset of the field member from
     the start of the structure type.

     This macro is useful because the sizes of the fields that compose
     a structure can vary across implementations, and compilers may
     insert different numbers of padding bytes between fields.
     Consequently, an element's offset is not necessarily given by the
     sum of the sizes of the previous elements.

     A compiler error will result if member is not aligned to a byte
     boundary (i.e., it is a bit field).

    More information: https://man7.org/linux/man-pages/man3/offsetof.3.html
  }];
  let assemblyFormat = [{
    `(` $type `,` $member `)` attr-dict `:` functional-type( operands, results )
  }];
}

def ContainerOf :
  Kernel_Op<"container_of">
  , Arguments<(ins
    AnyType:$ptr,
    TypeAttr:$type,
    StrAttr:$member
  )>
  , Results<(outs AnyType:$result)> {
  let summary = "An expansion of the Linux kernel macro, container_of()";
  let description = [{
     The macro container_of() returns an address pointing to the container of
     of type type that contains a instance ptr of the member named member named
     member.
  }];
  let assemblyFormat = [{
    `(` $ptr `,` $type `,` $member `)`
    attr-dict `:` functional-type( operands, results )
  }];
}

def ListForEach :
  Kernel_Op<"list_for_each()", [SingleBlock, NoTerminator] >
  , Arguments<(ins
    AnyType:$pos,
    AnyType:$head
  )> {
  let summary = "An expansion of the Linux kernel macro, list_for_each()";
  let description = [{
     The macro list_for_each() uses a temporary variable pos to iterate the list
     starting at head. It performs the following statement for each node in the
     list.
  }];
  let regions = (region SizedRegion<1>:$bodyRegion);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "mlir::Value":$pos,
      "mlir::Value":$head,
      "std::unique_ptr< mlir::Region > &&":$bodyRegion
    )>
  ];
  let assemblyFormat = [{
    `(` $pos `,` $head `)` attr-dict
    `:` functional-type( operands, results ) $bodyRegion
  }];
}

def SMPMB :
  Kernel_Op<"smp_mb">
  , Results<(outs AnyType:$result)> {
  let summary = "An expansion of the Linux kernel macro, smp_mb()";
  let description = "smp_mb() activates a memory barrier";
  let assemblyFormat = [{
    `(` `)` attr-dict `:` functional-type( operands, results )
  }];
}

def RCUCriticalSection :
  Kernel_Op<"rcu.critical_section", [NoTerminator] > {
  let summary = "An RCU critical section";
  let description = [{
    An RCU critical section denoted by an rcu_read_lock() and rcu_read_unlock()
    pair. This operation only creates a section for lock/unlock that are called
    in the same region of C source code.
  }];
  let regions = (region AnyRegion:$bodyRegion);
  let assemblyFormat = "$bodyRegion attr-dict";
}

/* RCU macros */

class RCU_Op<string mnemonic, list<Trait> traits = []>
  : Kernel_Op<mnemonic, !listconcat(traits, [])> {
    let description = [{
      See https://www.kernel.org/doc/html/next/core-api/kernel-api.html
    }];
}

class RCU_Unary_Op<string mnemonic, list<Trait> traits = []>
  : RCU_Op<mnemonic, !listconcat(traits,
    [DeclareOpInterfaceMethods<RCU_Macro_Interface>])>
  , Arguments<(ins AnyType:$p)>
  , Results<(outs AnyType:$result)> {
    let summary = "An expansion of an RCU macro with one parameter, p";
    let assemblyFormat = [{
      `(` $p `)` attr-dict
      `:` functional-type( operands, results )
    }];
}

def RCUAccessPointer : RCU_Unary_Op<"rcu_access_pointer">;

class RCU_Deference_Op<string mnemonic, list<Trait> traits = []> :
  RCU_Unary_Op<mnemonic, !listconcat(traits,
    [DeclareOpInterfaceMethods<RCU_Dereference_Interface>])> {
    let summary = "An expansion of an rcu_dereference() macro variant";
}

def RCUDereference : RCU_Deference_Op<"rcu_dereference">;
def RCUDereferenceBH : RCU_Deference_Op<"rcu_dereference_bh">;
def RCUDereferenceSched : RCU_Deference_Op<"rcu_dereference_sched">;

class RCU_Deference_Check_Op<string mnemonic, list<Trait> traits = []>
  : RCU_Op<mnemonic, !listconcat(traits,
  [DeclareOpInterfaceMethods<RCU_Macro_Interface>])>
  , Arguments<(ins
    AnyType:$p,
    AnyType:$c)>
  , Results<(outs AnyType:$result)> {
    let summary = "An expansion of an rcu_dereference_check() macro variant";
    let description = [{
      See https://www.kernel.org/doc/html/next/core-api/kernel-api.html
    }];
    let assemblyFormat = [{
      `(` $p `,` $c `)` attr-dict
      `:` functional-type( operands, results )
    }];
}

def RCUDereferenceCheck : RCU_Deference_Check_Op<"rcu_dereference_check">;
def RCUDereferenceBHCheck : RCU_Deference_Check_Op<"rcu_dereference_bh_check">;
def RCUDereferenceSchedCheck :
  RCU_Deference_Check_Op<"rcu_dereference_sched_check">;
def RCUDereferenceProtected :
  RCU_Deference_Check_Op<"rcu_dereference_protected">;

def RCUAssignPointer :
  RCU_Op<"rcu_assign_pointer">
  , Arguments<(ins
    AnyType:$p
    , AnyType:$v)>
  , Results<(outs AnyType:$result)> {
  let summary = [{
    An expansion of the Linux kernel macro, rcu_assign_pointer()
  }];
  let description = [{
    See https://www.kernel.org/doc/html/next/core-api/kernel-api.html
  }];
  let assemblyFormat = [{
    `(` $p `,` $v `)` attr-dict
    `:` functional-type( operands, results )
  }];
}

def RCUReplacePointer :
  RCU_Op<"rcu_replace_pointer">
  , Arguments<(ins
    AnyType:$rcu_ptr
    , AnyType:$ptr
    , AnyType:$c)>
  , Results<(outs AnyType:$result)> {
  let summary = [{
    An expansion of the Linux kernel macro, rcu_replace_pointer()
  }];
  let description = [{
    See https://www.kernel.org/doc/html/next/core-api/kernel-api.html
  }];
  let assemblyFormat = [{
    `(` $rcu_ptr `,` $ptr `,` $c `)` attr-dict
    `:` functional-type( operands, results )
  }];
}

#endif // MACRONI_DIALECT_IR_KERNELOPS
