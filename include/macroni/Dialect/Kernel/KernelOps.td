// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef MACRONI_DIALECT_IR_KERNELOPS
#define MACRONI_DIALECT_IR_KERNELOPS

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

/* Special macros */

def GetUser :
  Kernel_Op< "get_user", [] >
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
    get_user returns zero on success, or -EFAULT on error.

    More information: https://www.cs.bham.ac.uk/~exr/lectures/opsys/12_13/docs/kernelAPI/r3776.html
  }];
  let assemblyFormat = "`get_user` `(` $x `,` $ptr `)` attr-dict `:` functional-type( operands, results )";
}

def OffsetOf :
  Kernel_Op< "offsetof", [] >
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
  let assemblyFormat = "`offsetof` `(` $type `,` $member `)` attr-dict `:` functional-type( operands, results )";
}

def ContainerOf :
  Kernel_Op< "container_of", [] >
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
    `container_of` `(` $ptr `,` $type `,` $member `)`
    attr-dict `:` functional-type( operands, results )
  }];
}

def ListForEach :
  Kernel_Op< "list_for_each()", [SingleBlock, NoTerminator] >
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
  let assemblyFormat = "`list_for_each` `(` $pos `,` $head `)` attr-dict `:` functional-type( operands, results ) $bodyRegion";
}

def RCUCriticalSection :
  Kernel_Op< "rcu.critical_section", [NoTerminator] >
{
  let summary = "An RCU critical section";
  let description = [{
    An RCU critical section denoted by an rcu_read_lock() and rcu_read_unlock()
    pair. This operation only creates a section for lock/unlock that are called
    in the same region of C source code.
  }];
  let regions = (region AnyRegion:$bodyRegion);
  let assemblyFormat = "$bodyRegion attr-dict";
}

#endif // MACRONI_DIALECT_IR_KERNELOPS