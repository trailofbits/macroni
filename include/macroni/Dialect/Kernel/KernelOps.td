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

def RCUDereference :
  Kernel_Op< "rcu_dereference", [] >
  , Arguments<(ins AnyType:$p)>
  , Results<(outs AnyType:$result)> {
  let summary = "An expansion of the Linux kernel macro, rcu_dereference()";
  let description = [{
    rcu_dereference() - fetch RCU-protected pointer for dereferencing
    @p: The pointer to read, prior to dereferencing
    See https://www.kernel.org/doc/html/next/core-api/kernel-api.html
  }];
  let assemblyFormat = [{
    `rcu_dereference` `(` $p `)` attr-dict
    `:` functional-type( operands, results )
  }];
}

def RCUDereferenceBH :
  Kernel_Op< "rcu_dereference_bh", [] >
  , Arguments<(ins AnyType:$p)>
  , Results<(outs AnyType:$result)> {
  let summary = "An expansion of the Linux kernel macro, rcu_dereference_bh()";
  let description = [{
    rcu_dereference_bh() - fetch an RCU-bh-protected pointer for dereferencing
    @p: The pointer to read, prior to dereferencing
    See https://www.kernel.org/doc/html/next/core-api/kernel-api.html
  }];
  let assemblyFormat = [{
    `rcu_dereference_bh` `(` $p `)` attr-dict
    `:` functional-type( operands, results )
  }];
}

def RCUDereferenceSched :
  Kernel_Op< "rcu_dereference_sched", [] >
  , Arguments<(ins AnyType:$p)>
  , Results<(outs AnyType:$result)> {
  let summary = "An expansion of the Linux kernel macro, rcu_dereference_sched()";
  let description = [{
    rcu_dereference_sched() - fetch RCU-sched-protected pointer for dereferencing
    @p: The pointer to read, prior to dereferencing
    See https://www.kernel.org/doc/html/next/core-api/kernel-api.html
  }];
  let assemblyFormat = [{
    `rcu_dereference_sched` `(` $p `)` attr-dict
    `:` functional-type( operands, results )
  }];
}

def RCUDereferenceCheck :
  Kernel_Op< "rcu_dereference_check", [] >
  , Arguments<(ins
    AnyType:$p,
    AnyType:$c)>
  , Results<(outs AnyType:$result)> {
  let summary = [{
    An expansion of the Linux kernel macro, rcu_dereference_check()
  }];
  let description = [{
    rcu_dereference_check() - rcu_dereference with debug checking
    @p: The pointer to read, prior to dereferencing
    @c: The conditions under which the dereference will take place
    See https://www.kernel.org/doc/html/next/core-api/kernel-api.html
  }];
  let assemblyFormat = [{
    `rcu_dereference_check` `(` $p `,` $c `)` attr-dict
    `:` functional-type( operands, results )
  }];
}

def RCUDereferenceBHCheck :
  Kernel_Op< "rcu_dereference_bh_check", [] >
  , Arguments<(ins
    AnyType:$p,
    AnyType:$c)>
  , Results<(outs AnyType:$result)> {
  let summary = [{
    An expansion of the Linux kernel macro, rcu_dereference_bh_check()
  }];
  let description = [{
    rcu_dereference_bh_check() - rcu_dereference_bh with debug checking
    @p: The pointer to read, prior to dereferencing
    @c: The conditions under which the dereference will take place
    See https://www.kernel.org/doc/html/next/core-api/kernel-api.html
  }];
  let assemblyFormat = [{
    `rcu_dereference_bh_check` `(` $p `,` $c `)` attr-dict
    `:` functional-type( operands, results )
  }];
}

def RCUDereferenceSchedCheck :
  Kernel_Op< "rcu_dereference_sched_check", [] >
  , Arguments<(ins
    AnyType:$p,
    AnyType:$c)>
  , Results<(outs AnyType:$result)> {
  let summary = [{
    An expansion of the Linux kernel macro, rcu_dereference_sched_check()
  }];
  let description = [{
    rcu_dereference_sched_check() - rcu_dereference_sched with debug checking
    @p: The pointer to read, prior to dereferencing
    @c: The conditions under which the dereference will take place
    See https://www.kernel.org/doc/html/next/core-api/kernel-api.html
  }];
  let assemblyFormat = [{
    `rcu_dereference_sched_check` `(` $p `,` $c `)` attr-dict
    `:` functional-type( operands, results )
  }];
}

def RCUDereferenceProtected :
  Kernel_Op< "rcu_dereference_protected", [] >
  , Arguments<(ins
    AnyType:$p,
    AnyType:$c)>
  , Results<(outs AnyType:$result)> {
  let summary = [{
    An expansion of the Linux kernel macro, rcu_dereference_protected()
  }];
  let description = [{
    rcu_dereference_protected() - fetch RCU pointer when updates prevented
    @p: The pointer to read, prior to dereferencing
    @c: The conditions under which the dereference will take place
    See https://www.kernel.org/doc/html/next/core-api/kernel-api.html
  }];
  let assemblyFormat = [{
    `rcu_dereference_protected` `(` $p `,` $c `)` attr-dict
    `:` functional-type( operands, results )
  }];
}

def RCUAccessPointer :
  Kernel_Op< "rcu_access_pointer", [] >
  , Arguments<(ins AnyType:$p)>
  , Results<(outs AnyType:$result)> {
  let summary = [{
    An expansion of the Linux kernel macro, rcu_access_pointer()
  }];
  let description = [{
    rcu_access_pointer() - fetch RCU pointer with no dereferencing 
    @p: The pointer to read
    See https://www.kernel.org/doc/html/next/core-api/kernel-api.html
  }];
  let assemblyFormat = [{
    `rcu_access_pointer` `(` $p `)` attr-dict
    `:` functional-type( operands, results )
  }];
}

def SMPMB :
  Kernel_Op< "smp_mb", [] >
  , Results<(outs AnyType:$result)> {
  let summary = "An expansion of the Linux kernel macro, smp_mb()";
  let description = "smp_mb() activates a memory barrier";
  let assemblyFormat = [{
    `smp_mb` `(` `)` attr-dict `:` functional-type( operands, results )
  }];
}

#endif // MACRONI_DIALECT_IR_KERNELOPS