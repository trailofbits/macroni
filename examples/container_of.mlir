// Before
hl.func external @main () -> !hl.int {
  hl.scope {
    %0 = hl.var "container_instance" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"container">>>>
    %1 = hl.stmt.expr : !hl.ptr<!hl.elaborated<!hl.record<"container">>> {
      %3 = hl.var "__mptr" : !hl.lvalue<!hl.ptr<!hl.typeof.expr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>>> = {
        %6 = hl.expr : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> {
          %8 = hl.ref %0 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"container">>>>
          %9 = hl.implicit_cast %8 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"container">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"container">>>
          %10 = hl.member %9 at "contained_member" : !hl.ptr<!hl.elaborated<!hl.record<"container">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
          %11 = hl.addressof %10 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> -> !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
          hl.value.yield %11 : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
        }
        %7 = hl.implicit_cast %6 NoOp : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> -> !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
        hl.value.yield %7 : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
      }
      %4 = hl.expr : !hl.ptr<!hl.char> {
        %6 = hl.ref %3 : !hl.lvalue<!hl.ptr<!hl.typeof.expr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>>>
        %7 = hl.implicit_cast %6 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.typeof.expr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>>> -> !hl.ptr<!hl.typeof.expr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>>
        %8 = hl.cstyle_cast %7 BitCast : !hl.ptr<!hl.typeof.expr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>> -> !hl.ptr<!hl.char>
        %9 = hl.expr : !hl.elaborated<!hl.typedef<"size_t">> {
          %11 = hl.expr : !hl.ptr<!hl.elaborated<!hl.record<"container">>> {
            %15 = hl.const #hl.integer<0> : !hl.int
            %16 = hl.implicit_cast %15 NullToPointer : !hl.int -> !hl.ptr<!hl.elaborated<!hl.record<"container">>>
            %17 = hl.cstyle_cast %16 NoOp : !hl.ptr<!hl.elaborated<!hl.record<"container">>> -> !hl.ptr<!hl.elaborated<!hl.record<"container">>>
            hl.value.yield %17 : !hl.ptr<!hl.elaborated<!hl.record<"container">>>
          }
          %12 = hl.member %11 at "contained_member" : !hl.ptr<!hl.elaborated<!hl.record<"container">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
          %13 = hl.addressof %12 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> -> !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
          %14 = hl.cstyle_cast %13 PointerToIntegral : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> -> !hl.elaborated<!hl.typedef<"size_t">>
          hl.value.yield %14 : !hl.elaborated<!hl.typedef<"size_t">>
        }
        %10 = hl.sub %8, %9 : (!hl.ptr<!hl.char>, !hl.elaborated<!hl.typedef<"size_t">>) -> !hl.ptr<!hl.char>
        hl.value.yield %10 : !hl.ptr<!hl.char>
      }
      %5 = hl.cstyle_cast %4 BitCast : !hl.ptr<!hl.char> -> !hl.ptr<!hl.elaborated<!hl.record<"container">>>
      hl.value.yield %5 : !hl.ptr<!hl.elaborated<!hl.record<"container">>>
    }
    %2 = hl.const #hl.integer<0> : !hl.int
    hl.return %2 : !hl.int
  }
}

// After
hl.func external @main () -> !hl.int {
  hl.scope {
    %0 = hl.var "container_instance" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"container">>>>
    %1 = macroni.parameter "ptr" : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> {
      %4 = hl.ref %0 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"container">>>>
      %5 = hl.implicit_cast %4 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"container">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"container">>>
      %6 = hl.member %5 at "contained_member" : !hl.ptr<!hl.elaborated<!hl.record<"container">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
      %7 = hl.addressof %6 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>> -> !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
      hl.value.yield %7 : !hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>
    }
    %2 = kernel.container_of container_of(%1, !hl.elaborated<!hl.record<"container">>, "contained_member") : (!hl.ptr<!hl.ptr<!hl.elaborated<!hl.record<"contained">>>>) -> !hl.ptr<!hl.elaborated<!hl.record<"container">>>
    %3 = hl.const #hl.integer<0> : !hl.int
    hl.return %3 : !hl.int
  }
}
