// Before
hl.func external @main () -> !hl.int {
  hl.scope {
    %0 = hl.expr : !hl.elaborated<!hl.typedef<"size_t">> {
      %2 = hl.expr : !hl.ptr<!hl.elaborated<!hl.record<"A">>> {
        %6 = hl.const #hl.integer<0> : !hl.int
        %7 = hl.implicit_cast %6 NullToPointer : !hl.int -> !hl.ptr<!hl.elaborated<!hl.record<"A">>>
        %8 = hl.cstyle_cast %7 NoOp : !hl.ptr<!hl.elaborated<!hl.record<"A">>> -> !hl.ptr<!hl.elaborated<!hl.record<"A">>>
        hl.value.yield %8 : !hl.ptr<!hl.elaborated<!hl.record<"A">>>
      }
      %3 = hl.member %2 at "x" : !hl.ptr<!hl.elaborated<!hl.record<"A">>> -> !hl.lvalue<!hl.int>
      %4 = hl.addressof %3 : !hl.lvalue<!hl.int> -> !hl.ptr<!hl.int>
      %5 = hl.cstyle_cast %4 PointerToIntegral : !hl.ptr<!hl.int> -> !hl.elaborated<!hl.typedef<"size_t">>
      hl.value.yield %5 : !hl.elaborated<!hl.typedef<"size_t">>
    }
    %1 = hl.const #hl.integer<0> : !hl.int
    hl.return %1 : !hl.int
  }
}

// After
hl.func external @main () -> !hl.int {
  hl.scope {
    %0 = macroni.offsetof offsetof(!hl.elaborated<!hl.record<"A">>, "x") : () -> !hl.elaborated<!hl.typedef<"size_t">>
    %1 = hl.const #hl.integer<0> : !hl.int
    hl.return %1 : !hl.int
  }
}
