// Before
hl.func external @main () -> !hl.int {
  hl.scope {
    %0 = hl.var "head" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
    %1 = hl.var "pos" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
    hl.scope {
      %3 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
      %4 = hl.expr : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
        %9 = hl.ref %0 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
        hl.value.yield %9 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
      }
      %5 = hl.implicit_cast %4 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
      %6 = hl.member %5 at "next" : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
      %7 = hl.implicit_cast %6 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
      %8 = hl.assign %7 to %3 : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>, !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
      hl.for {
        %9 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
        %10 = hl.implicit_cast %9 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
        %11 = hl.implicit_cast %10 NoOp : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
        %12 = hl.expr : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
          %18 = hl.ref %0 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
          hl.value.yield %18 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
        }
        %13 = hl.implicit_cast %12 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
        %14 = hl.implicit_cast %13 NoOp : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
        %15 = hl.call @list_is_head(%11, %14) : (!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>, !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>) -> !hl.int
        %16 = hl.implicit_cast %15 IntegralToBoolean : !hl.int -> !hl.bool
        %17 = hl.lnot %16 : !hl.bool -> !hl.bool
        hl.cond.yield %17 : !hl.bool
      } incr {
        %9 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
        %10 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
        %11 = hl.implicit_cast %10 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
        %12 = hl.member %11 at "next" : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
        %13 = hl.implicit_cast %12 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
        %14 = hl.assign %13 to %9 : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>, !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
      } do {
        hl.scope {
          %9 = hl.var "prev" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> = {
            %10 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
            %11 = hl.implicit_cast %10 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
            %12 = hl.member %11 at "prev" : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
            %13 = hl.implicit_cast %12 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
            hl.value.yield %13 : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
          }
        }
      }
    }
    %2 = hl.const #hl.integer<0> : !hl.int
    hl.return %2 : !hl.int
  }
}

// After
hl.func external @main () -> !hl.int {
  hl.scope {
    %0 = hl.var "head" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
    %1 = hl.var "pos" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
    hl.scope {
      %3 = macroni.parameter "pos" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
        %11 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
        hl.value.yield %11 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
      }
      %4 = hl.expr : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
        %11 = macroni.parameter "head" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
          %12 = hl.ref %0 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
          hl.value.yield %12 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
        }
        hl.value.yield %11 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
      }
      %5 = hl.implicit_cast %4 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
      %6 = hl.member %5 at "next" : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
      %7 = hl.implicit_cast %6 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
      %8 = hl.assign %7 to %3 : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>, !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
      %9 = macroni.parameter "pos" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
        %11 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
        hl.value.yield %11 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
      }
      %10 = macroni.parameter "head" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
        %11 = hl.ref %0 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
        hl.value.yield %11 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
      }
      macroni.list_for_each() list_for_each(%9, %10) : (!hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>, !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>) -> () {
        hl.scope {
          %11 = hl.var "prev" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> = {
            %12 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
            %13 = hl.implicit_cast %12 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
            %14 = hl.member %13 at "prev" : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
            %15 = hl.implicit_cast %14 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
            hl.value.yield %15 : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
          }
        }
      }
    }
    %2 = hl.const #hl.integer<0> : !hl.int
    hl.return %2 : !hl.int
  }
}
