// Before
hl.func external @main () -> !hl.int {
  %0 = hl.var "x" : !hl.lvalue<!hl.int>
  %1 = hl.var "ptr" : !hl.lvalue<!hl.ptr<!hl.int>>
  %2 = hl.stmt.expr : !hl.int {
    %4 = hl.var "res" : !hl.lvalue<!hl.int> = {
      %7 = hl.const #hl.integer<0> : !hl.int
      hl.value.yield %7 : !hl.int
    }
    hl.if {
      %7 = hl.const #hl.integer<1> : !hl.int
      %8 = hl.implicit_cast %7 IntegralToBoolean : !hl.int -> !hl.bool
      hl.cond.yield %8 : !hl.bool
    } then {
      %7 = hl.ref %0 : !hl.lvalue<!hl.int>
      %8 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.int>>
      %9 = hl.implicit_cast %8 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
      %10 = hl.deref %9 : !hl.ptr<!hl.int> -> !hl.lvalue<!hl.int>
      %11 = hl.implicit_cast %10 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
      %12 = hl.assign %11 to %7 : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
      %13 = hl.ref %4 : !hl.lvalue<!hl.int>
      %14 = hl.const #hl.integer<1> : !hl.int
      %15 = hl.assign %14 to %13 : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
    }
    %5 = hl.ref %4 : !hl.lvalue<!hl.int>
    %6 = hl.implicit_cast %5 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    hl.value.yield %6 : !hl.int
  }
  %3 = hl.const #hl.integer<0> : !hl.int
  hl.return %3 : !hl.int
}

// After
hl.func external @main () -> !hl.int {
  %0 = hl.var "x" : !hl.lvalue<!hl.int>
  %1 = hl.var "ptr" : !hl.lvalue<!hl.ptr<!hl.int>>
  %2 = macroni.parameter "x" : !hl.lvalue<!hl.int> {
    %6 = hl.ref %0 : !hl.lvalue<!hl.int>
    hl.value.yield %6 : !hl.lvalue<!hl.int>
  }
  %3 = macroni.parameter "ptr" : !hl.lvalue<!hl.ptr<!hl.int>> {
    %6 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.int>>
    hl.value.yield %6 : !hl.lvalue<!hl.ptr<!hl.int>>
  }
  %4 = macroni.get_user get_user(%2, %3) : (!hl.lvalue<!hl.int>, !hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.int
  %5 = hl.const #hl.integer<0> : !hl.int
  hl.return %5 : !hl.int
}
