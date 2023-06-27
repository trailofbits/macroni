// Before
%0 = hl.var "x" : !hl.lvalue<!hl.int> = {
  %1 = hl.const #hl.integer<1> : !hl.int
  %2 = hl.const #hl.integer<2> : !hl.int
  %3 = hl.mul %1, %2 : (!hl.int, !hl.int) -> !hl.int
  %4 = hl.const #hl.integer<3> : !hl.int
  %5 = hl.const #hl.integer<4> : !hl.int
  %6 = hl.mul %4, %5 : (!hl.int, !hl.int) -> !hl.int
  %7 = hl.add %3, %6 : (!hl.int, !hl.int) -> !hl.int
  hl.value.yield %7 : !hl.int
}

// After
%0 = hl.var "x" : !hl.lvalue<!hl.int> = {
  %1 = macroni.expansion "ADD(X, Y)" : !hl.int {
    %2 = macroni.parameter "X" : !hl.int {
      %5 = macroni.expansion "MUL(A, B)" : !hl.int {
        %6 = macroni.parameter "A" : !hl.int {
          %9 = hl.const #hl.integer<1> : !hl.int
          hl.value.yield %9 : !hl.int
        }
        %7 = macroni.parameter "B" : !hl.int {
          %9 = hl.const #hl.integer<2> : !hl.int
          hl.value.yield %9 : !hl.int
        }
        %8 = hl.mul %6, %7 : (!hl.int, !hl.int) -> !hl.int
        hl.value.yield %8 : !hl.int
      }
      hl.value.yield %5 : !hl.int
    }
    %3 = macroni.parameter "Y" : !hl.int {
      %5 = macroni.expansion "MUL(A, B)" : !hl.int {
        %6 = macroni.parameter "A" : !hl.int {
          %9 = hl.const #hl.integer<3> : !hl.int
          hl.value.yield %9 : !hl.int
        }
        %7 = macroni.parameter "B" : !hl.int {
          %9 = hl.const #hl.integer<4> : !hl.int
          hl.value.yield %9 : !hl.int
        }
        %8 = hl.mul %6, %7 : (!hl.int, !hl.int) -> !hl.int
        hl.value.yield %8 : !hl.int
      }
      hl.value.yield %5 : !hl.int
    }
    %4 = hl.add %2, %3 : (!hl.int, !hl.int) -> !hl.int
    hl.value.yield %4 : !hl.int
  }
  hl.value.yield %1 : !hl.int
}
