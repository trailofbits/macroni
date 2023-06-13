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
  %1 = hl.expr {MacroKind = "kExpansion", MacroName = "ADD", RawMacroID = 94145659118536 : ui64} : !hl.int {
    %2 = hl.expr {MacroKind = "kParameterSubstitution", ParameterIndex = 0 : ui64, ParameterName = "X", ParentRawMacroID = 94145659118536 : ui64, RawMacroID = 94145659957648 : ui64} : !hl.int {
      %5 = hl.expr {MacroKind = "kExpansion", MacroName = "MUL", RawMacroID = 94145659951816 : ui64} : !hl.int {
        %6 = hl.expr {MacroKind = "kParameterSubstitution", ParameterIndex = 0 : ui64, ParameterName = "A", ParentRawMacroID = 94145659951816 : ui64, RawMacroID = 94145658933376 : ui64} : !hl.int {
          %9 = hl.const #hl.integer<1> : !hl.int
          hl.value.yield %9 : !hl.int
        }
        %7 = hl.expr {MacroKind = "kParameterSubstitution", ParameterIndex = 1 : ui64, ParameterName = "B", ParentRawMacroID = 94145659951816 : ui64, RawMacroID = 94145658933552 : ui64} : !hl.int {
          %9 = hl.const #hl.integer<2> : !hl.int
          hl.value.yield %9 : !hl.int
        }
        %8 = hl.mul %6, %7 : (!hl.int, !hl.int) -> !hl.int
        hl.value.yield %8 : !hl.int
      }
      hl.value.yield %5 : !hl.int
    }
    %3 = hl.expr {MacroKind = "kParameterSubstitution", ParameterIndex = 1 : ui64, ParameterName = "Y", ParentRawMacroID = 94145659118536 : ui64, RawMacroID = 94145659957824 : ui64} : !hl.int {
      %5 = hl.expr {MacroKind = "kExpansion", MacroName = "MUL", RawMacroID = 94145659953432 : ui64} : !hl.int {
        %6 = hl.expr {MacroKind = "kParameterSubstitution", ParameterIndex = 0 : ui64, ParameterName = "A", ParentRawMacroID = 94145659953432 : ui64, RawMacroID = 94145659113152 : ui64} : !hl.int {
          %9 = hl.const #hl.integer<3> : !hl.int
          hl.value.yield %9 : !hl.int
        }
        %7 = hl.expr {MacroKind = "kParameterSubstitution", ParameterIndex = 1 : ui64, ParameterName = "B", ParentRawMacroID = 94145659953432 : ui64, RawMacroID = 94145659113328 : ui64} : !hl.int {
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
