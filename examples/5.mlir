// Before
%0 = hl.var "x" : !hl.lvalue<!hl.int> = {
  %1 = hl.const #hl.integer<1> : !hl.int
  %2 = hl.const #hl.integer<2> : !hl.int
  %3 = hl.add %1, %2 : (!hl.int, !hl.int) -> !hl.int
  hl.value.yield %3 : !hl.int
}

// After
%0 = hl.var "x" : !hl.lvalue<!hl.int> = {
  %1 = hl.expr {MacroKind = "kExpansion", MacroName = "ADD", RawMacroID = 94217236767688 : ui64} : !hl.int {
    %2 = hl.expr {MacroKind = "kParameterSubstitution", ParameterIndex = 0 : ui64, ParameterName = "X", ParentRawMacroID = 94217236767688 : ui64, RawMacroID = 94217236582528 : ui64} : !hl.int {
      %5 = hl.const #hl.integer<1> : !hl.int
      hl.value.yield %5 : !hl.int
    }
    %3 = hl.expr {MacroKind = "kParameterSubstitution", ParameterIndex = 1 : ui64, ParameterName = "Y", ParentRawMacroID = 94217236767688 : ui64, RawMacroID = 94217236582704 : ui64} : !hl.int {
      %5 = hl.const #hl.integer<2> : !hl.int
      hl.value.yield %5 : !hl.int
    }
    %4 = hl.add %2, %3 : (!hl.int, !hl.int) -> !hl.int
    hl.value.yield %4 : !hl.int
  }
  hl.value.yield %1 : !hl.int
}
