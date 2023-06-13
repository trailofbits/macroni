// Before
%0 = hl.var "x" : !hl.lvalue<!hl.int> = {
  %1 = hl.const #hl.integer<1> : !hl.int
  hl.value.yield %1 : !hl.int
}

// After
%0 = hl.var "x" : !hl.lvalue<!hl.int> = {
  %1 = hl.expr {MacroKind = "kExpansion", MacroName = "ID", RawMacroID = 94882571310024 : ui64} : !hl.int {
    %2 = hl.expr {MacroKind = "kParameterSubstitution", ParameterIndex = 0 : ui64, ParameterName = "X", ParentRawMacroID = 94882571310024 : ui64, RawMacroID = 94882571124864 : ui64} : !hl.int {
      %3 = hl.const #hl.integer<1> : !hl.int
      hl.value.yield %3 : !hl.int
    }
    hl.value.yield %2 : !hl.int
  }
  hl.value.yield %1 : !hl.int
}