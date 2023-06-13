// Before
%0 = hl.var "x" : !hl.lvalue<!hl.int> = {
  %1 = hl.const #hl.integer<1> : !hl.int
  hl.value.yield %1 : !hl.int
}

// After
%0 = hl.var "x" : !hl.lvalue<!hl.int> = {
  %1 = hl.expr {MacroKind = "kExpansion", MacroName = "A", RawMacroID = 94047482262224 : ui64} : !hl.int {
    %2 = hl.expr {MacroKind = "kExpansion", MacroName = "B", RawMacroID = 94047482262472 : ui64} : !hl.int {
      %3 = hl.const #hl.integer<1> : !hl.int
        hl.value.yield %3 : !hl.int
      }
    hl.value.yield %2 : !hl.int
  }
  hl.value.yield %1 : !hl.int
}
