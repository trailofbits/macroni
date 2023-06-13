// Before
%0 = hl.var "x" : !hl.lvalue<!hl.int> = {
  %1 = hl.const #hl.integer<1> : !hl.int
  hl.value.yield %1 : !hl.int
}

// After
%0 = hl.var "x" : !hl.lvalue<!hl.int> = {
  %1 = hl.expr {MacroKind = "kExpansion", MacroName = "ONE", RawMacroID = 93848280998608 : ui64} : !hl.int {
    %2 = hl.const #hl.integer<1> : !hl.int
    hl.value.yield %2 : !hl.int
  }
  hl.value.yield %1 : !hl.int
}
