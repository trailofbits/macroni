// Before
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!hl.int, 32 : i32>>} {
  %0 = hl.var "x" : !hl.lvalue<!hl.int> = {
    %1 = hl.const #hl.integer<1> : !hl.int
    %2 = hl.const #hl.integer<2> : !hl.int
    %3 = hl.add %1, %2 : (!hl.int, !hl.int) -> !hl.int
    hl.value.yield %3 : !hl.int
  }
}

// After
%0 = hl.var "x" : !hl.lvalue<!hl.int> = {
  %1 = hl.expr {MacroKind = "kExpansion", MacroName = "A", RawMacroID = 94788574859984 : ui64} : !hl.int {
    %2 = hl.const #hl.integer<1> : !hl.int
    %3 = hl.const #hl.integer<2> : !hl.int
    %4 = hl.add %2, %3 : (!hl.int, !hl.int) -> !hl.int
    hl.value.yield %4 : !hl.int
    }
  hl.value.yield %1 : !hl.int
}
