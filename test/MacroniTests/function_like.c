// RUN: macronify -xc %s | FileCheck %s --match-full-lines

//CHECK: hl.translation_unit {
//CHECK:   hl.typedef "__int128_t" : !hl.int128
//CHECK:   hl.typedef "__uint128_t" : !hl.int128< unsigned >
//CHECK:   hl.struct "__NSConstantString_tag" : {
//CHECK:     hl.field "isa" : !hl.ptr<!hl.int< const >>
//CHECK:     hl.field "flags" : !hl.int
//CHECK:     hl.field "str" : !hl.ptr<!hl.char< const >>
//CHECK:     hl.field "length" : !hl.long
//CHECK:   }
//CHECK:   hl.typedef "__NSConstantString" : !hl.record<"__NSConstantString_tag">
//CHECK:   hl.typedef "__builtin_ms_va_list" : !hl.ptr<!hl.char>
//CHECK:   hl.struct "__va_list_tag" : {
//CHECK:     hl.field "gp_offset" : !hl.int< unsigned >
//CHECK:     hl.field "fp_offset" : !hl.int< unsigned >
//CHECK:     hl.field "overflow_arg_area" : !hl.ptr<!hl.void>
//CHECK:     hl.field "reg_save_area" : !hl.ptr<!hl.void>
//CHECK:   }
//CHECK:   hl.typedef "__builtin_va_list" : !hl.array<1, !hl.record<"__va_list_tag">>
//CHECK:   %0 = hl.var "a" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ONE" : !hl.int {
//CHECK:       %15 = hl.const #hl.integer<1> : !hl.int
//CHECK:       hl.value.yield %15 : !hl.int
//CHECK:     }
//CHECK:     hl.value.yield %14 : !hl.int
//CHECK:   }
//CHECK:   %1 = hl.var "b" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ONE" : !hl.int {
//CHECK:       %17 = hl.const #hl.integer<1> : !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     %15 = macroni.expansion "ONE" : !hl.int {
//CHECK:       %17 = hl.const #hl.integer<1> : !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     %16 = hl.add %14, %15 : (!hl.int, !hl.int) -> !hl.int
//CHECK:     hl.value.yield %16 : !hl.int
//CHECK:   }
//CHECK:   %2 = hl.var "c" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ADD(X, Y)" : !hl.int {
//CHECK:       %15 = macroni.parameter "X" : !hl.int {
//CHECK:         %18 = hl.const #hl.integer<1> : !hl.int
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %16 = macroni.parameter "Y" : !hl.int {
//CHECK:         %18 = hl.const #hl.integer<1> : !hl.int
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %17 = hl.add %15, %16 : (!hl.int, !hl.int) -> !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     hl.value.yield %14 : !hl.int
//CHECK:   }
//CHECK:   %3 = hl.var "d" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ADD(X, Y)" : !hl.int {
//CHECK:       %15 = macroni.parameter "X" : !hl.int {
//CHECK:         %18 = hl.const #hl.integer<1> : !hl.int
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %16 = macroni.parameter "Y" : !hl.int {
//CHECK:         %18 = macroni.expansion "ONE" : !hl.int {
//CHECK:           %19 = hl.const #hl.integer<1> : !hl.int
//CHECK:           hl.value.yield %19 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %17 = hl.add %15, %16 : (!hl.int, !hl.int) -> !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     hl.value.yield %14 : !hl.int
//CHECK:   }
//CHECK:   %4 = hl.var "e" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ADD(X, Y)" : !hl.int {
//CHECK:       %15 = macroni.parameter "X" : !hl.int {
//CHECK:         %18 = macroni.expansion "ONE" : !hl.int {
//CHECK:           %19 = hl.const #hl.integer<1> : !hl.int
//CHECK:           hl.value.yield %19 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %16 = macroni.parameter "Y" : !hl.int {
//CHECK:         %18 = hl.const #hl.integer<1> : !hl.int
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %17 = hl.add %15, %16 : (!hl.int, !hl.int) -> !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     hl.value.yield %14 : !hl.int
//CHECK:   }
//CHECK:   %5 = hl.var "f" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ADD(X, Y)" : !hl.int {
//CHECK:       %15 = macroni.parameter "X" : !hl.int {
//CHECK:         %18 = macroni.expansion "ONE" : !hl.int {
//CHECK:           %19 = hl.const #hl.integer<1> : !hl.int
//CHECK:           hl.value.yield %19 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %16 = macroni.parameter "Y" : !hl.int {
//CHECK:         %18 = macroni.expansion "ONE" : !hl.int {
//CHECK:           %19 = hl.const #hl.integer<1> : !hl.int
//CHECK:           hl.value.yield %19 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %17 = hl.add %15, %16 : (!hl.int, !hl.int) -> !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     hl.value.yield %14 : !hl.int
//CHECK:   }
//CHECK:   %6 = hl.var "g" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ADD(X, Y)" : !hl.int {
//CHECK:       %15 = macroni.parameter "X" : !hl.int {
//CHECK:         %18 = macroni.expansion "MUL(A, B)" : !hl.int {
//CHECK:           %19 = macroni.parameter "A" : !hl.int {
//CHECK:             %22 = hl.const #hl.integer<1> : !hl.int
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %20 = macroni.parameter "B" : !hl.int {
//CHECK:             %22 = hl.const #hl.integer<2> : !hl.int
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %21 = hl.mul %19, %20 : (!hl.int, !hl.int) -> !hl.int
//CHECK:           hl.value.yield %21 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %16 = macroni.parameter "Y" : !hl.int {
//CHECK:         %18 = hl.const #hl.integer<3> : !hl.int
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %17 = hl.add %15, %16 : (!hl.int, !hl.int) -> !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     hl.value.yield %14 : !hl.int
//CHECK:   }
//CHECK:   %7 = hl.var "h" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ADD(X, Y)" : !hl.int {
//CHECK:       %15 = macroni.parameter "X" : !hl.int {
//CHECK:         %18 = hl.const #hl.integer<1> : !hl.int
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %16 = macroni.parameter "Y" : !hl.int {
//CHECK:         %18 = macroni.expansion "MUL(A, B)" : !hl.int {
//CHECK:           %19 = macroni.parameter "A" : !hl.int {
//CHECK:             %22 = hl.const #hl.integer<2> : !hl.int
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %20 = macroni.parameter "B" : !hl.int {
//CHECK:             %22 = hl.const #hl.integer<3> : !hl.int
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %21 = hl.mul %19, %20 : (!hl.int, !hl.int) -> !hl.int
//CHECK:           hl.value.yield %21 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %17 = hl.add %15, %16 : (!hl.int, !hl.int) -> !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     hl.value.yield %14 : !hl.int
//CHECK:   }
//CHECK:   %8 = hl.var "i" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ADD(X, Y)" : !hl.int {
//CHECK:       %15 = macroni.parameter "X" : !hl.int {
//CHECK:         %18 = macroni.expansion "ONE" : !hl.int {
//CHECK:           %19 = hl.const #hl.integer<1> : !hl.int
//CHECK:           hl.value.yield %19 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %16 = macroni.parameter "Y" : !hl.int {
//CHECK:         %18 = macroni.expansion "MUL(A, B)" : !hl.int {
//CHECK:           %19 = macroni.parameter "A" : !hl.int {
//CHECK:             %22 = hl.const #hl.integer<2> : !hl.int
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %20 = macroni.parameter "B" : !hl.int {
//CHECK:             %22 = hl.const #hl.integer<3> : !hl.int
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %21 = hl.mul %19, %20 : (!hl.int, !hl.int) -> !hl.int
//CHECK:           hl.value.yield %21 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %17 = hl.add %15, %16 : (!hl.int, !hl.int) -> !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     hl.value.yield %14 : !hl.int
//CHECK:   }
//CHECK:   %9 = hl.var "j" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ADD(X, Y)" : !hl.int {
//CHECK:       %15 = macroni.parameter "X" : !hl.int {
//CHECK:         %18 = macroni.expansion "ONE" : !hl.int {
//CHECK:           %19 = hl.const #hl.integer<1> : !hl.int
//CHECK:           hl.value.yield %19 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %16 = macroni.parameter "Y" : !hl.int {
//CHECK:         %18 = macroni.expansion "MUL(A, B)" : !hl.int {
//CHECK:           %19 = macroni.parameter "A" : !hl.int {
//CHECK:             %22 = macroni.expansion "ONE" : !hl.int {
//CHECK:               %23 = hl.const #hl.integer<1> : !hl.int
//CHECK:               hl.value.yield %23 : !hl.int
//CHECK:             }
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %20 = macroni.parameter "B" : !hl.int {
//CHECK:             %22 = macroni.expansion "ONE" : !hl.int {
//CHECK:               %23 = hl.const #hl.integer<1> : !hl.int
//CHECK:               hl.value.yield %23 : !hl.int
//CHECK:             }
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %21 = hl.mul %19, %20 : (!hl.int, !hl.int) -> !hl.int
//CHECK:           hl.value.yield %21 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %17 = hl.add %15, %16 : (!hl.int, !hl.int) -> !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     hl.value.yield %14 : !hl.int
//CHECK:   }
//CHECK:   %10 = hl.var "k" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ADD(X, Y)" : !hl.int {
//CHECK:       %15 = macroni.parameter "X" : !hl.int {
//CHECK:         %18 = macroni.expansion "MUL(A, B)" : !hl.int {
//CHECK:           %19 = macroni.parameter "A" : !hl.int {
//CHECK:             %22 = hl.const #hl.integer<1> : !hl.int
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %20 = macroni.parameter "B" : !hl.int {
//CHECK:             %22 = hl.const #hl.integer<2> : !hl.int
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %21 = hl.mul %19, %20 : (!hl.int, !hl.int) -> !hl.int
//CHECK:           hl.value.yield %21 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %16 = macroni.parameter "Y" : !hl.int {
//CHECK:         %18 = hl.const #hl.integer<3> : !hl.int
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %17 = hl.add %15, %16 : (!hl.int, !hl.int) -> !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     hl.value.yield %14 : !hl.int
//CHECK:   }
//CHECK:   %11 = hl.var "l" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ADD(X, Y)" : !hl.int {
//CHECK:       %15 = macroni.parameter "X" : !hl.int {
//CHECK:         %18 = macroni.expansion "MUL(A, B)" : !hl.int {
//CHECK:           %19 = macroni.parameter "A" : !hl.int {
//CHECK:             %22 = hl.const #hl.integer<1> : !hl.int
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %20 = macroni.parameter "B" : !hl.int {
//CHECK:             %22 = hl.const #hl.integer<2> : !hl.int
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %21 = hl.mul %19, %20 : (!hl.int, !hl.int) -> !hl.int
//CHECK:           hl.value.yield %21 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %16 = macroni.parameter "Y" : !hl.int {
//CHECK:         %18 = macroni.expansion "ONE" : !hl.int {
//CHECK:           %19 = hl.const #hl.integer<1> : !hl.int
//CHECK:           hl.value.yield %19 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %17 = hl.add %15, %16 : (!hl.int, !hl.int) -> !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     hl.value.yield %14 : !hl.int
//CHECK:   }
//CHECK:   %12 = hl.var "m" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ADD(X, Y)" : !hl.int {
//CHECK:       %15 = macroni.parameter "X" : !hl.int {
//CHECK:         %18 = macroni.expansion "MUL(A, B)" : !hl.int {
//CHECK:           %19 = macroni.parameter "A" : !hl.int {
//CHECK:             %22 = hl.const #hl.integer<1> : !hl.int
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %20 = macroni.parameter "B" : !hl.int {
//CHECK:             %22 = hl.const #hl.integer<2> : !hl.int
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %21 = hl.mul %19, %20 : (!hl.int, !hl.int) -> !hl.int
//CHECK:           hl.value.yield %21 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %16 = macroni.parameter "Y" : !hl.int {
//CHECK:         %18 = macroni.expansion "MUL(A, B)" : !hl.int {
//CHECK:           %19 = macroni.parameter "A" : !hl.int {
//CHECK:             %22 = macroni.expansion "ONE" : !hl.int {
//CHECK:               %23 = hl.const #hl.integer<1> : !hl.int
//CHECK:               hl.value.yield %23 : !hl.int
//CHECK:             }
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %20 = macroni.parameter "B" : !hl.int {
//CHECK:             %22 = macroni.expansion "ONE" : !hl.int {
//CHECK:               %23 = hl.const #hl.integer<1> : !hl.int
//CHECK:               hl.value.yield %23 : !hl.int
//CHECK:             }
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %21 = hl.mul %19, %20 : (!hl.int, !hl.int) -> !hl.int
//CHECK:           hl.value.yield %21 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %17 = hl.add %15, %16 : (!hl.int, !hl.int) -> !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     hl.value.yield %14 : !hl.int
//CHECK:   }
//CHECK:   %13 = hl.var "n" : !hl.lvalue<!hl.int> = {
//CHECK:     %14 = macroni.expansion "ADD(X, Y)" : !hl.int {
//CHECK:       %15 = macroni.parameter "X" : !hl.int {
//CHECK:         %18 = macroni.expansion "MUL(A, B)" : !hl.int {
//CHECK:           %19 = macroni.parameter "A" : !hl.int {
//CHECK:             %22 = macroni.expansion "ONE" : !hl.int {
//CHECK:               %23 = hl.const #hl.integer<1> : !hl.int
//CHECK:               hl.value.yield %23 : !hl.int
//CHECK:             }
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %20 = macroni.parameter "B" : !hl.int {
//CHECK:             %22 = macroni.expansion "ONE" : !hl.int {
//CHECK:               %23 = hl.const #hl.integer<1> : !hl.int
//CHECK:               hl.value.yield %23 : !hl.int
//CHECK:             }
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %21 = hl.mul %19, %20 : (!hl.int, !hl.int) -> !hl.int
//CHECK:           hl.value.yield %21 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %16 = macroni.parameter "Y" : !hl.int {
//CHECK:         %18 = macroni.expansion "MUL(A, B)" : !hl.int {
//CHECK:           %19 = macroni.parameter "A" : !hl.int {
//CHECK:             %22 = macroni.expansion "ONE" : !hl.int {
//CHECK:               %23 = hl.const #hl.integer<1> : !hl.int
//CHECK:               hl.value.yield %23 : !hl.int
//CHECK:             }
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %20 = macroni.parameter "B" : !hl.int {
//CHECK:             %22 = macroni.expansion "ONE" : !hl.int {
//CHECK:               %23 = hl.const #hl.integer<1> : !hl.int
//CHECK:               hl.value.yield %23 : !hl.int
//CHECK:             }
//CHECK:             hl.value.yield %22 : !hl.int
//CHECK:           }
//CHECK:           %21 = hl.mul %19, %20 : (!hl.int, !hl.int) -> !hl.int
//CHECK:           hl.value.yield %21 : !hl.int
//CHECK:         }
//CHECK:         hl.value.yield %18 : !hl.int
//CHECK:       }
//CHECK:       %17 = hl.add %15, %16 : (!hl.int, !hl.int) -> !hl.int
//CHECK:       hl.value.yield %17 : !hl.int
//CHECK:     }
//CHECK:     hl.value.yield %14 : !hl.int
//CHECK:   }
//CHECK:   hl.func external @main (%arg0: !hl.lvalue<!hl.int>, %arg1: !hl.lvalue<!hl.decayed<!hl.ptr<!hl.ptr<!hl.char< const >>>>>) -> !hl.int {
//CHECK:     %14 = hl.var "x" : !hl.lvalue<!hl.int>
//CHECK:     macroni.expansion "DO_NO_TRAILING_SEMI(STMT)" :  {
//CHECK:       hl.do {
//CHECK:         %16 = macroni.parameter "STMT" : !hl.int {
//CHECK:           %17 = hl.ref %14 : !hl.lvalue<!hl.int>
//CHECK:           %18 = hl.const #hl.integer<0> : !hl.int
//CHECK:           %19 = hl.assign %18 to %17 : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
//CHECK:           hl.value.yield %19 : !hl.int
//CHECK:         }
//CHECK:       } while {
//CHECK:         %16 = hl.const #hl.integer<0> : !hl.int
//CHECK:         %17 = hl.implicit_cast %16 IntegralToBoolean : !hl.int -> !hl.bool
//CHECK:         hl.cond.yield %17 : !hl.bool
//CHECK:       }
//CHECK:     }
//CHECK:     macroni.expansion "DO_NO_TRAILING_SEMI(STMT)" :  {
//CHECK:       hl.do {
//CHECK:         macroni.parameter "STMT" :  {
//CHECK:           macroni.expansion "DO_NO_TRAILING_SEMI(STMT)" :  {
//CHECK:             hl.do {
//CHECK:               %16 = macroni.parameter "STMT" : !hl.int {
//CHECK:                 %17 = hl.ref %14 : !hl.lvalue<!hl.int>
//CHECK:                 %18 = hl.const #hl.integer<0> : !hl.int
//CHECK:                 %19 = hl.assign %18 to %17 : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
//CHECK:                 hl.value.yield %19 : !hl.int
//CHECK:               }
//CHECK:             } while {
//CHECK:               %16 = hl.const #hl.integer<0> : !hl.int
//CHECK:               %17 = hl.implicit_cast %16 IntegralToBoolean : !hl.int -> !hl.bool
//CHECK:               hl.cond.yield %17 : !hl.bool
//CHECK:             }
//CHECK:           }
//CHECK:         }
//CHECK:       } while {
//CHECK:         %16 = hl.const #hl.integer<0> : !hl.int
//CHECK:         %17 = hl.implicit_cast %16 IntegralToBoolean : !hl.int -> !hl.bool
//CHECK:         hl.cond.yield %17 : !hl.bool
//CHECK:       }
//CHECK:     }
//CHECK:     macroni.expansion "DO_TRAILING_SEMI(STMT)" :  {
//CHECK:       hl.do {
//CHECK:         %16 = macroni.parameter "STMT" : !hl.int {
//CHECK:           %17 = hl.ref %14 : !hl.lvalue<!hl.int>
//CHECK:           %18 = hl.const #hl.integer<1> : !hl.int
//CHECK:           %19 = hl.assign %18 to %17 : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
//CHECK:           hl.value.yield %19 : !hl.int
//CHECK:         }
//CHECK:       } while {
//CHECK:         %16 = hl.const #hl.integer<0> : !hl.int
//CHECK:         %17 = hl.implicit_cast %16 IntegralToBoolean : !hl.int -> !hl.bool
//CHECK:         hl.cond.yield %17 : !hl.bool
//CHECK:       }
//CHECK:     }
//CHECK:     macroni.expansion "DO_TRAILING_SEMI(STMT)" :  {
//CHECK:       hl.do {
//CHECK:         macroni.parameter "STMT" :  {
//CHECK:           macroni.expansion "DO_TRAILING_SEMI(STMT)" :  {
//CHECK:             hl.do {
//CHECK:               %16 = macroni.parameter "STMT" : !hl.int {
//CHECK:                 %17 = hl.ref %14 : !hl.lvalue<!hl.int>
//CHECK:                 %18 = hl.const #hl.integer<1> : !hl.int
//CHECK:                 %19 = hl.assign %18 to %17 : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
//CHECK:                 hl.value.yield %19 : !hl.int
//CHECK:               }
//CHECK:             } while {
//CHECK:               %16 = hl.const #hl.integer<0> : !hl.int
//CHECK:               %17 = hl.implicit_cast %16 IntegralToBoolean : !hl.int -> !hl.bool
//CHECK:               hl.cond.yield %17 : !hl.bool
//CHECK:             }
//CHECK:           }
//CHECK:         }
//CHECK:       } while {
//CHECK:         %16 = hl.const #hl.integer<0> : !hl.int
//CHECK:         %17 = hl.implicit_cast %16 IntegralToBoolean : !hl.int -> !hl.bool
//CHECK:         hl.cond.yield %17 : !hl.bool
//CHECK:       }
//CHECK:     }
//CHECK:     %15 = hl.const #hl.integer<0> : !hl.int
//CHECK:     hl.return %15 : !hl.int
//CHECK:   }
//CHECK: }


#define ONE 1
#define ADD(X, Y) X + Y
#define MUL(A, B) A * B

int a = ONE;
int b = ONE + ONE;
int c = ADD(1, 1);
int d = ADD(1, ONE);
int e = ADD(ONE, 1);
int f = ADD(ONE, ONE);
int g = ADD(MUL(1, 2), 3);
int h = ADD(1, MUL(2, 3));
int i = ADD(ONE, MUL(2, 3));
int j = ADD(ONE, MUL(ONE, ONE));
int k = ADD(MUL(1, 2), 3);
int l = ADD(MUL(1, 2), ONE);
int m = ADD(MUL(1, 2), MUL(ONE, ONE));
int n = ADD(MUL(ONE, ONE), MUL(ONE, ONE));

#define DO_NO_TRAILING_SEMI(STMT) do { STMT; } while (0)
#define DO_TRAILING_SEMI(STMT) do { STMT; } while (0)

int main(int argc, char const *argv[]) {
        int x;
        
        DO_NO_TRAILING_SEMI(x = 0);
        DO_NO_TRAILING_SEMI(DO_NO_TRAILING_SEMI(x = 0));
        
        DO_TRAILING_SEMI(x = 1);
        DO_TRAILING_SEMI(DO_TRAILING_SEMI(x = 1));

        return 0;
}
