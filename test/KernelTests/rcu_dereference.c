// RUN: kernelize -x c %s | FileCheck %s --match-full-lines

// CHECK: hl.translation_unit {
// CHECK:   hl.typedef "__int128_t" : !hl.int128
// CHECK:   hl.typedef "__uint128_t" : !hl.int128< unsigned >
// CHECK:   hl.struct "__NSConstantString_tag" : {
// CHECK:     hl.field "isa" : !hl.ptr<!hl.int< const >>
// CHECK:     hl.field "flags" : !hl.int
// CHECK:     hl.field "str" : !hl.ptr<!hl.char< const >>
// CHECK:     hl.field "length" : !hl.long
// CHECK:   }
// CHECK:   hl.typedef "__NSConstantString" : !hl.record<"__NSConstantString_tag">
// CHECK:   hl.typedef "__builtin_ms_va_list" : !hl.ptr<!hl.char>
// CHECK:   hl.struct "__va_list_tag" : {
// CHECK:     hl.field "gp_offset" : !hl.int< unsigned >
// CHECK:     hl.field "fp_offset" : !hl.int< unsigned >
// CHECK:     hl.field "overflow_arg_area" : !hl.ptr<!hl.void>
// CHECK:     hl.field "reg_save_area" : !hl.ptr<!hl.void>
// CHECK:   }
// CHECK:   hl.typedef "__builtin_va_list" : !hl.array<1, !hl.record<"__va_list_tag">>
// CHECK:   hl.func external @main () -> !hl.int {
// CHECK:     hl.scope {
// CHECK:       %0 = hl.var "p" : !hl.lvalue<!hl.ptr<!hl.int>> = {
// CHECK:         %4 = hl.const #hl.integer<12648430> : !hl.int
// CHECK:         %5 = hl.cstyle_cast %4 IntegralToPointer : !hl.int -> !hl.ptr<!hl.int>
// CHECK:         hl.value.yield %5 : !hl.ptr<!hl.int>
// CHECK:       }
// CHECK:       %1 = hl.var "x" : !hl.lvalue<!hl.int> = {
// CHECK:         %4 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:           %7 = hl.ref %0 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:           hl.value.yield %7 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:         }
// CHECK:         %5 = kernel.rcu_dereference rcu_dereference(%4) : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.int>
// CHECK:         %6 = hl.implicit_cast %5 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
// CHECK:         hl.value.yield %6 : !hl.int
// CHECK:       }
// CHECK:       %2 = hl.var "y" : !hl.lvalue<!hl.int> = {
// CHECK:         %4 = macroni.expansion "deref(p)" : !hl.lvalue<!hl.int> {
// CHECK:           %6 = hl.expr : !hl.lvalue<!hl.int> {
// CHECK:             %7 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:               %9 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:                 %10 = hl.ref %0 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:                 hl.value.yield %10 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:               }
// CHECK:               hl.value.yield %9 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:             }
// CHECK:             %8 = kernel.rcu_dereference rcu_dereference(%7) : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.int>
// CHECK:             hl.value.yield %8 : !hl.lvalue<!hl.int>
// CHECK:           }
// CHECK:           hl.value.yield %6 : !hl.lvalue<!hl.int>
// CHECK:         }
// CHECK:         %5 = hl.implicit_cast %4 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
// CHECK:         hl.value.yield %5 : !hl.int
// CHECK:       }
// CHECK:       %3 = hl.const #hl.integer<0> : !hl.int
// CHECK:       hl.return %3 : !hl.int
// CHECK:     }
// CHECK:   }
// CHECK: }

#define rcu_dereference(p) (*p)

#define deref(p) (rcu_dereference(p))

int main(void) {
        int *p = (int *) 0xC0FFEE,
                x = rcu_dereference(p),
                y = deref(p);
        return 0;
}
