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
// CHECK:   hl.func @main () -> !hl.int {
// CHECK:     core.scope {
// CHECK:       %0 = hl.var "p" : !hl.lvalue<!hl.ptr<!hl.int>> = {
// CHECK:         %5 = hl.const #core.integer<12648430> : !hl.int
// CHECK:         %6 = hl.cstyle_cast %5 IntegralToPointer : !hl.int -> !hl.ptr<!hl.void>
// CHECK:         %7 = hl.implicit_cast %6 BitCast : !hl.ptr<!hl.void> -> !hl.ptr<!hl.int>
// CHECK:         hl.value.yield %7 : !hl.ptr<!hl.int>
// CHECK:       }
// CHECK:       %1 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:         %5 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:         hl.value.yield %5 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:       }
// CHECK:       %2 = kernel.rcu_access_pointer rcu_access_pointer(%1) : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.int>
// CHECK:       %3 = hl.implicit_cast %2 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
// CHECK:       %4 = hl.const #core.integer<0> : !hl.int
// CHECK:       hl.return %4 : !hl.int
// CHECK:     }
// CHECK:   }
// CHECK: }

#define rcu_access_pointer(p) (*p)

int main(void) {
  int *p = (void *)0xc0ffee;
  rcu_access_pointer(p);
  return 0;
}
