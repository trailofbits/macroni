// RUN: kernelize -x c %s | FileCheck %s --match-full-lines

#define rcu_assign_pointer(p, v) ((p) = (v))

int main(void) {
  int *p, *v;
  rcu_assign_pointer(p, v);
  return 0;
}

// CHECK:hl.translation_unit {
// CHECK:  hl.typedef "__int128_t" : !hl.int128
// CHECK:  hl.typedef "__uint128_t" : !hl.int128< unsigned >
// CHECK:  hl.struct "__NSConstantString_tag" {type_visibility = #unsup<attr "type_visibility">} : {
// CHECK:    hl.field "isa" : !hl.ptr<!hl.int< const >>
// CHECK:    hl.field "flags" : !hl.int
// CHECK:    hl.field "str" : !hl.ptr<!hl.char< const >>
// CHECK:    hl.field "length" : !hl.long
// CHECK:  }
// CHECK:  hl.typedef "__NSConstantString" : !hl.record<"__NSConstantString_tag">
// CHECK:  hl.typedef "__builtin_ms_va_list" : !hl.ptr<!hl.char>
// CHECK:  hl.struct "__va_list_tag" {type_visibility = #unsup<attr "type_visibility">} : {
// CHECK:    hl.field "gp_offset" : !hl.int< unsigned >
// CHECK:    hl.field "fp_offset" : !hl.int< unsigned >
// CHECK:    hl.field "overflow_arg_area" : !hl.ptr<!hl.void>
// CHECK:    hl.field "reg_save_area" : !hl.ptr<!hl.void>
// CHECK:  }
// CHECK:  hl.typedef "__builtin_va_list" : !hl.array<1, !hl.record<"__va_list_tag">>
// CHECK:  hl.func @main () -> !hl.int {
// CHECK:    core.scope {
// CHECK:      %0 = hl.var "p" : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      %1 = hl.var "v" : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      %2 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:        %6 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:        hl.value.yield %6 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      }
// CHECK:      %3 = macroni.parameter "v" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:        %6 = hl.ref %1 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:        hl.value.yield %6 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      }
// CHECK:      %4 = kernel.rcu_assign_pointer(%2, %3) : (!hl.lvalue<!hl.ptr<!hl.int>>, !hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.ptr<!hl.int>
// CHECK:      %5 = hl.const #core.integer<0> : !hl.int
// CHECK:      hl.return %5 : !hl.int
// CHECK:    }
// CHECK:  }
// CHECK:}

