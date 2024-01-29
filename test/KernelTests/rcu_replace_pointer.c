// RUN: kernelize -x c %s | FileCheck %s --match-full-lines

#define rcu_replace_pointer(rcu_ptr, ptr, c) (c && ((rcu_ptr) = (ptr)))

int main(void) {
  int *rcu_ptr, *ptr;
  rcu_replace_pointer(rcu_ptr, ptr, 1);
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
// CHECK:      %0 = hl.var "rcu_ptr" : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      %1 = hl.var "ptr" : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      %2 = macroni.parameter "rcu_ptr" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:        %7 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:        hl.value.yield %7 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      }
// CHECK:      %3 = macroni.parameter "ptr" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:        %7 = hl.ref %1 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:        hl.value.yield %7 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      }
// CHECK:      %4 = macroni.parameter "c" : !hl.int {
// CHECK:        %7 = hl.const #core.integer<1> : !hl.int
// CHECK:        hl.value.yield %7 : !hl.int
// CHECK:      }
// CHECK:      %5 = kernel.rcu_replace_pointer(%2, %3, %4) : (!hl.lvalue<!hl.ptr<!hl.int>>, !hl.lvalue<!hl.ptr<!hl.int>>, !hl.int) -> !hl.int
// CHECK:      %6 = hl.const #core.integer<0> : !hl.int
// CHECK:      hl.return %6 : !hl.int
// CHECK:    }
// CHECK:  }
// CHECK:}

