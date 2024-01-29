// RUN: kernelize -x c %s | FileCheck %s --match-full-lines

#define __rcu_dereference_check(p, local, c, space)                            \
  ({                                                                           \
    typeof(*p) *local = (typeof(*p) *)(p);                                     \
    do {                                                                       \
    } while (0 && (c));                                                        \
    ((typeof(*p) *)(local));                                                   \
  })
#define rcu_dereference_check(p, c)                                            \
  __rcu_dereference_check((p), unique_id, (c), __rcu)
#define rcu_dereference(p) rcu_dereference_check(p, 0)

int main(void) {
  int *x;
  x = rcu_dereference(x);
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
// CHECK:      %0 = hl.var "x" : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      %1 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      %2 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:        %6 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:        hl.value.yield %6 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      }
// CHECK:      %3 = kernel.rcu_dereference(%2) : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.ptr<!hl.typeof.expr<"(*(x))">>
// CHECK:      %4 = hl.assign %3 to %1 : !hl.ptr<!hl.typeof.expr<"(*(x))">>, !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
// CHECK:      %5 = hl.const #core.integer<0> : !hl.int
// CHECK:      hl.return %5 : !hl.int
// CHECK:    }
// CHECK:  }
// CHECK:}

