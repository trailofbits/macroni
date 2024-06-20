// RUN: kernelize %s -- | FileCheck %s --match-full-lines

#include "rcu.h"
#include "test_structs.h"

int main(void) {
  int *p = 0, *v = 0;
  char *new_begin = 0;
  struct string *s = {0};

  rcu_assign_pointer(p, v);
  rcu_assign_pointer(s->begin, new_begin);
  return 0;
}

// CHECK: #void_value = #core.void : !hl.void
// CHECK:   hl.translation_unit {
// CHECK:     hl.typedef "__int128_t" : !hl.int128
// CHECK:     hl.typedef "__uint128_t" : !hl.int128< unsigned >
// CHECK:     hl.typedef "__NSConstantString" : !hl.record<"__NSConstantString_tag">
// CHECK:     hl.typedef "__builtin_ms_va_list" : !hl.ptr<!hl.char>
// CHECK:     hl.typedef "__builtin_va_list" : !hl.array<1, !hl.record<"__va_list_tag">>
// CHECK:     hl.func @main external () -> !hl.int {
// CHECK:       core.scope {
// CHECK:         %0 = hl.var "p" : !hl.lvalue<!hl.ptr<!hl.int>> = {
// CHECK:           %13 = hl.const #core.integer<0> : !hl.int
// CHECK:           %14 = hl.implicit_cast %13 NullToPointer : !hl.int -> !hl.ptr<!hl.int>
// CHECK:           hl.value.yield %14 : !hl.ptr<!hl.int>
// CHECK:         }
// CHECK:         %1 = hl.var "v" : !hl.lvalue<!hl.ptr<!hl.int>> = {
// CHECK:           %13 = hl.const #core.integer<0> : !hl.int
// CHECK:           %14 = hl.implicit_cast %13 NullToPointer : !hl.int -> !hl.ptr<!hl.int>
// CHECK:           hl.value.yield %14 : !hl.ptr<!hl.int>
// CHECK:         }
// CHECK:         %2 = hl.var "new_begin" : !hl.lvalue<!hl.ptr<!hl.char>> = {
// CHECK:           %13 = hl.const #core.integer<0> : !hl.int
// CHECK:           %14 = hl.implicit_cast %13 NullToPointer : !hl.int -> !hl.ptr<!hl.char>
// CHECK:           hl.value.yield %14 : !hl.ptr<!hl.char>
// CHECK:         }
// CHECK:         %3 = hl.var "s" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"string">>>> = {
// CHECK:           %13 = hl.const #core.integer<0> : !hl.int
// CHECK:           %14 = hl.implicit_cast %13 NullToPointer : !hl.int -> !hl.ptr<!hl.elaborated<!hl.record<"string">>>
// CHECK:           %15 = hl.initlist %14 : (!hl.ptr<!hl.elaborated<!hl.record<"string">>>) -> !hl.ptr<!hl.elaborated<!hl.record<"string">>>
// CHECK:           hl.value.yield %15 : !hl.ptr<!hl.elaborated<!hl.record<"string">>>
// CHECK:         }
// CHECK:         %4 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:         %5 = hl.ref %1 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:         %6 = kernel.rcu_assign_pointer(%4, %5) : (!hl.lvalue<!hl.ptr<!hl.int>>, !hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.void
// CHECK:         %7 = hl.ref %3 : (!hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"string">>>>) -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"string">>>>
// CHECK:         %8 = hl.implicit_cast %7 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"string">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"string">>>
// CHECK:         %9 = hl.member %8 at "begin" : !hl.ptr<!hl.elaborated<!hl.record<"string">>> -> !hl.lvalue<!hl.ptr<!hl.char>>
// CHECK:         %10 = hl.ref %2 : (!hl.lvalue<!hl.ptr<!hl.char>>) -> !hl.lvalue<!hl.ptr<!hl.char>>
// CHECK:         %11 = kernel.rcu_assign_pointer(%9, %10) : (!hl.lvalue<!hl.ptr<!hl.char>>, !hl.lvalue<!hl.ptr<!hl.char>>) -> !hl.void
// CHECK:         %12 = hl.const #core.integer<0> : !hl.int
// CHECK:         hl.return %12 : !hl.int
// CHECK:       }
// CHECK:     }
// CHECK:     hl.func @__builtin_constant_p external () -> !hl.int attributes {hl.builtin = #hl.builtin<411>, hl.const = #hl.const, hl.nothrow = #hl.nothrow, sym_visibility = "private"}
// CHECK:   }
// CHECK: }
