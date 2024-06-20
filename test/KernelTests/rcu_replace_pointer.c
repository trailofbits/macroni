// RUN: kernelize %s -- | FileCheck %s --match-full-lines

#include "rcu.h"
#include "test_structs.h"

int main(void) {
  int *rcu_ptr = 0, *ptr = 0;
  int c = 1;
  char *new_begin = 0;
  struct string s;
  rcu_replace_pointer(rcu_ptr, ptr, c);
  rcu_replace_pointer(s.begin, new_begin, 0);
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
// CHECK:         %0 = hl.var "rcu_ptr" : !hl.lvalue<!hl.ptr<!hl.int>> = {
// CHECK:           %15 = hl.const #core.integer<0> : !hl.int
// CHECK:           %16 = hl.implicit_cast %15 NullToPointer : !hl.int -> !hl.ptr<!hl.int>
// CHECK:           hl.value.yield %16 : !hl.ptr<!hl.int>
// CHECK:         }
// CHECK:         %1 = hl.var "ptr" : !hl.lvalue<!hl.ptr<!hl.int>> = {
// CHECK:           %15 = hl.const #core.integer<0> : !hl.int
// CHECK:           %16 = hl.implicit_cast %15 NullToPointer : !hl.int -> !hl.ptr<!hl.int>
// CHECK:           hl.value.yield %16 : !hl.ptr<!hl.int>
// CHECK:         }
// CHECK:         %2 = hl.var "c" : !hl.lvalue<!hl.int> = {
// CHECK:           %15 = hl.const #core.integer<1> : !hl.int
// CHECK:           hl.value.yield %15 : !hl.int
// CHECK:         }
// CHECK:         %3 = hl.var "new_begin" : !hl.lvalue<!hl.ptr<!hl.char>> = {
// CHECK:           %15 = hl.const #core.integer<0> : !hl.int
// CHECK:           %16 = hl.implicit_cast %15 NullToPointer : !hl.int -> !hl.ptr<!hl.char>
// CHECK:           hl.value.yield %16 : !hl.ptr<!hl.char>
// CHECK:         }
// CHECK:         %4 = hl.var "s" : !hl.lvalue<!hl.elaborated<!hl.record<"string">>>
// CHECK:         hl.typeof.expr "(ptr)" {
// CHECK:           %15 = hl.expr : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:             %16 = hl.ref %1 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:             hl.value.yield %16 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:           }
// CHECK:           hl.type.yield %15 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:         } : !hl.ptr<!hl.int>
// CHECK:         %5 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:         %6 = hl.ref %1 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:         %7 = hl.ref %2 : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
// CHECK:         %8 = kernel.rcu_replace_pointer(%5, %6, %7) : (!hl.lvalue<!hl.ptr<!hl.int>>, !hl.lvalue<!hl.ptr<!hl.int>>, !hl.lvalue<!hl.int>) -> !hl.typeof.expr<"(ptr)">
// CHECK:         hl.typeof.expr "(new_begin)" {
// CHECK:           %15 = hl.expr : !hl.lvalue<!hl.ptr<!hl.char>> {
// CHECK:             %16 = hl.ref %3 : (!hl.lvalue<!hl.ptr<!hl.char>>) -> !hl.lvalue<!hl.ptr<!hl.char>>
// CHECK:             hl.value.yield %16 : !hl.lvalue<!hl.ptr<!hl.char>>
// CHECK:           }
// CHECK:           hl.type.yield %15 : !hl.lvalue<!hl.ptr<!hl.char>>
// CHECK:         } : !hl.ptr<!hl.char>
// CHECK:         %9 = hl.ref %4 : (!hl.lvalue<!hl.elaborated<!hl.record<"string">>>) -> !hl.lvalue<!hl.elaborated<!hl.record<"string">>>
// CHECK:         %10 = hl.member %9 at "begin" : !hl.lvalue<!hl.elaborated<!hl.record<"string">>> -> !hl.lvalue<!hl.ptr<!hl.char>>
// CHECK:         %11 = hl.ref %3 : (!hl.lvalue<!hl.ptr<!hl.char>>) -> !hl.lvalue<!hl.ptr<!hl.char>>
// CHECK:         %12 = hl.const #core.integer<0> : !hl.int
// CHECK:         %13 = kernel.rcu_replace_pointer(%10, %11, %12) : (!hl.lvalue<!hl.ptr<!hl.char>>, !hl.lvalue<!hl.ptr<!hl.char>>, !hl.int) -> !hl.typeof.expr<"(new_begin)">
// CHECK:         %14 = hl.const #core.integer<0> : !hl.int
// CHECK:         hl.return %14 : !hl.int
// CHECK:       }
// CHECK:     }
// CHECK:     hl.func @__builtin_constant_p external () -> !hl.int attributes {hl.builtin = #hl.builtin<411>, hl.const = #hl.const, hl.nothrow = #hl.nothrow, sym_visibility = "private"}
// CHECK:   }
// CHECK: }
