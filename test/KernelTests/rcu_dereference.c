// RUN: kernelize %s -- | FileCheck %s --match-full-lines

#include "rcu.h"
#include "test_structs.h"

int main(void) {
  int *x = 0;
  struct string *s = {0};
  rcu_dereference(x);
  rcu_dereference(s->begin);
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
// CHECK:         %0 = hl.var "x" : !hl.lvalue<!hl.ptr<!hl.int>> = {
// CHECK:           %9 = hl.const #core.integer<0> : !hl.int
// CHECK:           %10 = hl.implicit_cast %9 NullToPointer : !hl.int -> !hl.ptr<!hl.int>
// CHECK:           hl.value.yield %10 : !hl.ptr<!hl.int>
// CHECK:         }
// CHECK:         %1 = hl.var "s" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"string">>>> = {
// CHECK:           %9 = hl.const #core.integer<0> : !hl.int
// CHECK:           %10 = hl.implicit_cast %9 NullToPointer : !hl.int -> !hl.ptr<!hl.elaborated<!hl.record<"string">>>
// CHECK:           %11 = hl.initlist %10 : (!hl.ptr<!hl.elaborated<!hl.record<"string">>>) -> !hl.ptr<!hl.elaborated<!hl.record<"string">>>
// CHECK:           hl.value.yield %11 : !hl.ptr<!hl.elaborated<!hl.record<"string">>>
// CHECK:         }
// CHECK:         hl.typeof.expr "(*(x))" {
// CHECK:           %9 = hl.expr : !hl.lvalue<!hl.int> {
// CHECK:             %10 = hl.expr : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:               %13 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:               hl.value.yield %13 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:             }
// CHECK:             %11 = hl.implicit_cast %10 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
// CHECK:             %12 = hl.deref %11 : !hl.ptr<!hl.int> -> !hl.lvalue<!hl.int>
// CHECK:             hl.value.yield %12 : !hl.lvalue<!hl.int>
// CHECK:           }
// CHECK:           hl.type.yield %9 : !hl.lvalue<!hl.int>
// CHECK:         } : !hl.int
// CHECK:         %2 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:         %3 = kernel.rcu_dereference(%2) : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.ptr<!hl.typeof.expr<"(*(x))">>
// CHECK:         hl.typeof.expr "(*(s->begin))" {
// CHECK:           %9 = hl.expr : !hl.lvalue<!hl.char> {
// CHECK:             %10 = hl.expr : !hl.lvalue<!hl.ptr<!hl.char>> {
// CHECK:               %13 = hl.ref %1 : (!hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"string">>>>) -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"string">>>>
// CHECK:               %14 = hl.implicit_cast %13 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"string">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"string">>>
// CHECK:               %15 = hl.member %14 at "begin" : !hl.ptr<!hl.elaborated<!hl.record<"string">>> -> !hl.lvalue<!hl.ptr<!hl.char>>
// CHECK:               hl.value.yield %15 : !hl.lvalue<!hl.ptr<!hl.char>>
// CHECK:             }
// CHECK:             %11 = hl.implicit_cast %10 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.char>> -> !hl.ptr<!hl.char>
// CHECK:             %12 = hl.deref %11 : !hl.ptr<!hl.char> -> !hl.lvalue<!hl.char>
// CHECK:             hl.value.yield %12 : !hl.lvalue<!hl.char>
// CHECK:           }
// CHECK:           hl.type.yield %9 : !hl.lvalue<!hl.char>
// CHECK:         } : !hl.char
// CHECK:         %4 = hl.ref %1 : (!hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"string">>>>) -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"string">>>>
// CHECK:         %5 = hl.implicit_cast %4 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"string">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"string">>>
// CHECK:         %6 = hl.member %5 at "begin" : !hl.ptr<!hl.elaborated<!hl.record<"string">>> -> !hl.lvalue<!hl.ptr<!hl.char>>
// CHECK:         %7 = kernel.rcu_dereference(%6) : (!hl.lvalue<!hl.ptr<!hl.char>>) -> !hl.ptr<!hl.typeof.expr<"(*(s->begin))">>
// CHECK:         %8 = hl.const #core.integer<0> : !hl.int
// CHECK:         hl.return %8 : !hl.int
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }
