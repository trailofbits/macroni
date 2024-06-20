// RUN: kernelize %s -- | FileCheck %s --match-full-lines

#include "rcu.h"
#include "test_structs.h"

int main(void) {
  int *p = 0;
  struct string s = {0};

  rcu_access_pointer(p);
  rcu_access_pointer(s.begin);

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
// CHECK:           %8 = hl.const #core.integer<0> : !hl.int
// CHECK:           %9 = hl.implicit_cast %8 NullToPointer : !hl.int -> !hl.ptr<!hl.int>
// CHECK:           hl.value.yield %9 : !hl.ptr<!hl.int>
// CHECK:         }
// CHECK:         %1 = hl.var "s" : !hl.lvalue<!hl.elaborated<!hl.record<"string">>> = {
// CHECK:           %8 = hl.const #core.integer<0> : !hl.int
// CHECK:           %9 = hl.implicit_cast %8 NullToPointer : !hl.int -> !hl.ptr<!hl.char>
// CHECK:           %10 = hl.initlist  : () -> !hl.long< unsigned >
// CHECK:           %11 = hl.initlist %9, %10 : (!hl.ptr<!hl.char>, !hl.long< unsigned >) -> !hl.elaborated<!hl.record<"string">>
// CHECK:           hl.value.yield %11 : !hl.elaborated<!hl.record<"string">>
// CHECK:         }
// CHECK:         hl.typeof.expr "(*(p))" {
// CHECK:           %8 = hl.expr : !hl.lvalue<!hl.int> {
// CHECK:             %9 = hl.expr : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:               %12 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:               hl.value.yield %12 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:             }
// CHECK:             %10 = hl.implicit_cast %9 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
// CHECK:             %11 = hl.deref %10 : !hl.ptr<!hl.int> -> !hl.lvalue<!hl.int>
// CHECK:             hl.value.yield %11 : !hl.lvalue<!hl.int>
// CHECK:           }
// CHECK:           hl.type.yield %8 : !hl.lvalue<!hl.int>
// CHECK:         } : !hl.int
// CHECK:         %2 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:         %3 = kernel.rcu_access_pointer(%2) : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.ptr<!hl.typeof.expr<"(*(p))">>
// CHECK:         hl.typeof.expr "(*(s.begin))" {
// CHECK:           %8 = hl.expr : !hl.lvalue<!hl.char> {
// CHECK:             %9 = hl.expr : !hl.lvalue<!hl.ptr<!hl.char>> {
// CHECK:               %12 = hl.ref %1 : (!hl.lvalue<!hl.elaborated<!hl.record<"string">>>) -> !hl.lvalue<!hl.elaborated<!hl.record<"string">>>
// CHECK:               %13 = hl.member %12 at "begin" : !hl.lvalue<!hl.elaborated<!hl.record<"string">>> -> !hl.lvalue<!hl.ptr<!hl.char>>
// CHECK:               hl.value.yield %13 : !hl.lvalue<!hl.ptr<!hl.char>>
// CHECK:             }
// CHECK:             %10 = hl.implicit_cast %9 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.char>> -> !hl.ptr<!hl.char>
// CHECK:             %11 = hl.deref %10 : !hl.ptr<!hl.char> -> !hl.lvalue<!hl.char>
// CHECK:             hl.value.yield %11 : !hl.lvalue<!hl.char>
// CHECK:           }
// CHECK:           hl.type.yield %8 : !hl.lvalue<!hl.char>
// CHECK:         } : !hl.char
// CHECK:         %4 = hl.ref %1 : (!hl.lvalue<!hl.elaborated<!hl.record<"string">>>) -> !hl.lvalue<!hl.elaborated<!hl.record<"string">>>
// CHECK:         %5 = hl.member %4 at "begin" : !hl.lvalue<!hl.elaborated<!hl.record<"string">>> -> !hl.lvalue<!hl.ptr<!hl.char>>
// CHECK:         %6 = kernel.rcu_access_pointer(%5) : (!hl.lvalue<!hl.ptr<!hl.char>>) -> !hl.ptr<!hl.typeof.expr<"(*(s.begin))">>
// CHECK:         %7 = hl.const #core.integer<0> : !hl.int
// CHECK:         hl.return %7 : !hl.int
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }
