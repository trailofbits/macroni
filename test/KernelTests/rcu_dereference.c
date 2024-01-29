// RUN: kernelize -x c %s | FileCheck %s --match-full-lines

#define rcu_dereference(p) (*p)
#define rcu_dereference_bh(p) (*p)
#define rcu_dereference_sched(p) (*p)
#define rcu_dereference_check(p, c) (c && *p)
#define rcu_dereference_bh_check(p, c) (c && *p)
#define rcu_dereference_sched_check(p, c) (c && *p)
#define rcu_dereference_protected(p, c) (c && *p)

#define deref(p) (rcu_dereference(p))

int main(void) {
        int *p = (int *) 0xC0FFEE;
        rcu_dereference(p);
        rcu_dereference_bh(p);
        rcu_dereference_sched(p);
        rcu_dereference_check(p, 1);
        rcu_dereference_bh_check(p, 1);
        rcu_dereference_sched_check(p, 1);
        rcu_dereference_protected(p, 1);
        deref(p);
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
// CHECK:      %0 = hl.var "p" : !hl.lvalue<!hl.ptr<!hl.int>> = {
// CHECK:        %25 = hl.const #core.integer<12648430> : !hl.int
// CHECK:        %26 = hl.cstyle_cast %25 IntegralToPointer : !hl.int -> !hl.ptr<!hl.int>
// CHECK:        hl.value.yield %26 : !hl.ptr<!hl.int>
// CHECK:      }
// CHECK:      %1 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:        %25 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:        hl.value.yield %25 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      }
// CHECK:      %2 = kernel.rcu_dereference(%1) : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.int>
// CHECK:      %3 = hl.implicit_cast %2 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
// CHECK:      %4 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:        %25 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:        hl.value.yield %25 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      }
// CHECK:      %5 = kernel.rcu_dereference_bh(%4) : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.int>
// CHECK:      %6 = hl.implicit_cast %5 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
// CHECK:      %7 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:        %25 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:        hl.value.yield %25 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      }
// CHECK:      %8 = kernel.rcu_dereference_sched(%7) : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.int>
// CHECK:      %9 = hl.implicit_cast %8 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
// CHECK:      %10 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:        %25 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:        hl.value.yield %25 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      }
// CHECK:      %11 = macroni.parameter "c" : !hl.int {
// CHECK:        %25 = hl.const #core.integer<1> : !hl.int
// CHECK:        hl.value.yield %25 : !hl.int
// CHECK:      }
// CHECK:      %12 = kernel.rcu_dereference_check(%10, %11) : (!hl.lvalue<!hl.ptr<!hl.int>>, !hl.int) -> !hl.int
// CHECK:      %13 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:        %25 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:        hl.value.yield %25 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      }
// CHECK:      %14 = macroni.parameter "c" : !hl.int {
// CHECK:        %25 = hl.const #core.integer<1> : !hl.int
// CHECK:        hl.value.yield %25 : !hl.int
// CHECK:      }
// CHECK:      %15 = kernel.rcu_dereference_bh_check(%13, %14) : (!hl.lvalue<!hl.ptr<!hl.int>>, !hl.int) -> !hl.int
// CHECK:      %16 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:        %25 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:        hl.value.yield %25 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      }
// CHECK:      %17 = macroni.parameter "c" : !hl.int {
// CHECK:        %25 = hl.const #core.integer<1> : !hl.int
// CHECK:        hl.value.yield %25 : !hl.int
// CHECK:      }
// CHECK:      %18 = kernel.rcu_dereference_sched_check(%16, %17) : (!hl.lvalue<!hl.ptr<!hl.int>>, !hl.int) -> !hl.int
// CHECK:      %19 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:        %25 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:        hl.value.yield %25 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:      }
// CHECK:      %20 = macroni.parameter "c" : !hl.int {
// CHECK:        %25 = hl.const #core.integer<1> : !hl.int
// CHECK:        hl.value.yield %25 : !hl.int
// CHECK:      }
// CHECK:      %21 = kernel.rcu_dereference_protected(%19, %20) : (!hl.lvalue<!hl.ptr<!hl.int>>, !hl.int) -> !hl.int
// CHECK:      %22 = macroni.expansion "deref(p)" : !hl.lvalue<!hl.int> {
// CHECK:        %25 = hl.expr : !hl.lvalue<!hl.int> {
// CHECK:          %26 = macroni.parameter "p" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:            %28 = hl.ref %0 : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:            hl.value.yield %28 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:          }
// CHECK:          %27 = kernel.rcu_dereference(%26) : (!hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.lvalue<!hl.int>
// CHECK:          hl.value.yield %27 : !hl.lvalue<!hl.int>
// CHECK:        }
// CHECK:        hl.value.yield %25 : !hl.lvalue<!hl.int>
// CHECK:      }
// CHECK:      %23 = hl.implicit_cast %22 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
// CHECK:      %24 = hl.const #core.integer<0> : !hl.int
// CHECK:      hl.return %24 : !hl.int
// CHECK:    }
// CHECK:  }
// CHECK:}

