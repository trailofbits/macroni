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
// CHECK:   hl.func @rcu_read_lock () -> !hl.void {
// CHECK:     core.scope {
// CHECK:       %0 = hl.const #void_value
// CHECK:       core.implicit.return %0 : !hl.void
// CHECK:     }
// CHECK:   }
// CHECK:   hl.func @rcu_read_unlock () -> !hl.void {
// CHECK:     core.scope {
// CHECK:       %0 = hl.const #void_value
// CHECK:       core.implicit.return %0 : !hl.void
// CHECK:     }
// CHECK:   }
// CHECK:   hl.func @lock () -> !hl.void {
// CHECK:     core.scope {
// CHECK:       %0 = hl.call @rcu_read_lock() {lock_level = 0 : i64} : () -> !hl.void
// CHECK:       %1 = hl.const #void_value
// CHECK:       core.implicit.return %1 : !hl.void
// CHECK:     }
// CHECK:   }
// CHECK:   hl.func @unlock () -> !hl.void {
// CHECK:     core.scope {
// CHECK:       %0 = hl.call @rcu_read_unlock() {lock_level = 0 : i64} : () -> !hl.void
// CHECK:       %1 = hl.const #void_value
// CHECK:       core.implicit.return %1 : !hl.void
// CHECK:     }
// CHECK:   }
// CHECK:   hl.func @foo () -> !hl.void {
// CHECK:     core.scope {
// CHECK:       %0 = hl.const #void_value
// CHECK:       core.implicit.return %0 : !hl.void
// CHECK:     }
// CHECK:   }
// CHECK:   hl.func @main () -> !hl.int {
// CHECK:     core.scope {
// CHECK:       %0 = hl.var "i" : !hl.lvalue<!hl.int>
// CHECK:       kernel.rcu.critical_section {
// CHECK:       }
// CHECK:       kernel.rcu.critical_section {
// CHECK:         %2 = hl.call @foo() : () -> !hl.void
// CHECK:       }
// CHECK:       kernel.rcu.critical_section {
// CHECK:         core.scope {
// CHECK:           %2 = hl.ref %0 : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
// CHECK:           %3 = hl.const #core.integer<0> : !hl.int
// CHECK:           %4 = hl.assign %3 to %2 : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
// CHECK:           hl.for {
// CHECK:             %5 = hl.ref %0 : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
// CHECK:             %6 = hl.implicit_cast %5 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
// CHECK:             %7 = hl.const #core.integer<10> : !hl.int
// CHECK:             %8 = hl.cmp slt %6, %7 : !hl.int, !hl.int -> !hl.int
// CHECK:             hl.cond.yield %8 : !hl.int
// CHECK:           } incr {
// CHECK:             %5 = hl.ref %0 : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
// CHECK:             %6 = hl.post.inc %5 : !hl.lvalue<!hl.int> -> !hl.int
// CHECK:           } do {
// CHECK:             core.scope {
// CHECK:               %5 = hl.call @foo() : () -> !hl.void
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:       kernel.rcu.critical_section {
// CHECK:         core.scope {
// CHECK:           kernel.rcu.critical_section {
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:       kernel.rcu.critical_section {
// CHECK:         hl.do {
// CHECK:           core.scope {
// CHECK:             %2 = hl.var "x" : !hl.lvalue<!hl.int>
// CHECK:           }
// CHECK:         } while {
// CHECK:           %2 = hl.const #core.integer<0> : !hl.int
// CHECK:           hl.cond.yield %2 : !hl.int
// CHECK:         }
// CHECK:       }
// CHECK:       kernel.rcu.critical_section {
// CHECK:         kernel.rcu.critical_section {
// CHECK:           kernel.rcu.critical_section {
// CHECK:             kernel.rcu.critical_section {
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:       kernel.rcu.critical_section {
// CHECK:         hl.do {
// CHECK:           core.scope {
// CHECK:             kernel.rcu.critical_section {
// CHECK:               %2 = hl.var "x" : !hl.lvalue<!hl.int>
// CHECK:             }
// CHECK:           }
// CHECK:         } while {
// CHECK:           %2 = hl.const #core.integer<0> : !hl.int
// CHECK:           hl.cond.yield %2 : !hl.int
// CHECK:         }
// CHECK:       }
// CHECK:       hl.do {
// CHECK:         core.scope {
// CHECK:           kernel.rcu.critical_section {
// CHECK:             %2 = hl.var "x" : !hl.lvalue<!hl.int> = {
// CHECK:               %4 = hl.const #core.integer<1> : !hl.int
// CHECK:               hl.value.yield %4 : !hl.int
// CHECK:             }
// CHECK:             kernel.rcu.critical_section {
// CHECK:             }
// CHECK:             %3 = hl.var "y" : !hl.lvalue<!hl.int> = {
// CHECK:               %4 = hl.const #core.integer<2> : !hl.int
// CHECK:               hl.value.yield %4 : !hl.int
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       } while {
// CHECK:         %2 = hl.const #core.integer<0> : !hl.int
// CHECK:         hl.cond.yield %2 : !hl.int
// CHECK:       }
// CHECK:       kernel.rcu.critical_section {
// CHECK:         hl.if {
// CHECK:           %2 = hl.const #core.integer<1> : !hl.int
// CHECK:           hl.cond.yield %2 : !hl.int
// CHECK:         } then {
// CHECK:           core.scope {
// CHECK:             %2 = hl.call @rcu_read_unlock() {lock_level = 0 : i64} : () -> !hl.void
// CHECK:           }
// CHECK:         } else {
// CHECK:           core.scope {
// CHECK:             %2 = hl.call @rcu_read_lock() {lock_level = 0 : i64} : () -> !hl.void
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:       %1 = hl.const #core.integer<0> : !hl.int
// CHECK:       hl.return %1 : !hl.int
// CHECK:     }
// CHECK:   }
// CHECK: }

void rcu_read_lock(void) {}
void rcu_read_unlock(void) {}

void lock(void) {
        rcu_read_lock();
}

void unlock(void) {
        rcu_read_unlock();
}

void foo(void) {}

int main(void) {
        int i;

        rcu_read_lock();
        rcu_read_unlock();

        rcu_read_lock();
        foo();
        rcu_read_unlock();

        rcu_read_lock();
        for (i = 0; i < 10; i++) {
                foo();
        }
        rcu_read_unlock();


        rcu_read_lock();
        {
                rcu_read_lock();
                rcu_read_unlock();
        }
        rcu_read_unlock();

        rcu_read_lock();
        do {
                int x;
        } while (0);
        rcu_read_unlock();

        rcu_read_lock();
        rcu_read_lock();
        rcu_read_lock();
        rcu_read_lock();
        rcu_read_unlock();
        rcu_read_unlock();
        rcu_read_unlock();
        rcu_read_unlock();

        rcu_read_lock();
        do {
                rcu_read_lock();
                int x;
                rcu_read_unlock();
        } while (0);
        rcu_read_unlock();

        do {
                rcu_read_lock();
                int x = 1;
                rcu_read_lock();
                rcu_read_unlock();
                int y = 2;
                rcu_read_unlock();
        } while (0);

        rcu_read_lock();
        if (1) {
                rcu_read_unlock();
        } else {
                rcu_read_lock();
        }
        rcu_read_unlock();

        return 0;
}
