// RUN: kernelize %s -- | FileCheck %s --match-full-lines

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

        rcu_read_lock();
a_label:
        rcu_read_unlock();

        rcu_read_lock();
        1;
b_label:
        rcu_read_unlock();

        return 0;
}

// CHECK: #void_value = #core.void : !hl.void
// CHECK:   hl.translation_unit {
// CHECK:     hl.typedef "__int128_t" : !hl.int128
// CHECK:     hl.typedef "__uint128_t" : !hl.int128< unsigned >
// CHECK:     hl.typedef "__NSConstantString" : !hl.record<"__NSConstantString_tag">
// CHECK:     hl.typedef "__builtin_ms_va_list" : !hl.ptr<!hl.char>
// CHECK:     hl.typedef "__builtin_va_list" : !hl.array<1, !hl.record<"__va_list_tag">>
// CHECK:     hl.func @rcu_read_lock external () -> !hl.void {
// CHECK:       %0 = hl.const #void_value
// CHECK:       core.implicit.return %0 : !hl.void
// CHECK:     }
// CHECK:     hl.func @rcu_read_unlock external () -> !hl.void {
// CHECK:       %0 = hl.const #void_value
// CHECK:       core.implicit.return %0 : !hl.void
// CHECK:     }
// CHECK:     hl.func @lock external () -> !hl.void {
// CHECK:       core.scope {
// CHECK:         %1 = hl.call @rcu_read_lock() {lock_level = 0 : i64} : () -> !hl.void
// CHECK:       }
// CHECK:       %0 = hl.const #void_value
// CHECK:       core.implicit.return %0 : !hl.void
// CHECK:     }
// CHECK:     hl.func @unlock external () -> !hl.void {
// CHECK:       core.scope {
// CHECK:         %1 = hl.call @rcu_read_unlock() {lock_level = -1 : i64} : () -> !hl.void
// CHECK:       }
// CHECK:       %0 = hl.const #void_value
// CHECK:       core.implicit.return %0 : !hl.void
// CHECK:     }
// CHECK:     hl.func @foo external () -> !hl.void {
// CHECK:       %0 = hl.const #void_value
// CHECK:       core.implicit.return %0 : !hl.void
// CHECK:     }
// CHECK:     hl.func @main external () -> !hl.int {
// CHECK:       %0 = hl.label.decl "a_label" : !hl.label
// CHECK:       %1 = hl.label.decl "b_label" : !hl.label
// CHECK:       core.scope {
// CHECK:         %2 = hl.var "i" : !hl.lvalue<!hl.int>
// CHECK:         kernel.rcu.critical_section {
// CHECK:         }
// CHECK:         kernel.rcu.critical_section {
// CHECK:           %4 = hl.call @foo() : () -> !hl.void
// CHECK:         }
// CHECK:         kernel.rcu.critical_section {
// CHECK:           %4 = hl.ref %2 : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
// CHECK:           %5 = hl.const #core.integer<0> : !hl.int
// CHECK:           %6 = hl.assign %5 to %4 : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
// CHECK:           hl.for {
// CHECK:             %7 = hl.ref %2 : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
// CHECK:             %8 = hl.implicit_cast %7 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
// CHECK:             %9 = hl.const #core.integer<10> : !hl.int
// CHECK:             %10 = hl.cmp slt %8, %9 : !hl.int, !hl.int -> !hl.int
// CHECK:             hl.cond.yield %10 : !hl.int
// CHECK:           } incr {
// CHECK:             %7 = hl.ref %2 : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
// CHECK:             %8 = hl.post.inc %7 : !hl.lvalue<!hl.int> -> !hl.int
// CHECK:           } do {
// CHECK:             core.scope {
// CHECK:               %7 = hl.call @foo() : () -> !hl.void
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:         kernel.rcu.critical_section {
// CHECK:           core.scope {
// CHECK:             kernel.rcu.critical_section {
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:         kernel.rcu.critical_section {
// CHECK:           hl.do {
// CHECK:             core.scope {
// CHECK:               %4 = hl.var "x" : !hl.lvalue<!hl.int>
// CHECK:             }
// CHECK:           } while {
// CHECK:             %4 = hl.const #core.integer<0> : !hl.int
// CHECK:             hl.cond.yield %4 : !hl.int
// CHECK:           }
// CHECK:         }
// CHECK:         kernel.rcu.critical_section {
// CHECK:           kernel.rcu.critical_section {
// CHECK:             kernel.rcu.critical_section {
// CHECK:               kernel.rcu.critical_section {
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:         kernel.rcu.critical_section {
// CHECK:           hl.do {
// CHECK:             core.scope {
// CHECK:               kernel.rcu.critical_section {
// CHECK:                 %4 = hl.var "x" : !hl.lvalue<!hl.int>
// CHECK:               }
// CHECK:             }
// CHECK:           } while {
// CHECK:             %4 = hl.const #core.integer<0> : !hl.int
// CHECK:             hl.cond.yield %4 : !hl.int
// CHECK:           }
// CHECK:         }
// CHECK:         hl.do {
// CHECK:           core.scope {
// CHECK:             kernel.rcu.critical_section {
// CHECK:               %4 = hl.var "x" : !hl.lvalue<!hl.int> = {
// CHECK:                 %6 = hl.const #core.integer<1> : !hl.int
// CHECK:                 hl.value.yield %6 : !hl.int
// CHECK:               }
// CHECK:               kernel.rcu.critical_section {
// CHECK:               }
// CHECK:               %5 = hl.var "y" : !hl.lvalue<!hl.int> = {
// CHECK:                 %6 = hl.const #core.integer<2> : !hl.int
// CHECK:                 hl.value.yield %6 : !hl.int
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         } while {
// CHECK:           %4 = hl.const #core.integer<0> : !hl.int
// CHECK:           hl.cond.yield %4 : !hl.int
// CHECK:         }
// CHECK:         kernel.rcu.critical_section {
// CHECK:           hl.if {
// CHECK:             %4 = hl.const #core.integer<1> : !hl.int
// CHECK:             hl.cond.yield %4 : !hl.int
// CHECK:           } then {
// CHECK:             core.scope {
// CHECK:               %4 = hl.call @rcu_read_unlock() {lock_level = 0 : i64} : () -> !hl.void
// CHECK:             }
// CHECK:           } else {
// CHECK:             core.scope {
// CHECK:               %4 = hl.call @rcu_read_lock() {lock_level = 0 : i64} : () -> !hl.void
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:         kernel.rcu.critical_section {
// CHECK:         }
// CHECK:         hl.label %0 {
// CHECK:         }
// CHECK:         kernel.rcu.critical_section {
// CHECK:           %4 = hl.const #core.integer<1> : !hl.int
// CHECK:         }
// CHECK:         hl.label %1 {
// CHECK:         }
// CHECK:         %3 = hl.const #core.integer<0> : !hl.int
// CHECK:         hl.return %3 : !hl.int
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }
