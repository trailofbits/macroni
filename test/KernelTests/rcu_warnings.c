// RUN: kernelize -x c %s 2>&1 1>/dev/null | FileCheck %s

#define __must_hold(x) __attribute__((context(x, 1, 1)))
#define __acquires(x) __attribute__((context(x, 0, 1)))
#define __releases(x) __attribute__((context(x, 1, 0)))

#define rcu_dereference(p) (*p)
#define rcu_dereference_bh(p) (*p)
#define rcu_dereference_sched(p) (*p)
#define rcu_dereference_check(p, c) (c && *p)
#define rcu_dereference_bh_check(p, c) (c && *p)
#define rcu_dereference_sched_check(p, c) (c && *p)
#define rcu_dereference_protected(p, c) (c && *p)
#define rcu_access_pointer(p) (*p)
#define rcu_assign_pointer(p, v) (((p) = (v)))
#define rcu_replace_pointer(rcu_ptr, ptr, c) (c && ((rcu_ptr) = (ptr)))

#define deref(p) (rcu_dereference(p))

void rcu_read_lock(void) {}
void rcu_read_unlock(void) {}

void must_hold_test(int lock) __must_hold(lock)  {
        int *p = (int *) 0xC0FFEE;
        int *v;

        rcu_dereference(p);
        rcu_dereference_bh(p);
        rcu_dereference_sched(p);
        rcu_dereference_check(p, 1);
        rcu_dereference_bh_check(p, 1);
        rcu_dereference_sched_check(p, 1);
        rcu_dereference_protected(p, 1);
        rcu_access_pointer(p);
        rcu_assign_pointer(p, v);
        rcu_replace_pointer(p, v, 1);
        deref(p);
}

void acquires_test(int lock) __acquires(lock)  {
        int *p = (int *) 0xC0FFEE;
        int *v;

        rcu_dereference(p);
        rcu_dereference_bh(p);
        rcu_dereference_sched(p);
        rcu_dereference_check(p, 1);
        rcu_dereference_bh_check(p, 1);
        rcu_dereference_sched_check(p, 1);
        rcu_dereference_protected(p, 1);
        rcu_access_pointer(p);
        rcu_assign_pointer(p, v);
        rcu_replace_pointer(p, v, 1);
        deref(p);

        rcu_read_lock();

        rcu_dereference(p);
        rcu_dereference_bh(p);
        rcu_dereference_sched(p);
        rcu_dereference_check(p, 1);
        rcu_dereference_bh_check(p, 1);
        rcu_dereference_sched_check(p, 1);
        rcu_dereference_protected(p, 1);
        rcu_access_pointer(p);
        rcu_assign_pointer(p, v);
        rcu_replace_pointer(p, v, 1);
        deref(p);
}

void releases_test(int lock) __releases(lock)  {
        int *p = (int *) 0xC0FFEE;
        int *v;

        rcu_dereference(p);
        rcu_dereference_bh(p);
        rcu_dereference_sched(p);
        rcu_dereference_check(p, 1);
        rcu_dereference_bh_check(p, 1);
        rcu_dereference_sched_check(p, 1);
        rcu_dereference_protected(p, 1);
        rcu_access_pointer(p);
        rcu_assign_pointer(p, v);
        rcu_replace_pointer(p, v, 1);
        deref(p);

        rcu_read_unlock();

        rcu_dereference(p);
        rcu_dereference_bh(p);
        rcu_dereference_sched(p);
        rcu_dereference_check(p, 1);
        rcu_dereference_bh_check(p, 1);
        rcu_dereference_sched_check(p, 1);
        rcu_dereference_protected(p, 1);
        rcu_access_pointer(p);
        rcu_assign_pointer(p, v);
        rcu_replace_pointer(p, v, 1);
        deref(p);
}

int main(void) {
        int *p = (int *) 0xC0FFEE;
        int *v;

        rcu_dereference(p);
        rcu_dereference_bh(p);
        rcu_dereference_sched(p);
        rcu_dereference_check(p, 1);
        rcu_dereference_bh_check(p, 1);
        rcu_dereference_sched_check(p, 1);
        rcu_dereference_protected(p, 1);
        rcu_access_pointer(p);
        rcu_assign_pointer(p, v);
        rcu_replace_pointer(p, v, 1);
        deref(p);

        rcu_read_lock();
        rcu_dereference(p);
        rcu_dereference_bh(p);
        rcu_dereference_sched(p);
        rcu_dereference_check(p, 1);
        rcu_dereference_bh_check(p, 1);
        rcu_dereference_sched_check(p, 1);
        rcu_dereference_protected(p, 1);
        rcu_access_pointer(p);
        rcu_assign_pointer(p, v);
        rcu_replace_pointer(p, v, 1);
        deref(p);
        rcu_read_unlock();

        return 0;
}

// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK:v
// CHECK:p
// CHECK:v
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK:v
// CHECK:p
// CHECK:v
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK:v
// CHECK:p
// CHECK:v
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK:v
// CHECK:p
// CHECK:v
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK:v
// CHECK:p
// CHECK:v
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK:v
// CHECK:p
// CHECK:v
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK:v
// CHECK:p
// CHECK:v
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:rcu_ptr
// CHECK:ptr
// CHECK:c
// CHECK:p
// CHECK:p
// CHECK::44:9: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK::45:9: warning: Invocation of rcu_dereference_bh() outside of RCU critical section
// CHECK::46:9: warning: Invocation of rcu_dereference_sched() outside of RCU critical section
// CHECK::18:19: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK::58:9: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK::59:9: warning: Invocation of rcu_dereference_bh() outside of RCU critical section
// CHECK::60:9: warning: Invocation of rcu_dereference_sched() outside of RCU critical section
// CHECK::18:19: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK::18:19: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK::91:9: warning: Invocation of rcu_dereference_sched() outside of RCU critical section
// CHECK::90:9: warning: Invocation of rcu_dereference_bh() outside of RCU critical section
// CHECK::89:9: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK::18:19: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK::77:9: warning: Invocation of rcu_dereference_sched() outside of RCU critical section
// CHECK::76:9: warning: Invocation of rcu_dereference_bh() outside of RCU critical section
// CHECK::75:9: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK::106:9: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK::107:9: warning: Invocation of rcu_dereference_bh() outside of RCU critical section
// CHECK::108:9: warning: Invocation of rcu_dereference_sched() outside of RCU critical section
// CHECK::18:19: warning: Invocation of rcu_dereference() outside of RCU critical section

