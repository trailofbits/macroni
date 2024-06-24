// RUN: kernelize %s -- 2>/dev/null | kernelcheck 2>&1 1>/dev/null | FileCheck %s

#include "rcu.h"

#define __must_hold(x) __attribute__((context(x, 1, 1)))
#define __acquires(x) __attribute__((context(x, 0, 1)))
#define __releases(x) __attribute__((context(x, 1, 0)))

#define deref(p) (rcu_dereference(p))

void rcu_read_lock(void) {}
void rcu_read_unlock(void) {}

void must_hold_test(int lock) __must_hold(lock) {
  int *p = (int *)0xC0FFEE;
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

void acquires_test(int lock) __acquires(lock) {
  int *p = (int *)0xC0FFEE;
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

void releases_test(int lock) __releases(lock) {
  int *p = (int *)0xC0FFEE;
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
  int *p = (int *)0xC0FFEE;
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

// CHECK: {{.*}}: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK: {{.*}}: warning: Invocation of rcu_dereference_bh() outside of RCU critical section
// CHECK: {{.*}}: warning: Invocation of rcu_dereference_sched() outside of RCU critical section
// CHECK: {{.*}}: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK: {{.*}}: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK: {{.*}}: warning: Invocation of rcu_dereference_sched() outside of RCU critical section
// CHECK: {{.*}}: warning: Invocation of rcu_dereference_bh() outside of RCU critical section
// CHECK: {{.*}}: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK: {{.*}}: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK: {{.*}}: warning: Invocation of rcu_dereference_bh() outside of RCU critical section
// CHECK: {{.*}}: warning: Invocation of rcu_dereference_sched() outside of RCU critical section
// CHECK: {{.*}}: warning: Invocation of rcu_dereference() outside of RCU critical section
// CHECK: {{.*}}: info: Use rcu_dereference_protected() instead of rcu_access_pointer() in critical section
