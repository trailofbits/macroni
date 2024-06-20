#include "rcu.h"
#include "test_structs.h"

int main(void) {
  int *x = 0;
  struct string *s = {0};
  rcu_dereference_bh(x);
  rcu_dereference_bh(s->begin);
  return 0;
}
