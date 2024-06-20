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
