#include "rcu.h"
#include "test_structs.h"

int main(void) {
  int *p = 0;
  struct string s = {0};

  rcu_access_pointer(p);
  rcu_access_pointer(s.begin);

  return 0;
}
