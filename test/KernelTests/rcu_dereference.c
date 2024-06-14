#include "rcu.h"

struct string {
  char *begin;
  unsigned long size;
};

int main(void) {
  int *x = 0;
  struct string *s = {0};
  rcu_dereference(x);
  rcu_dereference(s->begin);
  return 0;
}
