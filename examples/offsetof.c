typedef unsigned long int size_t;

#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

struct A { int x; };

int main(void) {
        offsetof(struct A, x);
        return 0;
}
