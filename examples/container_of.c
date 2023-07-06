typedef unsigned long size_t;

#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

#define container_of(ptr, type, member) ({          \
	const typeof(((type *)0)->member)*__mptr = (ptr);    \
		     (type *)((char *)__mptr - offsetof(type, member)); })

struct contained {};

struct container { struct contained *contained_member; };

int main(void) {
        struct container *container_instance;
        container_of(&container_instance->contained_member,
                     struct container, contained_member);
        return 0;
}
