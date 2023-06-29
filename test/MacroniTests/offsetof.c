typedef unsigned long int size_t;

#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

struct location {
        char *file;
        int line;
        int col;
};

int main(void) {
        offsetof(struct location, line);
        return 0;
}
