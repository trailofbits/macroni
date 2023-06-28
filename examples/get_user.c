#define get_user(x, ptr) ({ int res = 0; if (1) { x = *ptr; res = 1; } res; })

int main(void) {
        int x, *ptr;
        get_user(x, ptr);
        return 0;
}