void rcu_read_lock(void) {}
void rcu_read_unlock(void) {}

void foo(void) {}

int main(void) {
        int i = 0;

        rcu_read_lock();
        for (i = 0; i < 10; i++) {
                foo();
        }
        rcu_read_unlock();

        return 0;
}
