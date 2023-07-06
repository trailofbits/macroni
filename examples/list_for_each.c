typedef unsigned long size_t;

struct list_head { struct list_head *next, *prev; };

static inline int list_is_head(const struct list_head *list,
                               const struct list_head *head) {
        return list == head;
}

#define list_for_each(pos, head) \
	for (pos = (head)->next; !list_is_head(pos, (head)); pos = pos->next)

int main(void) {
        struct list_head *head, *pos;
        list_for_each(pos, head) {
                struct list_head *prev = pos->prev;
        }
        return 0;
}
