#include <solv/pool.h>

int main() {
    Pool* pool = pool_create();
    pool_str2id(pool, "hello", /* create= */ 1);
    pool_free(pool);
}
