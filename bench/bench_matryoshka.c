/*
 * bench_matryoshka.c — Throughput benchmark for the matryoshka B+ tree.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>
#include "matryoshka.h"

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Simple xorshift32 PRNG for reproducible random queries. */
static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return (*state = x);
}

int main(void)
{
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int nqueries = 5000000;

    printf("Matryoshka B+ tree benchmark\n");
    printf("%-12s  %-12s  %-10s  %-10s\n",
           "Size", "Build (ms)", "Mq/s", "ns/query");
    printf("%-12s  %-12s  %-10s  %-10s\n",
           "----", "----------", "----", "--------");

    for (int si = 0; si < nsizes; si++) {
        int n = sizes[si];
        int32_t *keys = malloc((size_t)n * sizeof(int32_t));
        for (int i = 0; i < n; i++)
            keys[i] = i * 2 + 1;  /* odd keys */

        double t0 = now_sec();
        matryoshka_tree_t *tree = matryoshka_bulk_load(keys, (size_t)n);
        double build_ms = (now_sec() - t0) * 1000.0;

        /* Generate random queries. */
        int32_t *queries = malloc((size_t)nqueries * sizeof(int32_t));
        uint32_t rng = 42;
        for (int i = 0; i < nqueries; i++)
            queries[i] = (int32_t)(xorshift32(&rng) % (uint32_t)(n * 2));

        /* Warm up. */
        volatile int32_t sink = 0;
        for (int i = 0; i < 100000; i++) {
            int32_t r;
            if (matryoshka_search(tree, queries[i % nqueries], &r))
                sink = r;
        }
        (void)sink;

        /* Timed search. */
        t0 = now_sec();
        for (int i = 0; i < nqueries; i++) {
            int32_t r;
            if (matryoshka_search(tree, queries[i], &r))
                sink = r;
        }
        double elapsed = now_sec() - t0;
        double mqs = nqueries / elapsed / 1e6;
        double ns_per = elapsed / nqueries * 1e9;

        printf("%-12d  %-12.1f  %-10.2f  %-10.1f\n",
               n, build_ms, mqs, ns_per);

        matryoshka_destroy(tree);
        free(keys);
        free(queries);
    }

    return 0;
}
