/*
 * test_matryoshka.c — Unit tests for the matryoshka B+ tree.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "matryoshka.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name)  do { printf("  %-45s ", #name); tests_run++; } while (0)
#define PASS()      do { tests_passed++; printf("PASS\n"); return; } while (0)
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); return; } while (0)
#define ASSERT(c, m) do { if (!(c)) FAIL(m); } while (0)

/* ── Basic lifecycle ──────────────────────────────────────────── */

static void test_create_destroy(void)
{
    TEST(create_destroy);
    matryoshka_tree_t *t = matryoshka_create();
    ASSERT(t != NULL, "create returned NULL");
    ASSERT(matryoshka_size(t) == 0, "empty tree has non-zero size");
    matryoshka_destroy(t);
    PASS();
}

/* ── Single insert ────────────────────────────────────────────── */

static void test_insert_single(void)
{
    TEST(insert_single);
    matryoshka_tree_t *t = matryoshka_create();
    ASSERT(matryoshka_insert(t, 42), "insert failed");
    ASSERT(matryoshka_size(t) == 1, "size != 1");
    ASSERT(matryoshka_contains(t, 42), "key 42 not found");
    ASSERT(!matryoshka_contains(t, 41), "phantom key 41");
    ASSERT(!matryoshka_contains(t, 43), "phantom key 43");
    matryoshka_destroy(t);
    PASS();
}

/* ── Duplicate rejection ──────────────────────────────────────── */

static void test_insert_duplicate(void)
{
    TEST(insert_duplicate);
    matryoshka_tree_t *t = matryoshka_create();
    ASSERT(matryoshka_insert(t, 42), "first insert failed");
    ASSERT(!matryoshka_insert(t, 42), "dup insert succeeded");
    ASSERT(matryoshka_size(t) == 1, "size != 1 after dup");
    matryoshka_destroy(t);
    PASS();
}

/* ── Many inserts (ascending) ─────────────────────────────────── */

static void test_insert_ascending(void)
{
    TEST(insert_ascending_1000);
    matryoshka_tree_t *t = matryoshka_create();
    for (int i = 0; i < 1000; i++)
        ASSERT(matryoshka_insert(t, i * 3), "insert failed");
    ASSERT(matryoshka_size(t) == 1000, "wrong size");
    for (int i = 0; i < 1000; i++)
        ASSERT(matryoshka_contains(t, i * 3), "key not found");
    ASSERT(!matryoshka_contains(t, 1), "phantom key 1");
    ASSERT(!matryoshka_contains(t, 2), "phantom key 2");
    matryoshka_destroy(t);
    PASS();
}

/* ── Many inserts (descending) ────────────────────────────────── */

static void test_insert_descending(void)
{
    TEST(insert_descending_1000);
    matryoshka_tree_t *t = matryoshka_create();
    for (int i = 999; i >= 0; i--)
        ASSERT(matryoshka_insert(t, i), "insert failed");
    ASSERT(matryoshka_size(t) == 1000, "wrong size");
    for (int i = 0; i < 1000; i++)
        ASSERT(matryoshka_contains(t, i), "key not found");
    matryoshka_destroy(t);
    PASS();
}

/* ── Insert triggers leaf split ───────────────────────────────── */

static void test_insert_split(void)
{
    TEST(insert_leaf_split_600);
    matryoshka_tree_t *t = matryoshka_create();
    for (int i = 0; i < 600; i++)
        ASSERT(matryoshka_insert(t, i * 2), "insert failed");
    ASSERT(matryoshka_size(t) == 600, "wrong size");
    for (int i = 0; i < 600; i++)
        ASSERT(matryoshka_contains(t, i * 2), "key not found after split");
    ASSERT(!matryoshka_contains(t, 1), "phantom key");
    matryoshka_destroy(t);
    PASS();
}

/* ── Predecessor search ───────────────────────────────────────── */

static void test_search_predecessor(void)
{
    TEST(search_predecessor);
    matryoshka_tree_t *t = matryoshka_create();
    for (int i = 0; i < 100; i++)
        matryoshka_insert(t, i * 10);

    int32_t result;

    /* Exact match. */
    ASSERT(matryoshka_search(t, 50, &result) && result == 50,
           "exact match 50 failed");

    /* Predecessor. */
    ASSERT(matryoshka_search(t, 55, &result) && result == 50,
           "pred(55) != 50");

    /* Max key exact. */
    ASSERT(matryoshka_search(t, 990, &result) && result == 990,
           "exact match 990 failed");

    /* Past max key. */
    ASSERT(matryoshka_search(t, 999, &result) && result == 990,
           "pred(999) != 990");

    /* Below min key. */
    ASSERT(!matryoshka_search(t, -1, &result),
           "pred(-1) should not exist");

    matryoshka_destroy(t);
    PASS();
}

/* ── Bulk load ────────────────────────────────────────────────── */

static void test_bulk_load_small(void)
{
    TEST(bulk_load_100);
    int n = 100;
    int32_t keys[100];
    for (int i = 0; i < n; i++) keys[i] = i * 2;

    matryoshka_tree_t *t = matryoshka_bulk_load(keys, n);
    ASSERT(t != NULL, "bulk load returned NULL");
    ASSERT(matryoshka_size(t) == (size_t)n, "wrong size");
    for (int i = 0; i < n; i++)
        ASSERT(matryoshka_contains(t, i * 2), "key not found");
    ASSERT(!matryoshka_contains(t, 1), "phantom key");

    matryoshka_destroy(t);
    PASS();
}

static void test_bulk_load_medium(void)
{
    TEST(bulk_load_10000);
    int n = 10000;
    int32_t *keys = malloc((size_t)n * sizeof(int32_t));
    for (int i = 0; i < n; i++) keys[i] = i * 2;

    matryoshka_tree_t *t = matryoshka_bulk_load(keys, (size_t)n);
    ASSERT(t != NULL, "bulk load returned NULL");
    ASSERT(matryoshka_size(t) == (size_t)n, "wrong size");

    for (int i = 0; i < n; i++)
        ASSERT(matryoshka_contains(t, i * 2), "key not found");
    ASSERT(!matryoshka_contains(t, 1), "phantom key");

    matryoshka_destroy(t);
    free(keys);
    PASS();
}

static void test_bulk_load_large(void)
{
    TEST(bulk_load_100000);
    int n = 100000;
    int32_t *keys = malloc((size_t)n * sizeof(int32_t));
    for (int i = 0; i < n; i++) keys[i] = i;

    matryoshka_tree_t *t = matryoshka_bulk_load(keys, (size_t)n);
    ASSERT(t != NULL, "bulk load returned NULL");
    ASSERT(matryoshka_size(t) == (size_t)n, "wrong size");

    /* Spot check. */
    for (int i = 0; i < n; i += 97)
        ASSERT(matryoshka_contains(t, i), "key not found");

    matryoshka_destroy(t);
    free(keys);
    PASS();
}

/* ── Bulk load + predecessor search ───────────────────────────── */

static void test_bulk_load_search(void)
{
    TEST(bulk_load_predecessor_search);
    int n = 5000;
    int32_t *keys = malloc((size_t)n * sizeof(int32_t));
    for (int i = 0; i < n; i++) keys[i] = i * 4;

    matryoshka_tree_t *t = matryoshka_bulk_load(keys, (size_t)n);
    int32_t result;

    ASSERT(matryoshka_search(t, 100, &result) && result == 100,
           "exact match 100");
    ASSERT(matryoshka_search(t, 101, &result) && result == 100,
           "pred(101) != 100");
    ASSERT(matryoshka_search(t, 103, &result) && result == 100,
           "pred(103) != 100");
    ASSERT(matryoshka_search(t, 104, &result) && result == 104,
           "exact match 104");
    ASSERT(!matryoshka_search(t, -1, &result),
           "pred(-1) should not exist");

    matryoshka_destroy(t);
    free(keys);
    PASS();
}

/* ── Delete ───────────────────────────────────────────────────── */

static void test_delete_basic(void)
{
    TEST(delete_basic);
    matryoshka_tree_t *t = matryoshka_create();
    for (int i = 0; i < 100; i++)
        matryoshka_insert(t, i);

    ASSERT(matryoshka_delete(t, 50), "delete 50 failed");
    ASSERT(!matryoshka_contains(t, 50), "deleted key 50 still found");
    ASSERT(matryoshka_size(t) == 99, "wrong size after delete");
    ASSERT(!matryoshka_delete(t, 50), "double delete succeeded");
    ASSERT(matryoshka_contains(t, 49), "key 49 missing");
    ASSERT(matryoshka_contains(t, 51), "key 51 missing");

    matryoshka_destroy(t);
    PASS();
}

static void test_delete_many(void)
{
    TEST(delete_half);
    matryoshka_tree_t *t = matryoshka_create();
    for (int i = 0; i < 200; i++)
        matryoshka_insert(t, i);

    /* Delete even keys. */
    for (int i = 0; i < 200; i += 2)
        ASSERT(matryoshka_delete(t, i), "delete failed");

    ASSERT(matryoshka_size(t) == 100, "wrong size");

    for (int i = 0; i < 200; i++) {
        if (i % 2 == 0)
            ASSERT(!matryoshka_contains(t, i), "deleted key found");
        else
            ASSERT(matryoshka_contains(t, i), "remaining key missing");
    }

    matryoshka_destroy(t);
    PASS();
}

/* ── Iterator ─────────────────────────────────────────────────── */

static void test_iterator_full(void)
{
    TEST(iterator_full_scan);
    int n = 500;
    int32_t *keys = malloc((size_t)n * sizeof(int32_t));
    for (int i = 0; i < n; i++) keys[i] = i * 3;

    matryoshka_tree_t *t = matryoshka_bulk_load(keys, (size_t)n);

    matryoshka_iter_t *it = matryoshka_iter_from(t, INT32_MIN);
    ASSERT(it != NULL, "iter_from returned NULL");

    int count = 0;
    int32_t key;
    while (matryoshka_iter_next(it, &key)) {
        ASSERT(key == count * 3, "wrong key in iteration");
        count++;
    }
    ASSERT(count == n, "wrong iteration count");

    matryoshka_iter_destroy(it);
    matryoshka_destroy(t);
    free(keys);
    PASS();
}

static void test_iterator_from_midpoint(void)
{
    TEST(iterator_from_midpoint);
    matryoshka_tree_t *t = matryoshka_create();
    for (int i = 0; i < 100; i++)
        matryoshka_insert(t, i * 10);

    /* Start at 50. */
    matryoshka_iter_t *it = matryoshka_iter_from(t, 50);
    int32_t key;
    ASSERT(matryoshka_iter_next(it, &key) && key == 50,
           "first key from iter_from(50) != 50");
    ASSERT(matryoshka_iter_next(it, &key) && key == 60,
           "second key != 60");
    matryoshka_iter_destroy(it);

    /* Start between keys. */
    it = matryoshka_iter_from(t, 55);
    ASSERT(matryoshka_iter_next(it, &key) && key == 60,
           "iter_from(55) first key != 60");
    matryoshka_iter_destroy(it);

    matryoshka_destroy(t);
    PASS();
}

static void test_iterator_empty(void)
{
    TEST(iterator_empty_tree);
    matryoshka_tree_t *t = matryoshka_create();
    matryoshka_iter_t *it = matryoshka_iter_from(t, 0);
    ASSERT(it != NULL, "iter returned NULL");
    int32_t key;
    ASSERT(!matryoshka_iter_next(it, &key), "next on empty tree");
    matryoshka_iter_destroy(it);
    matryoshka_destroy(t);
    PASS();
}

/* ── Multi-leaf iterator (verifies leaf linking) ──────────────── */

static void test_iterator_across_leaves(void)
{
    TEST(iterator_across_leaves_2000);
    int n = 2000;
    int32_t *keys = malloc((size_t)n * sizeof(int32_t));
    for (int i = 0; i < n; i++) keys[i] = i;

    matryoshka_tree_t *t = matryoshka_bulk_load(keys, (size_t)n);

    matryoshka_iter_t *it = matryoshka_iter_from(t, INT32_MIN);
    int count = 0;
    int32_t key, prev = INT32_MIN;
    while (matryoshka_iter_next(it, &key)) {
        if (count > 0)
            ASSERT(key > prev, "keys not strictly increasing");
        prev = key;
        count++;
    }
    ASSERT(count == n, "wrong count across leaves");

    matryoshka_iter_destroy(it);
    matryoshka_destroy(t);
    free(keys);
    PASS();
}

/* ── Bulk load empty ──────────────────────────────────────────── */

static void test_bulk_load_empty(void)
{
    TEST(bulk_load_empty);
    matryoshka_tree_t *t = matryoshka_bulk_load(NULL, 0);
    ASSERT(t != NULL, "bulk load empty returned NULL");
    ASSERT(matryoshka_size(t) == 0, "empty tree size != 0");
    ASSERT(!matryoshka_contains(t, 0), "phantom key in empty tree");
    matryoshka_destroy(t);
    PASS();
}

/* ── Bulk load single ─────────────────────────────────────────── */

static void test_bulk_load_single(void)
{
    TEST(bulk_load_single);
    int32_t key = 42;
    matryoshka_tree_t *t = matryoshka_bulk_load(&key, 1);
    ASSERT(t != NULL, "bulk load single returned NULL");
    ASSERT(matryoshka_size(t) == 1, "size != 1");
    ASSERT(matryoshka_contains(t, 42), "key not found");
    matryoshka_destroy(t);
    PASS();
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(void)
{
    printf("Matryoshka tree tests:\n\n");

    test_create_destroy();
    test_insert_single();
    test_insert_duplicate();
    test_insert_ascending();
    test_insert_descending();
    test_insert_split();
    test_search_predecessor();
    test_bulk_load_empty();
    test_bulk_load_single();
    test_bulk_load_small();
    test_bulk_load_medium();
    test_bulk_load_large();
    test_bulk_load_search();
    test_delete_basic();
    test_delete_many();
    test_iterator_empty();
    test_iterator_full();
    test_iterator_from_midpoint();
    test_iterator_across_leaves();

    printf("\n  %d/%d tests passed.\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
