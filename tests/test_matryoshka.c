/*
 * test_matryoshka.c — Unit tests for the matryoshka B+ tree.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "matryoshka.h"
#include "matryoshka_internal.h"

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
    TEST(insert_leaf_split_2000);
    matryoshka_tree_t *t = matryoshka_create();
    for (int i = 0; i < 2000; i++)
        ASSERT(matryoshka_insert(t, i * 2), "insert failed");
    ASSERT(matryoshka_size(t) == 2000, "wrong size");
    for (int i = 0; i < 2000; i++)
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

/* ── Eager deletion: heavy deletion ───────────────────────────── */

static void test_delete_heavy(void)
{
    TEST(delete_heavy_900_of_1000);
    matryoshka_tree_t *t = matryoshka_create();
    for (int i = 0; i < 1000; i++)
        ASSERT(matryoshka_insert(t, i), "insert failed");

    /* Delete 900 keys (every key except multiples of 10). */
    int deleted = 0;
    for (int i = 0; i < 1000; i++) {
        if (i % 10 != 0) {
            ASSERT(matryoshka_delete(t, i), "delete failed");
            deleted++;
        }
    }
    ASSERT(matryoshka_size(t) == 100, "wrong size after heavy delete");

    /* Verify remaining keys. */
    for (int i = 0; i < 1000; i++) {
        if (i % 10 == 0)
            ASSERT(matryoshka_contains(t, i), "remaining key missing");
        else
            ASSERT(!matryoshka_contains(t, i), "deleted key found");
    }

    matryoshka_destroy(t);
    PASS();
}

/* ── Eager deletion: delete all ──────────────────────────────── */

static void test_delete_all(void)
{
    TEST(delete_all_500);
    matryoshka_tree_t *t = matryoshka_create();
    for (int i = 0; i < 500; i++)
        matryoshka_insert(t, i * 2);

    for (int i = 0; i < 500; i++)
        ASSERT(matryoshka_delete(t, i * 2), "delete failed");

    ASSERT(matryoshka_size(t) == 0, "size != 0 after deleting all");
    ASSERT(!matryoshka_contains(t, 0), "phantom key in empty tree");
    matryoshka_destroy(t);
    PASS();
}

/* ── Eager deletion: cascading merges (bulk load then deplete) ── */

static void test_delete_cascading(void)
{
    TEST(delete_cascading_merges);
    int n = 5000;
    int32_t *keys = malloc((size_t)n * sizeof(int32_t));
    for (int i = 0; i < n; i++) keys[i] = i;

    matryoshka_tree_t *t = matryoshka_bulk_load(keys, (size_t)n);

    /* Delete from the middle outward to trigger cascading merges. */
    for (int i = n / 2; i < n; i++)
        ASSERT(matryoshka_delete(t, i), "delete right failed");
    for (int i = n / 2 - 1; i >= 0; i--)
        ASSERT(matryoshka_delete(t, i), "delete left failed");

    ASSERT(matryoshka_size(t) == 0, "size != 0");
    matryoshka_destroy(t);
    free(keys);
    PASS();
}

/* ── Eager deletion: interleaved insert/delete ───────────────── */

static void test_delete_interleaved(void)
{
    TEST(delete_interleaved_insert_delete);
    matryoshka_tree_t *t = matryoshka_create();

    /* Insert 2000, delete 1500, insert 1000, verify. */
    for (int i = 0; i < 2000; i++)
        matryoshka_insert(t, i);
    for (int i = 0; i < 1500; i++)
        matryoshka_delete(t, i);

    ASSERT(matryoshka_size(t) == 500, "wrong size after partial delete");

    for (int i = 1500; i < 2000; i++)
        ASSERT(matryoshka_contains(t, i), "remaining key missing");

    /* Insert more keys. */
    for (int i = 3000; i < 4000; i++)
        matryoshka_insert(t, i);

    ASSERT(matryoshka_size(t) == 1500, "wrong size after re-insert");

    /* Verify via iteration. */
    matryoshka_iter_t *it = matryoshka_iter_from(t, INT32_MIN);
    int count = 0;
    int32_t key, prev = INT32_MIN;
    while (matryoshka_iter_next(it, &key)) {
        if (count > 0)
            ASSERT(key > prev, "keys not strictly increasing");
        prev = key;
        count++;
    }
    ASSERT(count == 1500, "iteration count wrong");
    matryoshka_iter_destroy(it);

    matryoshka_destroy(t);
    PASS();
}

/* ── Hierarchy: create_with (default) ─────────────────────────── */

static void test_create_with_default(void)
{
    TEST(create_with_default_hierarchy);
    mt_hierarchy_t hier;
    mt_hierarchy_init_default(&hier);
    matryoshka_tree_t *t = matryoshka_create_with(&hier);
    ASSERT(t != NULL, "create_with returned NULL");
    for (int i = 0; i < 500; i++)
        ASSERT(matryoshka_insert(t, i * 2), "insert failed");
    ASSERT(matryoshka_size(t) == 500, "wrong size");
    for (int i = 0; i < 500; i++)
        ASSERT(matryoshka_contains(t, i * 2), "key not found");
    matryoshka_destroy(t);
    PASS();
}

/* ── Hierarchy: bulk_load_with (default) ──────────────────────── */

static void test_bulk_load_with_default(void)
{
    TEST(bulk_load_with_default_hierarchy);
    mt_hierarchy_t hier;
    mt_hierarchy_init_default(&hier);
    int n = 10000;
    int32_t *keys = malloc((size_t)n * sizeof(int32_t));
    for (int i = 0; i < n; i++) keys[i] = i;
    matryoshka_tree_t *t = matryoshka_bulk_load_with(keys, (size_t)n, &hier);
    ASSERT(t != NULL, "bulk_load_with returned NULL");
    ASSERT(matryoshka_size(t) == (size_t)n, "wrong size");
    for (int i = 0; i < n; i += 97)
        ASSERT(matryoshka_contains(t, i), "key not found");
    matryoshka_destroy(t);
    free(keys);
    PASS();
}

/* ── Hierarchy: superpage configuration ──────────────────────── */

static void test_hierarchy_superpage(void)
{
    TEST(hierarchy_superpage);
    mt_hierarchy_t h;
    mt_hierarchy_init_superpage(&h);
    ASSERT(h.leaf_alloc == 2u * 1024 * 1024, "superpage leaf_alloc wrong");
    ASSERT(h.cl_key_cap == MT_CL_KEY_CAP, "cl_key_cap wrong");
    ASSERT(h.page_max_keys > 0, "page_max_keys is 0");
    PASS();
}

/* ── Hierarchy: custom leaf allocation ───────────────────────── */

static void test_hierarchy_custom(void)
{
    TEST(hierarchy_custom_leaf_alloc);
    mt_hierarchy_t h;
    mt_hierarchy_init_custom(&h, 8192);
    ASSERT(h.leaf_alloc == 8192, "custom leaf_alloc wrong");
    ASSERT(h.cl_key_cap == MT_CL_KEY_CAP, "cl_key_cap wrong");
    ASSERT(h.page_max_keys > 0, "page_max_keys is 0");
    PASS();
}

/* ── Page sub-tree: CL capacity ───────────────────────────────── */

static void test_page_subtree_capacity(void)
{
    TEST(page_subtree_capacity);
    mt_hierarchy_t h;
    mt_hierarchy_init_default(&h);
    ASSERT(h.cl_key_cap == 15, "cl_key_cap != 15");
    ASSERT(h.cl_sep_cap == 12, "cl_sep_cap != 12");
    ASSERT(h.cl_child_cap == 13, "cl_child_cap != 13");
    ASSERT(h.page_slots == 63, "page_slots != 63");
    ASSERT(h.min_cl_keys == 7, "min_cl_keys != 7");
    ASSERT(h.min_cl_children == 7, "min_cl_children != 7");
    PASS();
}

/* ── Arena allocator ──────────────────────────────────────────── */

static void test_arena_basic(void)
{
    TEST(arena_allocator_basic);
    /* Create an allocator with 64 KiB arenas, 4 KiB pages. */
    mt_allocator_t *alloc = mt_allocator_create(65536, 4096);
    ASSERT(alloc != NULL, "allocator create failed");

    /* Allocate 16 pages (fills one arena). */
    void *pages[16];
    for (int i = 0; i < 16; i++) {
        pages[i] = mt_allocator_alloc(alloc);
        ASSERT(pages[i] != NULL, "alloc returned NULL");
    }

    /* All pages should be distinct. */
    for (int i = 0; i < 16; i++)
        for (int j = i + 1; j < 16; j++)
            ASSERT(pages[i] != pages[j], "duplicate page pointers");

    /* Free some and reallocate. */
    mt_allocator_free(alloc, pages[5]);
    mt_allocator_free(alloc, pages[10]);
    void *p1 = mt_allocator_alloc(alloc);
    void *p2 = mt_allocator_alloc(alloc);
    ASSERT(p1 != NULL && p2 != NULL, "realloc after free failed");

    /* Allocate more — should trigger a second arena. */
    void *extra = mt_allocator_alloc(alloc);
    ASSERT(extra != NULL, "alloc from second arena failed");

    mt_allocator_destroy(alloc);
    PASS();
}

static void test_arena_co_location(void)
{
    TEST(arena_co_location);
    /* Allocations from the same arena should be within arena_size of each other. */
    size_t arena_size = 65536;
    mt_allocator_t *alloc = mt_allocator_create(arena_size, 4096);
    void *p1 = mt_allocator_alloc(alloc);
    void *p2 = mt_allocator_alloc(alloc);
    ASSERT(p1 != NULL && p2 != NULL, "alloc failed");

    size_t diff = ((char *)p2 > (char *)p1)
                  ? (size_t)((char *)p2 - (char *)p1)
                  : (size_t)((char *)p1 - (char *)p2);
    ASSERT(diff < arena_size, "co-located pages not in same arena");

    mt_allocator_destroy(alloc);
    PASS();
}

/* ── Batch tests ─────────────────────────────────────────────── */

static void test_batch_insert_basic(void)
{
    TEST(batch_insert_basic);
    matryoshka_tree_t *t = matryoshka_create();
    int32_t keys[] = {50, 10, 30, 20, 40};
    size_t inserted = matryoshka_insert_batch(t, keys, 5);
    ASSERT(inserted == 5, "wrong insert count");
    ASSERT(matryoshka_size(t) == 5, "wrong size");
    for (int i = 0; i < 5; i++)
        ASSERT(matryoshka_contains(t, keys[i]), "key not found");
    matryoshka_destroy(t);
    PASS();
}

static void test_batch_insert_duplicates(void)
{
    TEST(batch_insert_with_duplicates);
    matryoshka_tree_t *t = matryoshka_create();
    matryoshka_insert(t, 10);
    int32_t keys[] = {10, 20, 20, 30};
    size_t inserted = matryoshka_insert_batch(t, keys, 4);
    ASSERT(inserted == 2, "should insert 20,30 only");
    ASSERT(matryoshka_size(t) == 3, "wrong size");
    matryoshka_destroy(t);
    PASS();
}

static void test_batch_insert_splits(void)
{
    TEST(batch_insert_triggers_splits);
    matryoshka_tree_t *t = matryoshka_create();
    int32_t keys[5000];
    for (int i = 0; i < 5000; i++) keys[i] = i * 2;
    size_t inserted = matryoshka_insert_batch(t, keys, 5000);
    ASSERT(inserted == 5000, "wrong insert count");
    ASSERT(matryoshka_size(t) == 5000, "wrong size");
    for (int i = 0; i < 5000; i++)
        ASSERT(matryoshka_contains(t, i * 2), "key not found");
    matryoshka_destroy(t);
    PASS();
}

static void test_batch_insert_into_existing(void)
{
    TEST(batch_insert_into_existing_tree);
    int32_t initial[1000];
    for (int i = 0; i < 1000; i++) initial[i] = i * 4;
    matryoshka_tree_t *t = matryoshka_bulk_load(initial, 1000);
    int32_t batch[1000];
    for (int i = 0; i < 1000; i++) batch[i] = i * 4 + 2;
    size_t inserted = matryoshka_insert_batch(t, batch, 1000);
    ASSERT(inserted == 1000, "wrong count");
    ASSERT(matryoshka_size(t) == 2000, "wrong size");
    matryoshka_iter_t *it = matryoshka_iter_from(t, INT32_MIN);
    int count = 0;
    int32_t key, prev = INT32_MIN;
    while (matryoshka_iter_next(it, &key)) {
        ASSERT(key > prev, "not strictly increasing");
        prev = key; count++;
    }
    ASSERT(count == 2000, "iteration count wrong");
    matryoshka_iter_destroy(it);
    matryoshka_destroy(t);
    PASS();
}

static void test_batch_delete_basic(void)
{
    TEST(batch_delete_basic);
    matryoshka_tree_t *t = matryoshka_create();
    for (int i = 0; i < 100; i++) matryoshka_insert(t, i);
    int32_t to_delete[] = {10, 50, 99, 0, 75};
    size_t deleted = matryoshka_delete_batch(t, to_delete, 5);
    ASSERT(deleted == 5, "wrong delete count");
    ASSERT(matryoshka_size(t) == 95, "wrong size");
    for (int i = 0; i < 5; i++)
        ASSERT(!matryoshka_contains(t, to_delete[i]), "deleted key found");
    matryoshka_destroy(t);
    PASS();
}

static void test_batch_delete_heavy(void)
{
    TEST(batch_delete_heavy_3000);
    int32_t all[5000];
    for (int i = 0; i < 5000; i++) all[i] = i;
    matryoshka_tree_t *t = matryoshka_bulk_load(all, 5000);

    int32_t to_delete[3000];
    for (int i = 0; i < 3000; i++) to_delete[i] = i * 2 + 1; /* odd numbers < 6000 */
    size_t deleted = matryoshka_delete_batch(t, to_delete, 3000);
    /* Only odd numbers 1,3,...,4999 exist = 2500 */
    ASSERT(deleted == 2500, "wrong delete count");
    ASSERT(matryoshka_size(t) == 2500, "wrong size");

    /* All even numbers should remain. */
    for (int i = 0; i < 5000; i += 2)
        ASSERT(matryoshka_contains(t, i), "even key missing");

    matryoshka_destroy(t);
    PASS();
}

/* ── Superpage tests ──────────────────────────────────────────── */

static void test_sp_create_insert(void)
{
    TEST(superpage_create_insert);
    mt_hierarchy_t h;
    mt_hierarchy_init_superpage(&h);
    matryoshka_tree_t *t = matryoshka_create_with(&h);
    ASSERT(t != NULL, "create failed");
    ASSERT(matryoshka_size(t) == 0, "not empty");

    for (int i = 0; i < 1000; i++)
        ASSERT(matryoshka_insert(t, i * 3), "insert failed");
    ASSERT(matryoshka_size(t) == 1000, "wrong size");
    for (int i = 0; i < 1000; i++)
        ASSERT(matryoshka_contains(t, i * 3), "key not found");
    ASSERT(!matryoshka_contains(t, 1), "phantom key");
    matryoshka_destroy(t);
    PASS();
}

static void test_sp_bulk_load(void)
{
    TEST(superpage_bulk_load_10000);
    mt_hierarchy_t h;
    mt_hierarchy_init_superpage(&h);
    int32_t *keys = malloc(10000 * sizeof(int32_t));
    for (int i = 0; i < 10000; i++) keys[i] = i;
    matryoshka_tree_t *t = matryoshka_bulk_load_with(keys, 10000, &h);
    ASSERT(t != NULL, "bulk_load failed");
    ASSERT(matryoshka_size(t) == 10000, "wrong size");
    for (int i = 0; i < 10000; i++)
        ASSERT(matryoshka_contains(t, i), "key not found");
    matryoshka_destroy(t);
    free(keys);
    PASS();
}

static void test_sp_page_split(void)
{
    TEST(superpage_page_split);
    mt_hierarchy_t h;
    mt_hierarchy_init_superpage(&h);
    matryoshka_tree_t *t = matryoshka_create_with(&h);

    /* Insert enough keys to force page splits within the superpage.
       Each page holds ~855 keys; insert 5000 to use ~6 pages. */
    for (int i = 0; i < 5000; i++)
        ASSERT(matryoshka_insert(t, i), "insert failed");
    ASSERT(matryoshka_size(t) == 5000, "wrong size");

    /* Verify all keys via iteration. */
    matryoshka_iter_t *it = matryoshka_iter_from(t, INT32_MIN);
    int count = 0;
    int32_t key, prev = INT32_MIN;
    while (matryoshka_iter_next(it, &key)) {
        ASSERT(key > prev, "not strictly increasing");
        prev = key; count++;
    }
    ASSERT(count == 5000, "iteration count wrong");
    matryoshka_iter_destroy(it);
    matryoshka_destroy(t);
    PASS();
}

static void test_sp_delete(void)
{
    TEST(superpage_delete);
    mt_hierarchy_t h;
    mt_hierarchy_init_superpage(&h);
    int32_t keys[2000];
    for (int i = 0; i < 2000; i++) keys[i] = i;
    matryoshka_tree_t *t = matryoshka_bulk_load_with(keys, 2000, &h);

    /* Delete odd keys. */
    for (int i = 1; i < 2000; i += 2)
        ASSERT(matryoshka_delete(t, i), "delete failed");
    ASSERT(matryoshka_size(t) == 1000, "wrong size");

    /* Even keys should remain. */
    for (int i = 0; i < 2000; i += 2)
        ASSERT(matryoshka_contains(t, i), "even key missing");
    for (int i = 1; i < 2000; i += 2)
        ASSERT(!matryoshka_contains(t, i), "deleted odd key found");

    matryoshka_destroy(t);
    PASS();
}

static void test_sp_iterator(void)
{
    TEST(superpage_iterator);
    mt_hierarchy_t h;
    mt_hierarchy_init_superpage(&h);
    int32_t keys[3000];
    for (int i = 0; i < 3000; i++) keys[i] = i * 2;
    matryoshka_tree_t *t = matryoshka_bulk_load_with(keys, 3000, &h);

    /* Iterate from midpoint. */
    matryoshka_iter_t *it = matryoshka_iter_from(t, 3000);
    int count = 0;
    int32_t key;
    while (matryoshka_iter_next(it, &key)) count++;
    ASSERT(count == 1500, "wrong count from midpoint");
    matryoshka_iter_destroy(it);
    matryoshka_destroy(t);
    PASS();
}

static void test_sp_predecessor_search(void)
{
    TEST(superpage_predecessor_search);
    mt_hierarchy_t h;
    mt_hierarchy_init_superpage(&h);
    int32_t keys[100];
    for (int i = 0; i < 100; i++) keys[i] = i * 10;
    matryoshka_tree_t *t = matryoshka_bulk_load_with(keys, 100, &h);

    int32_t result;
    ASSERT(matryoshka_search(t, 55, &result) && result == 50,
           "predecessor of 55 should be 50");
    ASSERT(matryoshka_search(t, 990, &result) && result == 990,
           "predecessor of 990 should be 990");
    ASSERT(!matryoshka_search(t, -1, &result),
           "no predecessor below 0");

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
    test_delete_heavy();
    test_delete_all();
    test_delete_cascading();
    test_delete_interleaved();
    test_create_with_default();
    test_bulk_load_with_default();
    test_hierarchy_superpage();
    test_hierarchy_custom();
    test_page_subtree_capacity();
    test_arena_basic();
    test_arena_co_location();
    test_batch_insert_basic();
    test_batch_insert_duplicates();
    test_batch_insert_splits();
    test_batch_insert_into_existing();
    test_batch_delete_basic();
    test_batch_delete_heavy();
    test_sp_create_insert();
    test_sp_bulk_load();
    test_sp_page_split();
    test_sp_delete();
    test_sp_iterator();
    test_sp_predecessor_search();

    printf("\n  %d/%d tests passed.\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
