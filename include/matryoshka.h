/*
 * matryoshka.h — Matryoshka Tree: B+ tree with nested sub-tree nodes
 *
 * A cache-conscious B+ tree where each leaf node contains a B+ sub-tree
 * of cache-line-sized sub-nodes.  This "matryoshka" nesting maps the
 * data structure to the memory hierarchy:
 *
 *   Level 0: SSE2 register (16 B)  — SIMD search within sub-nodes
 *   Level 1: Cache line (64 B)     — CL sub-node (15 keys or 12 seps)
 *   Level 2: Page (4 KiB)          — B+ tree of CL sub-nodes
 *   Level 3: Superpage (2 MiB)     — future: B+ tree of page sub-nodes
 *   Level 4: Main memory           — outer B+ tree
 *
 * Operations within a leaf page modify only the affected CL sub-nodes,
 * giving O(log b) insert/delete per node instead of O(B) flat-array
 * rebuilds, where b is the CL branching factor.
 */

#ifndef MATRYOSHKA_H
#define MATRYOSHKA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque tree handle. */
typedef struct matryoshka_tree matryoshka_tree_t;

/* Forward declare hierarchy types for the _with variants.
   Full definitions are in matryoshka_internal.h. */
typedef struct mt_hierarchy mt_hierarchy_t;

/* ── Lifecycle ──────────────────────────────────────────────── */

/* Create an empty tree (default x86-64 hierarchy). */
matryoshka_tree_t *matryoshka_create(void);

/* Create an empty tree with a specific hierarchy configuration. */
matryoshka_tree_t *matryoshka_create_with(const mt_hierarchy_t *hier);

/* Create a tree bulk-loaded from sorted keys (default hierarchy).
   Keys must be in ascending order with no duplicates.  O(n) construction. */
matryoshka_tree_t *matryoshka_bulk_load(const int32_t *sorted_keys, size_t n);

/* Bulk-load with a specific hierarchy configuration. */
matryoshka_tree_t *matryoshka_bulk_load_with(const int32_t *sorted_keys,
                                              size_t n,
                                              const mt_hierarchy_t *hier);

/* Destroy a tree and free all associated memory. */
void matryoshka_destroy(matryoshka_tree_t *tree);

/* ── Query ──────────────────────────────────────────────────── */

/* Predecessor search: find the largest key <= query.
   If found, writes the key to *result and returns true.
   If no key <= query exists, returns false. */
bool matryoshka_search(const matryoshka_tree_t *tree, int32_t key,
                        int32_t *result);

/* Membership test. */
bool matryoshka_contains(const matryoshka_tree_t *tree, int32_t key);

/* Return the number of keys in the tree. */
size_t matryoshka_size(const matryoshka_tree_t *tree);

/* ── Modification ───────────────────────────────────────────── */

/* Insert a key.  Returns true if the key was inserted, false if it
   already existed.  O(log b · log_B n) where b is the CL branching
   factor and B is the page-level key capacity. */
bool matryoshka_insert(matryoshka_tree_t *tree, int32_t key);

/* Delete a key.  Returns true if the key was found and removed, false
   if it was not present.  O(log b · log_B n). */
bool matryoshka_delete(matryoshka_tree_t *tree, int32_t key);

/* ── Iteration ──────────────────────────────────────────────── */

/* Iterator for in-order traversal. */
typedef struct matryoshka_iter matryoshka_iter_t;

/* Create an iterator positioned at the first key >= `start`.
   Pass INT32_MIN for the beginning. */
matryoshka_iter_t *matryoshka_iter_from(const matryoshka_tree_t *tree,
                                         int32_t start);

/* Advance the iterator.  Returns true and writes *key if a key was
   available, false at end-of-tree. */
bool matryoshka_iter_next(matryoshka_iter_t *iter, int32_t *key);

/* Destroy an iterator. */
void matryoshka_iter_destroy(matryoshka_iter_t *iter);

#ifdef __cplusplus
}
#endif

#endif /* MATRYOSHKA_H */
