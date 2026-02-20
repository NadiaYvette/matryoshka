/*
 * hierarchy.c — Blocking hierarchy configuration for matryoshka trees.
 *
 * Defines the nesting parameters: cache-line sub-node capacities and
 * page-level capacity limits for the matryoshka nested B+ tree.
 */

#include "matryoshka_internal.h"

/*
 * Compute page_max_keys: the maximum number of keys that can fit in
 * `page_slots` CL slots arranged as a B+ sub-tree.
 *
 * At sub-height 0: 1 CL leaf = 15 keys.  Slots = 1.
 * At sub-height 1: 1 CL internal root + up to 13 CL leaves = 14 slots.
 *   Max keys = 13 × 15 = 195.
 * At sub-height 2: 1 root + m internals + n leaves.
 *   Constraints: m ≤ cl_child_cap, n ≤ m × cl_child_cap, 1+m+n ≤ page_slots.
 *   Optimal: m=5, n=57 → 855 keys in 63 slots.
 *
 * We enumerate sub-height 2 configurations to find the optimum.
 */
static int compute_page_max_keys(int cl_key_cap, int cl_child_cap,
                                  int page_slots)
{
    /* Sub-height 0: single CL leaf. */
    int best = cl_key_cap;

    /* Sub-height 1: 1 root internal + n leaves, n ≤ cl_child_cap. */
    int h1_leaves = cl_child_cap;
    if (1 + h1_leaves > page_slots) h1_leaves = page_slots - 1;
    if (h1_leaves * cl_key_cap > best) best = h1_leaves * cl_key_cap;

    /* Sub-height 2: 1 root + m internals + n leaves.
       m ≤ cl_child_cap, n ≤ m × cl_child_cap, 1 + m + n ≤ page_slots. */
    for (int m = 1; m <= cl_child_cap && 1 + m < page_slots; m++) {
        int max_n = page_slots - 1 - m;
        if (max_n > m * cl_child_cap) max_n = m * cl_child_cap;
        int keys = max_n * cl_key_cap;
        if (keys > best) best = keys;
    }

    return best;
}

void mt_hierarchy_init_default(mt_hierarchy_t *h)
{
    h->leaf_alloc      = MT_PAGE_SIZE;
    h->cl_key_cap      = MT_CL_KEY_CAP;
    h->cl_sep_cap      = MT_CL_SEP_CAP;
    h->cl_child_cap    = MT_CL_CHILD_CAP;
    h->page_slots      = MT_PAGE_SLOTS;
    h->min_cl_keys     = MT_CL_MIN_KEYS;
    h->min_cl_children = MT_CL_MIN_CHILDREN;
    h->page_max_keys   = compute_page_max_keys(MT_CL_KEY_CAP,
                                                MT_CL_CHILD_CAP,
                                                MT_PAGE_SLOTS);
    h->min_page_keys   = h->page_max_keys / 4;  /* looser min for page splits */
}

void mt_hierarchy_init_superpage(mt_hierarchy_t *h)
{
    /* Superpage leaves: same CL-level structure but within 2 MiB pages.
       The outer tree treats each 2 MiB superpage as a single leaf.
       Within the superpage, we use a two-level nesting:
       4 KiB page sub-nodes → CL sub-nodes → keys.
       For now, use the same single-level CL nesting. */
    mt_hierarchy_init_default(h);
    h->leaf_alloc = 2 * 1024 * 1024;
    /* With 2 MiB / 64 B = 32768 CL slots, capacity is enormous.
       page_slots is still 63 for now (TODO: multi-page nesting). */
}

void mt_hierarchy_init_custom(mt_hierarchy_t *h, size_t leaf_alloc)
{
    mt_hierarchy_init_default(h);
    h->leaf_alloc = leaf_alloc;
}
