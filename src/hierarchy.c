/*
 * hierarchy.c — Blocking hierarchy configuration for matryoshka trees.
 *
 * Factory functions that produce mt_hierarchy_t configurations mapping
 * the FAST blocking scheme to different memory hierarchy levels.
 */

#include "matryoshka_internal.h"

void mt_hierarchy_init_default(mt_hierarchy_t *h)
{
    /* x86-64 default: SIMD (d=2, 16B) + cache-line (d=4, 64B).
       Leaves are 4 KiB pages.  Tree depth = 9, capacity = 511 keys.
       With int16_t sorted_rank: 24 + 512*4 + 512*2 = 3096 ≤ 4096. */
    h->num_levels = 2;
    h->levels[0] = (mt_level_t){ .depth = MT_DK, .hw_size = 16 };
    h->levels[1] = (mt_level_t){ .depth = MT_DL, .hw_size = 64 };
    h->leaf_alloc = MT_PAGE_SIZE;
    h->leaf_depth = 9;
    h->leaf_cap   = 511;     /* 2^9 - 1 */
    h->tree_cap   = 512;     /* 2^9 */
    h->min_lkeys  = 511 / 2;
    h->rank_wide  = false;
}

void mt_hierarchy_init_superpage(mt_hierarchy_t *h)
{
    /* x86-64 superpage: SIMD (d=2) + CL (d=4) + page (d=10).
       Leaves are 2 MiB superpages.  Tree depth = 18, capacity = 262143.
       With int32_t sorted_rank: header + N*4 + N*4 ≤ 2 MiB. */
    h->num_levels = 3;
    h->levels[0] = (mt_level_t){ .depth = 2,  .hw_size = 16 };
    h->levels[1] = (mt_level_t){ .depth = 4,  .hw_size = 64 };
    h->levels[2] = (mt_level_t){ .depth = 10, .hw_size = 4096 };
    h->leaf_alloc = 2 * 1024 * 1024;  /* 2 MiB */
    h->leaf_depth = 18;
    h->leaf_cap   = (1 << 18) - 1;    /* 262143 */
    h->tree_cap   = (1 << 18);        /* 262144 */
    h->min_lkeys  = ((1 << 18) - 1) / 2;
    h->rank_wide  = true;             /* >32767 keys need int32_t rank */
}

void mt_hierarchy_init_custom(mt_hierarchy_t *h, const mt_level_t *levels,
                               int num_levels, size_t leaf_alloc)
{
    h->num_levels = (num_levels > MT_MAX_LEVELS) ? MT_MAX_LEVELS : num_levels;
    for (int i = 0; i < h->num_levels; i++)
        h->levels[i] = levels[i];

    h->leaf_alloc = leaf_alloc;

    /* leaf_depth: find the largest tree depth that fits in leaf_alloc.
       Header = 24 B.  Per node: 4 B (int32_t key) + rank_bytes.
       rank_bytes = 2 for int16_t (leaf_cap ≤ 32767) or 4 for int32_t.
       tree_nodes = 2^d - 1.  header + tree_nodes * bytes_per ≤ leaf_alloc. */
    size_t header = 24;
    size_t avail = leaf_alloc - header;

    /* First try with int16_t rank (6 bytes/node). */
    int bytes_per = 6;
    size_t max_nodes = avail / (size_t)bytes_per;

    int d = 0;
    while (((size_t)1 << (d + 1)) - 1 <= max_nodes)
        d++;

    int cap = (1 << d) - 1;

    if (cap > 32767) {
        /* Need int32_t rank.  Recompute with 8 bytes/node. */
        bytes_per = 8;
        max_nodes = avail / (size_t)bytes_per;
        d = 0;
        while (((size_t)1 << (d + 1)) - 1 <= max_nodes)
            d++;
        cap = (1 << d) - 1;
        h->rank_wide = true;
    } else {
        h->rank_wide = false;
    }

    h->leaf_depth = d;
    h->leaf_cap   = cap;
    h->tree_cap   = (1 << d);
    h->min_lkeys  = cap / 2;
}
