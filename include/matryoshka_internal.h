/*
 * matryoshka_internal.h — Internal data structures for the matryoshka tree.
 *
 * Node layout:
 *
 *   Internal node (4 KiB page):
 *     ┌───────────────────────────────────────────────┐
 *     │ header: node_type, nkeys, padding              │  16 B
 *     ├───────────────────────────────────────────────┤
 *     │ keys[MAX_IKEYS]: sorted int32 array            │  ≤1360 B
 *     ├───────────────────────────────────────────────┤
 *     │ children[MAX_IKEYS+1]: child page pointers     │  ≤2728 B
 *     └───────────────────────────────────────────────┘
 *     Search: SIMD-accelerated binary search on sorted keys[].
 *     4k + 8(k+1) + 16 ≤ 4096  →  k ≤ 339
 *
 *   Leaf node (4 KiB page):
 *     ┌───────────────────────────────────────────────┐
 *     │ header: node_type, nkeys, prev, next           │  32 B
 *     ├───────────────────────────────────────────────┤
 *     │ keys[MAX_LKEYS]: FAST blocked layout           │  ≤2016 B
 *     ├───────────────────────────────────────────────┤
 *     │ sorted_rank[MAX_LKEYS]: layout→sorted mapping  │  ≤2016 B
 *     └───────────────────────────────────────────────┘
 *     Search: full FAST hierarchical SIMD blocking.
 *     2 * 4k + 32 ≤ 4096  →  k ≤ 508
 *     (With tree padding to power of 2, effective capacity ~500 keys.)
 */

#ifndef MATRYOSHKA_INTERNAL_H
#define MATRYOSHKA_INTERNAL_H

#include "matryoshka.h"
#include <immintrin.h>

/* ── Compile-time constants ─────────────────────────────────── */

#define MT_PAGE_SIZE       4096

/* FAST blocking depths (same as FAST) */
#define MT_DK              2       /* SIMD block depth */
#define MT_NK              3       /* Keys per SIMD block (2^DK - 1) */
#define MT_DL              4       /* Cache-line block depth */
#define MT_NL              15      /* Keys per cache-line block */

/* Internal node capacity.
   Header = 16 B.  Per key: 4 B key + 8 B pointer = 12 B, plus 1 extra ptr.
   (4096 - 16 - 8) / 12 = 339 keys, 340 children. */
#define MT_INODE_HEADER    16
#define MT_MAX_IKEYS       339
#define MT_MIN_IKEYS       ((MT_MAX_IKEYS) / 2)   /* B+ tree min fill */

/* Leaf node capacity.
   Header = 24 B (type, nkeys, tree_depth, pad, prev, next).
   Tree arrays: layout[512] (int32_t) + sorted_rank[512] (int16_t).
   24 + 512*4 + 512*2 = 3096 ≤ 4096.
   Max tree depth = 9, tree_nodes = 511, max actual keys = 511. */
#define MT_LNODE_TREE_CAP  512      /* Array capacity (≥ max tree nodes) */
#define MT_MAX_LKEYS       511
#define MT_MIN_LKEYS       ((MT_MAX_LKEYS) / 2)

/* SIMD lookup table (same as FAST). */
static const int MT_SIMD_LOOKUP[16] = {
    0, -1, 1, 2, -1, -1, -1, 3,
    0, -1, 1, 2, -1, -1, -1, 3
};

#define MT_KEY_MAX         INT32_MAX

/* ── Node types ─────────────────────────────────────────────── */

typedef enum {
    MT_NODE_INTERNAL = 0,
    MT_NODE_LEAF     = 1,
} mt_node_type_t;

/* Forward declaration so struct members can hold pointers. */
union mt_node;

/* Internal node: sorted keys + child pointers.
   Fits in one 4 KiB page. */
typedef struct mt_inode {
    /* Header (16 bytes) */
    uint16_t        type;           /* MT_NODE_INTERNAL */
    uint16_t        nkeys;          /* Number of keys currently stored */
    uint32_t        _pad;
    uint64_t        _reserved;

    /* Sorted key array.  Searched via SIMD-accelerated binary search. */
    int32_t         keys[MT_MAX_IKEYS];

    /* Child pointers.  children[i] is the subtree for keys < keys[i];
       children[nkeys] is the subtree for keys >= keys[nkeys-1]. */
    union mt_node  *children[MT_MAX_IKEYS + 1];
} mt_inode_t;

/* Leaf node: FAST-blocked keys + sorted_rank + sibling pointers.
   Fits in one 4 KiB page (3096 B actual). */
typedef struct mt_lnode {
    /* Header (24 bytes) */
    uint16_t         type;          /* MT_NODE_LEAF */
    uint16_t         nkeys;         /* Number of keys stored */
    uint16_t         tree_depth;    /* Depth of FAST tree within this leaf */
    uint16_t         _pad;
    struct mt_lnode *prev;          /* Previous leaf (for range scan) */
    struct mt_lnode *next;          /* Next leaf (for range scan) */

    /* FAST hierarchically blocked key layout. */
    int32_t          layout[MT_LNODE_TREE_CAP];

    /* sorted_rank[i] = index in logical sorted order for layout[i].
       int16_t suffices (max value = 511). */
    int16_t          sorted_rank[MT_LNODE_TREE_CAP];
} mt_lnode_t;

/* Generic node pointer (tagged by type field at offset 0). */
typedef union mt_node {
    mt_node_type_t type;
    mt_inode_t     inode;
    mt_lnode_t     lnode;
} mt_node_t;

/* ── Tree root ──────────────────────────────────────────────── */

struct matryoshka_tree {
    mt_node_t *root;
    size_t     n;            /* Total number of keys */
    int        height;       /* Tree height (0 = single leaf) */
};

/* ── Iterator ───────────────────────────────────────────────── */

struct matryoshka_iter {
    const matryoshka_tree_t *tree;
    mt_lnode_t              *leaf;    /* Current leaf */
    int                      pos;     /* Position within leaf's sorted keys */
    int                      nkeys;   /* Number of keys in current leaf */
    int32_t                  sorted[MT_MAX_LKEYS]; /* Sorted keys extracted from leaf */
};

/* ── Internal functions (implemented in separate .c files) ──── */

/* Leaf FAST layout: build blocked layout from sorted keys within a leaf. */
void mt_leaf_build(mt_lnode_t *leaf, const int32_t *sorted_keys, int nkeys);

/* Extract sorted keys from a leaf's FAST blocked layout into `out`.
   `out` must have room for at least leaf->nkeys elements. */
void mt_leaf_extract_sorted(const mt_lnode_t *leaf, int32_t *out);

/* Leaf FAST search: predecessor search within a single leaf.
   Returns sorted index within the leaf, or -1. */
int mt_leaf_search(const mt_lnode_t *leaf, int32_t key);

/* Internal node search: find child index for the given key.
   Returns i such that children[i] should be followed. */
int mt_inode_search(const mt_inode_t *node, int32_t key);

/* Node allocation (page-aligned). */
mt_node_t *mt_alloc_inode(void);
mt_node_t *mt_alloc_lnode(void);
void mt_free_node(mt_node_t *node);

#endif /* MATRYOSHKA_INTERNAL_H */
