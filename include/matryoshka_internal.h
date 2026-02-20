/*
 * matryoshka_internal.h — Internal data structures for the matryoshka tree.
 *
 * Node layout (matryoshka nesting):
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
 *
 *   Leaf node (4 KiB page): Matryoshka-nested B+ sub-tree
 *     ┌───────────────────────────────────────────────┐
 *     │ slot 0: page header (type, nkeys, bitmap, ...)│  64 B
 *     ├───────────────────────────────────────────────┤
 *     │ slots 1–63: cache-line-sized sub-nodes         │  63 × 64 B
 *     │   ├─ CL leaf:  sorted int32 keys (up to 15)   │
 *     │   └─ CL inode: separator keys + child slots    │
 *     └───────────────────────────────────────────────┘
 *     Within the page, a B+ tree of CL sub-nodes.
 *     Insert/delete operate on individual CL leaves: O(log b).
 */

#ifndef MATRYOSHKA_INTERNAL_H
#define MATRYOSHKA_INTERNAL_H

#include "matryoshka.h"
#include <immintrin.h>

/* ── Compile-time constants ─────────────────────────────────── */

#define MT_PAGE_SIZE       4096
#define MT_CL_SIZE         64       /* Cache-line size in bytes */

/* Maximum hierarchy levels */
#define MT_MAX_LEVELS      8

/* ── Cache-line sub-node capacities ─────────────────────────── */

/* CL leaf: 4 B header + 15 × 4 B keys = 64 B */
#define MT_CL_KEY_CAP      15
#define MT_CL_MIN_KEYS     7       /* floor(15/2) */

/* CL internal: 2 B header + 13 B children + 1 B pad + 12 × 4 B keys = 64 B */
#define MT_CL_SEP_CAP      12
#define MT_CL_CHILD_CAP    13
#define MT_CL_MIN_CHILDREN 7       /* ceil(13/2) */

/* Page: 64 CL slots.  Slot 0 = header; slots 1–63 usable. */
#define MT_PAGE_SLOTS      63

/* Internal node capacity (outer B+ tree).
   Header = 16 B.  Per key: 4 B key + 8 B pointer = 12 B, plus 1 extra ptr.
   (4096 - 16 - 8) / 12 = 339 keys, 340 children. */
#define MT_INODE_HEADER    16
#define MT_MAX_IKEYS       339
#define MT_MIN_IKEYS       ((MT_MAX_IKEYS) / 2)

/* ── Hierarchy configuration ───────────────────────────────── */

typedef struct mt_hierarchy {
    size_t  leaf_alloc;       /* Allocation size for leaf pages (4096 or 2 MiB) */
    int     cl_key_cap;       /* Keys per CL leaf (15) */
    int     cl_sep_cap;       /* Separator keys per CL internal (12) */
    int     cl_child_cap;     /* Children per CL internal (13) */
    int     page_slots;       /* Usable CL slots per page (63) */
    int     page_max_keys;    /* Max keys per page (depends on sub-tree height) */
    int     min_page_keys;    /* Minimum leaf occupancy for outer B+ tree */
    int     min_cl_keys;      /* Minimum CL leaf occupancy (7) */
    int     min_cl_children;  /* Minimum CL internal children (7) */
} mt_hierarchy_t;

/* ── Arena allocator types ──────────────────────────────────── */

typedef struct mt_arena {
    void            *base;
    size_t           size;
    size_t           page_size;
    int              num_pages;
    bool             is_mmap;
    uint64_t        *bitmap;
    struct mt_arena *next;
} mt_arena_t;

typedef struct mt_allocator {
    mt_arena_t *arenas;
    size_t      arena_size;
    size_t      page_size;
} mt_allocator_t;

#define MT_KEY_MAX         INT32_MAX

/* ── Node types ─────────────────────────────────────────────── */

/* Outer B+ tree node types (at offset 0 of a 4 KiB page). */
typedef enum {
    MT_NODE_INTERNAL = 0,
    MT_NODE_LEAF     = 1,
} mt_node_type_t;

/* CL sub-node types (at offset 0 of a 64 B slot within a leaf page). */
typedef enum {
    MT_CL_FREE     = 0,
    MT_CL_LEAF     = 1,
    MT_CL_INTERNAL = 2,
} mt_cl_type_t;

/* ── CL sub-node structures (64 B each) ────────────────────── */

/* Cache-line leaf: sorted array of up to 15 int32_t keys. */
typedef struct mt_cl_leaf {
    uint8_t  type;            /* MT_CL_LEAF */
    uint8_t  nkeys;           /* 0–15 */
    uint8_t  _pad[2];
    int32_t  keys[MT_CL_KEY_CAP]; /* Sorted keys */
} mt_cl_leaf_t;

_Static_assert(sizeof(mt_cl_leaf_t) == MT_CL_SIZE,
               "mt_cl_leaf_t must be exactly 64 bytes");

/* Cache-line internal: separator keys + child slot indices. */
typedef struct mt_cl_inode {
    uint8_t  type;                     /* MT_CL_INTERNAL */
    uint8_t  nkeys;                    /* 0–12 separator keys */
    uint8_t  children[MT_CL_CHILD_CAP]; /* Slot indices (1–63) */
    uint8_t  _pad;
    int32_t  keys[MT_CL_SEP_CAP];     /* Separator keys */
} mt_cl_inode_t;

_Static_assert(sizeof(mt_cl_inode_t) == MT_CL_SIZE,
               "mt_cl_inode_t must be exactly 64 bytes");

/* Generic CL slot — tagged union at offset 0. */
typedef union mt_cl_slot {
    uint8_t        type;       /* Discriminant: MT_CL_FREE/LEAF/INTERNAL */
    mt_cl_leaf_t   leaf;
    mt_cl_inode_t  inode;
    uint8_t        raw[MT_CL_SIZE];
} mt_cl_slot_t;

_Static_assert(sizeof(mt_cl_slot_t) == MT_CL_SIZE,
               "mt_cl_slot_t must be exactly 64 bytes");

/* ── Page header (slot 0 of a leaf page) ────────────────────── */

typedef struct mt_page_header {
    uint16_t         type;          /* MT_NODE_LEAF (outer tree perspective) */
    uint16_t         nkeys;         /* Total keys in this page */
    uint8_t          root_slot;     /* CL slot index of sub-tree root (1–63) */
    uint8_t          sub_height;    /* Sub-tree height (0 = single CL leaf) */
    uint8_t          nslots_used;   /* Number of CL slots allocated */
    uint8_t          _pad;
    uint64_t         slot_bitmap;   /* Bits 1–63: CL slot allocation */
    struct mt_lnode *prev;          /* Previous leaf (outer tree linked list) */
    struct mt_lnode *next;          /* Next leaf (outer tree linked list) */
    uint8_t          _reserved[32]; /* Pad to 64 B */
} mt_page_header_t;

_Static_assert(sizeof(mt_page_header_t) == MT_CL_SIZE,
               "mt_page_header_t must be exactly 64 bytes");

/* ── Leaf node (4 KiB page with matryoshka-nested sub-tree) ── */

/* Forward declaration for pointer usage. */
struct mt_lnode;

typedef struct mt_lnode {
    mt_page_header_t  header;               /* Slot 0: page header */
    mt_cl_slot_t      slots[MT_PAGE_SLOTS]; /* Slots 1–63: CL sub-nodes */
} mt_lnode_t;

_Static_assert(sizeof(mt_lnode_t) == MT_PAGE_SIZE,
               "mt_lnode_t must be exactly 4096 bytes");

/* ── Outer B+ tree nodes ────────────────────────────────────── */

/* Forward declaration so struct members can hold pointers. */
union mt_node;

/* Internal node: sorted keys + child pointers.
   Fits in one 4 KiB page. */
typedef struct mt_inode {
    /* Header (16 bytes) */
    uint16_t        type;           /* MT_NODE_INTERNAL */
    uint16_t        nkeys;
    uint32_t        _pad;
    uint64_t        _reserved;

    /* Sorted key array. */
    int32_t         keys[MT_MAX_IKEYS];

    /* Child pointers. */
    union mt_node  *children[MT_MAX_IKEYS + 1];
} mt_inode_t;

/* Generic node pointer (tagged by type field at offset 0). */
typedef union mt_node {
    mt_node_type_t type;
    mt_inode_t     inode;
    mt_lnode_t     lnode;
} mt_node_t;

/* ── Tree root ──────────────────────────────────────────────── */

struct matryoshka_tree {
    mt_node_t      *root;
    size_t          n;            /* Total number of keys */
    int             height;       /* Outer tree height (0 = single leaf) */
    mt_hierarchy_t  hier;
    mt_allocator_t *alloc;        /* Arena allocator for leaf nodes */
};

/* ── Iterator ───────────────────────────────────────────────── */

struct matryoshka_iter {
    const matryoshka_tree_t *tree;
    mt_lnode_t              *leaf;    /* Current leaf page */
    int                      pos;     /* Position within extracted sorted keys */
    int                      nkeys;   /* Number of keys in current leaf */
    int32_t                 *sorted;  /* Heap-allocated sorted keys buffer */
};

/* ── Hierarchy factory functions ───────────────────────────── */

void mt_hierarchy_init_default(mt_hierarchy_t *h);
void mt_hierarchy_init_superpage(mt_hierarchy_t *h);
void mt_hierarchy_init_custom(mt_hierarchy_t *h, size_t leaf_alloc);

/* ── Page sub-tree operations (leaf.c) ─────────────────────── */

/* Status codes for page-level operations. */
typedef enum {
    MT_OK        = 0,   /* Success */
    MT_DUPLICATE = 1,   /* Key already exists (insert) */
    MT_NOT_FOUND = 2,   /* Key not found (delete) */
    MT_PAGE_FULL = 3,   /* Page has no free CL slots (insert) */
    MT_UNDERFLOW = 4,   /* Page fell below minimum fill (delete) */
} mt_status_t;

/* Initialise an empty leaf page (one empty CL leaf). */
void mt_page_init(mt_lnode_t *page);

/* Search for predecessor of `key` within a leaf page.
   Returns the sorted index (0-based), or -1 if no predecessor. */
int mt_page_search(const mt_lnode_t *page, int32_t key);

/* Search for predecessor, writing the result key to *result.
   Returns true if found. */
bool mt_page_search_key(const mt_lnode_t *page, int32_t key, int32_t *result);

/* Insert a key into a leaf page.
   Returns MT_OK, MT_DUPLICATE, or MT_PAGE_FULL. */
mt_status_t mt_page_insert(mt_lnode_t *page, int32_t key);

/* Delete a key from a leaf page.
   Returns MT_OK, MT_NOT_FOUND, or MT_UNDERFLOW. */
mt_status_t mt_page_delete(mt_lnode_t *page, int32_t key,
                            const mt_hierarchy_t *hier);

/* Extract all keys from a leaf page in sorted order.
   `out` must have room for page->header.nkeys elements.
   Returns the number of keys extracted. */
int mt_page_extract_sorted(const mt_lnode_t *page, int32_t *out);

/* Bulk-load sorted keys into an empty page.  O(n). */
void mt_page_bulk_load(mt_lnode_t *page, const int32_t *sorted_keys, int nkeys);

/* Split a page: move approximately half of the keys to `new_page`.
   Returns the separator key (first key of new_page). */
int32_t mt_page_split(mt_lnode_t *page, mt_lnode_t *new_page);

/* Return the minimum (first) key in a page. */
int32_t mt_page_min_key(const mt_lnode_t *page);

/* Check if a key exists in the page. */
bool mt_page_contains(const mt_lnode_t *page, int32_t key);

/* ── Internal node search (inode.c) ───────────────────────── */

int mt_inode_search(const mt_inode_t *node, int32_t key);

/* ── Node allocation (alloc.c) ─────────────────────────────── */

mt_node_t *mt_alloc_inode(void);
mt_node_t *mt_alloc_lnode(const mt_hierarchy_t *hier, mt_allocator_t *alloc);
void mt_free_inode(mt_node_t *node);
void mt_free_lnode(mt_node_t *node, mt_allocator_t *alloc);

/* ── Arena allocator (arena.c) ─────────────────────────────── */

mt_allocator_t *mt_allocator_create(size_t arena_size, size_t page_size);
void            mt_allocator_destroy(mt_allocator_t *alloc);
void           *mt_allocator_alloc(mt_allocator_t *alloc);
void            mt_allocator_free(mt_allocator_t *alloc, void *ptr);

#endif /* MATRYOSHKA_INTERNAL_H */
