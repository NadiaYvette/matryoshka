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

/* C/C++ compatibility for static assertions. */
#ifdef __cplusplus
#define MT_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define MT_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

#ifdef __cplusplus
extern "C" {
#endif

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

/* Eytzinger CL internal: 4 B header + 15 × 4 B keys = 64 B (no children[]) */
#define MT_CL_EYTZ_SEP_CAP    15
#define MT_CL_EYTZ_CHILD_CAP  16

/* Fence keys embedded in page header (6 keys in 32 spare bytes). */
#define MT_FENCE_KEY_CAP   6
#define MT_FENCE_SLOT_CAP  7       /* MT_FENCE_KEY_CAP + 1 */

/* Page: 64 CL slots.  Slot 0 = header; slots 1–63 usable. */
#define MT_PAGE_SLOTS      63

/* Internal node capacity (outer B+ tree).
   Header = 16 B.  Per key: 4 B key + 8 B pointer = 12 B, plus 1 extra ptr.
   (4096 - 16 - 8) / 12 = 339 keys, 340 children. */
#define MT_INODE_HEADER    16
#define MT_MAX_IKEYS       339
#define MT_MIN_IKEYS       ((MT_MAX_IKEYS) / 2)

/* ── CL sub-tree strategy ──────────────────────────────────── */

typedef enum {
    MT_CL_STRAT_DEFAULT = 0,   /* Slot-indexed CL sub-tree (baseline) */
    MT_CL_STRAT_FENCE   = 1,   /* + fence keys in page header */
    MT_CL_STRAT_EYTZ    = 2,   /* Eytzinger dense BFS layout, height ≤ 1 */
} mt_cl_strategy_t;

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
    bool    use_superpages;   /* Whether leaves are 2 MiB superpages */
    int     sp_max_keys;      /* Max keys per superpage (~436K) */
    int     min_sp_keys;      /* Min superpage occupancy for outer tree */
    int     cl_strategy;      /* mt_cl_strategy_t: DEFAULT, FENCE, or EYTZ */
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

MT_STATIC_ASSERT(sizeof(mt_cl_leaf_t) == MT_CL_SIZE,
               "mt_cl_leaf_t must be exactly 64 bytes");

/* Cache-line internal: separator keys + child slot indices. */
typedef struct mt_cl_inode {
    uint8_t  type;                     /* MT_CL_INTERNAL */
    uint8_t  nkeys;                    /* 0–12 separator keys */
    uint8_t  children[MT_CL_CHILD_CAP]; /* Slot indices (1–63) */
    uint8_t  _pad;
    int32_t  keys[MT_CL_SEP_CAP];     /* Separator keys */
} mt_cl_inode_t;

MT_STATIC_ASSERT(sizeof(mt_cl_inode_t) == MT_CL_SIZE,
               "mt_cl_inode_t must be exactly 64 bytes");

/* Eytzinger CL internal: no children[] array, 15 separator keys.
   Children are at implicit BFS positions: root at slot R, children
   at slots R+1 .. R+nchildren.  */
typedef struct mt_cl_inode_eytz {
    uint8_t  type;                             /* MT_CL_INTERNAL */
    uint8_t  nkeys;                            /* 0–15 separator keys */
    uint8_t  nchildren;                        /* 1–16 children */
    uint8_t  _pad;
    int32_t  keys[MT_CL_EYTZ_SEP_CAP];        /* 15 × 4 B = 60 B */
} mt_cl_inode_eytz_t;

MT_STATIC_ASSERT(sizeof(mt_cl_inode_eytz_t) == MT_CL_SIZE,
               "mt_cl_inode_eytz_t must be exactly 64 bytes");

/* Generic CL slot — tagged union at offset 0. */
typedef union mt_cl_slot {
    uint8_t              type;       /* Discriminant: MT_CL_FREE/LEAF/INTERNAL */
    mt_cl_leaf_t         leaf;
    mt_cl_inode_t        inode;
    mt_cl_inode_eytz_t   inode_eytz;
    uint8_t              raw[MT_CL_SIZE];
} mt_cl_slot_t;

MT_STATIC_ASSERT(sizeof(mt_cl_slot_t) == MT_CL_SIZE,
               "mt_cl_slot_t must be exactly 64 bytes");

/* ── Superpage constants ────────────────────────────────────── */

#define MT_SP_SIZE         (2u * 1024 * 1024)   /* 2 MiB */
#define MT_SP_PAGES        (MT_SP_SIZE / MT_PAGE_SIZE)  /* 512 */

/* Page-level internal within superpage:
   8 B header + 681 × 4 B keys + 682 × 2 B children = 4096 B */
#define MT_SP_MAX_IKEYS    681
#define MT_SP_MIN_IKEYS    (MT_SP_MAX_IKEYS / 2)

/* ── Superpage header (page 0 of a 2 MiB region) ──────────── */

struct mt_sp_header;  /* forward decl */

typedef struct mt_sp_header {
    uint16_t  type;            /* MT_NODE_LEAF from outer tree */
    uint16_t  _pad0;
    uint32_t  nkeys;           /* total keys across all page leaves */
    uint16_t  root_page;       /* page index of sub-tree root (1–511) */
    uint8_t   sub_height;      /* page sub-tree height (0 or 1) */
    uint8_t   _pad1;
    uint16_t  npages_used;     /* number of pages allocated */
    uint16_t  _pad2;
    uint64_t  page_bitmap[8];  /* 512 bits for page allocation */
    struct mt_sp_header *prev; /* previous superpage (outer tree) */
    struct mt_sp_header *next; /* next superpage (outer tree) */
    uint8_t   _reserved[4000]; /* pad to 4096 */
} mt_sp_header_t;

MT_STATIC_ASSERT(sizeof(mt_sp_header_t) == MT_PAGE_SIZE,
               "mt_sp_header_t must be exactly 4096 bytes");

/* ── Page-level internal node within superpage ─────────────── */

typedef struct mt_sp_inode {
    uint16_t  type;                         /* 2 = SP internal */
    uint16_t  nkeys;
    uint32_t  _pad;
    int32_t   keys[MT_SP_MAX_IKEYS];       /* separator keys */
    uint16_t  children[MT_SP_MAX_IKEYS+1]; /* page indices (0–511) */
} mt_sp_inode_t;

MT_STATIC_ASSERT(sizeof(mt_sp_inode_t) == MT_PAGE_SIZE,
               "mt_sp_inode_t must be exactly 4096 bytes");

/* ── Page header (slot 0 of a leaf page) ────────────────────── */

typedef struct mt_page_header {
    uint16_t         type;          /* MT_NODE_LEAF (outer tree perspective) */
    uint16_t         nkeys;         /* Total keys in this page */
    uint8_t          root_slot;     /* CL slot index of sub-tree root (1–63) */
    uint8_t          sub_height;    /* Sub-tree height (0 = single CL leaf) */
    uint8_t          nslots_used;   /* Number of CL slots allocated */
    uint8_t          flags;         /* Bit 0: Eytzinger layout */
    uint64_t         slot_bitmap;   /* Bits 1–63: CL slot allocation */
    struct mt_lnode *prev;          /* Previous leaf (outer tree linked list) */
    struct mt_lnode *next;          /* Next leaf (outer tree linked list) */
    /* Fence keys: CL root internal separators cached in the header.
       Valid when nfence > 0 && nfence == root internal's nkeys.
       fence_slots[i] = CL slot for the i-th child (0 ≤ i ≤ nfence). */
    int32_t          fence_keys[MT_FENCE_KEY_CAP];   /* +32, 24 B */
    uint8_t          fence_slots[MT_FENCE_SLOT_CAP]; /* +56,  7 B */
    uint8_t          nfence;                         /* +63,  1 B */
} mt_page_header_t;

MT_STATIC_ASSERT(sizeof(mt_page_header_t) == MT_CL_SIZE,
               "mt_page_header_t must be exactly 64 bytes");

/* ── Leaf node (4 KiB page with matryoshka-nested sub-tree) ── */

/* Forward declaration for pointer usage. */
struct mt_lnode;

typedef struct mt_lnode {
    mt_page_header_t  header;               /* Slot 0: page header */
    mt_cl_slot_t      slots[MT_PAGE_SLOTS]; /* Slots 1–63: CL sub-nodes */
} mt_lnode_t;

MT_STATIC_ASSERT(sizeof(mt_lnode_t) == MT_PAGE_SIZE,
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

/* ── Pointer tagging ───────────────────────────────────────────── */
/*
 * Leaf pointers in outer-tree children[] arrays encode metadata in
 * the low 12 bits (guaranteed zero by 4096-byte alignment):
 *   bits 0–5: root_slot  (CL sub-tree root slot index, 1–63)
 *   bits 6–8: sub_height (CL sub-tree height, 0–7)
 *
 * This lets find_leaf() prefetch the CL root cache line one outer-tree
 * level earlier — overlapping the prefetch with the inode search at the
 * next level — instead of waiting until the page header is loaded.
 */

#define MT_PTR_TAG_MASK       ((uintptr_t)0xFFF)
#define MT_PTR_SLOT_MASK      ((uintptr_t)0x03F)   /* bits 0–5 */
#define MT_PTR_HEIGHT_SHIFT   6
#define MT_PTR_HEIGHT_MASK    ((uintptr_t)0x1C0)   /* bits 6–8 */

/* Strip tag bits, returning the raw page-aligned pointer. */
static inline mt_node_t *mt_untag(mt_node_t *ptr)
{
    return (mt_node_t *)((uintptr_t)ptr & ~MT_PTR_TAG_MASK);
}

/* Tag a leaf pointer with the page's current root_slot and sub_height. */
static inline mt_node_t *mt_tag_leaf_ptr(mt_node_t *ptr)
{
    mt_lnode_t *leaf = &ptr->lnode;
    uintptr_t tag = (uintptr_t)leaf->header.root_slot |
                    ((uintptr_t)leaf->header.sub_height << MT_PTR_HEIGHT_SHIFT);
    return (mt_node_t *)((uintptr_t)ptr | tag);
}

/* Extract root_slot from a tagged pointer. */
static inline uint8_t mt_ptr_root_slot(mt_node_t *ptr)
{
    return (uint8_t)((uintptr_t)ptr & MT_PTR_SLOT_MASK);
}

/* Extract sub_height from a tagged pointer. */
static inline uint8_t mt_ptr_sub_height(mt_node_t *ptr)
{
    return (uint8_t)(((uintptr_t)ptr >> MT_PTR_HEIGHT_SHIFT) & 0x7);
}

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
void mt_hierarchy_init_fence(mt_hierarchy_t *h);
void mt_hierarchy_init_eytzinger(mt_hierarchy_t *h);
void mt_hierarchy_init_superpage(mt_hierarchy_t *h);
void mt_hierarchy_init_custom(mt_hierarchy_t *h, size_t leaf_alloc);

/* Page header flags (mt_page_header_t.flags). */
#define MT_PAGE_FLAG_EYTZ  0x01   /* Eytzinger dense BFS layout */

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
mt_status_t mt_page_insert(mt_lnode_t *page, int32_t key,
                            const mt_hierarchy_t *hier);

/* Delete a key from a leaf page.
   Returns MT_OK, MT_NOT_FOUND, or MT_UNDERFLOW. */
mt_status_t mt_page_delete(mt_lnode_t *page, int32_t key,
                            const mt_hierarchy_t *hier);

/* Extract all keys from a leaf page in sorted order.
   `out` must have room for page->header.nkeys elements.
   Returns the number of keys extracted. */
int mt_page_extract_sorted(const mt_lnode_t *page, int32_t *out);

/* Bulk-load sorted keys into an empty page.  O(n).
   Uses hier->cl_strategy to select sub-tree layout. */
void mt_page_bulk_load(mt_lnode_t *page, const int32_t *sorted_keys, int nkeys,
                        const mt_hierarchy_t *hier);

/* Initialise an empty leaf page using the given hierarchy. */
void mt_page_init_with(mt_lnode_t *page, const mt_hierarchy_t *hier);

/* Split a page: move approximately half of the keys to `new_page`.
   Returns the separator key (first key of new_page). */
int32_t mt_page_split(mt_lnode_t *page, mt_lnode_t *new_page,
                       const mt_hierarchy_t *hier);

/* Return the minimum (first) key in a page. */
int32_t mt_page_min_key(const mt_lnode_t *page);

/* Membership test within a leaf page. */
bool mt_page_contains(const mt_lnode_t *page, int32_t key);

/* ── Internal node search (inode.c) ───────────────────────── */

int mt_inode_search(const mt_inode_t *node, int32_t key);

/* ── Node allocation (alloc.c) ─────────────────────────────── */

mt_node_t *mt_alloc_inode(void);
mt_node_t *mt_alloc_lnode(const mt_hierarchy_t *hier, mt_allocator_t *alloc);
void mt_free_inode(mt_node_t *node);
void mt_free_lnode(mt_node_t *node, mt_allocator_t *alloc);

/* ── Superpage operations (superpage.c) ────────────────────── */

void        mt_sp_init(void *sp);
mt_status_t mt_sp_insert(void *sp, int32_t key, const mt_hierarchy_t *hier);
mt_status_t mt_sp_delete(void *sp, int32_t key, const mt_hierarchy_t *hier);
bool        mt_sp_search_key(const void *sp, int32_t key, int32_t *result);
bool        mt_sp_contains(const void *sp, int32_t key);
int32_t     mt_sp_split(void *sp, void *new_sp, const mt_hierarchy_t *hier);
void        mt_sp_bulk_load(void *sp, const int32_t *keys, int nkeys,
                             const mt_hierarchy_t *hier);
int         mt_sp_extract_sorted(const void *sp, int32_t *out);
int32_t     mt_sp_min_key(const void *sp);
int32_t     mt_sp_max_key(const void *sp);

/* Get the first page leaf in a superpage (for iterator start). */
mt_lnode_t *mt_sp_first_leaf(void *sp);

/* Find the page leaf containing `key` in a superpage (for iterator seek). */
mt_lnode_t *mt_sp_find_leaf(void *sp, int32_t key);

/* ── Arena allocator (arena.c) ─────────────────────────────── */

mt_allocator_t *mt_allocator_create(size_t arena_size, size_t page_size);
void            mt_allocator_destroy(mt_allocator_t *alloc);
void           *mt_allocator_alloc(mt_allocator_t *alloc);
void            mt_allocator_free(mt_allocator_t *alloc, void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* MATRYOSHKA_INTERNAL_H */
