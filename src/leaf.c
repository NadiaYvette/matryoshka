/*
 * leaf.c — Page-level matryoshka sub-tree operations.
 *
 * Each leaf page (4 KiB) contains a B+ tree of cache-line-sized (64 B)
 * sub-nodes.  CL leaf nodes hold up to 15 sorted int32_t keys; CL
 * internal nodes hold up to 12 separator keys with 13 child slot indices.
 *
 * Operations within a page modify only the affected CL sub-nodes,
 * giving O(log b) insert/delete instead of O(B) flat-array rebuild.
 */

#include "matryoshka_internal.h"
#include <string.h>

/* ── Slot allocator ────────────────────────────────────────── */

/* Allocate a CL slot from the page's bitmap.  Returns slot index (1–63)
   or 0 if no slots available. */
static int slot_alloc(mt_lnode_t *page)
{
    /* Bits 1–63 track slots; bit 0 is always set (header). */
    uint64_t avail = ~page->header.slot_bitmap & ~1ULL;
    if (avail == 0)
        return 0;
    int slot = __builtin_ctzll(avail);
    page->header.slot_bitmap |= (1ULL << slot);
    page->header.nslots_used++;
    return slot;
}

/* Free a CL slot back to the bitmap. */
static void slot_free(mt_lnode_t *page, int slot)
{
    page->header.slot_bitmap &= ~(1ULL << slot);
    page->header.nslots_used--;
}

/* Get a CL slot by index (1-based; slot 0 is the header). */
static mt_cl_slot_t *get_slot(mt_lnode_t *page, int slot)
{
    return &page->slots[slot - 1];
}

static const mt_cl_slot_t *get_slot_c(const mt_lnode_t *page, int slot)
{
    return &page->slots[slot - 1];
}

/* ── CL leaf operations ────────────────────────────────────── */

static void cl_leaf_init(mt_cl_slot_t *s)
{
    memset(s, 0, MT_CL_SIZE);
    s->leaf.type = MT_CL_LEAF;
    s->leaf.nkeys = 0;
}

/* Binary search in a CL leaf for the insertion point of `key`.
   Returns the index where key should be / is. */
static int cl_leaf_lower_bound(const mt_cl_leaf_t *cl, int32_t key)
{
    int lo = 0, hi = cl->nkeys;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (cl->keys[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/* SIMD predecessor search within a CL leaf.
   Returns the index of the largest key <= `key`, or -1 if none. */
static int cl_leaf_predecessor(const mt_cl_leaf_t *cl, int32_t key)
{
    int n = cl->nkeys;
    if (n == 0) return -1;

#if defined(__AVX512F__)
    /* AVX-512: single masked 16-lane compare covers all 15 keys. */
    __mmask16 valid = (__mmask16)((1u << n) - 1);
    __m512i vkey = _mm512_set1_epi32(key);
    __m512i vtree = _mm512_loadu_si512((const void *)cl->keys);
    __mmask16 gt = _mm512_mask_cmpgt_epi32_mask(valid, vtree, vkey);
    if (gt != 0)
        return __builtin_ctz(gt) - 1;
    return n - 1;

#elif defined(__AVX2__)
    /* AVX2: one 8-key load + scalar tail for remaining 7. */
    __m256i vkey256 = _mm256_set1_epi32(key);
    int result = -1;

    if (n > 0) {
        __m256i vtree = _mm256_loadu_si256((const __m256i *)cl->keys);
        __m256i vcmp = _mm256_cmpgt_epi32(vtree, vkey256);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(vcmp));
        int count = (n < 8) ? n : 8;
        mask &= (1 << count) - 1;
        if (mask != 0)
            return __builtin_ctz(mask) - 1;
        result = count - 1;
    }
    for (int i = 8; i < n; i++) {
        if (cl->keys[i] > key)
            return i - 1;
        result = i;
    }
    return result;

#else
    /* SSE2: compare 4 keys at a time. */
    __m128i vkey = _mm_set1_epi32(key);
    int result = -1;

    int i = 0;
    for (; i + 3 < n; i += 4) {
        __m128i vtree = _mm_loadu_si128((const __m128i *)(cl->keys + i));
        __m128i vcmp = _mm_cmpgt_epi32(vtree, vkey);
        int mask = _mm_movemask_ps(_mm_castsi128_ps(vcmp));
        if (mask != 0) {
            int first_gt = i + __builtin_ctz(mask);
            return first_gt - 1;
        }
        result = i + 3;
    }
    for (; i < n; i++) {
        if (cl->keys[i] > key)
            return i - 1;
        result = i;
    }
    return result;
#endif
}

/* Insert key into CL leaf.  Returns 0 on success, -1 if duplicate,
   -2 if full.  Caller must check nkeys < MT_CL_KEY_CAP before calling
   unless it wants the full status. */
static int cl_leaf_insert(mt_cl_leaf_t *cl, int32_t key)
{
    int pos = cl_leaf_lower_bound(cl, key);
    if (pos < cl->nkeys && cl->keys[pos] == key)
        return -1;  /* duplicate */
    if (cl->nkeys >= MT_CL_KEY_CAP)
        return -2;  /* full */

    /* Shift right and insert. */
    int n = cl->nkeys;
    memmove(cl->keys + pos + 1, cl->keys + pos,
            (size_t)(n - pos) * sizeof(int32_t));
    cl->keys[pos] = key;
    cl->nkeys = (uint8_t)(n + 1);
    return 0;
}

/* Delete key from CL leaf.  Returns 0 on success, -1 if not found. */
static int cl_leaf_delete(mt_cl_leaf_t *cl, int32_t key)
{
    int pos = cl_leaf_lower_bound(cl, key);
    if (pos >= cl->nkeys || cl->keys[pos] != key)
        return -1;

    int n = cl->nkeys;
    memmove(cl->keys + pos, cl->keys + pos + 1,
            (size_t)(n - pos - 1) * sizeof(int32_t));
    cl->nkeys = (uint8_t)(n - 1);
    return 0;
}

/* Split a full CL leaf into two halves.
   `right` is a freshly initialized CL leaf slot.
   Returns the separator key (first key of right). */
static int32_t cl_leaf_split(mt_cl_leaf_t *left, mt_cl_leaf_t *right)
{
    int total = left->nkeys;
    int left_n = total / 2;
    int right_n = total - left_n;

    memcpy(right->keys, left->keys + left_n,
           (size_t)right_n * sizeof(int32_t));
    right->nkeys = (uint8_t)right_n;
    left->nkeys = (uint8_t)left_n;

    return right->keys[0];
}

/* ── CL internal operations ────────────────────────────────── */

static void cl_inode_init(mt_cl_slot_t *s)
{
    memset(s, 0, MT_CL_SIZE);
    s->inode.type = MT_CL_INTERNAL;
    s->inode.nkeys = 0;
}

/* Find child index in CL internal for the given key.
   Returns i such that children[i] should be followed. */
static int cl_inode_search(const mt_cl_inode_t *cl, int32_t key)
{
    int n = cl->nkeys;
    if (n == 0) return 0;

#if defined(__AVX512F__)
    /* AVX-512: single masked compare for all 12 keys. */
    __mmask16 valid = (__mmask16)((1u << n) - 1);
    __m512i vkey = _mm512_set1_epi32(key);
    __m512i vtree = _mm512_loadu_si512((const void *)cl->keys);
    __mmask16 gt = _mm512_mask_cmpgt_epi32_mask(valid, vtree, vkey);
    if (gt != 0)
        return __builtin_ctz(gt);
    return n;

#elif defined(__AVX2__)
    /* AVX2: one 8-key load + SSE2 tail for remaining 4. */
    {
        __m256i vkey256 = _mm256_set1_epi32(key);
        __m256i vtree = _mm256_loadu_si256((const __m256i *)cl->keys);
        __m256i vcmp = _mm256_cmpgt_epi32(vtree, vkey256);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(vcmp));
        int count = (n < 8) ? n : 8;
        mask &= (1 << count) - 1;
        if (mask != 0)
            return __builtin_ctz(mask);
    }
    if (n > 8) {
        __m128i vkey128 = _mm_set1_epi32(key);
        __m128i vtree = _mm_loadu_si128((const __m128i *)(cl->keys + 8));
        __m128i vcmp = _mm_cmpgt_epi32(vtree, vkey128);
        int mask = _mm_movemask_ps(_mm_castsi128_ps(vcmp));
        mask &= (1 << (n - 8)) - 1;
        if (mask != 0)
            return 8 + __builtin_ctz(mask);
    }
    return n;

#else
    /* SSE2: scan 4 keys at a time. */
    __m128i vkey = _mm_set1_epi32(key);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        __m128i vtree = _mm_loadu_si128((const __m128i *)(cl->keys + i));
        __m128i vcmp = _mm_cmpgt_epi32(vtree, vkey);
        int mask = _mm_movemask_ps(_mm_castsi128_ps(vcmp));
        if (mask != 0)
            return i + __builtin_ctz(mask);
    }
    for (; i < n; i++) {
        if (cl->keys[i] > key)
            return i;
    }
    return n;
#endif
}

/* Insert a separator key and right child into a CL internal node at `pos`.
   Caller must ensure there is room (nkeys < MT_CL_SEP_CAP). */
static void cl_inode_insert_at(mt_cl_inode_t *cl, int pos,
                                int32_t key, uint8_t right_child)
{
    int n = cl->nkeys;
    /* Shift keys right. */
    memmove(cl->keys + pos + 1, cl->keys + pos,
            (size_t)(n - pos) * sizeof(int32_t));
    /* Shift children right (from pos+1 onward). */
    memmove(cl->children + pos + 2, cl->children + pos + 1,
            (size_t)(n - pos) * sizeof(uint8_t));
    cl->keys[pos] = key;
    cl->children[pos + 1] = right_child;
    cl->nkeys = (uint8_t)(n + 1);
}

/* Remove separator at `pos` and child at `pos+1` from a CL internal. */
static void cl_inode_remove_at(mt_cl_inode_t *cl, int pos)
{
    int n = cl->nkeys;
    memmove(cl->keys + pos, cl->keys + pos + 1,
            (size_t)(n - pos - 1) * sizeof(int32_t));
    memmove(cl->children + pos + 1, cl->children + pos + 2,
            (size_t)(n - pos - 1) * sizeof(uint8_t));
    cl->nkeys = (uint8_t)(n - 1);
}

/* Split a CL internal node.  `right` is freshly initialized.
   Returns the median key that should be promoted to the parent. */
static int32_t cl_inode_split(mt_cl_inode_t *left, mt_cl_inode_t *right)
{
    int total = left->nkeys;
    int left_n = total / 2;
    int right_n = total - left_n - 1;  /* middle key goes up */
    int32_t median = left->keys[left_n];

    /* Copy right half of keys. */
    memcpy(right->keys, left->keys + left_n + 1,
           (size_t)right_n * sizeof(int32_t));
    /* Copy right half of children. */
    memcpy(right->children, left->children + left_n + 1,
           (size_t)(right_n + 1) * sizeof(uint8_t));

    right->nkeys = (uint8_t)right_n;
    left->nkeys = (uint8_t)left_n;

    return median;
}

/* ── Path tracking for sub-tree traversal ──────────────────── */

#define MT_SUB_MAX_HEIGHT 8

typedef struct {
    uint8_t slot;     /* CL slot index of this internal node */
    uint8_t child_idx; /* child index taken */
} mt_sub_path_t;

/* ── Page-level search ─────────────────────────────────────── */

/* Find the CL leaf slot and position for a predecessor search.
   Returns the slot index of the CL leaf containing the predecessor,
   or 0 if the page is empty.  `leaf_pos` is set to the position
   within the CL leaf. */
static int page_find_leaf(const mt_lnode_t *page, int32_t key,
                           mt_sub_path_t *path, int *path_len)
{
    int slot = page->header.root_slot;
    int height = page->header.sub_height;
    *path_len = 0;

    /* Prefetch the root CL node — it's on a different cache line than
       the page header we just read (root_slot). */
    __builtin_prefetch(get_slot_c(page, slot), 0, 1);

    for (int i = 0; i < height; i++) {
        const mt_cl_slot_t *s = get_slot_c(page, slot);
        int ci = cl_inode_search(&s->inode, key);
        path[*path_len].slot = (uint8_t)slot;
        path[*path_len].child_idx = (uint8_t)ci;
        (*path_len)++;
        slot = s->inode.children[ci];
        /* Prefetch the child CL node's cache line so the next
           iteration (or the caller) finds it warm in L2. */
        __builtin_prefetch(get_slot_c(page, slot), 0, 1);
    }

    return slot;
}

int mt_page_search(const mt_lnode_t *page, int32_t key)
{
    if (page->header.nkeys == 0)
        return -1;

    mt_sub_path_t path[MT_SUB_MAX_HEIGHT];
    int path_len;
    int leaf_slot = page_find_leaf(page, key, path, &path_len);

    const mt_cl_leaf_t *cl = &get_slot_c(page, leaf_slot)->leaf;
    int pos = cl_leaf_predecessor(cl, key);

    if (pos >= 0)
        return pos;  /* Found predecessor within this CL leaf. */

    /* Key is smaller than all keys in this CL leaf.
       Walk left: find the previous CL leaf in the sub-tree. */
    for (int i = path_len - 1; i >= 0; i--) {
        if (path[i].child_idx > 0) {
            /* Go to the rightmost key of the left sibling subtree. */
            const mt_cl_inode_t *parent =
                &get_slot_c(page, path[i].slot)->inode;
            int prev_child = path[i].child_idx - 1;
            int s = parent->children[prev_child];
            /* Descend to rightmost leaf. */
            const mt_cl_slot_t *slot = get_slot_c(page, s);
            while (slot->type == MT_CL_INTERNAL) {
                s = slot->inode.children[slot->inode.nkeys];
                slot = get_slot_c(page, s);
            }
            /* Return last key of this leaf. */
            if (slot->leaf.nkeys > 0)
                return -(slot->leaf.nkeys);  /* encode: negative = from-prev */
            break;
        }
    }

    return -1;
}

bool mt_page_search_key(const mt_lnode_t *page, int32_t key, int32_t *result)
{
    if (page->header.nkeys == 0)
        return false;

    mt_sub_path_t path[MT_SUB_MAX_HEIGHT];
    int path_len;
    int leaf_slot = page_find_leaf(page, key, path, &path_len);

    const mt_cl_leaf_t *cl = &get_slot_c(page, leaf_slot)->leaf;
    int pos = cl_leaf_predecessor(cl, key);

    if (pos >= 0) {
        if (result) *result = cl->keys[pos];
        return true;
    }

    /* Walk left to find predecessor in previous CL leaf. */
    for (int i = path_len - 1; i >= 0; i--) {
        if (path[i].child_idx > 0) {
            const mt_cl_inode_t *parent =
                &get_slot_c(page, path[i].slot)->inode;
            int prev_child = path[i].child_idx - 1;
            int s = parent->children[prev_child];
            const mt_cl_slot_t *slot = get_slot_c(page, s);
            while (slot->type == MT_CL_INTERNAL) {
                s = slot->inode.children[slot->inode.nkeys];
                slot = get_slot_c(page, s);
            }
            if (slot->leaf.nkeys > 0) {
                if (result)
                    *result = slot->leaf.keys[slot->leaf.nkeys - 1];
                return true;
            }
            break;
        }
    }

    return false;
}

bool mt_page_contains(const mt_lnode_t *page, int32_t key)
{
    if (page->header.nkeys == 0)
        return false;

    mt_sub_path_t path[MT_SUB_MAX_HEIGHT];
    int path_len;
    int leaf_slot = page_find_leaf(page, key, path, &path_len);

    const mt_cl_leaf_t *cl = &get_slot_c(page, leaf_slot)->leaf;
    int pos = cl_leaf_lower_bound(cl, key);
    return (pos < cl->nkeys && cl->keys[pos] == key);
}

/* ── Page-level insert ─────────────────────────────────────── */

mt_status_t mt_page_insert(mt_lnode_t *page, int32_t key)
{
    mt_sub_path_t path[MT_SUB_MAX_HEIGHT];
    int path_len;
    int leaf_slot = page_find_leaf(page, key, path, &path_len);

    mt_cl_leaf_t *cl = &get_slot(page, leaf_slot)->leaf;

    /* Try inserting into the CL leaf. */
    if (cl->nkeys < MT_CL_KEY_CAP) {
        int rc = cl_leaf_insert(cl, key);
        if (rc == -1) return MT_DUPLICATE;
        page->header.nkeys++;
        return MT_OK;
    }

    /* CL leaf is full.  Check for duplicate first. */
    int pos = cl_leaf_lower_bound(cl, key);
    if (pos < cl->nkeys && cl->keys[pos] == key)
        return MT_DUPLICATE;

    /* Need to split the CL leaf. */
    int new_slot = slot_alloc(page);
    if (new_slot == 0)
        return MT_PAGE_FULL;

    mt_cl_slot_t *new_s = get_slot(page, new_slot);
    cl_leaf_init(new_s);

    int32_t sep = cl_leaf_split(cl, &new_s->leaf);

    /* Insert the key into the appropriate half. */
    if (key < sep)
        cl_leaf_insert(cl, key);
    else
        cl_leaf_insert(&new_s->leaf, key);
    page->header.nkeys++;

    /* Propagate the split upward through CL internal nodes. */
    uint8_t right_slot = (uint8_t)new_slot;

    for (int i = path_len - 1; i >= 0; i--) {
        mt_cl_inode_t *parent = &get_slot(page, path[i].slot)->inode;

        if (parent->nkeys < MT_CL_SEP_CAP) {
            cl_inode_insert_at(parent, path[i].child_idx, sep, right_slot);
            return MT_OK;
        }

        /* CL internal is full — split it. */
        int split_slot = slot_alloc(page);
        if (split_slot == 0) {
            /* Out of slots.  Key is already in a CL leaf.
               Signal page full — caller will split the page,
               which extracts all keys and rebuilds. */
            return MT_PAGE_FULL;
        }

        mt_cl_slot_t *new_inode = get_slot(page, split_slot);
        cl_inode_init(new_inode);

        /* Build merged key/child arrays (without modifying parent). */
        int pn = parent->nkeys;
        int32_t all_keys[MT_CL_SEP_CAP + 1];
        uint8_t all_children[MT_CL_CHILD_CAP + 1];
        int ci = path[i].child_idx;

        memcpy(all_keys, parent->keys, (size_t)ci * sizeof(int32_t));
        all_keys[ci] = sep;
        memcpy(all_keys + ci + 1, parent->keys + ci,
               (size_t)(pn - ci) * sizeof(int32_t));

        memcpy(all_children, parent->children,
               (size_t)(ci + 1) * sizeof(uint8_t));
        all_children[ci + 1] = right_slot;
        memcpy(all_children + ci + 2, parent->children + ci + 1,
               (size_t)(pn - ci) * sizeof(uint8_t));

        int total = pn + 1;
        int left_n = total / 2;
        int right_n = total - left_n - 1;
        sep = all_keys[left_n];

        /* Rebuild left (reuse parent). */
        memcpy(parent->keys, all_keys, (size_t)left_n * sizeof(int32_t));
        memcpy(parent->children, all_children,
               (size_t)(left_n + 1) * sizeof(uint8_t));
        parent->nkeys = (uint8_t)left_n;

        /* Build right. */
        memcpy(new_inode->inode.keys, all_keys + left_n + 1,
               (size_t)right_n * sizeof(int32_t));
        memcpy(new_inode->inode.children, all_children + left_n + 1,
               (size_t)(right_n + 1) * sizeof(uint8_t));
        new_inode->inode.nkeys = (uint8_t)right_n;

        right_slot = (uint8_t)split_slot;
        /* Continue propagating upward. */
    }

    /* Split reached the sub-tree root — create new root. */
    int new_root_slot = slot_alloc(page);
    if (new_root_slot == 0) {
        /* Extremely unlikely: out of slots for a new root.
           The data is still valid — signal page full. */
        return MT_PAGE_FULL;
    }

    mt_cl_slot_t *new_root = get_slot(page, new_root_slot);
    cl_inode_init(new_root);
    new_root->inode.keys[0] = sep;
    new_root->inode.children[0] = page->header.root_slot;
    new_root->inode.children[1] = right_slot;
    new_root->inode.nkeys = 1;
    page->header.root_slot = (uint8_t)new_root_slot;
    page->header.sub_height++;

    return MT_OK;
}

/* ── Page-level delete ─────────────────────────────────────── */

mt_status_t mt_page_delete(mt_lnode_t *page, int32_t key,
                            const mt_hierarchy_t *hier)
{
    mt_sub_path_t path[MT_SUB_MAX_HEIGHT];
    int path_len;
    int leaf_slot = page_find_leaf(page, key, path, &path_len);

    mt_cl_leaf_t *cl = &get_slot(page, leaf_slot)->leaf;

    /* Try deleting from the CL leaf. */
    int rc = cl_leaf_delete(cl, key);
    if (rc < 0)
        return MT_NOT_FOUND;
    page->header.nkeys--;

    /* Check for CL leaf underflow. */
    if (path_len == 0) {
        /* Root is a CL leaf — no underflow possible. */
        return (page->header.nkeys < hier->min_page_keys)
               ? MT_UNDERFLOW : MT_OK;
    }

    if (cl->nkeys >= MT_CL_MIN_KEYS)
        goto check_page_underflow;

    /* CL leaf underflow — try redistribute from siblings. */
    for (int level = path_len - 1; level >= 0; level--) {
        mt_cl_inode_t *parent = &get_slot(page, path[level].slot)->inode;
        int cidx = path[level].child_idx;
        int cur_slot = (level == path_len - 1) ? leaf_slot
            : path[level + 1].slot;
        mt_cl_slot_t *cur = get_slot(page, cur_slot);

        int cur_nkeys;
        if (cur->type == MT_CL_LEAF)
            cur_nkeys = cur->leaf.nkeys;
        else
            cur_nkeys = cur->inode.nkeys;

        int min_keys = (cur->type == MT_CL_LEAF) ? MT_CL_MIN_KEYS
                                                   : (MT_CL_MIN_CHILDREN - 1);

        if (cur_nkeys >= min_keys)
            break;  /* No underflow at this level. */

        /* Try redistribute from left sibling. */
        if (cidx > 0 && cur->type == MT_CL_LEAF) {
            int left_slot_idx = parent->children[cidx - 1];
            mt_cl_leaf_t *left = &get_slot(page, left_slot_idx)->leaf;
            if (left->nkeys > MT_CL_MIN_KEYS) {
                /* Move last key from left to current. */
                int32_t moved = left->keys[left->nkeys - 1];
                left->nkeys--;
                cl_leaf_insert(&cur->leaf, moved);
                /* Update separator in parent. */
                parent->keys[cidx - 1] = cur->leaf.keys[0];
                break;
            }
        }

        /* Try redistribute from right sibling. */
        if (cidx < parent->nkeys && cur->type == MT_CL_LEAF) {
            int right_slot_idx = parent->children[cidx + 1];
            mt_cl_leaf_t *right = &get_slot(page, right_slot_idx)->leaf;
            if (right->nkeys > MT_CL_MIN_KEYS) {
                int32_t moved = right->keys[0];
                cl_leaf_delete(right, moved);
                cl_leaf_insert(&cur->leaf, moved);
                parent->keys[cidx] = right->keys[0];
                break;
            }
        }

        /* Merge CL leaves. */
        if (cur->type == MT_CL_LEAF) {
            if (cidx > 0) {
                /* Merge current into left sibling. */
                int left_slot_idx = parent->children[cidx - 1];
                mt_cl_leaf_t *left = &get_slot(page, left_slot_idx)->leaf;
                /* Copy current's keys into left. */
                memcpy(left->keys + left->nkeys, cur->leaf.keys,
                       (size_t)cur->leaf.nkeys * sizeof(int32_t));
                left->nkeys = (uint8_t)(left->nkeys + cur->leaf.nkeys);
                /* Free current slot, remove from parent. */
                slot_free(page, cur_slot);
                cl_inode_remove_at(parent, cidx - 1);
            } else {
                /* Merge right sibling into current. */
                int right_slot_idx = parent->children[cidx + 1];
                mt_cl_leaf_t *right = &get_slot(page, right_slot_idx)->leaf;
                memcpy(cur->leaf.keys + cur->leaf.nkeys, right->keys,
                       (size_t)right->nkeys * sizeof(int32_t));
                cur->leaf.nkeys = (uint8_t)(cur->leaf.nkeys + right->nkeys);
                slot_free(page, right_slot_idx);
                cl_inode_remove_at(parent, cidx);
            }
            /* Check if parent underflows — continue loop. */
            /* Update path to point to parent for next iteration. */
            if (level > 0) {
                /* The current node for the next level up is the parent. */
                /* The path already has the right info. */
            }
            continue;
        }

        /* CL internal underflow — similar logic but with key rotation. */
        if (cidx > 0) {
            int left_slot_idx = parent->children[cidx - 1];
            mt_cl_inode_t *left = &get_slot(page, left_slot_idx)->inode;
            if (left->nkeys > (MT_CL_MIN_CHILDREN - 1)) {
                /* Rotate right. */
                memmove(cur->inode.keys + 1, cur->inode.keys,
                        (size_t)cur->inode.nkeys * sizeof(int32_t));
                memmove(cur->inode.children + 1, cur->inode.children,
                        (size_t)(cur->inode.nkeys + 1) * sizeof(uint8_t));
                cur->inode.keys[0] = parent->keys[cidx - 1];
                cur->inode.children[0] = left->children[left->nkeys];
                cur->inode.nkeys++;
                parent->keys[cidx - 1] = left->keys[left->nkeys - 1];
                left->nkeys--;
                break;
            }
        }

        if (cidx < parent->nkeys) {
            int right_slot_idx = parent->children[cidx + 1];
            mt_cl_inode_t *right_in = &get_slot(page, right_slot_idx)->inode;
            if (right_in->nkeys > (MT_CL_MIN_CHILDREN - 1)) {
                /* Rotate left. */
                cur->inode.keys[cur->inode.nkeys] = parent->keys[cidx];
                cur->inode.children[cur->inode.nkeys + 1] = right_in->children[0];
                cur->inode.nkeys++;
                parent->keys[cidx] = right_in->keys[0];
                memmove(right_in->keys, right_in->keys + 1,
                        (size_t)(right_in->nkeys - 1) * sizeof(int32_t));
                memmove(right_in->children, right_in->children + 1,
                        (size_t)right_in->nkeys * sizeof(uint8_t));
                right_in->nkeys--;
                break;
            }
        }

        /* Merge CL internal nodes. */
        if (cidx > 0) {
            int left_slot_idx = parent->children[cidx - 1];
            mt_cl_inode_t *left = &get_slot(page, left_slot_idx)->inode;
            /* Pull separator down. */
            left->keys[left->nkeys] = parent->keys[cidx - 1];
            memcpy(left->keys + left->nkeys + 1, cur->inode.keys,
                   (size_t)cur->inode.nkeys * sizeof(int32_t));
            memcpy(left->children + left->nkeys + 1, cur->inode.children,
                   (size_t)(cur->inode.nkeys + 1) * sizeof(uint8_t));
            left->nkeys = (uint8_t)(left->nkeys + 1 + cur->inode.nkeys);
            slot_free(page, cur_slot);
            cl_inode_remove_at(parent, cidx - 1);
        } else {
            int right_slot_idx = parent->children[cidx + 1];
            mt_cl_inode_t *right_in = &get_slot(page, right_slot_idx)->inode;
            cur->inode.keys[cur->inode.nkeys] = parent->keys[cidx];
            memcpy(cur->inode.keys + cur->inode.nkeys + 1, right_in->keys,
                   (size_t)right_in->nkeys * sizeof(int32_t));
            memcpy(cur->inode.children + cur->inode.nkeys + 1, right_in->children,
                   (size_t)(right_in->nkeys + 1) * sizeof(uint8_t));
            cur->inode.nkeys = (uint8_t)(cur->inode.nkeys + 1 + right_in->nkeys);
            slot_free(page, right_slot_idx);
            cl_inode_remove_at(parent, cidx);
        }
    }

    /* Check if root internal has 0 keys — collapse. */
    {
        mt_cl_slot_t *root = get_slot(page, page->header.root_slot);
        if (root->type == MT_CL_INTERNAL && root->inode.nkeys == 0
            && page->header.sub_height > 0) {
            int old_root = page->header.root_slot;
            page->header.root_slot = root->inode.children[0];
            page->header.sub_height--;
            slot_free(page, old_root);
        }
    }

check_page_underflow:
    return (page->header.nkeys < hier->min_page_keys)
           ? MT_UNDERFLOW : MT_OK;
}

/* ── Page-level extract sorted ──────────────────────────────── */

/* Recursive in-order traversal of the CL sub-tree. */
static int extract_subtree(const mt_lnode_t *page, int slot,
                            int32_t *out, int pos)
{
    const mt_cl_slot_t *s = get_slot_c(page, slot);
    __builtin_prefetch(s, 0, 0);

    if (s->type == MT_CL_LEAF) {
        memcpy(out + pos, s->leaf.keys,
               (size_t)s->leaf.nkeys * sizeof(int32_t));
        return pos + s->leaf.nkeys;
    }

    /* Internal: in-order traversal. */
    const mt_cl_inode_t *in = &s->inode;
    for (int i = 0; i <= in->nkeys; i++) {
        pos = extract_subtree(page, in->children[i], out, pos);
        if (i < in->nkeys) {
            /* Separator keys are NOT stored in leaves — they're routing
               keys derived from leaf contents.  Skip them during extraction. */
        }
    }
    return pos;
}

int mt_page_extract_sorted(const mt_lnode_t *page, int32_t *out)
{
    if (page->header.nkeys == 0)
        return 0;
    return extract_subtree(page, page->header.root_slot, out, 0);
}

/* ── Page-level bulk load ──────────────────────────────────── */

void mt_page_bulk_load(mt_lnode_t *page, const int32_t *sorted_keys, int nkeys)
{
    /* Reset page to empty state. */
    memset(page, 0, MT_PAGE_SIZE);
    page->header.type = MT_NODE_LEAF;
    page->header.slot_bitmap = 1;  /* bit 0 = header */

    if (nkeys == 0) {
        /* Allocate one empty CL leaf as root. */
        int root = slot_alloc(page);
        cl_leaf_init(get_slot(page, root));
        page->header.root_slot = (uint8_t)root;
        page->header.sub_height = 0;
        page->header.nkeys = 0;
        return;
    }

    /* Fill CL leaves sequentially. */
    int nleaves = (nkeys + MT_CL_KEY_CAP - 1) / MT_CL_KEY_CAP;
    int keys_per = nkeys / nleaves;
    int extra = nkeys % nleaves;

    uint8_t leaf_slots[MT_PAGE_SLOTS];
    int32_t separators[MT_PAGE_SLOTS];  /* separator[i] = first key of leaf[i] */

    int offset = 0;
    for (int i = 0; i < nleaves; i++) {
        int k = keys_per + (i < extra ? 1 : 0);
        int slot = slot_alloc(page);
        mt_cl_slot_t *s = get_slot(page, slot);
        cl_leaf_init(s);
        memcpy(s->leaf.keys, sorted_keys + offset, (size_t)k * sizeof(int32_t));
        s->leaf.nkeys = (uint8_t)k;
        leaf_slots[i] = (uint8_t)slot;
        separators[i] = sorted_keys[offset];
        offset += k;
    }

    page->header.nkeys = (uint16_t)nkeys;

    if (nleaves == 1) {
        /* Single leaf is the root. */
        page->header.root_slot = leaf_slots[0];
        page->header.sub_height = 0;
        return;
    }

    /* Build internal nodes bottom-up. */
    uint8_t current_level_slots[MT_PAGE_SLOTS];
    int32_t current_level_seps[MT_PAGE_SLOTS];
    int level_count = nleaves;
    memcpy(current_level_slots, leaf_slots,
           (size_t)nleaves * sizeof(uint8_t));
    memcpy(current_level_seps, separators,
           (size_t)nleaves * sizeof(int32_t));
    int height = 0;

    while (level_count > 1) {
        int num_parents = (level_count + MT_CL_CHILD_CAP - 1) / MT_CL_CHILD_CAP;
        if (num_parents == 0) num_parents = 1;

        uint8_t next_slots[MT_PAGE_SLOTS];
        int32_t next_seps[MT_PAGE_SLOTS];
        int children_per = level_count / num_parents;
        int extra_c = level_count % num_parents;
        int ci = 0;

        for (int p = 0; p < num_parents; p++) {
            int nc = children_per + (p < extra_c ? 1 : 0);
            int pslot = slot_alloc(page);
            mt_cl_slot_t *ps = get_slot(page, pslot);
            cl_inode_init(ps);

            ps->inode.children[0] = current_level_slots[ci];
            for (int j = 1; j < nc; j++) {
                ps->inode.keys[j - 1] = current_level_seps[ci + j];
                ps->inode.children[j] = current_level_slots[ci + j];
            }
            ps->inode.nkeys = (uint8_t)(nc - 1);

            next_slots[p] = (uint8_t)pslot;
            next_seps[p] = current_level_seps[ci];
            ci += nc;
        }

        memcpy(current_level_slots, next_slots,
               (size_t)num_parents * sizeof(uint8_t));
        memcpy(current_level_seps, next_seps,
               (size_t)num_parents * sizeof(int32_t));
        level_count = num_parents;
        height++;
    }

    page->header.root_slot = current_level_slots[0];
    page->header.sub_height = (uint8_t)height;
}

/* ── Page initialisation ───────────────────────────────────── */

void mt_page_init(mt_lnode_t *page)
{
    mt_page_bulk_load(page, NULL, 0);
}

/* ── Page split ────────────────────────────────────────────── */

int32_t mt_page_split(mt_lnode_t *page, mt_lnode_t *new_page)
{
    int total = page->header.nkeys;
    int32_t all_keys[1024];  /* max possible keys in a page */
    int n = mt_page_extract_sorted(page, all_keys);
    (void)total;

    int left_n = n / 2;
    int right_n = n - left_n;

    mt_page_bulk_load(page, all_keys, left_n);
    mt_page_bulk_load(new_page, all_keys + left_n, right_n);

    return all_keys[left_n];  /* separator = first key of right page */
}

/* ── Page min key ──────────────────────────────────────────── */

int32_t mt_page_min_key(const mt_lnode_t *page)
{
    if (page->header.nkeys == 0)
        return MT_KEY_MAX;

    /* Walk to leftmost CL leaf. */
    int slot = page->header.root_slot;
    const mt_cl_slot_t *s = get_slot_c(page, slot);
    while (s->type == MT_CL_INTERNAL) {
        slot = s->inode.children[0];
        __builtin_prefetch(get_slot_c(page, slot), 0, 1);
        s = get_slot_c(page, slot);
    }
    return (s->leaf.nkeys > 0) ? s->leaf.keys[0] : MT_KEY_MAX;
}
