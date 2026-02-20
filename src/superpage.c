/*
 * superpage.c — Superpage-level nesting for matryoshka trees.
 *
 * A 2 MiB superpage contains a B+ tree of 4 KiB page sub-nodes.
 * Page 0 is the superpage header; pages 1–511 are either page-level
 * internal nodes (mt_sp_inode_t) or page-level leaves (mt_lnode_t).
 * Each page leaf contains a CL sub-tree (as implemented in leaf.c).
 *
 * Three-level matryoshka nesting:
 *   Level 0: SSE2/AVX2 register — SIMD search within CL sub-nodes
 *   Level 1: Cache line (64 B)  — B+ tree of CL sub-nodes within page
 *   Level 2: Page (4 KiB)       — B+ tree of page sub-nodes within superpage
 *   Level 3: Main memory        — outer B+ tree of superpages
 */

#include "matryoshka_internal.h"
#include <string.h>
#include <stdlib.h>

/* Maximum keys extractable from a single page (all 63 slots as leaves). */
#define SP_MAX_PAGE_KEYS  (MT_PAGE_SLOTS * MT_CL_KEY_CAP)  /* 945 */

/* ── Page access helpers ─────────────────────────────────────── */

static inline void *sp_page(void *sp, int idx)
{
    return (char *)sp + (size_t)idx * MT_PAGE_SIZE;
}

static inline const void *sp_page_c(const void *sp, int idx)
{
    return (const char *)sp + (size_t)idx * MT_PAGE_SIZE;
}

static inline mt_sp_header_t *sp_hdr(void *sp)
{
    return (mt_sp_header_t *)sp;
}

static inline const mt_sp_header_t *sp_hdr_c(const void *sp)
{
    return (const mt_sp_header_t *)sp;
}

/* ── Page allocator within superpage ─────────────────────────── */

/* Allocate a page from the superpage's bitmap.  Returns page index
   (1–511) or 0 if no pages available. */
static int sp_page_alloc(mt_sp_header_t *hdr)
{
    for (int w = 0; w < 8; w++) {
        uint64_t avail = ~hdr->page_bitmap[w];
        if (avail) {
            int bit = __builtin_ctzll(avail);
            int idx = w * 64 + bit;
            if (idx >= (int)MT_SP_PAGES) return 0;
            hdr->page_bitmap[w] |= (1ULL << bit);
            hdr->npages_used++;
            return idx;
        }
    }
    return 0;
}

static void sp_page_free(mt_sp_header_t *hdr, int idx)
{
    hdr->page_bitmap[idx / 64] &= ~(1ULL << (idx % 64));
    hdr->npages_used--;
}

/* ── Page-level internal search ──────────────────────────────── */

/* Binary search in a page-level internal node.
   Returns child index i such that children[i] should be followed. */
static int sp_inode_search(const mt_sp_inode_t *node, int32_t key)
{
    int lo = 0, hi = node->nkeys;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (node->keys[mid] <= key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

/* Insert separator and right child at pos in a page-level internal. */
static void sp_inode_insert_at(mt_sp_inode_t *node, int pos,
                                int32_t key, uint16_t right_child)
{
    int n = node->nkeys;
    memmove(node->keys + pos + 1, node->keys + pos,
            (size_t)(n - pos) * sizeof(int32_t));
    memmove(node->children + pos + 2, node->children + pos + 1,
            (size_t)(n - pos) * sizeof(uint16_t));
    node->keys[pos] = key;
    node->children[pos + 1] = right_child;
    node->nkeys = (uint16_t)(n + 1);
}

/* Remove separator at pos and child at pos+1. */
static void sp_inode_remove_at(mt_sp_inode_t *node, int pos)
{
    int n = node->nkeys;
    memmove(node->keys + pos, node->keys + pos + 1,
            (size_t)(n - pos - 1) * sizeof(int32_t));
    memmove(node->children + pos + 1, node->children + pos + 2,
            (size_t)(n - pos - 1) * sizeof(uint16_t));
    node->nkeys = (uint16_t)(n - 1);
}

/* ── Path tracking for page-level sub-tree ───────────────────── */

#define MT_SP_MAX_HEIGHT 4  /* height 0 or 1 in practice */

typedef struct {
    uint16_t page_idx;
    uint16_t child_idx;
} mt_sp_path_t;

/* Navigate from root to the page leaf containing key. */
static int sp_find_leaf(const void *sp, int32_t key,
                         mt_sp_path_t *path, int *path_len)
{
    const mt_sp_header_t *hdr = sp_hdr_c(sp);
    int page_idx = hdr->root_page;
    int height = hdr->sub_height;
    *path_len = 0;

    for (int i = 0; i < height; i++) {
        const mt_sp_inode_t *inode =
            (const mt_sp_inode_t *)sp_page_c(sp, page_idx);
        int ci = sp_inode_search(inode, key);
        path[*path_len].page_idx = (uint16_t)page_idx;
        path[*path_len].child_idx = (uint16_t)ci;
        (*path_len)++;
        page_idx = inode->children[ci];
    }

    return page_idx;
}

/* Navigate to the leftmost page leaf. */
static int sp_leftmost_leaf(const void *sp)
{
    const mt_sp_header_t *hdr = sp_hdr_c(sp);
    int page_idx = hdr->root_page;
    for (int i = 0; i < hdr->sub_height; i++) {
        const mt_sp_inode_t *inode =
            (const mt_sp_inode_t *)sp_page_c(sp, page_idx);
        page_idx = inode->children[0];
    }
    return page_idx;
}

/* Navigate to the rightmost page leaf. */
static int sp_rightmost_leaf(const void *sp)
{
    const mt_sp_header_t *hdr = sp_hdr_c(sp);
    int page_idx = hdr->root_page;
    for (int i = 0; i < hdr->sub_height; i++) {
        const mt_sp_inode_t *inode =
            (const mt_sp_inode_t *)sp_page_c(sp, page_idx);
        page_idx = inode->children[inode->nkeys];
    }
    return page_idx;
}

/* ── Initialisation ──────────────────────────────────────────── */

void mt_sp_init(void *sp)
{
    memset(sp, 0, MT_SP_SIZE);
    mt_sp_header_t *hdr = sp_hdr(sp);
    hdr->type = MT_NODE_LEAF;
    hdr->page_bitmap[0] = 1;  /* bit 0 = header page */

    int root = sp_page_alloc(hdr);
    mt_lnode_t *page = (mt_lnode_t *)sp_page(sp, root);
    mt_page_init(page);
    hdr->root_page = (uint16_t)root;
    hdr->sub_height = 0;
    hdr->nkeys = 0;
}

/* ── Search ──────────────────────────────────────────────────── */

bool mt_sp_search_key(const void *sp, int32_t key, int32_t *result)
{
    const mt_sp_header_t *hdr = sp_hdr_c(sp);
    if (hdr->nkeys == 0) return false;

    mt_sp_path_t path[MT_SP_MAX_HEIGHT];
    int path_len;
    int leaf_idx = sp_find_leaf(sp, key, path, &path_len);
    const mt_lnode_t *page = (const mt_lnode_t *)sp_page_c(sp, leaf_idx);

    if (mt_page_search_key(page, key, result))
        return true;

    /* Key smaller than all in this page.  Check prev page leaf. */
    if (page->header.prev) {
        const mt_lnode_t *prev = page->header.prev;
        if (prev->header.nkeys > 0) {
            /* Get max key of previous page via rightmost CL leaf. */
            int slot = prev->header.root_slot;
            const mt_cl_slot_t *s = &prev->slots[slot - 1];
            while (s->type == MT_CL_INTERNAL) {
                slot = s->inode.children[s->inode.nkeys];
                s = &prev->slots[slot - 1];
            }
            if (s->leaf.nkeys > 0) {
                if (result) *result = s->leaf.keys[s->leaf.nkeys - 1];
                return true;
            }
        }
    }

    return false;
}

bool mt_sp_contains(const void *sp, int32_t key)
{
    const mt_sp_header_t *hdr = sp_hdr_c(sp);
    if (hdr->nkeys == 0) return false;

    mt_sp_path_t path[MT_SP_MAX_HEIGHT];
    int path_len;
    int leaf_idx = sp_find_leaf(sp, key, path, &path_len);
    const mt_lnode_t *page = (const mt_lnode_t *)sp_page_c(sp, leaf_idx);
    return mt_page_contains(page, key);
}

/* ── Insert ──────────────────────────────────────────────────── */

mt_status_t mt_sp_insert(void *sp, int32_t key, const mt_hierarchy_t *hier)
{
    (void)hier;
    mt_sp_header_t *hdr = sp_hdr(sp);
    mt_sp_path_t path[MT_SP_MAX_HEIGHT];
    int path_len;
    int leaf_idx = sp_find_leaf(sp, key, path, &path_len);
    mt_lnode_t *page = (mt_lnode_t *)sp_page(sp, leaf_idx);

    mt_status_t status = mt_page_insert(page, key);

    if (status == MT_DUPLICATE) return MT_DUPLICATE;
    if (status == MT_OK) {
        hdr->nkeys++;
        return MT_OK;
    }

    /* MT_PAGE_FULL: split the page leaf within the superpage. */
    int new_idx = sp_page_alloc(hdr);
    if (new_idx == 0)
        return MT_PAGE_FULL;  /* superpage out of pages */

    mt_lnode_t *new_page = (mt_lnode_t *)sp_page(sp, new_idx);

    /* Save linked list pointers (bulk_load zeroes the page). */
    mt_lnode_t *saved_prev = page->header.prev;
    mt_lnode_t *saved_next = page->header.next;

    int32_t sep = mt_page_split(page, new_page);

    if (key < sep)
        mt_page_insert(page, key);
    else
        mt_page_insert(new_page, key);
    hdr->nkeys++;

    /* Restore linked list: splice new_page after page. */
    page->header.prev = saved_prev;
    new_page->header.next = saved_next;
    new_page->header.prev = page;
    if (saved_next)
        saved_next->header.prev = new_page;
    page->header.next = new_page;

    /* Propagate split through page-level internals. */
    sep = mt_page_min_key(new_page);
    uint16_t right_page = (uint16_t)new_idx;

    for (int level = path_len - 1; level >= 0; level--) {
        mt_sp_inode_t *parent =
            (mt_sp_inode_t *)sp_page(sp, path[level].page_idx);

        if (parent->nkeys < MT_SP_MAX_IKEYS) {
            int pos = 0;
            while (pos < parent->nkeys && parent->keys[pos] < sep)
                pos++;
            sp_inode_insert_at(parent, pos, sep, right_page);
            return MT_OK;
        }

        /* Page-level internal overflow — split it.
           In practice this never happens (one internal can hold 682
           children, but we only have 511 usable pages). */
        int pn = parent->nkeys;
        int pos = 0;
        while (pos < pn && parent->keys[pos] < sep)
            pos++;

        int total = pn + 1;
        int left_keys = total / 2;
        int right_keys = total - left_keys - 1;

        int new_inode_idx = sp_page_alloc(hdr);
        if (new_inode_idx == 0)
            return MT_PAGE_FULL;

        mt_sp_inode_t *new_inode =
            (mt_sp_inode_t *)sp_page(sp, new_inode_idx);
        memset(new_inode, 0, MT_PAGE_SIZE);
        new_inode->type = 2;

        /* Build merged arrays on heap (too large for stack). */
        int32_t *all_keys = malloc((size_t)(pn + 1) * sizeof(int32_t));
        uint16_t *all_ch = malloc((size_t)(pn + 2) * sizeof(uint16_t));
        if (!all_keys || !all_ch) {
            free(all_keys); free(all_ch);
            return MT_PAGE_FULL;
        }

        memcpy(all_keys, parent->keys, (size_t)pos * sizeof(int32_t));
        all_keys[pos] = sep;
        memcpy(all_keys + pos + 1, parent->keys + pos,
               (size_t)(pn - pos) * sizeof(int32_t));

        memcpy(all_ch, parent->children,
               (size_t)(pos + 1) * sizeof(uint16_t));
        all_ch[pos + 1] = right_page;
        memcpy(all_ch + pos + 2, parent->children + pos + 1,
               (size_t)(pn - pos) * sizeof(uint16_t));

        sep = all_keys[left_keys];

        memcpy(parent->keys, all_keys,
               (size_t)left_keys * sizeof(int32_t));
        memcpy(parent->children, all_ch,
               (size_t)(left_keys + 1) * sizeof(uint16_t));
        parent->nkeys = (uint16_t)left_keys;

        memcpy(new_inode->keys, all_keys + left_keys + 1,
               (size_t)right_keys * sizeof(int32_t));
        memcpy(new_inode->children, all_ch + left_keys + 1,
               (size_t)(right_keys + 1) * sizeof(uint16_t));
        new_inode->nkeys = (uint16_t)right_keys;

        free(all_keys); free(all_ch);
        right_page = (uint16_t)new_inode_idx;
    }

    /* Split reached root — create new root internal. */
    int new_root_idx = sp_page_alloc(hdr);
    if (new_root_idx == 0)
        return MT_PAGE_FULL;

    mt_sp_inode_t *new_root = (mt_sp_inode_t *)sp_page(sp, new_root_idx);
    memset(new_root, 0, MT_PAGE_SIZE);
    new_root->type = 2;
    new_root->keys[0] = sep;
    new_root->children[0] = hdr->root_page;
    new_root->children[1] = right_page;
    new_root->nkeys = 1;
    hdr->root_page = (uint16_t)new_root_idx;
    hdr->sub_height++;

    return MT_OK;
}

/* ── Delete ──────────────────────────────────────────────────── */

mt_status_t mt_sp_delete(void *sp, int32_t key, const mt_hierarchy_t *hier)
{
    mt_sp_header_t *hdr = sp_hdr(sp);
    mt_sp_path_t path[MT_SP_MAX_HEIGHT];
    int path_len;
    int leaf_idx = sp_find_leaf(sp, key, path, &path_len);
    mt_lnode_t *page = (mt_lnode_t *)sp_page(sp, leaf_idx);

    mt_status_t status = mt_page_delete(page, key, hier);
    if (status == MT_NOT_FOUND) return MT_NOT_FOUND;

    hdr->nkeys--;

    if (status == MT_OK || path_len == 0)
        goto check_sp_underflow;

    /* MT_UNDERFLOW at page level — redistribute or merge pages. */
    {
        mt_sp_inode_t *parent =
            (mt_sp_inode_t *)sp_page(sp, path[path_len - 1].page_idx);
        int cidx = path[path_len - 1].child_idx;

        /* Try redistribute from left page sibling. */
        if (cidx > 0) {
            int left_idx = parent->children[cidx - 1];
            mt_lnode_t *left = (mt_lnode_t *)sp_page(sp, left_idx);
            if (left->header.nkeys > hier->min_page_keys) {
                int32_t lkeys[SP_MAX_PAGE_KEYS], rkeys[SP_MAX_PAGE_KEYS];
                int ln = mt_page_extract_sorted(left, lkeys);
                int rn = mt_page_extract_sorted(page, rkeys);
                int total = ln + rn;
                int new_ln = total / 2;
                int move = ln - new_ln;

                int32_t new_right[SP_MAX_PAGE_KEYS];
                memcpy(new_right, lkeys + new_ln,
                       (size_t)move * sizeof(int32_t));
                memcpy(new_right + move, rkeys,
                       (size_t)rn * sizeof(int32_t));

                mt_lnode_t *lp = left->header.prev;
                mt_lnode_t *ln_next = left->header.next;
                mt_lnode_t *rp = page->header.prev;
                mt_lnode_t *rn_next = page->header.next;

                mt_page_bulk_load(left, lkeys, new_ln);
                mt_page_bulk_load(page, new_right, move + rn);

                left->header.prev = lp;
                left->header.next = ln_next;
                page->header.prev = rp;
                page->header.next = rn_next;

                parent->keys[cidx - 1] = new_right[0];
                goto check_sp_underflow;
            }
        }

        /* Try redistribute from right page sibling. */
        if (cidx < parent->nkeys) {
            int right_idx = parent->children[cidx + 1];
            mt_lnode_t *right = (mt_lnode_t *)sp_page(sp, right_idx);
            if (right->header.nkeys > hier->min_page_keys) {
                int32_t lkeys[SP_MAX_PAGE_KEYS], rkeys[SP_MAX_PAGE_KEYS];
                int ln = mt_page_extract_sorted(page, lkeys);
                int rn = mt_page_extract_sorted(right, rkeys);
                int total = ln + rn;
                int new_ln = total / 2;
                int move = new_ln - ln;

                int32_t new_left[SP_MAX_PAGE_KEYS];
                memcpy(new_left, lkeys, (size_t)ln * sizeof(int32_t));
                memcpy(new_left + ln, rkeys, (size_t)move * sizeof(int32_t));

                int32_t new_right_keys[SP_MAX_PAGE_KEYS];
                int new_rn = rn - move;
                memcpy(new_right_keys, rkeys + move,
                       (size_t)new_rn * sizeof(int32_t));

                mt_lnode_t *lp = page->header.prev;
                mt_lnode_t *ln_next = page->header.next;
                mt_lnode_t *rp = right->header.prev;
                mt_lnode_t *rn_next = right->header.next;

                mt_page_bulk_load(page, new_left, new_ln);
                mt_page_bulk_load(right, new_right_keys, new_rn);

                page->header.prev = lp;
                page->header.next = ln_next;
                right->header.prev = rp;
                right->header.next = rn_next;

                parent->keys[cidx] = new_right_keys[0];
                goto check_sp_underflow;
            }
        }

        /* Merge pages. */
        if (cidx > 0) {
            /* Merge current into left sibling. */
            int left_idx = parent->children[cidx - 1];
            mt_lnode_t *left = (mt_lnode_t *)sp_page(sp, left_idx);

            int32_t lkeys[SP_MAX_PAGE_KEYS], rkeys[SP_MAX_PAGE_KEYS];
            int ln = mt_page_extract_sorted(left, lkeys);
            int rn = mt_page_extract_sorted(page, rkeys);

            int32_t merged[SP_MAX_PAGE_KEYS * 2];
            memcpy(merged, lkeys, (size_t)ln * sizeof(int32_t));
            memcpy(merged + ln, rkeys, (size_t)rn * sizeof(int32_t));

            mt_lnode_t *lp = left->header.prev;

            mt_page_bulk_load(left, merged, ln + rn);

            left->header.prev = lp;
            left->header.next = page->header.next;
            if (page->header.next)
                page->header.next->header.prev = left;

            sp_inode_remove_at(parent, cidx - 1);
            sp_page_free(hdr, leaf_idx);
        } else {
            /* Merge right sibling into current. */
            int right_idx = parent->children[cidx + 1];
            mt_lnode_t *right = (mt_lnode_t *)sp_page(sp, right_idx);

            int32_t lkeys[SP_MAX_PAGE_KEYS], rkeys[SP_MAX_PAGE_KEYS];
            int ln = mt_page_extract_sorted(page, lkeys);
            int rn = mt_page_extract_sorted(right, rkeys);

            int32_t merged[SP_MAX_PAGE_KEYS * 2];
            memcpy(merged, lkeys, (size_t)ln * sizeof(int32_t));
            memcpy(merged + ln, rkeys, (size_t)rn * sizeof(int32_t));

            mt_lnode_t *lp = page->header.prev;

            mt_page_bulk_load(page, merged, ln + rn);

            page->header.prev = lp;
            page->header.next = right->header.next;
            if (right->header.next)
                right->header.next->header.prev = page;

            sp_inode_remove_at(parent, cidx);
            sp_page_free(hdr, right_idx);
        }

        /* Collapse root if it has 0 keys. */
        if (hdr->sub_height > 0) {
            mt_sp_inode_t *root =
                (mt_sp_inode_t *)sp_page(sp, hdr->root_page);
            if (root->nkeys == 0) {
                int old_root = hdr->root_page;
                hdr->root_page = root->children[0];
                hdr->sub_height--;
                sp_page_free(hdr, old_root);
            }
        }
    }

check_sp_underflow:
    return (hdr->nkeys < (uint32_t)hier->min_sp_keys) ? MT_UNDERFLOW : MT_OK;
}

/* ── Extract sorted ──────────────────────────────────────────── */

static int sp_extract_subtree(const void *sp, int page_idx, int height,
                               int32_t *out, int pos)
{
    if (height == 0) {
        const mt_lnode_t *page =
            (const mt_lnode_t *)sp_page_c(sp, page_idx);
        int n = mt_page_extract_sorted(page, out + pos);
        return pos + n;
    }

    const mt_sp_inode_t *inode =
        (const mt_sp_inode_t *)sp_page_c(sp, page_idx);
    for (int i = 0; i <= inode->nkeys; i++)
        pos = sp_extract_subtree(sp, inode->children[i], height - 1,
                                  out, pos);
    return pos;
}

int mt_sp_extract_sorted(const void *sp, int32_t *out)
{
    const mt_sp_header_t *hdr = sp_hdr_c(sp);
    if (hdr->nkeys == 0) return 0;
    return sp_extract_subtree(sp, hdr->root_page, hdr->sub_height, out, 0);
}

/* ── Bulk load ───────────────────────────────────────────────── */

void mt_sp_bulk_load(void *sp, const int32_t *keys, int nkeys,
                      const mt_hierarchy_t *hier)
{
    memset(sp, 0, MT_SP_SIZE);
    mt_sp_header_t *hdr = sp_hdr(sp);
    hdr->type = MT_NODE_LEAF;
    hdr->page_bitmap[0] = 1;  /* bit 0 = header */

    if (nkeys == 0) {
        int root = sp_page_alloc(hdr);
        mt_lnode_t *page = (mt_lnode_t *)sp_page(sp, root);
        mt_page_init(page);
        hdr->root_page = (uint16_t)root;
        hdr->sub_height = 0;
        hdr->nkeys = 0;
        return;
    }

    int max_per_page = hier->page_max_keys;
    int nleaves = (nkeys + max_per_page - 1) / max_per_page;
    int keys_per = nkeys / nleaves;
    int extra = nkeys % nleaves;

    uint16_t *leaf_pages = malloc((size_t)nleaves * sizeof(uint16_t));
    int32_t *seps = malloc((size_t)nleaves * sizeof(int32_t));
    if (!leaf_pages || !seps) {
        free(leaf_pages); free(seps);
        return;
    }

    int offset = 0;
    for (int i = 0; i < nleaves; i++) {
        int k = keys_per + (i < extra ? 1 : 0);
        int pidx = sp_page_alloc(hdr);
        mt_lnode_t *page = (mt_lnode_t *)sp_page(sp, pidx);
        mt_page_bulk_load(page, keys + offset, k);
        leaf_pages[i] = (uint16_t)pidx;
        seps[i] = keys[offset];
        offset += k;
    }

    hdr->nkeys = (uint32_t)nkeys;

    /* Link page leaves within superpage. */
    for (int i = 0; i < nleaves; i++) {
        mt_lnode_t *page = (mt_lnode_t *)sp_page(sp, leaf_pages[i]);
        page->header.prev = (i > 0)
            ? (mt_lnode_t *)sp_page(sp, leaf_pages[i - 1]) : NULL;
        page->header.next = (i < nleaves - 1)
            ? (mt_lnode_t *)sp_page(sp, leaf_pages[i + 1]) : NULL;
    }

    if (nleaves == 1) {
        hdr->root_page = leaf_pages[0];
        hdr->sub_height = 0;
        free(leaf_pages); free(seps);
        return;
    }

    /* Build page-level internals bottom-up. */
    uint16_t *cur_pages = leaf_pages;
    int32_t *cur_seps = seps;
    int level_count = nleaves;
    int height = 0;

    while (level_count > 1) {
        int cap = MT_SP_MAX_IKEYS + 1;
        int num_parents = (level_count + cap - 1) / cap;
        if (num_parents == 0) num_parents = 1;

        uint16_t *next_pages = malloc((size_t)num_parents * sizeof(uint16_t));
        int32_t *next_seps = malloc((size_t)num_parents * sizeof(int32_t));
        int children_per = level_count / num_parents;
        int extra_c = level_count % num_parents;
        int ci = 0;

        for (int p = 0; p < num_parents; p++) {
            int nc = children_per + (p < extra_c ? 1 : 0);
            int pidx = sp_page_alloc(hdr);
            mt_sp_inode_t *inode = (mt_sp_inode_t *)sp_page(sp, pidx);
            memset(inode, 0, MT_PAGE_SIZE);
            inode->type = 2;

            inode->children[0] = cur_pages[ci];
            for (int j = 1; j < nc; j++) {
                inode->keys[j - 1] = cur_seps[ci + j];
                inode->children[j] = cur_pages[ci + j];
            }
            inode->nkeys = (uint16_t)(nc - 1);

            next_pages[p] = (uint16_t)pidx;
            next_seps[p] = cur_seps[ci];
            ci += nc;
        }

        free(cur_pages); free(cur_seps);
        cur_pages = next_pages;
        cur_seps = next_seps;
        level_count = num_parents;
        height++;
    }

    hdr->root_page = cur_pages[0];
    hdr->sub_height = (uint8_t)height;
    free(cur_pages); free(cur_seps);
}

/* ── Split ───────────────────────────────────────────────────── */

int32_t mt_sp_split(void *sp, void *new_sp, const mt_hierarchy_t *hier)
{
    int total = (int)sp_hdr(sp)->nkeys;
    int32_t *all_keys = malloc((size_t)total * sizeof(int32_t));
    if (!all_keys) return 0;

    int n = mt_sp_extract_sorted(sp, all_keys);
    int left_n = n / 2;
    int right_n = n - left_n;

    mt_sp_bulk_load(sp, all_keys, left_n, hier);
    mt_sp_bulk_load(new_sp, all_keys + left_n, right_n, hier);

    int32_t sep = all_keys[left_n];
    free(all_keys);
    return sep;
}

/* ── Min key ─────────────────────────────────────────────────── */

int32_t mt_sp_min_key(const void *sp)
{
    const mt_sp_header_t *hdr = sp_hdr_c(sp);
    if (hdr->nkeys == 0) return MT_KEY_MAX;

    int leaf_idx = sp_leftmost_leaf(sp);
    const mt_lnode_t *page = (const mt_lnode_t *)sp_page_c(sp, leaf_idx);
    return mt_page_min_key(page);
}

int32_t mt_sp_max_key(const void *sp)
{
    const mt_sp_header_t *hdr = sp_hdr_c(sp);
    if (hdr->nkeys == 0) return INT32_MIN;

    int leaf_idx = sp_rightmost_leaf(sp);
    const mt_lnode_t *page = (const mt_lnode_t *)sp_page_c(sp, leaf_idx);

    /* Walk to rightmost CL leaf within the page. */
    int slot = page->header.root_slot;
    const mt_cl_slot_t *s = &page->slots[slot - 1];
    while (s->type == MT_CL_INTERNAL) {
        slot = s->inode.children[s->inode.nkeys];
        s = &page->slots[slot - 1];
    }
    return (s->leaf.nkeys > 0) ? s->leaf.keys[s->leaf.nkeys - 1] : INT32_MIN;
}

/* ── Iterator helpers ────────────────────────────────────────── */

mt_lnode_t *mt_sp_first_leaf(void *sp)
{
    int leaf_idx = sp_leftmost_leaf(sp);
    return (mt_lnode_t *)sp_page(sp, leaf_idx);
}

mt_lnode_t *mt_sp_find_leaf(void *sp, int32_t key)
{
    mt_sp_path_t path[MT_SP_MAX_HEIGHT];
    int path_len;
    int leaf_idx = sp_find_leaf(sp, key, path, &path_len);
    return (mt_lnode_t *)sp_page(sp, leaf_idx);
}
