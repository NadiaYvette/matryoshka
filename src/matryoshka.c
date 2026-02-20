/*
 * matryoshka.c — B+ tree operations with matryoshka-nested leaf nodes.
 *
 * Core operations: create, bulk_load, destroy, search, insert, delete,
 * and iteration.  Leaf nodes contain a B+ sub-tree of cache-line-sized
 * sub-nodes, giving O(log b) intra-node operations instead of O(B)
 * flat-array rebuilds.
 */

#include "matryoshka_internal.h"
#include <stdlib.h>
#include <string.h>

/* ── Constants ────────────────────────────────────────────────── */

#define MT_MAX_HEIGHT 32

/* Absolute maximum keys extractable from a page (all 63 slots as leaves). */
#define MT_MAX_PAGE_KEYS (MT_PAGE_SLOTS * MT_CL_KEY_CAP)  /* 945 */

/* ── Path tracking for insert/delete ──────────────────────────── */

typedef struct {
    mt_inode_t *node;
    int         idx;     /* child index taken */
} mt_path_t;

/* ── Internal helpers ─────────────────────────────────────────── */

/* Re-tag a leaf child pointer in its parent after the leaf's
   root_slot/sub_height may have changed (e.g. CL root split). */
static inline void retag_leaf_in_parent(mt_path_t *path, int height)
{
    if (height > 0) {
        mt_inode_t *parent = path[height - 1].node;
        int cidx = path[height - 1].idx;
        parent->children[cidx] = mt_tag_leaf_ptr(
            mt_untag(parent->children[cidx]));
    }
}

/* Walk the tree from root to the leaf that should contain `key`,
   recording the path of internal nodes and child indices taken. */
static mt_lnode_t *find_leaf(mt_node_t *root, int height, int32_t key,
                              mt_path_t *path)
{
    mt_node_t *node = root;
    for (int i = 0; i < height; i++) {
        mt_inode_t *in = &node->inode;
        int idx = mt_inode_search(in, key);
        path[i].node = in;
        path[i].idx = idx;
        mt_node_t *raw = in->children[idx];
        if (i == height - 1) {
            /* Child is a leaf — extract root_slot from tag and prefetch
               the CL root cache line in parallel with the page header. */
            uint8_t rs = mt_ptr_root_slot(raw);
            node = mt_untag(raw);
            __builtin_prefetch(node, 0, 1);  /* page header line */
            if (rs > 0)
                __builtin_prefetch(&node->lnode.slots[rs - 1], 0, 1);
        } else {
            node = raw;  /* internal nodes are untagged */
            __builtin_prefetch(node, 0, 1);
        }
    }
    return &node->lnode;
}

/* Insert a separator key and right child pointer into an internal node
   at the given position. Caller must ensure the node has room. */
static void inode_insert_at(mt_inode_t *node, int pos, int32_t key,
                             mt_node_t *right_child)
{
    int n = node->nkeys;
    memmove(node->keys + pos + 1, node->keys + pos,
            (size_t)(n - pos) * sizeof(int32_t));
    memmove(node->children + pos + 2, node->children + pos + 1,
            (size_t)(n - pos) * sizeof(mt_node_t *));
    node->keys[pos] = key;
    node->children[pos + 1] = right_child;
    node->nkeys = (uint16_t)(n + 1);
}

/* Remove a separator key and right child pointer from an internal node
   at the given position. */
static void inode_remove_at(mt_inode_t *node, int pos)
{
    int n = node->nkeys;
    memmove(node->keys + pos, node->keys + pos + 1,
            (size_t)(n - pos - 1) * sizeof(int32_t));
    memmove(node->children + pos + 1, node->children + pos + 2,
            (size_t)(n - pos - 1) * sizeof(mt_node_t *));
    node->nkeys = (uint16_t)(n - 1);
}

/* Get the maximum key in a leaf page by walking to the rightmost CL leaf. */
static int32_t page_max_key(const mt_lnode_t *page)
{
    int slot = page->header.root_slot;
    const mt_cl_slot_t *s = &page->slots[slot - 1];
    while (s->type == MT_CL_INTERNAL) {
        slot = s->inode.children[s->inode.nkeys];
        s = &page->slots[slot - 1];
    }
    return (s->leaf.nkeys > 0) ? s->leaf.keys[s->leaf.nkeys - 1] : MT_KEY_MAX;
}

/* ── Lifecycle ────────────────────────────────────────────────── */

matryoshka_tree_t *matryoshka_create_with(const mt_hierarchy_t *hier)
{
    matryoshka_tree_t *tree = malloc(sizeof(*tree));
    if (!tree) return NULL;
    tree->hier = *hier;
    tree->n = 0;
    tree->height = 0;

    /* Create arena allocator for superpage leaves. */
    if (hier->leaf_alloc > MT_PAGE_SIZE) {
        tree->alloc = mt_allocator_create(hier->leaf_alloc, hier->leaf_alloc);
    } else if (hier->leaf_alloc == MT_PAGE_SIZE) {
        /* Use arena for co-locating page-sized leaves within 2 MiB regions. */
        tree->alloc = mt_allocator_create(2u * 1024 * 1024, MT_PAGE_SIZE);
    } else {
        tree->alloc = NULL;
    }

    tree->root = mt_alloc_lnode(&tree->hier, tree->alloc);
    if (!tree->root) {
        if (tree->alloc) mt_allocator_destroy(tree->alloc);
        free(tree);
        return NULL;
    }
    if (hier->use_superpages)
        mt_sp_init(tree->root);
    else
        mt_page_init(&tree->root->lnode);
    return tree;
}

matryoshka_tree_t *matryoshka_create(void)
{
    mt_hierarchy_t hier;
    mt_hierarchy_init_default(&hier);
    return matryoshka_create_with(&hier);
}

static void free_subtree(mt_node_t *node, int height, mt_allocator_t *alloc)
{
    if (height > 0) {
        mt_inode_t *in = &node->inode;
        for (int i = 0; i <= in->nkeys; i++)
            free_subtree(mt_untag(in->children[i]), height - 1, alloc);
        mt_free_inode(node);
    } else {
        mt_free_lnode(node, alloc);
    }
}

void matryoshka_destroy(matryoshka_tree_t *tree)
{
    if (!tree) return;
    if (tree->root)
        free_subtree(tree->root, tree->height, tree->alloc);
    if (tree->alloc)
        mt_allocator_destroy(tree->alloc);
    free(tree);
}

/* ── Bulk load ────────────────────────────────────────────────── */

typedef struct {
    mt_node_t *node;
    int32_t    min_key;
} build_entry_t;

matryoshka_tree_t *matryoshka_bulk_load_with(const int32_t *sorted_keys,
                                              size_t n,
                                              const mt_hierarchy_t *hier)
{
    matryoshka_tree_t *tree = malloc(sizeof(*tree));
    if (!tree) return NULL;
    tree->hier = *hier;
    tree->n = n;

    /* Initialise arena allocator. */
    if (hier->leaf_alloc > MT_PAGE_SIZE) {
        tree->alloc = mt_allocator_create(hier->leaf_alloc, hier->leaf_alloc);
    } else if (hier->leaf_alloc == MT_PAGE_SIZE) {
        tree->alloc = mt_allocator_create(2u * 1024 * 1024, MT_PAGE_SIZE);
    } else {
        tree->alloc = NULL;
    }

    int max_lkeys = hier->use_superpages ? hier->sp_max_keys
                                          : hier->page_max_keys;

    if (n == 0) {
        tree->root = mt_alloc_lnode(&tree->hier, tree->alloc);
        if (hier->use_superpages)
            mt_sp_init(tree->root);
        else
            mt_page_init(&tree->root->lnode);
        tree->height = 0;
        return tree;
    }

    /* Distribute keys across leaves. */
    size_t nleaves = (n + (size_t)max_lkeys - 1) / (size_t)max_lkeys;
    size_t keys_per = n / nleaves;
    size_t extra = n % nleaves;

    build_entry_t *entries = malloc(nleaves * sizeof(build_entry_t));
    if (!entries) { free(tree); return NULL; }

    size_t offset = 0;
    for (size_t i = 0; i < nleaves; i++) {
        size_t k = keys_per + (i < extra ? 1 : 0);
        mt_node_t *lnode = mt_alloc_lnode(&tree->hier, tree->alloc);
        if (hier->use_superpages)
            mt_sp_bulk_load(lnode, sorted_keys + offset, (int)k, hier);
        else
            mt_page_bulk_load(&lnode->lnode, sorted_keys + offset, (int)k);
        entries[i].node = lnode;
        entries[i].min_key = sorted_keys[offset];
        offset += k;
    }

    /* Link leaves. */
    if (hier->use_superpages) {
        /* Link superpages via sp_header prev/next. */
        for (size_t i = 0; i < nleaves; i++) {
            mt_sp_header_t *sp = (mt_sp_header_t *)entries[i].node;
            sp->prev = (i > 0) ? (mt_sp_header_t *)entries[i - 1].node : NULL;
            sp->next = (i < nleaves - 1)
                ? (mt_sp_header_t *)entries[i + 1].node : NULL;
        }
        /* Link page leaves across superpage boundaries. */
        for (size_t i = 0; i + 1 < nleaves; i++) {
            mt_lnode_t *last = mt_sp_first_leaf(entries[i].node);
            while (last->header.next) last = last->header.next;
            mt_lnode_t *first = mt_sp_first_leaf(entries[i + 1].node);
            last->header.next = first;
            first->header.prev = last;
        }
    } else {
        for (size_t i = 0; i < nleaves; i++) {
            mt_lnode_t *l = &entries[i].node->lnode;
            l->header.prev = (i > 0) ? &entries[i - 1].node->lnode : NULL;
            l->header.next = (i < nleaves - 1)
                ? &entries[i + 1].node->lnode : NULL;
        }
    }

    /* Build internal levels bottom-up. */
    size_t level_count = nleaves;
    int height = 0;

    while (level_count > 1) {
        size_t num_parents = (level_count + MT_MAX_IKEYS) / (MT_MAX_IKEYS + 1);
        if (num_parents == 0) num_parents = 1;

        build_entry_t *new_entries = malloc(num_parents * sizeof(build_entry_t));
        if (!new_entries) {
            free(entries);
            free(tree);
            return NULL;
        }

        size_t children_per = level_count / num_parents;
        size_t extra_c = level_count % num_parents;

        size_t ci = 0;
        for (size_t p = 0; p < num_parents; p++) {
            size_t nc = children_per + (p < extra_c ? 1 : 0);

            mt_node_t *parent = mt_alloc_inode();
            mt_inode_t *in = &parent->inode;

            if (height == 0 && !hier->use_superpages) {
                /* Tag leaf pointers with root_slot/sub_height. */
                in->children[0] = mt_tag_leaf_ptr(entries[ci].node);
                for (size_t j = 1; j < nc; j++) {
                    in->keys[j - 1] = entries[ci + j].min_key;
                    in->children[j] = mt_tag_leaf_ptr(entries[ci + j].node);
                }
            } else {
                in->children[0] = entries[ci].node;
                for (size_t j = 1; j < nc; j++) {
                    in->keys[j - 1] = entries[ci + j].min_key;
                    in->children[j] = entries[ci + j].node;
                }
            }
            in->nkeys = (uint16_t)(nc - 1);

            new_entries[p].node = parent;
            new_entries[p].min_key = entries[ci].min_key;
            ci += nc;
        }

        free(entries);
        entries = new_entries;
        level_count = num_parents;
        height++;
    }

    tree->root = entries[0].node;
    tree->height = height;
    free(entries);
    return tree;
}

matryoshka_tree_t *matryoshka_bulk_load(const int32_t *sorted_keys, size_t n)
{
    mt_hierarchy_t hier;
    mt_hierarchy_init_default(&hier);
    return matryoshka_bulk_load_with(sorted_keys, n, &hier);
}

/* ── Query ────────────────────────────────────────────────────── */

size_t matryoshka_size(const matryoshka_tree_t *tree)
{
    return tree ? tree->n : 0;
}

bool matryoshka_search(const matryoshka_tree_t *tree, int32_t key,
                        int32_t *result)
{
    if (!tree || tree->n == 0)
        return false;

    /* Walk to leaf. */
    mt_node_t *node = tree->root;
    for (int i = 0; i < tree->height; i++) {
        int idx = mt_inode_search(&node->inode, key);
        node = mt_untag(node->inode.children[idx]);
    }

    if (tree->hier.use_superpages) {
        if (mt_sp_search_key(node, key, result))
            return true;
        mt_sp_header_t *sp = (mt_sp_header_t *)node;
        if (sp->prev && sp->prev->nkeys > 0) {
            if (result) *result = mt_sp_max_key(sp->prev);
            return true;
        }
        return false;
    }

    mt_lnode_t *leaf = &node->lnode;

    /* Predecessor search within the page sub-tree. */
    if (mt_page_search_key(leaf, key, result))
        return true;

    /* Key is smaller than all keys in this leaf.  Check previous leaf. */
    if (leaf->header.prev) {
        mt_lnode_t *prev = leaf->header.prev;
        if (prev->header.nkeys > 0) {
            if (result) *result = page_max_key(prev);
            return true;
        }
    }

    return false;
}

bool matryoshka_contains(const matryoshka_tree_t *tree, int32_t key)
{
    if (!tree || tree->n == 0)
        return false;

    /* Walk to leaf. */
    mt_node_t *node = tree->root;
    for (int i = 0; i < tree->height; i++) {
        int idx = mt_inode_search(&node->inode, key);
        node = mt_untag(node->inode.children[idx]);
    }

    if (tree->hier.use_superpages)
        return mt_sp_contains(node, key);
    return mt_page_contains(&node->lnode, key);
}

/* ── Split propagation helper ─────────────────────────────────── */

/* Propagate a leaf split up through internal nodes.
   `sep` is the separator key; `right_child` is the new right node. */
static void propagate_leaf_split(matryoshka_tree_t *tree, mt_path_t *path,
                                  int32_t sep, mt_node_t *right_child)
{
    for (int level = tree->height - 1; level >= 0; level--) {
        mt_inode_t *parent = path[level].node;

        if (parent->nkeys < MT_MAX_IKEYS) {
            int pos = 0;
            while (pos < parent->nkeys && parent->keys[pos] < sep)
                pos++;
            inode_insert_at(parent, pos, sep, right_child);
            return;
        }

        /* Internal node overflow: split it. */
        int pn = parent->nkeys;
        int32_t all_keys[MT_MAX_IKEYS + 1];
        mt_node_t *all_children[MT_MAX_IKEYS + 2];

        int pos = 0;
        while (pos < pn && parent->keys[pos] < sep)
            pos++;

        memcpy(all_keys, parent->keys, (size_t)pos * sizeof(int32_t));
        all_keys[pos] = sep;
        memcpy(all_keys + pos + 1, parent->keys + pos,
               (size_t)(pn - pos) * sizeof(int32_t));

        memcpy(all_children, parent->children,
               (size_t)(pos + 1) * sizeof(mt_node_t *));
        all_children[pos + 1] = right_child;
        memcpy(all_children + pos + 2, parent->children + pos + 1,
               (size_t)(pn - pos) * sizeof(mt_node_t *));

        int total = pn + 1;
        int left_keys = total / 2;
        int right_keys = total - left_keys - 1;
        sep = all_keys[left_keys];

        memcpy(parent->keys, all_keys, (size_t)left_keys * sizeof(int32_t));
        memcpy(parent->children, all_children,
               (size_t)(left_keys + 1) * sizeof(mt_node_t *));
        parent->nkeys = (uint16_t)left_keys;

        mt_node_t *new_rinode = mt_alloc_inode();
        mt_inode_t *ri = &new_rinode->inode;
        memcpy(ri->keys, all_keys + left_keys + 1,
               (size_t)right_keys * sizeof(int32_t));
        memcpy(ri->children, all_children + left_keys + 1,
               (size_t)(right_keys + 1) * sizeof(mt_node_t *));
        ri->nkeys = (uint16_t)right_keys;

        right_child = new_rinode;
    }

    /* Root split: create a new root. */
    mt_node_t *new_root = mt_alloc_inode();
    mt_inode_t *nr = &new_root->inode;
    nr->keys[0] = sep;
    /* When height == 0, old root and right_child are leaves — tag them.
       (right_child is already tagged by caller for non-superpage splits.) */
    nr->children[0] = (tree->height == 0 && !tree->hier.use_superpages)
                       ? mt_tag_leaf_ptr(tree->root) : tree->root;
    nr->children[1] = right_child;
    nr->nkeys = 1;
    tree->root = new_root;
    tree->height++;
}

/* Split a full superpage, insert key into the correct half,
   link the new superpage, and propagate the split upward. */
static void split_sp_and_insert(matryoshka_tree_t *tree, mt_path_t *path,
                                  mt_node_t *sp_node, int32_t key)
{
    mt_sp_header_t *sp = (mt_sp_header_t *)sp_node;

    /* Save inter-superpage linked list pointers. */
    mt_sp_header_t *saved_prev = sp->prev;
    mt_sp_header_t *saved_next = sp->next;

    /* Save cross-superpage page-leaf links. */
    mt_lnode_t *first_leaf = mt_sp_first_leaf(sp_node);
    mt_lnode_t *fl_prev = first_leaf->header.prev;  /* prev sp's last page */

    mt_node_t *new_rnode = mt_alloc_lnode(&tree->hier, tree->alloc);
    mt_sp_header_t *new_right = (mt_sp_header_t *)new_rnode;

    int32_t sep = mt_sp_split(sp_node, new_rnode, &tree->hier);

    if (key < sep)
        mt_sp_insert(sp_node, key, &tree->hier);
    else
        mt_sp_insert(new_rnode, key, &tree->hier);

    /* Restore inter-superpage linked list. */
    sp->prev = saved_prev;
    new_right->next = saved_next;
    new_right->prev = sp;
    if (saved_next) saved_next->prev = new_right;
    sp->next = new_right;

    /* Restore cross-superpage page-leaf links. */
    mt_lnode_t *left_last = mt_sp_first_leaf(sp_node);
    /* Navigate to last page leaf of left sp. */
    while (left_last->header.next) left_last = left_last->header.next;
    mt_lnode_t *right_first = mt_sp_first_leaf(new_rnode);
    mt_lnode_t *right_last = right_first;
    while (right_last->header.next) right_last = right_last->header.next;

    /* Chain: prev_sp_last <-> left_first ... left_last <-> right_first ... right_last <-> next_sp_first */
    left_last->header.next = right_first;
    right_first->header.prev = left_last;

    /* Restore link from previous superpage. */
    mt_lnode_t *new_first = mt_sp_first_leaf(sp_node);
    new_first->header.prev = fl_prev;
    if (fl_prev) fl_prev->header.next = new_first;

    /* Link to next superpage's first page leaf. */
    if (saved_next && saved_next->nkeys > 0) {
        mt_lnode_t *next_first = mt_sp_first_leaf((void *)saved_next);
        right_last->header.next = next_first;
        next_first->header.prev = right_last;
    }

    sep = mt_sp_min_key(new_rnode);
    propagate_leaf_split(tree, path, sep, new_rnode);
}

/* Split a full leaf, insert key into the correct half,
   link the new leaf, and propagate the split upward. */
static void split_leaf_and_insert(matryoshka_tree_t *tree, mt_path_t *path,
                                    mt_lnode_t *leaf, int32_t key)
{
    /* Save linked list pointers before split (bulk_load zeroes the page). */
    mt_lnode_t *saved_prev = leaf->header.prev;
    mt_lnode_t *saved_next = leaf->header.next;

    mt_node_t *new_rnode = mt_alloc_lnode(&tree->hier, tree->alloc);
    mt_lnode_t *new_right = &new_rnode->lnode;

    int32_t sep = mt_page_split(leaf, new_right);

    if (key < sep)
        mt_page_insert(leaf, key);
    else
        mt_page_insert(new_right, key);

    /* Restore linked list and splice in new_right after leaf. */
    leaf->header.prev = saved_prev;
    new_right->header.next = saved_next;
    new_right->header.prev = leaf;
    if (saved_next)
        saved_next->header.prev = new_right;
    leaf->header.next = new_right;

    sep = mt_page_min_key(new_right);
    /* Re-tag existing left leaf (root_slot may have changed after insert). */
    retag_leaf_in_parent(path, tree->height);
    propagate_leaf_split(tree, path, sep, mt_tag_leaf_ptr(new_rnode));
}

/* ── Insert ───────────────────────────────────────────────────── */

bool matryoshka_insert(matryoshka_tree_t *tree, int32_t key)
{
    if (!tree) return false;

    mt_path_t path[MT_MAX_HEIGHT];

    if (tree->hier.use_superpages) {
        /* Walk to superpage leaf. */
        mt_node_t *node = tree->root;
        for (int i = 0; i < tree->height; i++) {
            int idx = mt_inode_search(&node->inode, key);
            path[i].node = &node->inode;
            path[i].idx = idx;
            node = mt_untag(node->inode.children[idx]);
        }

        mt_status_t status = mt_sp_insert(node, key, &tree->hier);
        if (status == MT_DUPLICATE) return false;
        if (status == MT_OK) { tree->n++; return true; }

        /* MT_PAGE_FULL: superpage out of pages — split. */
        split_sp_and_insert(tree, path, node, key);
        tree->n++;
        return true;
    }

    mt_lnode_t *leaf = find_leaf(tree->root, tree->height, key, path);

    mt_status_t status = mt_page_insert(leaf, key);

    if (status == MT_DUPLICATE)
        return false;

    if (status == MT_OK) {
        tree->n++;
        retag_leaf_in_parent(path, tree->height);
        return true;
    }

    /* MT_PAGE_FULL: split and insert. */
    split_leaf_and_insert(tree, path, leaf, key);
    tree->n++;
    return true;
}

/* ── Delete (Jannink eager deletion) ──────────────────────────── */

/* Rebalance after leaf underflow.  `level` is the path index of the
   leaf's parent (tree->height - 1).  Propagates upward as needed. */
static void rebalance_leaf(matryoshka_tree_t *tree, mt_path_t *path,
                            mt_lnode_t *leaf, int level)
{
    mt_inode_t *parent = path[level].node;
    int cidx = path[level].idx;
    int min_page = tree->hier.min_page_keys;

    /* Try redistribute from left sibling. */
    if (cidx > 0) {
        mt_lnode_t *left = &mt_untag(parent->children[cidx - 1])->lnode;
        if (left->header.nkeys > min_page) {
            int32_t lsorted[MT_MAX_PAGE_KEYS], rsorted[MT_MAX_PAGE_KEYS];
            int ln = mt_page_extract_sorted(left, lsorted);
            int rn = mt_page_extract_sorted(leaf, rsorted);

            int total = ln + rn;
            int new_ln = total / 2;
            int move = ln - new_ln;

            int32_t new_right[MT_MAX_PAGE_KEYS];
            memcpy(new_right, lsorted + new_ln, (size_t)move * sizeof(int32_t));
            memcpy(new_right + move, rsorted, (size_t)rn * sizeof(int32_t));
            int new_rn = move + rn;

            /* Save linked list pointers (bulk_load zeroes the page). */
            struct mt_lnode *lp = left->header.prev;
            struct mt_lnode *ln_next = left->header.next;
            struct mt_lnode *rp = leaf->header.prev;
            struct mt_lnode *rn_next = leaf->header.next;

            mt_page_bulk_load(left, lsorted, new_ln);
            mt_page_bulk_load(leaf, new_right, new_rn);

            left->header.prev = lp;
            left->header.next = ln_next;
            leaf->header.prev = rp;
            leaf->header.next = rn_next;

            parent->keys[cidx - 1] = new_right[0];
            return;
        }
    }

    /* Try redistribute from right sibling. */
    if (cidx < parent->nkeys) {
        mt_lnode_t *right = &mt_untag(parent->children[cidx + 1])->lnode;
        if (right->header.nkeys > min_page) {
            int32_t lsorted[MT_MAX_PAGE_KEYS], rsorted[MT_MAX_PAGE_KEYS];
            int ln = mt_page_extract_sorted(leaf, lsorted);
            int rn = mt_page_extract_sorted(right, rsorted);

            int total = ln + rn;
            int new_ln = total / 2;
            int move = new_ln - ln;

            int32_t new_left[MT_MAX_PAGE_KEYS];
            memcpy(new_left, lsorted, (size_t)ln * sizeof(int32_t));
            memcpy(new_left + ln, rsorted, (size_t)move * sizeof(int32_t));

            int32_t new_right_keys[MT_MAX_PAGE_KEYS];
            int new_rn = rn - move;
            memcpy(new_right_keys, rsorted + move,
                   (size_t)new_rn * sizeof(int32_t));

            struct mt_lnode *lp = leaf->header.prev;
            struct mt_lnode *ln_next = leaf->header.next;
            struct mt_lnode *rp = right->header.prev;
            struct mt_lnode *rn_next = right->header.next;

            mt_page_bulk_load(leaf, new_left, new_ln);
            mt_page_bulk_load(right, new_right_keys, new_rn);

            leaf->header.prev = lp;
            leaf->header.next = ln_next;
            right->header.prev = rp;
            right->header.next = rn_next;

            parent->keys[cidx] = new_right_keys[0];
            return;
        }
    }

    /* Cannot redistribute — merge.  Prefer merging with left sibling. */
    if (cidx > 0) {
        mt_lnode_t *left = &mt_untag(parent->children[cidx - 1])->lnode;
        int32_t lsorted[MT_MAX_PAGE_KEYS], rsorted[MT_MAX_PAGE_KEYS];
        int ln = mt_page_extract_sorted(left, lsorted);
        int rn = mt_page_extract_sorted(leaf, rsorted);

        int32_t merged[MT_MAX_PAGE_KEYS];
        memcpy(merged, lsorted, (size_t)ln * sizeof(int32_t));
        memcpy(merged + ln, rsorted, (size_t)rn * sizeof(int32_t));

        struct mt_lnode *lp = left->header.prev;

        mt_page_bulk_load(left, merged, ln + rn);

        /* Restore and update linked list. */
        left->header.prev = lp;
        left->header.next = leaf->header.next;
        if (leaf->header.next)
            leaf->header.next->header.prev = left;

        inode_remove_at(parent, cidx - 1);
        mt_free_lnode((mt_node_t *)leaf, tree->alloc);
    } else {
        /* Merge with right sibling. */
        mt_lnode_t *right = &mt_untag(parent->children[cidx + 1])->lnode;
        int32_t lsorted[MT_MAX_PAGE_KEYS], rsorted[MT_MAX_PAGE_KEYS];
        int ln = mt_page_extract_sorted(leaf, lsorted);
        int rn = mt_page_extract_sorted(right, rsorted);

        int32_t merged[MT_MAX_PAGE_KEYS];
        memcpy(merged, lsorted, (size_t)ln * sizeof(int32_t));
        memcpy(merged + ln, rsorted, (size_t)rn * sizeof(int32_t));

        struct mt_lnode *lp = leaf->header.prev;

        mt_page_bulk_load(leaf, merged, ln + rn);

        /* Restore and update linked list. */
        leaf->header.prev = lp;
        leaf->header.next = right->header.next;
        if (right->header.next)
            right->header.next->header.prev = leaf;

        inode_remove_at(parent, cidx);
        mt_free_lnode((mt_node_t *)right, tree->alloc);
    }

    /* Propagate internal underflow upward. */
    for (int lv = level; lv >= 0; lv--) {
        mt_inode_t *node = path[lv].node;

        /* Root can have fewer keys — only collapse if it has 0 keys. */
        if (lv == 0) {
            if (node->nkeys == 0 && tree->height > 0) {
                mt_node_t *child = mt_untag(node->children[0]);
                mt_free_inode((mt_node_t *)node);
                tree->root = child;
                tree->height--;
            }
            return;
        }

        if (node->nkeys >= MT_MIN_IKEYS)
            return;

        /* Internal node underflow.  Parent is path[lv-1]. */
        mt_inode_t *pp = path[lv - 1].node;
        int pi = path[lv - 1].idx;  /* child index of `node` in pp */

        /* Try redistribute from left internal sibling. */
        if (pi > 0) {
            mt_inode_t *lsib = &mt_untag(pp->children[pi - 1])->inode;
            if (lsib->nkeys > MT_MIN_IKEYS) {
                /* Rotate right: pull separator from parent down,
                   push last key of left sibling up. */
                memmove(node->keys + 1, node->keys,
                        (size_t)node->nkeys * sizeof(int32_t));
                memmove(node->children + 1, node->children,
                        (size_t)(node->nkeys + 1) * sizeof(mt_node_t *));
                node->keys[0] = pp->keys[pi - 1];
                node->children[0] = lsib->children[lsib->nkeys];
                node->nkeys++;

                pp->keys[pi - 1] = lsib->keys[lsib->nkeys - 1];
                lsib->nkeys--;
                return;
            }
        }

        /* Try redistribute from right internal sibling. */
        if (pi < pp->nkeys) {
            mt_inode_t *rsib = &mt_untag(pp->children[pi + 1])->inode;
            if (rsib->nkeys > MT_MIN_IKEYS) {
                /* Rotate left: pull separator down, push first key of
                   right sibling up. */
                node->keys[node->nkeys] = pp->keys[pi];
                node->children[node->nkeys + 1] = rsib->children[0];
                node->nkeys++;

                pp->keys[pi] = rsib->keys[0];

                memmove(rsib->keys, rsib->keys + 1,
                        (size_t)(rsib->nkeys - 1) * sizeof(int32_t));
                memmove(rsib->children, rsib->children + 1,
                        (size_t)rsib->nkeys * sizeof(mt_node_t *));
                rsib->nkeys--;
                return;
            }
        }

        /* Merge internal nodes. */
        if (pi > 0) {
            /* Merge node into left sibling. */
            mt_inode_t *lsib = &mt_untag(pp->children[pi - 1])->inode;
            int lnk = lsib->nkeys;

            /* Pull down separator from parent. */
            lsib->keys[lnk] = pp->keys[pi - 1];

            /* Copy node's keys and children. */
            memcpy(lsib->keys + lnk + 1, node->keys,
                   (size_t)node->nkeys * sizeof(int32_t));
            memcpy(lsib->children + lnk + 1, node->children,
                   (size_t)(node->nkeys + 1) * sizeof(mt_node_t *));
            lsib->nkeys = (uint16_t)(lnk + 1 + node->nkeys);

            /* Remove node (child[pi]) from parent. */
            inode_remove_at(pp, pi - 1);
            mt_free_inode((mt_node_t *)node);
        } else {
            /* Merge right sibling into node. */
            mt_inode_t *rsib = &mt_untag(pp->children[pi + 1])->inode;
            int nn = node->nkeys;

            node->keys[nn] = pp->keys[pi];
            memcpy(node->keys + nn + 1, rsib->keys,
                   (size_t)rsib->nkeys * sizeof(int32_t));
            memcpy(node->children + nn + 1, rsib->children,
                   (size_t)(rsib->nkeys + 1) * sizeof(mt_node_t *));
            node->nkeys = (uint16_t)(nn + 1 + rsib->nkeys);

            inode_remove_at(pp, pi);
            mt_free_inode((mt_node_t *)rsib);
        }

        /* Continue loop to check pp for underflow. */
    }
}

/* Rebalance after superpage underflow.  Mirrors rebalance_leaf but
   operates on superpages instead of page leaves. */
static void rebalance_sp(matryoshka_tree_t *tree, mt_path_t *path,
                           mt_node_t *sp_node, int level)
{
    mt_sp_header_t *sp = (mt_sp_header_t *)sp_node;
    mt_inode_t *parent = path[level].node;
    int cidx = path[level].idx;

    /* Try redistribute from left sibling. */
    if (cidx > 0) {
        mt_sp_header_t *left = (mt_sp_header_t *)mt_untag(parent->children[cidx - 1]);
        if ((int)left->nkeys > tree->hier.min_sp_keys) {
            int32_t *lkeys = malloc((size_t)left->nkeys * sizeof(int32_t));
            int32_t *rkeys = malloc((size_t)sp->nkeys * sizeof(int32_t));
            if (!lkeys || !rkeys) { free(lkeys); free(rkeys); return; }

            int ln = mt_sp_extract_sorted(left, lkeys);
            int rn = mt_sp_extract_sorted(sp_node, rkeys);
            int total = ln + rn;
            int new_ln = total / 2;

            int32_t *merged = malloc((size_t)total * sizeof(int32_t));
            if (!merged) { free(lkeys); free(rkeys); return; }
            memcpy(merged, lkeys, (size_t)ln * sizeof(int32_t));
            memcpy(merged + ln, rkeys, (size_t)rn * sizeof(int32_t));

            mt_sp_header_t *left_sp_prev = left->prev;
            mt_sp_header_t *sp_next = sp->next;

            mt_sp_bulk_load(left, merged, new_ln, &tree->hier);
            mt_sp_bulk_load(sp_node, merged + new_ln, total - new_ln, &tree->hier);

            /* Restore inter-superpage links. */
            left->prev = left_sp_prev;
            sp->next = sp_next;
            left->next = sp;
            sp->prev = left;
            if (left_sp_prev) left_sp_prev->next = left;
            if (sp_next) sp_next->prev = sp;

            /* Fix cross-superpage page-leaf links. */
            mt_lnode_t *ll = mt_sp_first_leaf((void *)left);
            while (ll->header.next) ll = ll->header.next;
            mt_lnode_t *rf = mt_sp_first_leaf(sp_node);
            ll->header.next = rf;
            rf->header.prev = ll;

            /* Fix boundary with prev superpage. */
            if (left_sp_prev && left_sp_prev->nkeys > 0) {
                mt_lnode_t *plast = mt_sp_first_leaf((void *)left_sp_prev);
                while (plast->header.next) plast = plast->header.next;
                mt_lnode_t *lf = mt_sp_first_leaf((void *)left);
                plast->header.next = lf;
                lf->header.prev = plast;
            }
            /* Fix boundary with next superpage. */
            if (sp_next && sp_next->nkeys > 0) {
                mt_lnode_t *rl = mt_sp_first_leaf(sp_node);
                while (rl->header.next) rl = rl->header.next;
                mt_lnode_t *nf = mt_sp_first_leaf((void *)sp_next);
                rl->header.next = nf;
                nf->header.prev = rl;
            }

            parent->keys[cidx - 1] = merged[new_ln];
            free(lkeys); free(rkeys); free(merged);
            return;
        }
    }

    /* Try redistribute from right sibling. */
    if (cidx < parent->nkeys) {
        mt_sp_header_t *right = (mt_sp_header_t *)mt_untag(parent->children[cidx + 1]);
        if ((int)right->nkeys > tree->hier.min_sp_keys) {
            int32_t *lkeys = malloc((size_t)sp->nkeys * sizeof(int32_t));
            int32_t *rkeys = malloc((size_t)right->nkeys * sizeof(int32_t));
            if (!lkeys || !rkeys) { free(lkeys); free(rkeys); return; }

            int ln = mt_sp_extract_sorted(sp_node, lkeys);
            int rn = mt_sp_extract_sorted(right, rkeys);
            int total = ln + rn;
            int new_ln = total / 2;

            int32_t *merged = malloc((size_t)total * sizeof(int32_t));
            if (!merged) { free(lkeys); free(rkeys); return; }
            memcpy(merged, lkeys, (size_t)ln * sizeof(int32_t));
            memcpy(merged + ln, rkeys, (size_t)rn * sizeof(int32_t));

            mt_sp_header_t *sp_prev = sp->prev;
            mt_sp_header_t *right_next = right->next;

            mt_sp_bulk_load(sp_node, merged, new_ln, &tree->hier);
            mt_sp_bulk_load(right, merged + new_ln, total - new_ln, &tree->hier);

            sp->prev = sp_prev;
            right->next = right_next;
            sp->next = right;
            right->prev = sp;
            if (sp_prev) sp_prev->next = sp;
            if (right_next) right_next->prev = right;

            mt_lnode_t *ll = mt_sp_first_leaf(sp_node);
            while (ll->header.next) ll = ll->header.next;
            mt_lnode_t *rf = mt_sp_first_leaf((void *)right);
            ll->header.next = rf;
            rf->header.prev = ll;

            if (sp_prev && sp_prev->nkeys > 0) {
                mt_lnode_t *plast = mt_sp_first_leaf((void *)sp_prev);
                while (plast->header.next) plast = plast->header.next;
                mt_lnode_t *sf = mt_sp_first_leaf(sp_node);
                plast->header.next = sf;
                sf->header.prev = plast;
            }
            if (right_next && right_next->nkeys > 0) {
                mt_lnode_t *rl = mt_sp_first_leaf((void *)right);
                while (rl->header.next) rl = rl->header.next;
                mt_lnode_t *nf = mt_sp_first_leaf((void *)right_next);
                rl->header.next = nf;
                nf->header.prev = rl;
            }

            parent->keys[cidx] = merged[new_ln];
            free(lkeys); free(rkeys); free(merged);
            return;
        }
    }

    /* Cannot redistribute — merge with a sibling.
       For simplicity, merge with left or right and remove from parent. */
    if (cidx > 0) {
        mt_sp_header_t *left = (mt_sp_header_t *)mt_untag(parent->children[cidx - 1]);
        int32_t *lkeys = malloc((size_t)left->nkeys * sizeof(int32_t));
        int32_t *rkeys = malloc((size_t)sp->nkeys * sizeof(int32_t));
        if (!lkeys || !rkeys) { free(lkeys); free(rkeys); return; }

        int ln = mt_sp_extract_sorted(left, lkeys);
        int rn = mt_sp_extract_sorted(sp_node, rkeys);

        int32_t *merged = malloc((size_t)(ln + rn) * sizeof(int32_t));
        if (!merged) { free(lkeys); free(rkeys); return; }
        memcpy(merged, lkeys, (size_t)ln * sizeof(int32_t));
        memcpy(merged + ln, rkeys, (size_t)rn * sizeof(int32_t));

        mt_sp_header_t *left_prev = left->prev;
        mt_sp_header_t *sp_next_save = sp->next;

        mt_sp_bulk_load(left, merged, ln + rn, &tree->hier);

        left->prev = left_prev;
        left->next = sp_next_save;
        if (left_prev) left_prev->next = left;
        if (sp_next_save) sp_next_save->prev = left;

        /* Fix page-leaf boundary links. */
        if (left_prev && left_prev->nkeys > 0) {
            mt_lnode_t *plast = mt_sp_first_leaf((void *)left_prev);
            while (plast->header.next) plast = plast->header.next;
            mt_lnode_t *lf = mt_sp_first_leaf((void *)left);
            plast->header.next = lf;
            lf->header.prev = plast;
        }
        mt_lnode_t *ll = mt_sp_first_leaf((void *)left);
        while (ll->header.next) ll = ll->header.next;
        if (sp_next_save && sp_next_save->nkeys > 0) {
            mt_lnode_t *nf = mt_sp_first_leaf((void *)sp_next_save);
            ll->header.next = nf;
            nf->header.prev = ll;
        }

        inode_remove_at(parent, cidx - 1);
        mt_free_lnode(sp_node, tree->alloc);
        free(lkeys); free(rkeys); free(merged);
    } else {
        mt_sp_header_t *right = (mt_sp_header_t *)mt_untag(parent->children[cidx + 1]);
        int32_t *lkeys = malloc((size_t)sp->nkeys * sizeof(int32_t));
        int32_t *rkeys = malloc((size_t)right->nkeys * sizeof(int32_t));
        if (!lkeys || !rkeys) { free(lkeys); free(rkeys); return; }

        int ln = mt_sp_extract_sorted(sp_node, lkeys);
        int rn = mt_sp_extract_sorted(right, rkeys);

        int32_t *merged = malloc((size_t)(ln + rn) * sizeof(int32_t));
        if (!merged) { free(lkeys); free(rkeys); return; }
        memcpy(merged, lkeys, (size_t)ln * sizeof(int32_t));
        memcpy(merged + ln, rkeys, (size_t)rn * sizeof(int32_t));

        mt_sp_header_t *sp_prev_save = sp->prev;
        mt_sp_header_t *right_next = right->next;

        mt_sp_bulk_load(sp_node, merged, ln + rn, &tree->hier);

        sp->prev = sp_prev_save;
        sp->next = right_next;
        if (sp_prev_save) sp_prev_save->next = sp;
        if (right_next) right_next->prev = sp;

        if (sp_prev_save && sp_prev_save->nkeys > 0) {
            mt_lnode_t *plast = mt_sp_first_leaf((void *)sp_prev_save);
            while (plast->header.next) plast = plast->header.next;
            mt_lnode_t *sf = mt_sp_first_leaf(sp_node);
            plast->header.next = sf;
            sf->header.prev = plast;
        }
        mt_lnode_t *sl = mt_sp_first_leaf(sp_node);
        while (sl->header.next) sl = sl->header.next;
        if (right_next && right_next->nkeys > 0) {
            mt_lnode_t *nf = mt_sp_first_leaf((void *)right_next);
            sl->header.next = nf;
            nf->header.prev = sl;
        }

        inode_remove_at(parent, cidx);
        mt_free_lnode((mt_node_t *)right, tree->alloc);
        free(lkeys); free(rkeys); free(merged);
    }

    /* Propagate internal underflow upward. */
    for (int lv = level; lv >= 0; lv--) {
        mt_inode_t *node = path[lv].node;
        if (lv == 0) {
            if (node->nkeys == 0 && tree->height > 0) {
                mt_node_t *child = mt_untag(node->children[0]);
                mt_free_inode((mt_node_t *)node);
                tree->root = child;
                tree->height--;
            }
            return;
        }
        if (node->nkeys >= MT_MIN_IKEYS) return;
        /* Internal node underflow — same logic as non-superpage. */
        mt_inode_t *pp = path[lv - 1].node;
        int pi = path[lv - 1].idx;

        if (pi > 0) {
            mt_inode_t *lsib = &mt_untag(pp->children[pi - 1])->inode;
            if (lsib->nkeys > MT_MIN_IKEYS) {
                memmove(node->keys + 1, node->keys,
                        (size_t)node->nkeys * sizeof(int32_t));
                memmove(node->children + 1, node->children,
                        (size_t)(node->nkeys + 1) * sizeof(mt_node_t *));
                node->keys[0] = pp->keys[pi - 1];
                node->children[0] = lsib->children[lsib->nkeys];
                node->nkeys++;
                pp->keys[pi - 1] = lsib->keys[lsib->nkeys - 1];
                lsib->nkeys--;
                return;
            }
        }
        if (pi < pp->nkeys) {
            mt_inode_t *rsib = &mt_untag(pp->children[pi + 1])->inode;
            if (rsib->nkeys > MT_MIN_IKEYS) {
                node->keys[node->nkeys] = pp->keys[pi];
                node->children[node->nkeys + 1] = rsib->children[0];
                node->nkeys++;
                pp->keys[pi] = rsib->keys[0];
                memmove(rsib->keys, rsib->keys + 1,
                        (size_t)(rsib->nkeys - 1) * sizeof(int32_t));
                memmove(rsib->children, rsib->children + 1,
                        (size_t)rsib->nkeys * sizeof(mt_node_t *));
                rsib->nkeys--;
                return;
            }
        }
        if (pi > 0) {
            mt_inode_t *lsib = &mt_untag(pp->children[pi - 1])->inode;
            int lnk = lsib->nkeys;
            lsib->keys[lnk] = pp->keys[pi - 1];
            memcpy(lsib->keys + lnk + 1, node->keys,
                   (size_t)node->nkeys * sizeof(int32_t));
            memcpy(lsib->children + lnk + 1, node->children,
                   (size_t)(node->nkeys + 1) * sizeof(mt_node_t *));
            lsib->nkeys = (uint16_t)(lnk + 1 + node->nkeys);
            inode_remove_at(pp, pi - 1);
            mt_free_inode((mt_node_t *)node);
        } else {
            mt_inode_t *rsib = &mt_untag(pp->children[pi + 1])->inode;
            int nn = node->nkeys;
            node->keys[nn] = pp->keys[pi];
            memcpy(node->keys + nn + 1, rsib->keys,
                   (size_t)rsib->nkeys * sizeof(int32_t));
            memcpy(node->children + nn + 1, rsib->children,
                   (size_t)(rsib->nkeys + 1) * sizeof(mt_node_t *));
            node->nkeys = (uint16_t)(nn + 1 + rsib->nkeys);
            inode_remove_at(pp, pi);
            mt_free_inode((mt_node_t *)rsib);
        }
    }
}

bool matryoshka_delete(matryoshka_tree_t *tree, int32_t key)
{
    if (!tree || tree->n == 0)
        return false;

    mt_path_t path[MT_MAX_HEIGHT];

    if (tree->hier.use_superpages) {
        mt_node_t *node = tree->root;
        for (int i = 0; i < tree->height; i++) {
            int idx = mt_inode_search(&node->inode, key);
            path[i].node = &node->inode;
            path[i].idx = idx;
            node = mt_untag(node->inode.children[idx]);
        }

        mt_status_t status = mt_sp_delete(node, key, &tree->hier);
        if (status == MT_NOT_FOUND) return false;
        tree->n--;
        if (status == MT_OK || tree->height == 0) return true;
        rebalance_sp(tree, path, node, tree->height - 1);
        return true;
    }

    /* Find the leaf. */
    mt_lnode_t *leaf = find_leaf(tree->root, tree->height, key, path);

    /* Delete from the page sub-tree. */
    mt_status_t status = mt_page_delete(leaf, key, &tree->hier);

    if (status == MT_NOT_FOUND)
        return false;

    tree->n--;

    /* If leaf is the root or no underflow, we're done. */
    if (status == MT_OK || tree->height == 0) {
        if (status == MT_OK)
            retag_leaf_in_parent(path, tree->height);
        return true;
    }

    /* MT_UNDERFLOW: eager rebalance (Jannink). */
    rebalance_leaf(tree, path, leaf, tree->height - 1);
    return true;
}

/* ── Batch insert / delete ────────────────────────────────────── */

static int cmp_int32(const void *a, const void *b)
{
    int32_t va = *(const int32_t *)a;
    int32_t vb = *(const int32_t *)b;
    return (va > vb) - (va < vb);
}

/* Helper: walk outer tree to a leaf, recording path. */
static mt_node_t *find_leaf_node(mt_node_t *root, int height, int32_t key,
                                   mt_path_t *path)
{
    mt_node_t *node = root;
    for (int i = 0; i < height; i++) {
        mt_inode_t *in = &node->inode;
        int idx = mt_inode_search(in, key);
        path[i].node = in;
        path[i].idx = idx;
        mt_node_t *raw = in->children[idx];
        if (i == height - 1) {
            uint8_t rs = mt_ptr_root_slot(raw);
            node = mt_untag(raw);
            __builtin_prefetch(node, 0, 1);
            if (rs > 0)
                __builtin_prefetch(&node->lnode.slots[rs - 1], 0, 1);
        } else {
            node = raw;
            __builtin_prefetch(node, 0, 1);
        }
    }
    return node;
}

size_t matryoshka_insert_batch(matryoshka_tree_t *tree,
                                const int32_t *keys, size_t n)
{
    if (!tree || n == 0) return 0;

    int32_t *sorted = malloc(n * sizeof(int32_t));
    if (!sorted) return 0;
    memcpy(sorted, keys, n * sizeof(int32_t));
    qsort(sorted, n, sizeof(int32_t), cmp_int32);

    size_t inserted = 0;
    size_t i = 0;
    bool sp = tree->hier.use_superpages;

    while (i < n) {
        if (i > 0 && sorted[i] == sorted[i - 1]) { i++; continue; }

        mt_path_t path[MT_MAX_HEIGHT];
        mt_node_t *leaf_node = find_leaf_node(tree->root, tree->height,
                                               sorted[i], path);

        int32_t upper = INT32_MAX;
        if (tree->height > 0) {
            mt_inode_t *parent = path[tree->height - 1].node;
            int cidx = path[tree->height - 1].idx;
            if (cidx < parent->nkeys)
                upper = parent->keys[cidx];
        }

        while (i < n) {
            if (i > 0 && sorted[i] == sorted[i - 1]) { i++; continue; }
            if (upper != INT32_MAX && sorted[i] >= upper) break;

            mt_status_t status;
            if (sp)
                status = mt_sp_insert(leaf_node, sorted[i], &tree->hier);
            else
                status = mt_page_insert(&leaf_node->lnode, sorted[i]);

            if (status == MT_DUPLICATE) { i++; continue; }

            if (status == MT_OK) {
                tree->n++;
                inserted++;
                if (!sp) retag_leaf_in_parent(path, tree->height);
                i++;
                continue;
            }

            /* MT_PAGE_FULL: split and insert, then re-navigate. */
            if (sp)
                split_sp_and_insert(tree, path, leaf_node, sorted[i]);
            else
                split_leaf_and_insert(tree, path, &leaf_node->lnode, sorted[i]);
            tree->n++;
            inserted++;
            i++;
            break;
        }
    }

    free(sorted);
    return inserted;
}

size_t matryoshka_delete_batch(matryoshka_tree_t *tree,
                                const int32_t *keys, size_t n)
{
    if (!tree || n == 0) return 0;

    int32_t *sorted = malloc(n * sizeof(int32_t));
    if (!sorted) return 0;
    memcpy(sorted, keys, n * sizeof(int32_t));
    qsort(sorted, n, sizeof(int32_t), cmp_int32);

    size_t deleted = 0;
    size_t i = 0;
    bool use_sp = tree->hier.use_superpages;

    while (i < n) {
        if (i > 0 && sorted[i] == sorted[i - 1]) { i++; continue; }

        mt_path_t path[MT_MAX_HEIGHT];
        mt_node_t *leaf_node = find_leaf_node(tree->root, tree->height,
                                               sorted[i], path);

        int32_t upper = INT32_MAX;
        if (tree->height > 0) {
            mt_inode_t *parent = path[tree->height - 1].node;
            int cidx = path[tree->height - 1].idx;
            if (cidx < parent->nkeys)
                upper = parent->keys[cidx];
        }

        bool need_rebalance = false;

        while (i < n) {
            if (i > 0 && sorted[i] == sorted[i - 1]) { i++; continue; }
            if (upper != INT32_MAX && sorted[i] >= upper) break;

            mt_status_t status;
            if (use_sp)
                status = mt_sp_delete(leaf_node, sorted[i], &tree->hier);
            else
                status = mt_page_delete(&leaf_node->lnode, sorted[i],
                                         &tree->hier);

            if (status == MT_NOT_FOUND) { i++; continue; }

            tree->n--;
            deleted++;
            if (status == MT_OK && !use_sp)
                retag_leaf_in_parent(path, tree->height);
            i++;

            if (status == MT_UNDERFLOW && tree->height > 0) {
                need_rebalance = true;
                break;
            }
        }

        if (need_rebalance) {
            if (use_sp)
                rebalance_sp(tree, path, leaf_node, tree->height - 1);
            else
                rebalance_leaf(tree, path, &leaf_node->lnode,
                                tree->height - 1);
        }
    }

    free(sorted);
    return deleted;
}

/* ── Iteration ────────────────────────────────────────────────── */

/* Load sorted keys from the current leaf into the iterator's buffer. */
static void iter_load_leaf(matryoshka_iter_t *iter)
{
    if (iter->leaf && iter->leaf->header.nkeys > 0) {
        int nkeys = iter->leaf->header.nkeys;
        int32_t *buf = realloc(iter->sorted, (size_t)nkeys * sizeof(int32_t));
        if (buf) {
            iter->sorted = buf;
            iter->nkeys = mt_page_extract_sorted(iter->leaf, iter->sorted);
        } else {
            iter->nkeys = 0;
        }
    } else {
        iter->nkeys = 0;
    }
}

matryoshka_iter_t *matryoshka_iter_from(const matryoshka_tree_t *tree,
                                         int32_t start)
{
    if (!tree) return NULL;

    matryoshka_iter_t *iter = malloc(sizeof(*iter));
    if (!iter) return NULL;
    iter->tree = tree;
    iter->sorted = NULL;

    if (tree->n == 0) {
        iter->leaf = NULL;
        iter->pos = 0;
        iter->nkeys = 0;
        return iter;
    }

    /* Walk to the leaf that should contain `start`. */
    mt_node_t *node = tree->root;
    for (int i = 0; i < tree->height; i++) {
        int idx = mt_inode_search(&node->inode, start);
        node = mt_untag(node->inode.children[idx]);
    }
    if (tree->hier.use_superpages)
        iter->leaf = mt_sp_find_leaf(node, start);
    else
        iter->leaf = &node->lnode;
    iter_load_leaf(iter);

    /* Find the first key >= start in the sorted keys. */
    int pos = 0;
    while (pos < iter->nkeys && iter->sorted[pos] < start)
        pos++;

    /* If past the end of this leaf, move to the next one. */
    if (pos >= iter->nkeys && iter->leaf->header.next) {
        iter->leaf = iter->leaf->header.next;
        iter_load_leaf(iter);
        pos = 0;
    }

    iter->pos = pos;
    return iter;
}

bool matryoshka_iter_next(matryoshka_iter_t *iter, int32_t *key)
{
    if (!iter || !iter->leaf)
        return false;

    while (iter->pos >= iter->nkeys) {
        /* Move to next leaf. */
        if (!iter->leaf->header.next)
            return false;
        iter->leaf = iter->leaf->header.next;
        iter_load_leaf(iter);
        iter->pos = 0;
    }

    if (key) *key = iter->sorted[iter->pos];
    iter->pos++;
    return true;
}

void matryoshka_iter_destroy(matryoshka_iter_t *iter)
{
    if (iter) {
        free(iter->sorted);
        free(iter);
    }
}
