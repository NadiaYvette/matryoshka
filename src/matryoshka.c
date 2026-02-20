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
        node = in->children[idx];
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
            free_subtree(in->children[i], height - 1, alloc);
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

    int max_lkeys = hier->page_max_keys;

    if (n == 0) {
        tree->root = mt_alloc_lnode(&tree->hier, tree->alloc);
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
        mt_page_bulk_load(&lnode->lnode, sorted_keys + offset, (int)k);
        entries[i].node = lnode;
        entries[i].min_key = sorted_keys[offset];
        offset += k;
    }

    /* Link leaves. */
    for (size_t i = 0; i < nleaves; i++) {
        mt_lnode_t *l = &entries[i].node->lnode;
        l->header.prev = (i > 0) ? &entries[i - 1].node->lnode : NULL;
        l->header.next = (i < nleaves - 1) ? &entries[i + 1].node->lnode : NULL;
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

            in->children[0] = entries[ci].node;
            for (size_t j = 1; j < nc; j++) {
                in->keys[j - 1] = entries[ci + j].min_key;
                in->children[j] = entries[ci + j].node;
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
        node = node->inode.children[idx];
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
        node = node->inode.children[idx];
    }

    return mt_page_contains(&node->lnode, key);
}

/* ── Insert ───────────────────────────────────────────────────── */

bool matryoshka_insert(matryoshka_tree_t *tree, int32_t key)
{
    if (!tree) return false;

    /* Find the leaf. */
    mt_path_t path[MT_MAX_HEIGHT];
    mt_lnode_t *leaf = find_leaf(tree->root, tree->height, key, path);

    /* Try inserting into the page sub-tree. */
    mt_status_t status = mt_page_insert(leaf, key);

    if (status == MT_DUPLICATE)
        return false;

    if (status == MT_OK) {
        tree->n++;
        return true;
    }

    /* MT_PAGE_FULL: split the leaf page. */
    mt_node_t *new_rnode = mt_alloc_lnode(&tree->hier, tree->alloc);
    mt_lnode_t *new_right = &new_rnode->lnode;

    int32_t sep = mt_page_split(leaf, new_right);

    /* Insert the key into the appropriate half. */
    if (key < sep)
        mt_page_insert(leaf, key);
    else
        mt_page_insert(new_right, key);

    /* Maintain linked list. */
    new_right->header.next = leaf->header.next;
    new_right->header.prev = leaf;
    if (leaf->header.next)
        leaf->header.next->header.prev = new_right;
    leaf->header.next = new_right;

    /* Separator = first key of the right leaf. */
    sep = mt_page_min_key(new_right);
    mt_node_t *right_child = new_rnode;

    /* Propagate split up through internal nodes. */
    for (int level = tree->height - 1; level >= 0; level--) {
        mt_inode_t *parent = path[level].node;

        if (parent->nkeys < MT_MAX_IKEYS) {
            /* Find correct insertion position for the separator. */
            int pos = 0;
            while (pos < parent->nkeys && parent->keys[pos] < sep)
                pos++;
            inode_insert_at(parent, pos, sep, right_child);
            tree->n++;
            return true;
        }

        /* Internal node overflow: split it. */
        int pn = parent->nkeys;
        int32_t all_keys[MT_MAX_IKEYS + 1];
        mt_node_t *all_children[MT_MAX_IKEYS + 2];

        /* Find position for new separator. */
        int pos = 0;
        while (pos < pn && parent->keys[pos] < sep)
            pos++;

        /* Merge existing keys + new separator. */
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
        int right_keys = total - left_keys - 1;  /* middle key goes up */
        sep = all_keys[left_keys];

        /* Rebuild left (reuse parent). */
        memcpy(parent->keys, all_keys, (size_t)left_keys * sizeof(int32_t));
        memcpy(parent->children, all_children,
               (size_t)(left_keys + 1) * sizeof(mt_node_t *));
        parent->nkeys = (uint16_t)left_keys;

        /* Build right. */
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
    nr->children[0] = tree->root;
    nr->children[1] = right_child;
    nr->nkeys = 1;
    tree->root = new_root;
    tree->height++;
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
        mt_lnode_t *left = &parent->children[cidx - 1]->lnode;
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
        mt_lnode_t *right = &parent->children[cidx + 1]->lnode;
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
        mt_lnode_t *left = &parent->children[cidx - 1]->lnode;
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
        mt_lnode_t *right = &parent->children[cidx + 1]->lnode;
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
                mt_node_t *child = node->children[0];
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
            mt_inode_t *lsib = &pp->children[pi - 1]->inode;
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
            mt_inode_t *rsib = &pp->children[pi + 1]->inode;
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
            mt_inode_t *lsib = &pp->children[pi - 1]->inode;
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
            mt_inode_t *rsib = &pp->children[pi + 1]->inode;
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

bool matryoshka_delete(matryoshka_tree_t *tree, int32_t key)
{
    if (!tree || tree->n == 0)
        return false;

    /* Find the leaf. */
    mt_path_t path[MT_MAX_HEIGHT];
    mt_lnode_t *leaf = find_leaf(tree->root, tree->height, key, path);

    /* Delete from the page sub-tree. */
    mt_status_t status = mt_page_delete(leaf, key, &tree->hier);

    if (status == MT_NOT_FOUND)
        return false;

    tree->n--;

    /* If leaf is the root or no underflow, we're done. */
    if (status == MT_OK || tree->height == 0)
        return true;

    /* MT_UNDERFLOW: eager rebalance (Jannink). */
    rebalance_leaf(tree, path, leaf, tree->height - 1);
    return true;
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
        node = node->inode.children[idx];
    }
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
