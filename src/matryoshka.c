/*
 * matryoshka.c — B+ tree operations with FAST-blocked leaf nodes.
 *
 * Core operations: create, bulk_load, destroy, search, insert, delete,
 * and iteration.  Leaf nodes use hierarchical FAST blocking for search;
 * modifications rebuild the affected leaf in O(B) time, giving overall
 * O(B · log_B n) modification cost.
 */

#include "matryoshka_internal.h"
#include <stdlib.h>
#include <string.h>

/* ── Constants ────────────────────────────────────────────────── */

#define MT_MAX_HEIGHT 32

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

/* Insert a key at position `pos` in a sorted array of `n` elements.
   Returns the insertion index, or -1 if the key already exists. */
static int sorted_insert(int32_t *arr, int n, int32_t key)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    if (lo < n && arr[lo] == key)
        return -1;
    memmove(arr + lo + 1, arr + lo, (size_t)(n - lo) * sizeof(int32_t));
    arr[lo] = key;
    return lo;
}

/* Remove a key from a sorted array of `n` elements.
   Returns the removed index, or -1 if not found. */
static int sorted_remove(int32_t *arr, int n, int32_t key)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= n || arr[lo] != key)
        return -1;
    memmove(arr + lo, arr + lo + 1, (size_t)(n - lo - 1) * sizeof(int32_t));
    return lo;
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

    int max_lkeys = hier->leaf_cap;

    if (n == 0) {
        tree->root = mt_alloc_lnode(&tree->hier, tree->alloc);
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
        mt_leaf_build(&lnode->lnode, sorted_keys + offset, (int)k, &tree->hier);
        entries[i].node = lnode;
        entries[i].min_key = sorted_keys[offset];
        offset += k;
    }

    /* Link leaves. */
    for (size_t i = 0; i < nleaves; i++) {
        mt_lnode_t *l = &entries[i].node->lnode;
        l->prev = (i > 0) ? &entries[i - 1].node->lnode : NULL;
        l->next = (i < nleaves - 1) ? &entries[i + 1].node->lnode : NULL;
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

    /* Predecessor search using FAST layout. */
    int pos = mt_leaf_search(leaf, key, &tree->hier);
    if (pos >= 0) {
        int32_t sorted[MT_MAX_LKEYS];
        mt_leaf_extract_sorted(leaf, sorted, &tree->hier);
        if (result) *result = sorted[pos];
        return true;
    }

    /* Key is smaller than all keys in this leaf.  Check previous leaf. */
    if (leaf->prev) {
        mt_lnode_t *prev = leaf->prev;
        if (prev->nkeys > 0) {
            int32_t sorted[MT_MAX_LKEYS];
            mt_leaf_extract_sorted(prev, sorted, &tree->hier);
            if (result) *result = sorted[prev->nkeys - 1];
            return true;
        }
    }

    return false;
}

bool matryoshka_contains(const matryoshka_tree_t *tree, int32_t key)
{
    int32_t found;
    if (!matryoshka_search(tree, key, &found))
        return false;
    return found == key;
}

/* ── Insert ───────────────────────────────────────────────────── */

bool matryoshka_insert(matryoshka_tree_t *tree, int32_t key)
{
    if (!tree) return false;

    /* Find the leaf. */
    mt_path_t path[MT_MAX_HEIGHT];
    mt_lnode_t *leaf = find_leaf(tree->root, tree->height, key, path);

    /* Extract sorted keys. */
    int32_t sorted[MT_MAX_LKEYS + 1];
    int n = leaf->nkeys;
    mt_leaf_extract_sorted(leaf, sorted, &tree->hier);

    /* Insert into sorted array. */
    int ins = sorted_insert(sorted, n, key);
    if (ins < 0)
        return false;  /* duplicate */
    n++;

    int max_lkeys = tree->hier.leaf_cap;
    if (n <= max_lkeys) {
        mt_leaf_build(leaf, sorted, n, &tree->hier);
        tree->n++;
        return true;
    }

    /* Leaf overflow: split. */
    int left_n = n / 2;
    int right_n = n - left_n;

    mt_node_t *new_rnode = mt_alloc_lnode(&tree->hier, tree->alloc);
    mt_lnode_t *new_right = &new_rnode->lnode;

    mt_leaf_build(leaf, sorted, left_n, &tree->hier);
    mt_leaf_build(new_right, sorted + left_n, right_n, &tree->hier);

    /* Maintain linked list. */
    new_right->next = leaf->next;
    new_right->prev = leaf;
    if (leaf->next)
        leaf->next->prev = new_right;
    leaf->next = new_right;

    /* Separator = first key of the right leaf. */
    int32_t sep = sorted[left_n];
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
    const mt_hierarchy_t *hier = &tree->hier;
    mt_inode_t *parent = path[level].node;
    int cidx = path[level].idx;
    int min_lkeys = hier->min_lkeys;

    /* Try redistribute from left sibling. */
    if (cidx > 0) {
        mt_lnode_t *left = &parent->children[cidx - 1]->lnode;
        if (left->nkeys > min_lkeys) {
            /* Extract both leaves' sorted keys. */
            int32_t lsorted[MT_MAX_LKEYS], rsorted[MT_MAX_LKEYS];
            int ln = left->nkeys, rn = leaf->nkeys;
            mt_leaf_extract_sorted(left, lsorted, hier);
            mt_leaf_extract_sorted(leaf, rsorted, hier);

            /* Move keys until balanced: target = (ln + rn) / 2 per side. */
            int total = ln + rn;
            int new_ln = total / 2;
            int move = ln - new_ln;

            /* Build combined array: move `move` keys from left to right. */
            int32_t new_right[MT_MAX_LKEYS + 1];
            memcpy(new_right, lsorted + new_ln, (size_t)move * sizeof(int32_t));
            memcpy(new_right + move, rsorted, (size_t)rn * sizeof(int32_t));
            int new_rn = move + rn;

            mt_leaf_build(left, lsorted, new_ln, hier);
            mt_leaf_build(leaf, new_right, new_rn, hier);

            /* Update separator: new sep = first key of right leaf. */
            parent->keys[cidx - 1] = new_right[0];
            return;
        }
    }

    /* Try redistribute from right sibling. */
    if (cidx < parent->nkeys) {
        mt_lnode_t *right = &parent->children[cidx + 1]->lnode;
        if (right->nkeys > min_lkeys) {
            int32_t lsorted[MT_MAX_LKEYS], rsorted[MT_MAX_LKEYS];
            int ln = leaf->nkeys, rn = right->nkeys;
            mt_leaf_extract_sorted(leaf, lsorted, hier);
            mt_leaf_extract_sorted(right, rsorted, hier);

            int total = ln + rn;
            int new_ln = total / 2;
            int move = new_ln - ln;

            /* Move `move` keys from right to left. */
            int32_t new_left[MT_MAX_LKEYS + 1];
            memcpy(new_left, lsorted, (size_t)ln * sizeof(int32_t));
            memcpy(new_left + ln, rsorted, (size_t)move * sizeof(int32_t));

            int32_t new_right_keys[MT_MAX_LKEYS];
            int new_rn = rn - move;
            memcpy(new_right_keys, rsorted + move,
                   (size_t)new_rn * sizeof(int32_t));

            mt_leaf_build(leaf, new_left, new_ln, hier);
            mt_leaf_build(right, new_right_keys, new_rn, hier);

            /* Update separator: new sep = first key of right leaf. */
            parent->keys[cidx] = new_right_keys[0];
            return;
        }
    }

    /* Cannot redistribute — merge.  Prefer merging with left sibling. */
    if (cidx > 0) {
        mt_lnode_t *left = &parent->children[cidx - 1]->lnode;
        int32_t lsorted[MT_MAX_LKEYS], rsorted[MT_MAX_LKEYS];
        int ln = left->nkeys, rn = leaf->nkeys;
        mt_leaf_extract_sorted(left, lsorted, hier);
        mt_leaf_extract_sorted(leaf, rsorted, hier);

        int32_t merged[MT_MAX_LKEYS + 1];
        memcpy(merged, lsorted, (size_t)ln * sizeof(int32_t));
        memcpy(merged + ln, rsorted, (size_t)rn * sizeof(int32_t));

        mt_leaf_build(left, merged, ln + rn, hier);

        /* Unlink `leaf` from the doubly linked list. */
        left->next = leaf->next;
        if (leaf->next) leaf->next->prev = left;

        /* Remove `leaf` (child[cidx]) and separator keys[cidx-1]
           from the parent. */
        inode_remove_at(parent, cidx - 1);
        mt_free_lnode((mt_node_t *)leaf, tree->alloc);
    } else {
        /* Merge with right sibling. */
        mt_lnode_t *right = &parent->children[cidx + 1]->lnode;
        int32_t lsorted[MT_MAX_LKEYS], rsorted[MT_MAX_LKEYS];
        int ln = leaf->nkeys, rn = right->nkeys;
        mt_leaf_extract_sorted(leaf, lsorted, hier);
        mt_leaf_extract_sorted(right, rsorted, hier);

        int32_t merged[MT_MAX_LKEYS + 1];
        memcpy(merged, lsorted, (size_t)ln * sizeof(int32_t));
        memcpy(merged + ln, rsorted, (size_t)rn * sizeof(int32_t));

        mt_leaf_build(leaf, merged, ln + rn, hier);

        /* Unlink `right` from the doubly linked list. */
        leaf->next = right->next;
        if (right->next) right->next->prev = leaf;

        /* Remove right (child[cidx+1]) and separator keys[cidx]. */
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
                /* Make room in node: shift keys/children right. */
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
            int ln = lsib->nkeys;

            /* Pull down separator from parent. */
            lsib->keys[ln] = pp->keys[pi - 1];

            /* Copy node's keys and children. */
            memcpy(lsib->keys + ln + 1, node->keys,
                   (size_t)node->nkeys * sizeof(int32_t));
            memcpy(lsib->children + ln + 1, node->children,
                   (size_t)(node->nkeys + 1) * sizeof(mt_node_t *));
            lsib->nkeys = (uint16_t)(ln + 1 + node->nkeys);

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

    /* Extract sorted keys. */
    int32_t sorted[MT_MAX_LKEYS];
    int n = leaf->nkeys;
    mt_leaf_extract_sorted(leaf, sorted, &tree->hier);

    /* Remove key. */
    int rem = sorted_remove(sorted, n, key);
    if (rem < 0)
        return false;
    n--;

    /* Rebuild the leaf. */
    mt_leaf_build(leaf, sorted, n, &tree->hier);
    tree->n--;

    /* If leaf is the root or has enough keys, we're done. */
    if (tree->height == 0 || n >= tree->hier.min_lkeys)
        return true;

    /* Leaf underflow: eager rebalance (Jannink). */
    rebalance_leaf(tree, path, leaf, tree->height - 1);
    return true;
}

/* ── Iteration ────────────────────────────────────────────────── */

/* Load sorted keys from the current leaf into the iterator's buffer. */
static void iter_load_leaf(matryoshka_iter_t *iter)
{
    if (iter->leaf && iter->leaf->nkeys > 0) {
        iter->nkeys = iter->leaf->nkeys;
        mt_leaf_extract_sorted(iter->leaf, iter->sorted, &iter->tree->hier);
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
    if (pos >= iter->nkeys && iter->leaf->next) {
        iter->leaf = iter->leaf->next;
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
        if (!iter->leaf->next)
            return false;
        iter->leaf = iter->leaf->next;
        iter_load_leaf(iter);
        iter->pos = 0;
    }

    if (key) *key = iter->sorted[iter->pos];
    iter->pos++;
    return true;
}

void matryoshka_iter_destroy(matryoshka_iter_t *iter)
{
    free(iter);
}
