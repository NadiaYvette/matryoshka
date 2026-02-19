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

matryoshka_tree_t *matryoshka_create(void)
{
    matryoshka_tree_t *tree = malloc(sizeof(*tree));
    if (!tree) return NULL;
    tree->root = mt_alloc_lnode();
    if (!tree->root) { free(tree); return NULL; }
    tree->n = 0;
    tree->height = 0;
    return tree;
}

static void free_subtree(mt_node_t *node, int height)
{
    if (height > 0) {
        mt_inode_t *in = &node->inode;
        for (int i = 0; i <= in->nkeys; i++)
            free_subtree(in->children[i], height - 1);
    }
    mt_free_node(node);
}

void matryoshka_destroy(matryoshka_tree_t *tree)
{
    if (!tree) return;
    if (tree->root)
        free_subtree(tree->root, tree->height);
    free(tree);
}

/* ── Bulk load ────────────────────────────────────────────────── */

typedef struct {
    mt_node_t *node;
    int32_t    min_key;
} build_entry_t;

matryoshka_tree_t *matryoshka_bulk_load(const int32_t *sorted_keys, size_t n)
{
    matryoshka_tree_t *tree = malloc(sizeof(*tree));
    if (!tree) return NULL;
    tree->n = n;

    if (n == 0) {
        tree->root = mt_alloc_lnode();
        tree->height = 0;
        return tree;
    }

    /* Distribute keys across leaves. */
    size_t nleaves = (n + MT_MAX_LKEYS - 1) / MT_MAX_LKEYS;
    size_t keys_per = n / nleaves;
    size_t extra = n % nleaves;

    build_entry_t *entries = malloc(nleaves * sizeof(build_entry_t));
    if (!entries) { free(tree); return NULL; }

    size_t offset = 0;
    for (size_t i = 0; i < nleaves; i++) {
        size_t k = keys_per + (i < extra ? 1 : 0);
        mt_node_t *lnode = mt_alloc_lnode();
        mt_leaf_build(&lnode->lnode, sorted_keys + offset, (int)k);
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
            /* Leak on error — acceptable for this initial implementation. */
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
    int pos = mt_leaf_search(leaf, key);
    if (pos >= 0) {
        int32_t sorted[MT_MAX_LKEYS];
        mt_leaf_extract_sorted(leaf, sorted);
        if (result) *result = sorted[pos];
        return true;
    }

    /* Key is smaller than all keys in this leaf.  Check previous leaf. */
    if (leaf->prev) {
        mt_lnode_t *prev = leaf->prev;
        if (prev->nkeys > 0) {
            int32_t sorted[MT_MAX_LKEYS];
            mt_leaf_extract_sorted(prev, sorted);
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
    mt_leaf_extract_sorted(leaf, sorted);

    /* Insert into sorted array. */
    int ins = sorted_insert(sorted, n, key);
    if (ins < 0)
        return false;  /* duplicate */
    n++;

    if (n <= MT_MAX_LKEYS) {
        mt_leaf_build(leaf, sorted, n);
        tree->n++;
        return true;
    }

    /* Leaf overflow: split. */
    int left_n = n / 2;
    int right_n = n - left_n;

    mt_node_t *new_rnode = mt_alloc_lnode();
    mt_lnode_t *new_right = &new_rnode->lnode;

    mt_leaf_build(leaf, sorted, left_n);
    mt_leaf_build(new_right, sorted + left_n, right_n);

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

/* ── Delete ───────────────────────────────────────────────────── */

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
    mt_leaf_extract_sorted(leaf, sorted);

    /* Remove key. */
    int rem = sorted_remove(sorted, n, key);
    if (rem < 0)
        return false;
    n--;

    /* Rebuild the leaf. */
    mt_leaf_build(leaf, sorted, n);
    tree->n--;

    /* If leaf is the root or has enough keys, we're done. */
    if (tree->height == 0 || n >= MT_MIN_LKEYS)
        return true;

    /* Leaf underflow.  Try to merge with a sibling.
       We use lazy deletion: allow underfull leaves for simplicity.
       The tree remains correct; it just wastes some space.
       Full merge/redistribute logic can be added later. */

    /* If the leaf is completely empty and not the root, we should at
       least remove it from the tree structure. */
    if (n == 0 && tree->height > 0) {
        mt_inode_t *parent = path[tree->height - 1].node;
        int cidx = path[tree->height - 1].idx;

        /* Unlink from leaf linked list. */
        if (leaf->prev) leaf->prev->next = leaf->next;
        if (leaf->next) leaf->next->prev = leaf->prev;

        /* Remove from parent. */
        if (cidx == 0 && parent->nkeys > 0) {
            /* Remove first child: shift keys and children left. */
            memmove(parent->keys, parent->keys + 1,
                    (size_t)(parent->nkeys - 1) * sizeof(int32_t));
            memmove(parent->children, parent->children + 1,
                    (size_t)parent->nkeys * sizeof(mt_node_t *));
            parent->nkeys--;
        } else if (cidx > 0) {
            inode_remove_at(parent, cidx - 1);
        }

        mt_free_node((mt_node_t *)leaf);

        /* If root becomes a single-child internal node, collapse. */
        if (tree->height > 0) {
            mt_inode_t *root = &tree->root->inode;
            if (root->nkeys == 0) {
                mt_node_t *child = root->children[0];
                mt_free_node(tree->root);
                tree->root = child;
                tree->height--;
            }
        }
    }

    return true;
}

/* ── Iteration ────────────────────────────────────────────────── */

/* Load sorted keys from the current leaf into the iterator's buffer. */
static void iter_load_leaf(matryoshka_iter_t *iter)
{
    if (iter->leaf && iter->leaf->nkeys > 0) {
        iter->nkeys = iter->leaf->nkeys;
        mt_leaf_extract_sorted(iter->leaf, iter->sorted);
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
