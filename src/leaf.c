/*
 * leaf.c — FAST-blocked leaf node: build and search.
 *
 * Leaf nodes use the full FAST hierarchical blocking scheme from
 * Kim et al., SIGMOD 2010.  Keys are stored in a blocked layout
 * with sorted_rank for predecessor resolution.
 */

#include "matryoshka_internal.h"
#include <string.h>

/* ── Build helpers ──────────────────────────────────────────── */

/*
 * Compute in-order traversal mapping for a complete binary tree.
 * bfs_to_sorted[bfs_index] = in-order position.
 */
static void build_inorder_map(size_t *bfs_to_sorted, size_t tree_nodes)
{
    for (size_t i = 0; i < tree_nodes; i++)
        bfs_to_sorted[i] = 0;

    /* Iterative in-order traversal via explicit stack. */
    struct { size_t node; int state; } stk[32];
    int sp = 0;
    stk[sp].node = 0;
    stk[sp].state = 0;
    size_t count = 0;

    while (sp >= 0) {
        size_t node = stk[sp].node;
        int state = stk[sp].state;

        if (node >= tree_nodes) {
            sp--;
            continue;
        }

        if (state == 0) {
            stk[sp].state = 1;
            size_t left = 2 * node + 1;
            if (left < tree_nodes) {
                sp++;
                stk[sp].node = left;
                stk[sp].state = 0;
            }
        } else if (state == 1) {
            bfs_to_sorted[node] = count++;
            stk[sp].state = 2;
            size_t right = 2 * node + 2;
            if (right < tree_nodes) {
                sp++;
                stk[sp].node = right;
                stk[sp].state = 0;
            }
        } else {
            sp--;
        }
    }
}

/*
 * Write a BFS block of `depth` levels starting at `bfs_root` into the
 * output layout, recording sorted_rank.
 */
static size_t write_bfs_block(const int32_t *bfs_tree,
                               int32_t *out, int16_t *rank_out,
                               const size_t *bfs_to_sorted, int nkeys,
                               size_t bfs_root, size_t out_pos, int depth,
                               size_t tree_nodes)
{
    /* BFS traversal of subtree rooted at bfs_root, depth levels.
       Queue must hold up to 2^depth entries at the widest level. */
    size_t queue[1024];
    int head = 0, tail = 0;
    queue[tail++] = bfs_root;
    int levels_done = 0;
    int level_remaining = 1;
    int next_level = 0;

    while (head < tail && levels_done < depth) {
        size_t node = queue[head++];
        level_remaining--;

        if (node < tree_nodes) {
            out[out_pos] = bfs_tree[node];
            size_t sorted_idx = bfs_to_sorted[node];
            rank_out[out_pos] = (sorted_idx < (size_t)nkeys)
                                    ? (int16_t)sorted_idx
                                    : (int16_t)nkeys;
            out_pos++;

            size_t left = 2 * node + 1;
            size_t right = 2 * node + 2;
            if (left < tree_nodes)  queue[tail++] = left;
            if (right < tree_nodes) queue[tail++] = right;
            next_level += 2;
        } else {
            out[out_pos] = MT_KEY_MAX;
            rank_out[out_pos] = (int16_t)nkeys;
            out_pos++;
        }

        if (level_remaining == 0) {
            levels_done++;
            level_remaining = next_level;
            next_level = 0;
        }
    }
    return out_pos;
}

/*
 * Recursively lay out a subtree in hierarchical blocked order.
 */
static void lay_out_subtree(const int32_t *bfs_tree, int32_t *out,
                             int16_t *rank_out, const size_t *bfs_to_sorted,
                             int nkeys, size_t bfs_root, size_t *out_pos,
                             int remaining, int blk_level,
                             const int *depths, size_t tree_nodes)
{
    if (remaining <= 0 || bfs_root >= tree_nodes)
        return;

    int block_depth = depths[blk_level];
    if (remaining <= block_depth || blk_level == 0) {
        int d = (remaining < block_depth) ? remaining : block_depth;
        *out_pos = write_bfs_block(bfs_tree, out, rank_out, bfs_to_sorted,
                                    nkeys, bfs_root, *out_pos, d, tree_nodes);
        int child_remaining = remaining - d;
        if (child_remaining > 0) {
            size_t n_children = (size_t)1 << d;
            size_t first_child = bfs_root;
            for (int i = 0; i < d; i++)
                first_child = 2 * first_child + 1;
            for (size_t c = 0; c < n_children; c++) {
                lay_out_subtree(bfs_tree, out, rank_out, bfs_to_sorted,
                                nkeys, first_child + c, out_pos,
                                child_remaining, blk_level, depths,
                                tree_nodes);
            }
        }
    } else {
        lay_out_subtree(bfs_tree, out, rank_out, bfs_to_sorted, nkeys,
                        bfs_root, out_pos, remaining, blk_level - 1,
                        depths, tree_nodes);
    }
}

/* ── Public leaf interface ──────────────────────────────────── */

void mt_leaf_extract_sorted(const mt_lnode_t *leaf, int32_t *out)
{
    int n = leaf->nkeys;
    if (n == 0) return;
    int d = leaf->tree_depth;
    size_t tree_nodes = ((size_t)1 << d) - 1;
    for (size_t i = 0; i < tree_nodes && i < MT_LNODE_TREE_CAP; i++) {
        int ri = leaf->sorted_rank[i];
        if (ri >= 0 && ri < n)
            out[ri] = leaf->layout[i];
    }
}

void mt_leaf_build(mt_lnode_t *leaf, const int32_t *sorted_keys, int nkeys)
{
    leaf->nkeys = (uint16_t)nkeys;

    if (nkeys == 0) {
        leaf->tree_depth = 0;
        return;
    }

    /* Compute tree depth. */
    int d_n = 0;
    size_t tmp = 1;
    while ((int)(tmp - 1) < nkeys) { d_n++; tmp <<= 1; }
    size_t tree_nodes = tmp - 1;
    leaf->tree_depth = (uint16_t)d_n;

    /* Build in-order map. */
    size_t bfs_to_sorted[512];
    build_inorder_map(bfs_to_sorted, tree_nodes);

    /* Build BFS tree from sorted keys. */
    int32_t bfs_tree[512];
    for (size_t i = 0; i < tree_nodes; i++) {
        size_t si = bfs_to_sorted[i];
        bfs_tree[i] = (si < (size_t)nkeys) ? sorted_keys[si] : MT_KEY_MAX;
    }

    /* SIMD-blocked layout: recursively decompose into depth-DK blocks.
       (Cache-line blocking would require matching search-side changes.) */
    int depths[1] = { MT_DK };
    size_t out_pos = 0;
    lay_out_subtree(bfs_tree, leaf->layout, leaf->sorted_rank, bfs_to_sorted,
                    nkeys, 0, &out_pos, d_n, 0, depths, tree_nodes);

    /* Zero-fill unused slots. */
    for (size_t i = out_pos; i < MT_LNODE_TREE_CAP; i++) {
        leaf->layout[i] = MT_KEY_MAX;
        leaf->sorted_rank[i] = (int16_t)nkeys;
    }
}

int mt_leaf_search(const mt_lnode_t *leaf, int32_t key)
{
    int nkeys = leaf->nkeys;
    if (nkeys == 0)
        return -1;

    const int32_t *tree = leaf->layout;
    const int16_t *rank = leaf->sorted_rank;
    int d_n = leaf->tree_depth;

    /* Boundary check: find min and max keys by scanning rank. */
    int32_t min_key = MT_KEY_MAX, max_key = INT32_MIN;
    size_t tree_nodes = ((size_t)1 << d_n) - 1;
    for (size_t i = 0; i < tree_nodes && i < MT_LNODE_TREE_CAP; i++) {
        if (rank[i] == 0) min_key = tree[i];
        if (rank[i] == nkeys - 1) max_key = tree[i];
    }

    if (key < min_key)
        return -1;
    if (key >= max_key)
        return nkeys - 1;

    /* SIMD traversal of the FAST blocked layout. */
    __m128i v_key = _mm_set1_epi32(key);
    size_t offset = 0;
    int depth_remaining = d_n;
    int child_index = 0;
    int last_block_type = 0;

    while (depth_remaining > 0) {
        int simd_depth = (depth_remaining >= MT_DK) ? MT_DK : depth_remaining;

        if (simd_depth == MT_DK) {
            __m128i v_tree = _mm_loadu_si128((const __m128i *)(tree + offset));
            __m128i v_cmp = _mm_cmpgt_epi32(v_key, v_tree);
            int mask = _mm_movemask_ps(_mm_castsi128_ps(v_cmp));
            child_index = MT_SIMD_LOOKUP[mask & 0x7];
            if (child_index < 0) child_index = 0;

            depth_remaining -= MT_DK;
            last_block_type = 0;

            if (depth_remaining <= 0) break;

            size_t child_subtree_size = ((size_t)1 << depth_remaining) - 1;
            offset = offset + MT_NK + (size_t)child_index * child_subtree_size;
        } else {
            child_index = (key > tree[offset]) ? 1 : 0;
            depth_remaining -= 1;
            last_block_type = 1;
            break;
        }
    }

    /* Resolve leaf: find predecessor in sorted keys. */
    int lo;
    if (last_block_type == 0) {
        int r;
        switch (child_index) {
            case 0:  r = rank[offset + 1]; lo = r - 1; break;
            case 1:  r = rank[offset + 1]; lo = r;     break;
            case 2:  r = rank[offset];     lo = r;     break;
            default: r = rank[offset + 2]; lo = r;     break;
        }
    } else {
        int r = rank[offset];
        lo = (child_index == 0) ? r - 1 : r;
    }

    if (lo < 0) lo = 0;

    /* Build sorted key array on stack from layout + rank. */
    int32_t sorted[512];
    for (size_t i = 0; i < tree_nodes && i < MT_LNODE_TREE_CAP; i++) {
        int ri = rank[i];
        if (ri >= 0 && ri < nkeys)
            sorted[ri] = tree[i];
    }

    /* Forward scan from lo. */
    while (lo + 1 < nkeys && sorted[lo + 1] <= key)
        lo++;

    /* Verify predecessor. */
    if (lo >= 0 && lo < nkeys && sorted[lo] <= key)
        return lo;

    return -1;
}
