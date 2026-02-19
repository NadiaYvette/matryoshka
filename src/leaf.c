/*
 * leaf.c — FAST-blocked leaf node: build and search.
 *
 * Leaf nodes use the full FAST hierarchical blocking scheme from
 * Kim et al., SIGMOD 2010.  Keys are stored in a blocked layout
 * with sorted_rank for predecessor resolution.
 *
 * The blocking hierarchy is currently SIMD (d_K=2) + cache-line (d_L=4).
 * Within each cache-line block, SIMD blocks are laid out contiguously.
 * Between cache-line blocks, subtrees follow in pre-order.
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
 *
 * The blocking hierarchy is defined by depths[0..blk_level], finest to
 * coarsest.  At blk_level 0 (SIMD), blocks are written as flat BFS.
 * At higher levels, each block is recursively decomposed at the next
 * finer level, then child subtrees are laid out at the SAME level.
 *
 * Example with depths=[2, 4] (SIMD=2, CL=4), tree depth 8:
 *   Level 1 (CL): root block of depth 4, decomposed at level 0 (SIMD).
 *     Level 0 (SIMD): root SIMD block (3 keys), 4 child SIMD blocks (3 each).
 *   Then 16 child CL blocks (each depth 4, decomposed at level 0).
 *   Total: 15 + 16*15 = 255 = 2^8 - 1. ✓
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

    if (blk_level == 0) {
        /* Finest level (SIMD): write BFS block, recurse children at level 0. */
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
                                child_remaining, 0, depths, tree_nodes);
            }
        }
    } else if (remaining <= block_depth) {
        /* Subtree fits within one block at this level.
           Decompose it at the next finer level. */
        lay_out_subtree(bfs_tree, out, rank_out, bfs_to_sorted,
                        nkeys, bfs_root, out_pos,
                        remaining, blk_level - 1, depths, tree_nodes);
    } else {
        /* remaining > block_depth: write one block (decomposed at finer
           level), then lay out each of the 2^block_depth child subtrees
           at THIS level. */
        lay_out_subtree(bfs_tree, out, rank_out, bfs_to_sorted,
                        nkeys, bfs_root, out_pos,
                        block_depth, blk_level - 1, depths, tree_nodes);

        int child_remaining = remaining - block_depth;
        size_t n_children = (size_t)1 << block_depth;
        size_t first_child = bfs_root;
        for (int i = 0; i < block_depth; i++)
            first_child = 2 * first_child + 1;
        for (size_t c = 0; c < n_children; c++) {
            lay_out_subtree(bfs_tree, out, rank_out, bfs_to_sorted,
                            nkeys, first_child + c, out_pos,
                            child_remaining, blk_level, depths,
                            tree_nodes);
        }
    }
}

/* ── Public leaf interface ──────────────────────────────────── */

void mt_leaf_extract_sorted(const mt_lnode_t *leaf, int32_t *out,
                             const mt_hierarchy_t *hier)
{
    int n = leaf->nkeys;
    if (n == 0) return;
    int d = leaf->tree_depth;
    size_t tree_nodes = ((size_t)1 << d) - 1;
    size_t cap = (size_t)hier->tree_cap;
    for (size_t i = 0; i < tree_nodes && i < cap; i++) {
        int ri = leaf->sorted_rank[i];
        if (ri >= 0 && ri < n)
            out[ri] = leaf->layout[i];
    }
}

void mt_leaf_build(mt_lnode_t *leaf, const int32_t *sorted_keys, int nkeys,
                   const mt_hierarchy_t *hier)
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

    /* Hierarchical blocked layout using the hierarchy's blocking depths.
       depths[] is finest-to-coarsest; blk_level starts at the coarsest. */
    int depths[MT_MAX_LEVELS];
    int num_levels = hier->num_levels;
    for (int i = 0; i < num_levels; i++)
        depths[i] = hier->levels[i].depth;

    size_t out_pos = 0;
    lay_out_subtree(bfs_tree, leaf->layout, leaf->sorted_rank, bfs_to_sorted,
                    nkeys, 0, &out_pos, d_n, num_levels - 1, depths,
                    tree_nodes);

    /* Zero-fill unused slots. */
    size_t cap = (size_t)hier->leaf_cap + 1;
    if (cap > MT_LNODE_TREE_CAP) cap = MT_LNODE_TREE_CAP;
    for (size_t i = out_pos; i < cap; i++) {
        leaf->layout[i] = MT_KEY_MAX;
        leaf->sorted_rank[i] = (int16_t)nkeys;
    }
}

/* ── Search helpers ────────────────────────────────────────── */

/* State tracked during hierarchical search for predecessor resolution. */
typedef struct {
    size_t last_offset;   /* Layout offset of the last SIMD block processed */
    int    last_child;    /* Child index from the last SIMD comparison */
    int    last_depth;    /* Depth of the last block (2=SIMD, 1=scalar) */
} mt_search_state_t;

/*
 * Search within a hierarchically blocked subtree.
 *
 * Mirrors lay_out_subtree exactly: the recursive decomposition of the
 * search matches the recursive decomposition of the layout.
 *
 * Returns the combined child index for the block at this level
 * (in [0, 2^depth - 1]).  Also updates `state` with the last SIMD
 * block info for predecessor resolution.
 */
static int search_blocked(const int32_t *tree, int32_t key, __m128i v_key,
                          size_t offset, int depth, int level,
                          const int *depths, mt_search_state_t *state)
{
    if (depth <= 0)
        return 0;

    int block_depth = depths[level];

    if (level == 0) {
        /* Finest level (SIMD): navigate through SIMD blocks.
           This matches the level-0 case of lay_out_subtree. */
        int combined = 0;
        int remaining = depth;
        size_t off = offset;

        while (remaining > 0) {
            int d = (remaining < block_depth) ? remaining : block_depth;
            state->last_offset = off;
            state->last_depth = d;

            int child;
            if (d >= MT_DK) {
                __m128i v_tree = _mm_loadu_si128(
                    (const __m128i *)(tree + off));
                __m128i v_cmp = _mm_cmpgt_epi32(v_key, v_tree);
                int mask = _mm_movemask_ps(_mm_castsi128_ps(v_cmp));
                child = MT_SIMD_LOOKUP[mask & 0x7];
                if (child < 0) child = 0;
            } else {
                /* d == 1: single scalar comparison */
                child = (key > tree[off]) ? 1 : 0;
            }

            state->last_child = child;
            combined = combined * (1 << d) + child;

            remaining -= d;
            if (remaining <= 0)
                break;

            size_t child_subtree = ((size_t)1 << remaining) - 1;
            off += ((size_t)1 << d) - 1 + (size_t)child * child_subtree;
        }

        return combined;
    }

    if (depth <= block_depth) {
        /* Subtree fits in one block: decompose at finer level.
           Matches the (remaining <= block_depth) case of lay_out_subtree. */
        return search_blocked(tree, key, v_key, offset, depth, level - 1,
                              depths, state);
    }

    /* depth > block_depth: search within a block, then jump to child.
       Matches the else branch of lay_out_subtree. */
    int block_child = search_blocked(tree, key, v_key, offset, block_depth,
                                     level - 1, depths, state);

    int remaining = depth - block_depth;
    size_t block_size = ((size_t)1 << block_depth) - 1;
    size_t child_subtree = ((size_t)1 << remaining) - 1;
    size_t child_offset = offset + block_size +
                          (size_t)block_child * child_subtree;

    return search_blocked(tree, key, v_key, child_offset, remaining,
                          level, depths, state);
}

/* ── Public search ──────────────────────────────────────────── */

int mt_leaf_search(const mt_lnode_t *leaf, int32_t key,
                   const mt_hierarchy_t *hier)
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
    size_t cap = (size_t)hier->tree_cap;
    for (size_t i = 0; i < tree_nodes && i < cap; i++) {
        if (rank[i] == 0) min_key = tree[i];
        if (rank[i] == nkeys - 1) max_key = tree[i];
    }

    if (key < min_key)
        return -1;
    if (key >= max_key)
        return nkeys - 1;

    /* Hierarchical SIMD traversal of the FAST blocked layout. */
    int depths[MT_MAX_LEVELS];
    int num_levels = hier->num_levels;
    for (int i = 0; i < num_levels; i++)
        depths[i] = hier->levels[i].depth;

    __m128i v_key = _mm_set1_epi32(key);
    mt_search_state_t state = { 0, 0, 0 };

    search_blocked(tree, key, v_key, 0, d_n, num_levels - 1,
                   depths, &state);

    /* Resolve predecessor using the last SIMD block's position and child. */
    size_t off = state.last_offset;
    int child_index = state.last_child;
    int lo;

    if (state.last_depth >= MT_DK) {
        /* Last block was a full SIMD block (depth 2, 3 keys). */
        int r;
        switch (child_index) {
            case 0:  r = rank[off + 1]; lo = r - 1; break;
            case 1:  r = rank[off + 1]; lo = r;     break;
            case 2:  r = rank[off];     lo = r;     break;
            default: r = rank[off + 2]; lo = r;     break;
        }
    } else {
        /* Last block was a single scalar comparison (depth 1). */
        int r = rank[off];
        lo = (child_index == 0) ? r - 1 : r;
    }

    if (lo < 0) lo = 0;

    /* Build sorted key array on stack from layout + rank. */
    int32_t sorted[512];
    for (size_t i = 0; i < tree_nodes && i < cap; i++) {
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
