/*
 * inode.c — Internal node search using SIMD-accelerated binary search.
 *
 * Internal nodes store keys in sorted order (not FAST-blocked) so that
 * the search result directly yields the child pointer index without
 * needing a sorted_rank mapping.
 */

#include "matryoshka_internal.h"

/*
 * Find the child index to follow for `key` in an internal node.
 *
 * Returns i such that:
 *   - keys[i-1] <= key < keys[i]  (for 0 < i < nkeys)
 *   - i == 0 if key < keys[0]
 *   - i == nkeys if key >= keys[nkeys-1]
 *
 * Uses SIMD to compare 4 keys at a time within a binary search.
 */
int mt_inode_search(const mt_inode_t *node, int32_t key)
{
    const int32_t *keys = node->keys;
    int n = node->nkeys;

    if (n == 0)
        return 0;

    /* SIMD-accelerated linear scan for small n. */
    if (n <= 32) {
        __m128i vkey = _mm_set1_epi32(key);
        int i = 0;
        /* Process 4 keys at a time. */
        for (; i + 3 < n; i += 4) {
            __m128i vtree = _mm_loadu_si128((const __m128i *)(keys + i));
            __m128i vcmp = _mm_cmpgt_epi32(vtree, vkey); /* tree[j] > key */
            int mask = _mm_movemask_ps(_mm_castsi128_ps(vcmp));
            if (mask != 0) {
                /* First element where tree[j] > key. */
                return i + __builtin_ctz(mask);
            }
        }
        /* Scalar tail. */
        for (; i < n; i++) {
            if (keys[i] > key)
                return i;
        }
        return n;
    }

    /* Binary search for larger nodes. */
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (keys[mid] <= key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}
