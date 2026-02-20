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

#if defined(__AVX512F__)
    /* AVX-512: linear scan 16 keys at a time, profitable up to 64. */
    if (n <= 64) {
        __m512i vkey = _mm512_set1_epi32(key);
        int i = 0;
        for (; i + 15 < n; i += 16) {
            __m512i vtree = _mm512_loadu_si512((const void *)(keys + i));
            __mmask16 gt = _mm512_cmpgt_epi32_mask(vtree, vkey);
            if (gt != 0)
                return i + __builtin_ctz(gt);
        }
        if (i < n) {
            __mmask16 valid = (__mmask16)((1u << (n - i)) - 1);
            __m512i vtree = _mm512_maskz_loadu_epi32(valid, keys + i);
            __mmask16 gt = _mm512_mask_cmpgt_epi32_mask(valid, vtree, vkey);
            if (gt != 0)
                return i + __builtin_ctz(gt);
        }
        return n;
    }

#elif defined(__AVX2__)
    /* AVX2: linear scan 8 keys at a time, profitable up to 64. */
    if (n <= 64) {
        __m256i vkey = _mm256_set1_epi32(key);
        int i = 0;
        for (; i + 7 < n; i += 8) {
            __m256i vtree = _mm256_loadu_si256((const __m256i *)(keys + i));
            __m256i vcmp = _mm256_cmpgt_epi32(vtree, vkey);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(vcmp));
            if (mask != 0)
                return i + __builtin_ctz(mask);
        }
        for (; i < n; i++) {
            if (keys[i] > key)
                return i;
        }
        return n;
    }

#else
    /* SSE2: linear scan 4 keys at a time for small n. */
    if (n <= 32) {
        __m128i vkey = _mm_set1_epi32(key);
        int i = 0;
        for (; i + 3 < n; i += 4) {
            __m128i vtree = _mm_loadu_si128((const __m128i *)(keys + i));
            __m128i vcmp = _mm_cmpgt_epi32(vtree, vkey);
            int mask = _mm_movemask_ps(_mm_castsi128_ps(vcmp));
            if (mask != 0)
                return i + __builtin_ctz(mask);
        }
        for (; i < n; i++) {
            if (keys[i] > key)
                return i;
        }
        return n;
    }
#endif

    /* Binary search with prefetching for larger nodes. */
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        /* Prefetch the midpoints of both possible next halves so
           the next iteration's key load hits warm cache. */
        __builtin_prefetch(&keys[lo + (mid - lo) / 2], 0, 0);
        __builtin_prefetch(&keys[mid + 1 + (hi - mid - 1) / 2], 0, 0);
        if (keys[mid] <= key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}
