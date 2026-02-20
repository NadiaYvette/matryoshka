/*
 * arena.c — Superpage arena allocator for matryoshka trees.
 *
 * Allocates leaf nodes from superpage-aligned arenas using
 * mmap(MAP_HUGETLB) on Linux, falling back to posix_memalign.
 * Each arena is a contiguous, aligned region subdivided into
 * fixed-size pages tracked by a bitmap.
 *
 * For superpage-level leaves, the entire arena IS one leaf.
 * For page-level leaves, multiple leaves are co-located within
 * a single arena for TLB locality.
 */

#include "matryoshka_internal.h"
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <sys/mman.h>
#endif

/* ── Arena allocation ──────────────────────────────────────────── */

static mt_arena_t *arena_create(size_t arena_size, size_t page_size)
{
    /* Allocate the arena metadata. */
    int num_pages = (int)(arena_size / page_size);
    if (num_pages <= 0) num_pages = 1;

    int bitmap_words = (num_pages + 63) / 64;
    mt_arena_t *arena = malloc(sizeof(mt_arena_t) +
                               (size_t)bitmap_words * sizeof(uint64_t));
    if (!arena) return NULL;

    /* Try mmap with MAP_HUGETLB for large arenas (>= 2 MiB). */
    void *base = NULL;
#ifdef __linux__
    if (arena_size >= (2u << 20)) {
        base = mmap(NULL, arena_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (base == MAP_FAILED)
            base = NULL;
    }
#endif

    /* Fallback to posix_memalign. */
    if (!base) {
        size_t align = page_size;
        if (align < sizeof(void *)) align = sizeof(void *);
        if (posix_memalign(&base, align, arena_size) != 0) {
            free(arena);
            return NULL;
        }
        arena->is_mmap = false;
#ifdef __linux__
        /* Hint the kernel to back this with transparent huge pages. */
        madvise(base, arena_size, MADV_HUGEPAGE);
#endif
    }
#ifdef __linux__
    else {
        arena->is_mmap = true;
    }
#endif

    memset(base, 0, arena_size);

    arena->base = base;
    arena->size = arena_size;
    arena->page_size = page_size;
    arena->num_pages = num_pages;
    arena->bitmap = (uint64_t *)((char *)arena + sizeof(mt_arena_t));
    memset(arena->bitmap, 0, (size_t)bitmap_words * sizeof(uint64_t));
    arena->next = NULL;

    return arena;
}

static void arena_destroy(mt_arena_t *arena)
{
#ifdef __linux__
    if (arena->is_mmap) {
        munmap(arena->base, arena->size);
    } else {
        free(arena->base);
    }
#else
    free(arena->base);
#endif
    free(arena);
}

/* Find a free page in the arena.  Returns page index or -1. */
static int arena_find_free(mt_arena_t *arena)
{
    int bitmap_words = (arena->num_pages + 63) / 64;
    for (int w = 0; w < bitmap_words; w++) {
        if (arena->bitmap[w] != UINT64_MAX) {
            int bit = __builtin_ctzll(~arena->bitmap[w]);
            int page_idx = w * 64 + bit;
            if (page_idx < arena->num_pages)
                return page_idx;
        }
    }
    return -1;
}

static void *arena_alloc_page(mt_arena_t *arena)
{
    int idx = arena_find_free(arena);
    if (idx < 0) return NULL;

    arena->bitmap[idx / 64] |= (1ULL << (idx % 64));
    return (char *)arena->base + (size_t)idx * arena->page_size;
}

static void arena_free_page(mt_arena_t *arena, void *page)
{
    size_t offset = (size_t)((char *)page - (char *)arena->base);
    int idx = (int)(offset / arena->page_size);
    if (idx >= 0 && idx < arena->num_pages)
        arena->bitmap[idx / 64] &= ~(1ULL << (idx % 64));
}

/* Check if a pointer belongs to this arena. */
static bool arena_contains(const mt_arena_t *arena, const void *ptr)
{
    const char *p = (const char *)ptr;
    const char *base = (const char *)arena->base;
    return p >= base && p < base + arena->size;
}

/* ── Public allocator interface ────────────────────────────────── */

mt_allocator_t *mt_allocator_create(size_t arena_size, size_t page_size)
{
    mt_allocator_t *alloc = malloc(sizeof(mt_allocator_t));
    if (!alloc) return NULL;
    alloc->arenas = NULL;
    alloc->arena_size = arena_size;
    alloc->page_size = page_size;
    return alloc;
}

void mt_allocator_destroy(mt_allocator_t *alloc)
{
    if (!alloc) return;
    mt_arena_t *a = alloc->arenas;
    while (a) {
        mt_arena_t *next = a->next;
        arena_destroy(a);
        a = next;
    }
    free(alloc);
}

void *mt_allocator_alloc(mt_allocator_t *alloc)
{
    /* Try existing arenas first. */
    mt_arena_t *a = alloc->arenas;
    while (a) {
        void *p = arena_alloc_page(a);
        if (p) return p;
        a = a->next;
    }

    /* Create a new arena. */
    mt_arena_t *na = arena_create(alloc->arena_size, alloc->page_size);
    if (!na) return NULL;
    na->next = alloc->arenas;
    alloc->arenas = na;

    return arena_alloc_page(na);
}

void mt_allocator_free(mt_allocator_t *alloc, void *ptr)
{
    if (!ptr || !alloc) return;
    mt_arena_t *a = alloc->arenas;
    while (a) {
        if (arena_contains(a, ptr)) {
            arena_free_page(a, ptr);
            return;
        }
        a = a->next;
    }
    /* Pointer not from any arena — should not happen. */
}
