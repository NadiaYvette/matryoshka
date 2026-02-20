/*
 * alloc.c — Node allocation for matryoshka trees.
 *
 * Internal nodes use posix_memalign (always page-sized).
 * Leaf nodes use the arena allocator if provided, otherwise posix_memalign.
 */

#include "matryoshka_internal.h"
#include <stdlib.h>
#include <string.h>

mt_node_t *mt_alloc_inode(void)
{
    void *p = NULL;
    if (posix_memalign(&p, MT_PAGE_SIZE, MT_PAGE_SIZE) != 0)
        return NULL;
    memset(p, 0, MT_PAGE_SIZE);
    ((mt_inode_t *)p)->type = MT_NODE_INTERNAL;
    return (mt_node_t *)p;
}

mt_node_t *mt_alloc_lnode(const mt_hierarchy_t *hier, mt_allocator_t *alloc)
{
    size_t alloc_size = hier->leaf_alloc;
    void *p = NULL;

    if (alloc) {
        p = mt_allocator_alloc(alloc);
    } else {
        size_t align = (alloc_size >= MT_PAGE_SIZE) ? alloc_size : MT_PAGE_SIZE;
        if (posix_memalign(&p, align, alloc_size) != 0)
            return NULL;
    }

    if (!p) return NULL;
    memset(p, 0, alloc_size);
    ((mt_lnode_t *)p)->header.type = MT_NODE_LEAF;
    return (mt_node_t *)p;
}

void mt_free_inode(mt_node_t *node)
{
    free(node);
}

void mt_free_lnode(mt_node_t *node, mt_allocator_t *alloc)
{
    if (alloc) {
        mt_allocator_free(alloc, node);
    } else {
        free(node);
    }
}
