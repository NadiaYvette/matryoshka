/*
 * alloc.c — Page-aligned node allocation for matryoshka trees.
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

mt_node_t *mt_alloc_lnode(void)
{
    void *p = NULL;
    if (posix_memalign(&p, MT_PAGE_SIZE, MT_PAGE_SIZE) != 0)
        return NULL;
    memset(p, 0, MT_PAGE_SIZE);
    ((mt_lnode_t *)p)->type = MT_NODE_LEAF;
    return (mt_node_t *)p;
}

void mt_free_node(mt_node_t *node)
{
    free(node);
}
