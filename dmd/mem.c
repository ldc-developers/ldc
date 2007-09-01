
/* Copyright (c) 2000 Digital Mars	*/
/* All Rights Reserved 			*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gc.h"

#include "mem.h"

/* This implementation of the storage allocator uses the standard C allocation package.
 */

Mem mem;

void Mem::init()
{
    GC_init();
}

char *Mem::strdup(const char *s)
{
    char *p;

    if (s)
    {
	p = GC_strdup(s);
	if (p)
	    return p;
	error();
    }
    return NULL;
}

void *Mem::malloc(size_t size)
{   void *p;

    if (!size)
	p = NULL;
    else
    {
	p = GC_malloc(size);
	if (!p)
	    error();
    }
    return p;
}

void *Mem::calloc(size_t size, size_t n)
{   void *p;

    if (!size || !n)
	p = NULL;
    else
    {
	p = GC_malloc(size * n);
	if (!p)
	    error();
        memset(p, 0, size * n);
    }
    return p;
}

void *Mem::realloc(void *p, size_t size)
{
    if (!size)
    {	if (p)
	{   GC_free(p);
	    p = NULL;
	}
    }
    else if (!p)
    {
	p = GC_malloc(size);
	if (!p)
	    error();
    }
    else
    {
	p = GC_realloc(p, size);
	if (!p)
	    error();
    }
    return p;
}

void Mem::free(void *p)
{
    if (p)
	GC_free(p);
}

void *Mem::mallocdup(void *o, size_t size)
{   void *p;

    if (!size)
	p = NULL;
    else
    {
	p = GC_malloc(size);
	if (!p)
	    error();
	else
	    memcpy(p,o,size);
    }
    return p;
}

void Mem::error()
{
    printf("Error: out of memory\n");
    exit(EXIT_FAILURE);
}

void Mem::fullcollect()
{
    GC_gcollect();
}

void Mem::mark(void *pointer)
{
    (void) pointer;		// necessary for VC /W4
}

/* =================================================== */

void * operator new(size_t m_size)
{   
    void *p = GC_malloc(m_size);
    if (p)
	return p;
    printf("Error: out of memory\n");
    exit(EXIT_FAILURE);
    return p;
}

void operator delete(void *p)
{
    GC_free(p);
}


