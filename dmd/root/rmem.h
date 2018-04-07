/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 */

#ifndef ROOT_MEM_H
#define ROOT_MEM_H

#include <stdint.h>

#if __LP64__ || _M_X64
#if LDC_HOST_FE_VER >= 2079 || _M_X64
typedef uint64_t d_size_t;
#else
typedef unsigned long d_size_t;
#endif
#else
typedef uint32_t d_size_t;
#endif

struct Mem
{
    Mem() { }

    static char *xstrdup(const char *s);
    static void *xmalloc(d_size_t size);
    static void *xcalloc(d_size_t size, d_size_t n);
    static void *xrealloc(void *p, d_size_t size);
    static void xfree(void *p);
    static void *xmallocdup(void *o, d_size_t size);
    static void error();
};

extern Mem mem;

#endif /* ROOT_MEM_H */
