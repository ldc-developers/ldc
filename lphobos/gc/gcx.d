/*
 *  Copyright (C) 2004 by Digital Mars, www.digitalmars.com
 *  Written by Walter Bright
 *
 *  This software is provided 'as-is', without any express or implied
 *  warranty. In no event will the authors be held liable for any damages
 *  arising from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute it
 *  freely, subject to the following restrictions:
 *
 *  o  The origin of this software must not be misrepresented; you must not
 *     claim that you wrote the original software. If you use this software
 *     in a product, an acknowledgment in the product documentation would be
 *     appreciated but is not required.
 *  o  Altered source versions must be plainly marked as such, and must not
 *     be misrepresented as being the original software.
 *  o  This notice may not be removed or altered from any source
 *     distribution.
 */

// D Garbage Collector stub to prevent linking in gc

/*
 * Modified for use as the preliminary GC for LLVMDC (LLVM D Compiler)
 * by Tomas Lindquist Olsen, Dec 2007
 */

module gcx;

debug=PRINTF;

/***************************************************/


version (Win32)
{
    import win32;
}

version (linux)
{
    import gclinux;
}

import gcstats;
import stdc = std.c.stdlib;


//alias GC* gc_t;
alias GC gc_t;

//struct GCStats { }

/* ============================ GC =============================== */


//alias int size_t;
alias void (*GC_FINALIZER)(void *p, bool dummy);

const uint GCVERSION = 1;   // increment every time we change interface
                // to GC.

class GC
{
    uint gcversion = GCVERSION;

    void *gcx;          // implementation

    void initialize()
    {
    debug(PRINTF) printf("GC initialize()\n");
    }


    void Dtor()
    {
    debug(PRINTF) printf("GC Dtor()\n");
    }

    invariant
    {
    debug(PRINTF) printf("GC invariant()\n");
    }

    void *malloc(size_t size)
    {
    debug(PRINTF) printf("GC malloc()\n");
    return malloc(size);
    }

    void *mallocNoSync(size_t size)
    {
    debug(PRINTF) printf("GC mallocNoSync()\n");
    return malloc(size);
    }


    void *calloc(size_t size, size_t n)
    {
    debug(PRINTF) printf("GC calloc()\n");
    return calloc(n, size);
    }


    void *realloc(void *p, size_t size)
    {
    debug(PRINTF) printf("GC realloc()\n");
    return realloc(p, size);
    }


    void free(void *p)
    {
    debug(PRINTF) printf("GC free()\n");
    stdc.free(p);
    }

    size_t capacity(void *p)
    {
    debug(PRINTF) printf("GC capacity()\n");
    return 0;
    }

    void check(void *p)
    {
    debug(PRINTF) printf("GC check()\n");
    }


    void setStackBottom(void *p)
    {
    debug(PRINTF) printf("GC setStackBottom()\n");
    }

    static void scanStaticData(gc_t g)
    {
    void *pbot;
    void *ptop;
    uint nbytes;

    debug(PRINTF) printf("GC scanStaticData()\n");
    //debug(PRINTF) printf("+GC.scanStaticData()\n");
    os_query_staticdataseg(&pbot, &nbytes);
    ptop = pbot + nbytes;
    g.addRange(pbot, ptop);
    //debug(PRINTF) printf("-GC.scanStaticData()\n");
    }

    static void unscanStaticData(gc_t g)
    {
    void *pbot;
    uint nbytes;

    debug(PRINTF) printf("GC unscanStaticData()\n");
    os_query_staticdataseg(&pbot, &nbytes);
    g.removeRange(pbot);
    }


    void addRoot(void *p)   // add p to list of roots
    {
    debug(PRINTF) printf("GC addRoot()\n");
    }

    void removeRoot(void *p)    // remove p from list of roots
    {
    debug(PRINTF) printf("GC removeRoot()\n");
    }

    void addRange(void *pbot, void *ptop)   // add range to scan for roots
    {
    debug(PRINTF) printf("GC addRange()\n");
    }

    void removeRange(void *pbot)        // remove range
    {
    debug(PRINTF) printf("GC removeRange()\n");
    }

    void fullCollect()      // do full garbage collection
    {
    debug(PRINTF) printf("GC fullCollect()\n");
    }

    void fullCollectNoStack()       // do full garbage collection
    {
    debug(PRINTF) printf("GC fullCollectNoStack()\n");
    }

    void genCollect()   // do generational garbage collection
    {
    debug(PRINTF) printf("GC genCollect()\n");
    }

    void minimize() // minimize physical memory usage
    {
    debug(PRINTF) printf("GC minimize()\n");
    }

    void setFinalizer(void *p, GC_FINALIZER pFn)
    {
    debug(PRINTF) printf("GC setFinalizer()\n");
    }

    void enable()
    {
    debug(PRINTF) printf("GC enable()\n");
    }

    void disable()
    {
    debug(PRINTF) printf("GC disable()\n");
    }

    void getStats(out GCStats stats)
    {
    debug(PRINTF) printf("GC getStats()\n");
    }

    void hasPointers(void* p)
    {
    debug(PRINTF) printf("GC hasPointers()\n");
    }

    void hasNoPointers(void* p)
    {
    debug(PRINTF) printf("GC hasNoPointers()\n");
    }

    void setV1_0()
    {
    debug(PRINTF) printf("GC setV1_0()\n");
    assert(0);
    }

    size_t extend(void* p, size_t minsize, size_t maxsize)
    {
    debug(PRINTF) printf("GC extend()\n");
    assert(0);
    }
}
