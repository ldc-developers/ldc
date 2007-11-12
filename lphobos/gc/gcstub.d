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

module gcx;

//debug=PRINTF;

/***************************************************/

import object;

version (Win32)
{
    import win32;
}

version (linux)
{
    import gclinux;
}


//alias GC* gc_t;
alias GC gc_t;

struct GCStats { }

/* ============================ GC =============================== */


//alias int size_t;
alias void (*GC_FINALIZER)(void *p, void *dummy);

const uint GCVERSION = 1;	// increment every time we change interface
				// to GC.

class GC
{
    uint gcversion = GCVERSION;

    void *gcx;			// implementation

    void initialize()
    {
	debug(PRINTF) printf("initialize()\n");
    }


    void Dtor()
    {
	debug(PRINTF) printf("Dtor()\n");
    }

    /+invariant
    {
	debug(PRINTF) printf("invariant()\n");
    }+/

    void *malloc(size_t size)
    {
	debug(PRINTF) printf("malloc()\n");
	return null;
    }

    void *mallocNoSync(size_t size)
    {
	debug(PRINTF) printf("mallocNoSync()\n");
	return null;
    }


    void *calloc(size_t size, size_t n)
    {
	debug(PRINTF) printf("calloc()\n");
	return null;
    }


    void *realloc(void *p, size_t size)
    {
	debug(PRINTF) printf("realloc()\n");
	return null;
    }


    void free(void *p)
    {
	debug(PRINTF) printf("free()\n");
    }

    size_t capacity(void *p)
    {
	debug(PRINTF) printf("capacity()\n");
	return 0;
    }

    void check(void *p)
    {
	debug(PRINTF) printf("check()\n");
    }


    void setStackBottom(void *p)
    {
	debug(PRINTF) printf("setStackBottom()\n");
    }

    static void scanStaticData(gc_t g)
    {
	void *pbot;
	void *ptop;
	uint nbytes;

	debug(PRINTF) printf("scanStaticData()\n");
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

	debug(PRINTF) printf("unscanStaticData()\n");
	os_query_staticdataseg(&pbot, &nbytes);
	g.removeRange(pbot);
    }


    void addRoot(void *p)	// add p to list of roots
    {
	debug(PRINTF) printf("addRoot()\n");
    }

    void removeRoot(void *p)	// remove p from list of roots
    {
	debug(PRINTF) printf("removeRoot()\n");
    }

    void addRange(void *pbot, void *ptop)	// add range to scan for roots
    {
	debug(PRINTF) printf("addRange()\n");
    }

    void removeRange(void *pbot)		// remove range
    {
	debug(PRINTF) printf("removeRange()\n");
    }

    void fullCollect()		// do full garbage collection
    {
	debug(PRINTF) printf("fullCollect()\n");
    }

    void fullCollectNoStack()		// do full garbage collection
    {
	debug(PRINTF) printf("fullCollectNoStack()\n");
    }

    void genCollect()	// do generational garbage collection
    {
	debug(PRINTF) printf("genCollect()\n");
    }

    void minimize()	// minimize physical memory usage
    {
	debug(PRINTF) printf("minimize()\n");
    }

    void setFinalizer(void *p, GC_FINALIZER pFn)
    {
	debug(PRINTF) printf("setFinalizer()\n");
    }

    void enable()
    {
	debug(PRINTF) printf("enable()\n");
    }

    void disable()
    {
	debug(PRINTF) printf("disable()\n");
    }

    void getStats(out GCStats stats)
    {
	debug(PRINTF) printf("getStats()\n");
    }
}
