/**
 * Part of the D programming language runtime library.
 */

/*
 *  Copyright (C) 2004-2007 by Digital Mars, www.digitalmars.com
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


// Storage allocation

module std.gc;

//debug = PRINTF;

public import std.c.stdarg;
public import std.c.stdlib;
public import std.c.string;
public import gcx;
public import std.outofmemory;
public import gcstats;
public import std.thread;

version=GCCLASS;

version (GCCLASS)
    alias GC gc_t;
else
    alias GC* gc_t;

gc_t _gc;

void addRoot(void *p)		      { _gc.addRoot(p); }
void removeRoot(void *p)	      { _gc.removeRoot(p); }
void addRange(void *pbot, void *ptop) { _gc.addRange(pbot, ptop); }
void removeRange(void *pbot)	      { _gc.removeRange(pbot); }
void fullCollect()		      { _gc.fullCollect(); }
void fullCollectNoStack()	      { _gc.fullCollectNoStack(); }
void genCollect()		      { _gc.genCollect(); }
void minimize()			      { _gc.minimize(); }
void disable()			      { _gc.disable(); }
void enable()			      { _gc.enable(); }
void getStats(out GCStats stats)      { _gc.getStats(stats); }
void hasPointers(void* p)	      { _gc.hasPointers(p); }
void hasNoPointers(void* p)	      { _gc.hasNoPointers(p); }
void setV1_0()			      { _gc.setV1_0(); }

void[] malloc(size_t nbytes)
{
    void* p = _gc.malloc(nbytes);
    return p[0 .. nbytes];
}

void[] realloc(void* p, size_t nbytes)
{
    void* q = _gc.realloc(p, nbytes);
    return q[0 .. nbytes];
}

size_t extend(void* p, size_t minbytes, size_t maxbytes)
{
    return _gc.extend(p, minbytes, maxbytes);
}

size_t capacity(void* p)
{
    return _gc.capacity(p);
}

void setTypeInfo(TypeInfo ti, void* p)
{
    if (ti.flags() & 1)
	hasNoPointers(p);
    else
	hasPointers(p);
}

void* getGCHandle()
{
    return cast(void*)_gc;
}

void setGCHandle(void* p)
{
    void* oldp = getGCHandle();
    gc_t g = cast(gc_t)p;
    if (g.gcversion != gcx.GCVERSION)
	throw new Error("incompatible gc versions");

    // Add our static data to the new gc
    GC.scanStaticData(g);

    _gc = g;
//    return oldp;
}

void endGCHandle()
{
    GC.unscanStaticData(_gc);
}

extern (C)
{

void _d_monitorrelease(Object h);


void gc_init()
{
    version (GCCLASS)
    {	void* p;
	ClassInfo ci = GC.classinfo;

	p = std.c.stdlib.malloc(ci.init.length);
	(cast(byte*)p)[0 .. ci.init.length] = ci.init[];
	_gc = cast(GC)p;
    }
    else
    {
	_gc = cast(GC *) std.c.stdlib.calloc(1, GC.sizeof);
    }
    _gc.initialize();
    GC.scanStaticData(_gc);
    std.thread.Thread.thread_init();
}

void gc_term()
{
    _gc.fullCollectNoStack();
    _gc.Dtor();
}

}
