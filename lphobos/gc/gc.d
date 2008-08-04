/**
 * Part of the D programming language runtime library.
 */

/*
 *  Copyright (C) 2004-2008 by Digital Mars, www.digitalmars.com
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

/* NOTE: This file has been patched from the original DMD distribution to
   work with the GDC compiler.

   Modified by David Friedman, February 2007
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

void _d_monitorexit(Object h);


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

Object _d_newclass(ClassInfo ci)
{
    void *p;

    debug(PRINTF) printf("_d_newclass(ci = %p, %s)\n", ci, cast(char *)ci.name);
    if (ci.flags & 1)			// if COM object
    {
	p = std.c.stdlib.malloc(ci.init.length);
	if (!p)
	    _d_OutOfMemory();
	debug(PRINTF) printf(" COM object p = %p\n", p);
    }
    else
    {
	p = _gc.malloc(ci.init.length);
	debug(PRINTF) printf(" p = %p\n", p);
	_gc.setFinalizer(p, &new_finalizer);
	if (ci.flags & 2)
	    _gc.hasNoPointers(p);
    }

    debug (PRINTF)
    {
	printf("p = %p\n", p);
	printf("ci = %p, ci.init = %p, len = %d\n", ci, ci.init, ci.init.length);
	printf("vptr = %p\n", *cast(void **)ci.init);
	printf("vtbl[0] = %p\n", (*cast(void ***)ci.init)[0]);
	printf("vtbl[1] = %p\n", (*cast(void ***)ci.init)[1]);
	printf("init[0] = %x\n", (cast(uint *)ci.init)[0]);
	printf("init[1] = %x\n", (cast(uint *)ci.init)[1]);
	printf("init[2] = %x\n", (cast(uint *)ci.init)[2]);
	printf("init[3] = %x\n", (cast(uint *)ci.init)[3]);
	printf("init[4] = %x\n", (cast(uint *)ci.init)[4]);
    }


    // Initialize it
    (cast(byte*)p)[0 .. ci.init.length] = ci.init[];

    //printf("initialization done\n");
    return cast(Object)p;
}

extern (D) alias void (*fp_t)(Object);		// generic function pointer

void _d_delinterface(void** p)
{
    if (*p)
    {
	Interface *pi = **cast(Interface ***)*p;
	Object o;

	o = cast(Object)(*p - pi.offset);
	_d_delclass(&o);
	*p = null;
    }
}

void _d_delclass(Object *p)
{
    if (*p)
    {
	debug (PRINTF) printf("_d_delclass(%p)\n", *p);
	version(0)
	{
	    ClassInfo **pc = cast(ClassInfo **)*p;
	    if (*pc)
	    {
		ClassInfo c = **pc;

		if (c.deallocator)
		{
		    _d_callfinalizer(cast(void *)(*p));
		    fp_t fp = cast(fp_t)c.deallocator;
		    (*fp)(*p);			// call deallocator
		    *p = null;
		    return;
		}
	    }
	}
	_gc.free(cast(void*)(*p));
	*p = null;
    }
}

/******************************************
 * Allocate a new array of length elements.
 * ti is the type of the resulting array, or pointer to element.
 */

/* For when the array is initialized to 0 */
void* _d_newarrayT(TypeInfo ti, size_t length)
{
    void* result;
    auto size = ti.next.tsize();		// array element size

    debug(PRINTF) printf("_d_newarrayT(length = x%x, size = %d)\n", length, size);
    if (length && size)
    {
	/*version (D_InlineAsm_X86)
	{
	    asm
	    {
		mov	EAX,size	;
		mul	EAX,length	;
		mov	size,EAX	;
		jc	Loverflow	;
	    }
	}
	else*/
	    size *= length;
	result = cast(byte*) _gc.malloc(size + 1);
	if (!(ti.next.flags() & 1))
	    _gc.hasNoPointers(result);
	memset(result, 0, size);
    }
    return result;

Loverflow:
    _d_OutOfMemory();
}

/* For when the array has a non-zero initializer.
 */
void* _d_newarrayiT(TypeInfo ti, size_t length)
{
    void* result;
    auto size = ti.next.tsize();		// array element size

    debug(PRINTF)
	 printf("_d_newarrayiT(length = %d, size = %d)\n", length, size);
    if (length == 0 || size == 0)
	{ }
    else
    {
	auto initializer = ti.next.init();
	auto isize = initializer.length;
	auto q = initializer.ptr;
	/*version (D_InlineAsm_X86)
	{
	    asm
	    {
		mov	EAX,size	;
		mul	EAX,length	;
		mov	size,EAX	;
		jc	Loverflow	;
	    }
	}
	else*/
	    size *= length;
	auto p = _gc.malloc(size + 1);
	debug(PRINTF) printf(" p = %p\n", p);
	if (!(ti.next.flags() & 1))
	    _gc.hasNoPointers(p);
	if (isize == 1)
	    memset(p, *cast(ubyte*)q, size);
	else if (isize == int.sizeof)
	{
	    int init = *cast(int*)q;
	    size /= int.sizeof;
	    for (size_t u = 0; u < size; u++)
	    {
		(cast(int*)p)[u] = init;
	    }
	}
	else
	{
	    for (size_t u = 0; u < size; u += isize)
	    {
		memcpy(p + u, q, isize);
	    }
	}
	result = cast(byte*) p;
    }
    return result;

Loverflow:
    _d_OutOfMemory();
}

void[] _d_newarraymTp(TypeInfo ti, int ndims, size_t* pdim)
{
    void[] result = void;

    //debug(PRINTF)
	//printf("_d_newarraymT(ndims = %d)\n", ndims);
    if (ndims == 0)
	result = null;
    else
    {

	void[] foo(TypeInfo ti, size_t* pdim, int ndims)
	{
	    size_t dim = *pdim;
	    void[] p;

	    //printf("foo(ti = %p, ti.next = %p, dim = %d, ndims = %d\n", ti, ti.next, dim, ndims);
	    if (ndims == 1)
	    {
		auto r = _d_newarrayT(ti, dim);
		p = *cast(void[]*)(&r);
	    }
	    else
	    {
		p = _gc.malloc(dim * (void[]).sizeof + 1)[0 .. dim];
		for (int i = 0; i < dim; i++)
		{
		    (cast(void[]*)p.ptr)[i] = foo(ti.next, pdim + 1, ndims - 1);
		}
	    }
	    return p;
	}

	result = foo(ti, pdim, ndims);
	//printf("result = %llx\n", result);

	version (none)
	{
	    for (int i = 0; i < ndims; i++)
	    {
		printf("index %d: %d\n", i, pdim[i]);
	    }
	}
    }
    return result;
}

void[] _d_newarraymiTp(TypeInfo ti, int ndims, size_t* pdim)
{
    void[] result = void;

    //debug(PRINTF)
	//printf("_d_newarraymi(size = %d, ndims = %d)\n", size, ndims);
    if (ndims == 0)
	result = null;
    else
    {

	void[] foo(TypeInfo ti, size_t* pdim, int ndims)
	{
	    size_t dim = *pdim;
	    void[] p;

	    if (ndims == 1)
	    {
		auto r = _d_newarrayiT(ti, dim);
		p = *cast(void[]*)(&r);
	    }
	    else
	    {
		p = _gc.malloc(dim * (void[]).sizeof + 1)[0 .. dim];
		for (int i = 0; i < dim; i++)
		{
		    (cast(void[]*)p.ptr)[i] = foo(ti.next, pdim + 1, ndims - 1);
		}
	    }
	    return p;
	}

	result = foo(ti, pdim, ndims);
	//printf("result = %llx\n", result);

	version (none)
	{
	    for (int i = 0; i < ndims; i++)
	    {
		printf("index %d: %d\n", i, pdim[i]);
		printf("init = %d\n", *cast(int*)pinit);
	    }
	}
    }
    return result;
}

struct Array
{
    size_t length;
    byte *data;
};

// Perhaps we should get a a size argument like _d_new(), so we
// can zero out the array?

void _d_delarray(size_t plength, void* pdata)
{
    assert(!plength || pdata);
    if (pdata) _gc.free(pdata);
}


void _d_delmemory(void* *p)
{
    if (*p)
    {
	_gc.free(*p);
	*p = null;
    }
}


}

void new_finalizer(void *p, bool dummy)
{
    //printf("new_finalizer(p = %p)\n", p);
    _d_callfinalizer(p);
}

extern (C)
void _d_callinterfacefinalizer(void *p)
{
    //printf("_d_callinterfacefinalizer(p = %p)\n", p);
    if (p)
    {
	Interface *pi = **cast(Interface ***)p;
	Object o = cast(Object)(p - pi.offset);
	_d_callfinalizer(cast(void*)o);
    }
}

extern (C)
void _d_callfinalizer(void *p)
{
    //printf("_d_callfinalizer(p = %p)\n", p);
    if (p)	// not necessary if called from gc
    {
	ClassInfo **pc = cast(ClassInfo **)p;
	if (*pc)
	{
	    ClassInfo c = **pc;

	    try
	    {
		do
		{
		    if (c.destructor)
		    {
			fp_t fp = cast(fp_t)c.destructor;
			(*fp)(cast(Object)p);		// call destructor
		    }
		    c = c.base;
		} while (c);
		if ((cast(void**)p)[1])	// if monitor is not null
		    _d_monitorexit(cast(Object)p);
	    }
	    finally
	    {
		*pc = null;			// zero vptr
	    }
	}
    }
}

/+ ------------------------------------------------ +/


/******************************
 * Resize dynamic arrays with 0 initializers.
 */

extern (C)
byte* _d_arraysetlengthT(TypeInfo ti, size_t newlength, size_t plength, byte* pdata)
in
{
    assert(ti);
}
body
{
    byte* newdata;
    size_t sizeelem = ti.next.tsize();

    debug(PRINTF)
    {
	printf("_d_arraysetlengthT(p = %p, sizeelem = %d, newlength = %d)\n", p, sizeelem, newlength);
	if (p)
	    printf("\tpdata = %p, plength = %d\n", pdata, plength);
    }

    if (newlength)
    {
	version (GNU)
	{
	    // required to output the label;
	    static char x = 0;
	    if (x)
		goto Loverflow;
	}

	version (D_InlineAsm_X86)
	{
	    size_t newsize = void;

	    asm
	    {
		mov	EAX,newlength	;
		mul	EAX,sizeelem	;
		mov	newsize,EAX	;
		jc	Loverflow	;
	    }
	}
	else
	{
	    size_t newsize = sizeelem * newlength;

	    if (newsize / newlength != sizeelem)
		goto Loverflow;
	}
	//printf("newsize = %x, newlength = %x\n", newsize, newlength);

	if (pdata)
	{
	    newdata = pdata;
	    if (newlength > plength)
	    {
		size_t size = plength * sizeelem;
		size_t cap = _gc.capacity(pdata);

		if (cap <= newsize)
		{
		    if (cap >= 4096)
		    {	// Try to extend in-place
			auto u = _gc.extend(pdata, (newsize + 1) - cap, (newsize + 1) - cap);
			if (u)
			{
			    goto L1;
			}
		    }
		    newdata = cast(byte *)_gc.malloc(newsize + 1);
		    newdata[0 .. size] = pdata[0 .. size];
		    if (!(ti.next.flags() & 1))
			_gc.hasNoPointers(newdata);
		}
	     L1:
		newdata[size .. newsize] = 0;
	    }
	}
	else
	{
	    newdata = cast(byte *)_gc.calloc(newsize + 1, 1);
	    if (!(ti.next.flags() & 1))
		_gc.hasNoPointers(newdata);
	}
    }
    else
    {
	newdata = pdata;
    }

    pdata = newdata;
    plength = newlength;
    return newdata;

Loverflow:
    _d_OutOfMemory();
}

/**
 * Resize arrays for non-zero initializers.
 *	p		pointer to array lvalue to be updated
 *	newlength	new .length property of array
 *	sizeelem	size of each element of array
 *	initsize	size of initializer
 *	...		initializer
 */
extern (C)
byte* _d_arraysetlengthiT(TypeInfo ti, size_t newlength, size_t plength, byte* pdata)
in
{
    assert(!plength || pdata);
}
body
{
    byte* newdata;
    size_t sizeelem = ti.next.tsize();
    void[] initializer = ti.next.init();
    size_t initsize = initializer.length;

    assert(sizeelem);
    assert(initsize);
    assert(initsize <= sizeelem);
    assert((sizeelem / initsize) * initsize == sizeelem);

    debug(PRINTF)
    {
	printf("_d_arraysetlengthiT(p = %p, sizeelem = %d, newlength = %d, initsize = %d)\n", p, sizeelem, newlength, initsize);
	if (p)
	    printf("\tpdata = %p, plength = %d\n", pdata, plength);
    }

    if (newlength)
    {
	version (GNU)
	{
	    // required to output the label;
	    static char x = 0;
	    if (x)
		goto Loverflow;
	}

	version (D_InlineAsm_X86)
	{
	    size_t newsize = void;

	    asm
	    {
		mov	EAX,newlength	;
		mul	EAX,sizeelem	;
		mov	newsize,EAX	;
		jc	Loverflow	;
	    }
	}
	else
	{
	    size_t newsize = sizeelem * newlength;

	    if (newsize / newlength != sizeelem)
		goto Loverflow;
	}
	//printf("newsize = %x, newlength = %x\n", newsize, newlength);

	size_t size = plength * sizeelem;
	if (pdata)
	{
	    newdata = pdata;
	    if (newlength > plength)
	    {
		size_t cap = _gc.capacity(pdata);

		if (cap <= newsize)
		{
		    if (cap >= 4096)
		    {	// Try to extend in-place
			auto u = _gc.extend(pdata, (newsize + 1) - cap, (newsize + 1) - cap);
			if (u)
			{
			    goto L1;
			}
		    }
		    newdata = cast(byte *)_gc.malloc(newsize + 1);
		    newdata[0 .. size] = pdata[0 .. size];
		L1: ;
		}
	    }
	}
	else
	{
	    newdata = cast(byte *)_gc.malloc(newsize + 1);
	    if (!(ti.next.flags() & 1))
		_gc.hasNoPointers(newdata);
	}

	auto q = initializer.ptr;	// pointer to initializer

	if (newsize > size)
	{
	    if (initsize == 1)
	    {
		//printf("newdata = %p, size = %d, newsize = %d, *q = %d\n", newdata, size, newsize, *cast(byte*)q);
		newdata[size .. newsize] = *(cast(byte*)q);
	    }
	    else
	    {
		for (size_t u = size; u < newsize; u += initsize)
		{
		    memcpy(newdata + u, q, initsize);
		}
	    }
	}
    }
    else
    {
	newdata = pdata;
    }

    pdata = newdata;
    plength = newlength;
    return newdata;

Loverflow:
    _d_OutOfMemory();
}

/****************************************
 * Append y[] to array x[].
 * size is size of each array element.
 */

extern (C)
Array _d_arrayappendT(TypeInfo ti, Array *px, byte[] y)
{
    auto sizeelem = ti.next.tsize();		// array element size
    auto cap = _gc.capacity(px.data);
    auto length = px.length;
    auto newlength = length + y.length;
    auto newsize = newlength * sizeelem;
    if (newsize > cap)
    {   byte* newdata;

	if (cap >= 4096)
	{   // Try to extend in-place
	    auto u = _gc.extend(px.data, (newsize + 1) - cap, (newsize + 1) - cap);
	    if (u)
	    {
		goto L1;
	    }
	}

	newdata = cast(byte *)_gc.malloc(newCapacity(newlength, sizeelem) + 1);
	if (!(ti.next.flags() & 1))
	    _gc.hasNoPointers(newdata);
	memcpy(newdata, px.data, length * sizeelem);
	px.data = newdata;
    }
  L1:
    px.length = newlength;
    memcpy(px.data + length * sizeelem, y.ptr, y.length * sizeelem);
    return *px;
}

size_t newCapacity(size_t newlength, size_t size)
{
    version(none)
    {
	size_t newcap = newlength * size;
    }
    else
    {
	/*
	 * Better version by Dave Fladebo:
	 * This uses an inverse logorithmic algorithm to pre-allocate a bit more
	 * space for larger arrays.
	 * - Arrays smaller than 4096 bytes are left as-is, so for the most
	 * common cases, memory allocation is 1 to 1. The small overhead added
	 * doesn't effect small array perf. (it's virtually the same as
	 * current).
	 * - Larger arrays have some space pre-allocated.
	 * - As the arrays grow, the relative pre-allocated space shrinks.
	 * - The logorithmic algorithm allocates relatively more space for
	 * mid-size arrays, making it very fast for medium arrays (for
	 * mid-to-large arrays, this turns out to be quite a bit faster than the
	 * equivalent realloc() code in C, on Linux at least. Small arrays are
	 * just as fast as GCC).
	 * - Perhaps most importantly, overall memory usage and stress on the GC
	 * is decreased significantly for demanding environments.
	 */
	size_t newcap = newlength * size;
	size_t newext = 0;

	if (newcap > 4096)
	{
	    //double mult2 = 1.0 + (size / log10(pow(newcap * 2.0,2.0)));

	    // Redo above line using only integer math

	    static int log2plus1(size_t c)
	    {   int i;

		if (c == 0)
		    i = -1;
		else
		    for (i = 1; c >>= 1; i++)
			{   }
		return i;
	    }

	    /* The following setting for mult sets how much bigger
	     * the new size will be over what is actually needed.
	     * 100 means the same size, more means proportionally more.
	     * More means faster but more memory consumption.
	     */
	    //long mult = 100 + (1000L * size) / (6 * log2plus1(newcap));
	    long mult = 100 + (1000L * size) / log2plus1(newcap);

	    // testing shows 1.02 for large arrays is about the point of diminishing return
	    if (mult < 102)
		mult = 102;
	    newext = cast(size_t)((newcap * mult) / 100);
	    newext -= newext % size;
	    //printf("mult: %2.2f, mult2: %2.2f, alloc: %2.2f\n",mult/100.0,mult2,newext / cast(double)size);
	}
	newcap = newext > newcap ? newext : newcap;
	//printf("newcap = %d, newlength = %d, size = %d\n", newcap, newlength, size);
    }
    return newcap;
}

extern (C)
byte[] _d_arrayappendcTp(TypeInfo ti, inout byte[] x, byte *argp)
{
    auto sizeelem = ti.next.tsize();		// array element size
    auto cap = _gc.capacity(x.ptr);
    auto length = x.length;
    auto newlength = length + 1;
    auto newsize = newlength * sizeelem;

    assert(cap == 0 || length * sizeelem <= cap);

    //printf("_d_arrayappendc(sizeelem = %d, ptr = %p, length = %d, cap = %d)\n", sizeelem, x.ptr, x.length, cap);

    if (newsize >= cap)
    {   byte* newdata;

	if (cap >= 4096)
	{   // Try to extend in-place
	    auto u = _gc.extend(x.ptr, (newsize + 1) - cap, (newsize + 1) - cap);
	    if (u)
	    {
		goto L1;
	    }
	}

	//printf("_d_arrayappendc(sizeelem = %d, newlength = %d, cap = %d)\n", sizeelem, newlength, cap);
	cap = newCapacity(newlength, sizeelem);
	assert(cap >= newlength * sizeelem);
	newdata = cast(byte *)_gc.malloc(cap + 1);
	if (!(ti.next.flags() & 1))
	    _gc.hasNoPointers(newdata);
	memcpy(newdata, x.ptr, length * sizeelem);
	(cast(void **)(&x))[1] = newdata;
    }
  L1:

    *cast(size_t *)&x = newlength;
    x.ptr[length * sizeelem .. newsize] = argp[0 .. sizeelem];
    assert((cast(size_t)x.ptr & 15) == 0);
    assert(_gc.capacity(x.ptr) >= x.length * sizeelem);
    return x;
}

extern (C)
byte[] _d_arraycatT(TypeInfo ti, byte[] x, byte[] y)
out (result)
{
    auto sizeelem = ti.next.tsize();		// array element size
    //printf("_d_arraycatT(%d,%p ~ %d,%p sizeelem = %d => %d,%p)\n", x.length, x.ptr, y.length, y.ptr, sizeelem, result.length, result.ptr);
    assert(result.length == x.length + y.length);
    for (size_t i = 0; i < x.length * sizeelem; i++)
	assert((cast(byte*)result)[i] == (cast(byte*)x)[i]);
    for (size_t i = 0; i < y.length * sizeelem; i++)
	assert((cast(byte*)result)[x.length * sizeelem + i] == (cast(byte*)y)[i]);

    size_t cap = _gc.capacity(result.ptr);
    assert(!cap || cap > result.length * sizeelem);
}
body
{
    version (none)
    {
	/* Cannot use this optimization because:
	 *  char[] a, b;
	 *  char c = 'a';
	 *	b = a ~ c;
	 *	c = 'b';
	 * will change the contents of b.
	 */
	if (!y.length)
	    return x;
	if (!x.length)
	    return y;
    }

    //printf("_d_arraycatT(%d,%p ~ %d,%p)\n", x.length, x.ptr, y.length, y.ptr);
    auto sizeelem = ti.next.tsize();		// array element size
    //printf("_d_arraycatT(%d,%p ~ %d,%p sizeelem = %d)\n", x.length, x.ptr, y.length, y.ptr, sizeelem);
    size_t xlen = x.length * sizeelem;
    size_t ylen = y.length * sizeelem;
    size_t len = xlen + ylen;
    if (!len)
	return null;

    byte* p = cast(byte*)_gc.malloc(len + 1);
    if (!(ti.next.flags() & 1))
	_gc.hasNoPointers(p);
    memcpy(p, x.ptr, xlen);
    memcpy(p + xlen, y.ptr, ylen);
    p[len] = 0;

    return p[0 .. x.length + y.length];
}


extern (C)
byte[] _d_arraycatnT(TypeInfo ti, uint n, ...)
{   void* a;
    size_t length;
    byte[]* p;
    uint i;
    byte[] b;
    va_list va;
    auto sizeelem = ti.next.tsize();		// array element size

    va_start!(typeof(n))(va, n);

    for (i = 0; i < n; i++)
    {
	b = va_arg!(typeof(b))(va);
	length += b.length;
    }
    if (!length)
	return null;

    a = _gc.malloc(length * sizeelem);
    if (!(ti.next.flags() & 1))
	_gc.hasNoPointers(a);
    va_start!(typeof(n))(va, n);

    uint j = 0;
    for (i = 0; i < n; i++)
    {
	b = va_arg!(typeof(b))(va);
	if (b.length)
	{
	    memcpy(a + j, b.ptr, b.length * sizeelem);
	    j += b.length * sizeelem;
	}
    }

    return (cast(byte*)a)[0..length];
}

version (GNU) { } else
extern (C)
void* _d_arrayliteralT(TypeInfo ti, size_t length, ...)
{
    auto sizeelem = ti.next.tsize();		// array element size
    void* result;

    //printf("_d_arrayliteralT(sizeelem = %d, length = %d)\n", sizeelem, length);
    if (length == 0 || sizeelem == 0)
	result = null;
    else
    {
	result = _gc.malloc(length * sizeelem);
	if (!(ti.next.flags() & 1))
	{
	    _gc.hasNoPointers(result);
	}

	va_list q;
	va_start!(size_t)(q, length);

	size_t stacksize = (sizeelem + int.sizeof - 1) & ~(int.sizeof - 1);

	if (stacksize == sizeelem)
	{
	    memcpy(result, q, length * sizeelem);
	}
	else
	{
	    for (size_t i = 0; i < length; i++)
	    {
		memcpy(result + i * sizeelem, q, sizeelem);
		q += stacksize;
	    }
	}

	va_end(q);
    }
    return result;
}

/**********************************
 * Support for array.dup property.
 */

/*struct Array2
{
    size_t length;
    void* ptr;
}*/

extern(C) void* _d_allocmemoryT(size_t foo) { return malloc(foo).ptr; }
