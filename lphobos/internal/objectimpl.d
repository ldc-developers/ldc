
/**
 * Part of the D programming language runtime library.
 * Forms the symbols available to all D programs. Includes
 * Object, which is the root of the class object hierarchy.
 *
 * This module is implicitly imported.
 * Macros:
 *  WIKI = Phobos/Object
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
 *  freely, in both source and binary form, subject to the following
 *  restrictions:
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


module object;

//import std.outofmemory;

/**
 * An unsigned integral type large enough to span the memory space. Use for
 * array indices and pointer offsets for maximal portability to
 * architectures that have different memory address ranges. This is
 * analogous to C's size_t.
 */
alias typeof(int.sizeof) size_t;

/**
 * A signed integral type large enough to span the memory space. Use for
 * pointer differences and for size_t differences for maximal portability to
 * architectures that have different memory address ranges. This is
 * analogous to C's ptrdiff_t.
 */
alias typeof(cast(void*)0 - cast(void*)0) ptrdiff_t;

alias size_t hash_t;

extern (C)
{   /// C's printf function.
    int printf(char *, ...);
    void trace_term();

    int memcmp(void *, void *, size_t);
    void* memcpy(void *, void *, size_t);
    void* calloc(size_t, size_t);
    void* realloc(void*, size_t);
    void free(void*);

    //Object _d_newclass(ClassInfo ci);
}

/// Standard boolean type.
alias bool bit;

alias char[] string;
alias wchar[] wstring;
alias dchar[] dstring;

/+

/* *************************
 * Internal struct pointed to by the hidden .monitor member.
 */
struct Monitor
{
    void delegate(Object)[] delegates;

    /* More stuff goes here defined by internal/monitor.c */
}

+/

/******************
 * All D class objects inherit from Object.
 */
class Object
{
    void print()
    {
    char[] s = toString();
    printf("%.*s\n", s.length, s.ptr);
    }

    /**
     * Convert Object to a human readable string.
     */
    char[] toString()
    {
    //return this.classinfo.name;
    return "object.Object (no classinfo yet)";
    }

    /**
     * Compute hash function for Object.
     */
    hash_t toHash()
    {
    // BUG: this prevents a compacting GC from working, needs to be fixed
    return cast(hash_t)cast(void *)this;
    }

    /**
     * Compare with another Object obj.
     * Returns:
     *  $(TABLE
     *  $(TR $(TD this &lt; obj) $(TD &lt; 0))
     *  $(TR $(TD this == obj) $(TD 0))
     *  $(TR $(TD this &gt; obj) $(TD &gt; 0))
     *  )
     */
    int opCmp(Object o)
    {
    // BUG: this prevents a compacting GC from working, needs to be fixed
    //return cast(int)cast(void *)this - cast(int)cast(void *)o;

    //throw new Error("need opCmp for class " ~ this.classinfo.name);
    throw new Error("need opCmp for class unknown object.Object (no classinfo yet)");
    }

    /**
     * Returns !=0 if this object does have the same contents as obj.
     */
    int opEquals(Object o)
    {
    return cast(int)(this is o);
    }

    /* **
     * Call delegate dg, passing this to it, when this object gets destroyed.
     * Use extreme caution, as the list of delegates is stored in a place
     * not known to the gc. Thus, if any objects pointed to by one of these
     * delegates gets freed by the gc, calling the delegate will cause a
     * crash.
     * This is only for use by library developers, as it will need to be
     * redone if weak pointers are added or a moving gc is developed.
     */
    final void notifyRegister(void delegate(Object) dg)
    {
    /+
    //printf("notifyRegister(dg = %llx, o = %p)\n", dg, this);
    synchronized (this)
    {
        Monitor* m = cast(Monitor*)(cast(void**)this)[1];
        foreach (inout x; m.delegates)
        {
        if (!x || x == dg)
        {   x = dg;
            return;
        }
        }

        // Increase size of delegates[]
        auto len = m.delegates.length;
        auto startlen = len;
        if (len == 0)
        {
        len = 4;
        auto p = calloc((void delegate(Object)).sizeof, len);
        if (!p)
            _d_OutOfMemory();
        m.delegates = (cast(void delegate(Object)*)p)[0 .. len];
        }
        else
        {
        len += len + 4;
        auto p = realloc(m.delegates.ptr, (void delegate(Object)).sizeof * len);
        if (!p)
            _d_OutOfMemory();
        m.delegates = (cast(void delegate(Object)*)p)[0 .. len];
        m.delegates[startlen .. len] = null;
        }
        m.delegates[startlen] = dg;
    }
    +/
    }

    /* **
     * Remove delegate dg from the notify list.
     * This is only for use by library developers, as it will need to be
     * redone if weak pointers are added or a moving gc is developed.
     */
    final void notifyUnRegister(void delegate(Object) dg)
    {
    /+
    synchronized (this)
    {
        Monitor* m = cast(Monitor*)(cast(void**)this)[1];
        foreach (inout x; m.delegates)
        {
        if (x == dg)
            x = null;
        }
    }
    +/
    }

    /******
     * Create instance of class specified by classname.
     * The class must either have no constructors or have
     * a default constructor.
     * Returns:
     *  null if failed
     */
    static Object factory(char[] classname)
    {
    /+
    auto ci = ClassInfo.find(classname);
    if (ci)
    {
        return ci.create();
    }
    +/
    return null;
    }
}

/+
extern (C) void _d_notify_release(Object o)
{
    //printf("_d_notify_release(o = %p)\n", o);
    Monitor* m = cast(Monitor*)(cast(void**)o)[1];
    if (m.delegates.length)
    {
    auto dgs = m.delegates;
    synchronized (o)
    {
        dgs = m.delegates;
        m.delegates = null;
    }

    foreach (dg; dgs)
    {
        if (dg)
        {   //printf("calling dg = %llx (%p)\n", dg, o);
        dg(o);
        }
    }

    free(dgs.ptr);
    }
}

/**
 * Information about an interface.
 * A pointer to this appears as the first entry in the interface's vtbl[].
 */
struct Interface
{
    ClassInfo classinfo;    /// .classinfo for this interface (not for containing class)
    void *[] vtbl;
    int offset;         /// offset to Interface 'this' from Object 'this'
}

import std.moduleinit;
/**
 * Runtime type information about a class. Can be retrieved for any class type
 * or instance by using the .classinfo property.
 * A pointer to this appears as the first entry in the class's vtbl[].
 */
class ClassInfo : Object
{
    byte[] init;        /** class static initializer
                 * (init.length gives size in bytes of class)
                 */
    char[] name;        /// class name
    void *[] vtbl;      /// virtual function pointer table
    Interface[] interfaces; /// interfaces this class implements
    ClassInfo base;     /// base class
    void *destructor;
    void (*classInvariant)(Object);
    uint flags;
    //  1:          // IUnknown
    //  2:          // has no possible pointers into GC memory
    //  4:          // has offTi[] member
    //  8:          // has constructors
    void *deallocator;
    OffsetTypeInfo[] offTi;
    void function(Object) defaultConstructor;   // default Constructor

    /*************
     * Search all modules for ClassInfo corresponding to classname.
     * Returns: null if not found
     */
    static ClassInfo find(char[] classname)
    {
    foreach (m; ModuleInfo.modules())
    {
        //writefln("module %s, %d", m.name, m.localClasses.length);
        foreach (c; m.localClasses)
        {
        //writefln("\tclass %s", c.name);
        if (c.name == classname)
            return c;
        }
    }
    return null;
    }

    /********************
     * Create instance of Object represented by 'this'.
     */
    Object create()
    {
    if (flags & 8 && !defaultConstructor)
        return null;
    Object o = _d_newclass(this);
    if (flags & 8 && defaultConstructor)
    {
        defaultConstructor(o);
    }
    return o;
    }
}

private import std.string;

+/

/**
 * Array of pairs giving the offset and type information for each
 * member in an aggregate.
 */
struct OffsetTypeInfo
{
    size_t offset;  /// Offset of member from start of object
    TypeInfo ti;    /// TypeInfo for this member
}

private int string_cmp(char[] s1, char[] s2)
{
    auto len = s1.length;
    if (s2.length < len)
        len = s2.length;
    int result = memcmp(s1.ptr, s2.ptr, len);
    if (result == 0)
        result = cast(int)(cast(ptrdiff_t)s1.length - cast(ptrdiff_t)s2.length);
    return result;
}

/**
 * Runtime type information about a type.
 * Can be retrieved for any type using a
 * <a href="../expression.html#typeidexpression">TypeidExpression</a>.
 */
class TypeInfo
{
    hash_t toHash()
    {   hash_t hash;

    foreach (char c; this.toString())
        hash = hash * 9 + c;
    return hash;
    }

    int opCmp(Object o)
    {
    if (this is o)
        return 0;
    TypeInfo ti = cast(TypeInfo)o;
    if (ti is null)
        return 1;
    return string_cmp(this.toString(), ti.toString());
    }

    int opEquals(Object o)
    {
    /* TypeInfo instances are singletons, but duplicates can exist
     * across DLL's. Therefore, comparing for a name match is
     * sufficient.
     */
    if (this is o)
        return 1;
    TypeInfo ti = cast(TypeInfo)o;
    return cast(int)(ti && this.toString() == ti.toString());
    }

    /// Returns a hash of the instance of a type.
    hash_t getHash(void *p) { return cast(uint)p; }

    /// Compares two instances for equality.
    int equals(void *p1, void *p2) { return cast(int)(p1 == p2); }

    /// Compares two instances for &lt;, ==, or &gt;.
    int compare(void *p1, void *p2) { return 0; }

    /// Returns size of the type.
    size_t tsize() { return 0; }

    /// Swaps two instances of the type.
    void swap(void *p1, void *p2)
    {
    size_t n = tsize();
    for (size_t i = 0; i < n; i++)
    {   byte t;

        t = (cast(byte *)p1)[i];
        (cast(byte *)p1)[i] = (cast(byte *)p2)[i];
        (cast(byte *)p2)[i] = t;
    }
    }

    /// Get TypeInfo for 'next' type, as defined by what kind of type this is,
    /// null if none.
    TypeInfo next() { return null; }

    /// Return default initializer, null if default initialize to 0
    void[] init() { return null; }

    /// Get flags for type: 1 means GC should scan for pointers
    uint flags() { return 0; }

    /// Get type information on the contents of the type; null if not available
    OffsetTypeInfo[] offTi() { return null; }
}

class TypeInfo_Typedef : TypeInfo
{
    char[] toString() { return name; }

    int opEquals(Object o)
    {   TypeInfo_Typedef c;

    return cast(int)
        (this is o ||
        ((c = cast(TypeInfo_Typedef)o) !is null &&
         this.name == c.name &&
         this.base == c.base));
    }

    hash_t getHash(void *p) { return base.getHash(p); }
    int equals(void *p1, void *p2) { return base.equals(p1, p2); }
    int compare(void *p1, void *p2) { return base.compare(p1, p2); }
    size_t tsize() { return base.tsize(); }
    void swap(void *p1, void *p2) { return base.swap(p1, p2); }

    TypeInfo next() { return base; }
    uint flags() { return base.flags(); }
    void[] init() { return m_init.length ? m_init : base.init(); }

    TypeInfo base;
    char[] name;
    void[] m_init;
}

class TypeInfo_Enum : TypeInfo_Typedef
{
}

class TypeInfo_Pointer : TypeInfo
{
    char[] toString() { return m_next.toString() ~ "*"; }

    int opEquals(Object o)
    {   TypeInfo_Pointer c;

    return this is o ||
        ((c = cast(TypeInfo_Pointer)o) !is null &&
         this.m_next == c.m_next);
    }

    hash_t getHash(void *p)
    {
        return cast(uint)*cast(void* *)p;
    }

    int equals(void *p1, void *p2)
    {
        return cast(int)(*cast(void* *)p1 == *cast(void* *)p2);
    }

    int compare(void *p1, void *p2)
    {
    if (*cast(void* *)p1 < *cast(void* *)p2)
        return -1;
    else if (*cast(void* *)p1 > *cast(void* *)p2)
        return 1;
    else
        return 0;
    }

    size_t tsize()
    {
    return (void*).sizeof;
    }

    void swap(void *p1, void *p2)
    {   void* tmp;
    tmp = *cast(void**)p1;
    *cast(void**)p1 = *cast(void**)p2;
    *cast(void**)p2 = tmp;
    }

    TypeInfo next() { return m_next; }
    uint flags() { return 1; }

    TypeInfo m_next;
}

class TypeInfo_Array : TypeInfo
{
    char[] toString() { return value.toString() ~ "[]"; }

    int opEquals(Object o)
    {   TypeInfo_Array c;

    return cast(int)
           (this is o ||
        ((c = cast(TypeInfo_Array)o) !is null &&
         this.value == c.value));
    }

    hash_t getHash(void *p)
    {   size_t sz = value.tsize();
    hash_t hash = 0;
    void[] a = *cast(void[]*)p;
    for (size_t i = 0; i < a.length; i++)
        hash += value.getHash(a.ptr + i * sz);
        return hash;
    }

    int equals(void *p1, void *p2)
    {
    void[] a1 = *cast(void[]*)p1;
    void[] a2 = *cast(void[]*)p2;
    if (a1.length != a2.length)
        return 0;
    size_t sz = value.tsize();
    for (size_t i = 0; i < a1.length; i++)
    {
        if (!value.equals(a1.ptr + i * sz, a2.ptr + i * sz))
        return 0;
    }
        return 1;
    }

    int compare(void *p1, void *p2)
    {
    void[] a1 = *cast(void[]*)p1;
    void[] a2 = *cast(void[]*)p2;
    size_t sz = value.tsize();
    size_t len = a1.length;

        if (a2.length < len)
            len = a2.length;
        for (size_t u = 0; u < len; u++)
        {
            int result = value.compare(a1.ptr + u * sz, a2.ptr + u * sz);
            if (result)
                return result;
        }
        return cast(int)a1.length - cast(int)a2.length;
    }

    size_t tsize()
    {
    return (void[]).sizeof;
    }

    void swap(void *p1, void *p2)
    {   void[] tmp;
    tmp = *cast(void[]*)p1;
    *cast(void[]*)p1 = *cast(void[]*)p2;
    *cast(void[]*)p2 = tmp;
    }

    TypeInfo value;

    TypeInfo next()
    {
    return value;
    }

    uint flags() { return 1; }
}

private const char[10] digits    = "0123456789";            /// 0..9

private char[] lengthToString(uint u)
{   char[uint.sizeof * 3] buffer = void;
    int ndigits;
    char[] result;

    ndigits = 0;
    if (u < 10)
    // Avoid storage allocation for simple stuff
    result = digits[u .. u + 1];
    else
    {
    while (u)
    {
        uint c = (u % 10) + '0';
        u /= 10;
        ndigits++;
        buffer[buffer.length - ndigits] = cast(char)c;
    }
    result = new char[ndigits];
    result[] = buffer[buffer.length - ndigits .. buffer.length];
    }
    return result;
}

private char[] lengthToString(ulong u)
{   char[ulong.sizeof * 3] buffer;
    int ndigits;
    char[] result;

    if (u < 0x1_0000_0000)
    return lengthToString(cast(uint)u);
    ndigits = 0;
    while (u)
    {
    char c = cast(char)((u % 10) + '0');
    u /= 10;
    ndigits++;
    buffer[buffer.length - ndigits] = c;
    }
    result = new char[ndigits];
    result[] = buffer[buffer.length - ndigits .. buffer.length];
    return result;
}

class TypeInfo_StaticArray : TypeInfo
{
    char[] toString()
    {
    return value.toString() ~ "[" ~ lengthToString(len) ~ "]";
    }

    int opEquals(Object o)
    {   TypeInfo_StaticArray c;

    return cast(int)
           (this is o ||
        ((c = cast(TypeInfo_StaticArray)o) !is null &&
         this.len == c.len &&
         this.value == c.value));
    }

    hash_t getHash(void *p)
    {   size_t sz = value.tsize();
    hash_t hash = 0;
    for (size_t i = 0; i < len; i++)
        hash += value.getHash(p + i * sz);
        return hash;
    }

    int equals(void *p1, void *p2)
    {
    size_t sz = value.tsize();

        for (size_t u = 0; u < len; u++)
        {
        if (!value.equals(p1 + u * sz, p2 + u * sz))
        return 0;
        }
        return 1;
    }

    int compare(void *p1, void *p2)
    {
    size_t sz = value.tsize();

        for (size_t u = 0; u < len; u++)
        {
            int result = value.compare(p1 + u * sz, p2 + u * sz);
            if (result)
                return result;
        }
        return 0;
    }

    size_t tsize()
    {
    return len * value.tsize();
    }

    void swap(void *p1, void *p2)
    {   void* tmp;
    size_t sz = value.tsize();
    ubyte[16] buffer;
    void* pbuffer;

    if (sz < buffer.sizeof)
        tmp = buffer.ptr;
    else {
        if (value.flags() & 1)
        tmp = pbuffer = (new void[sz]).ptr;
        else
        tmp = pbuffer = (new byte[sz]).ptr;
    }

    for (size_t u = 0; u < len; u += sz)
    {   size_t o = u * sz;
        memcpy(tmp, p1 + o, sz);
        memcpy(p1 + o, p2 + o, sz);
        memcpy(p2 + o, tmp, sz);
    }
    if (pbuffer)
        delete pbuffer;
    }

    void[] init() { return value.init(); }
    TypeInfo next() { return value; }
    uint flags() { return value.flags(); }

    TypeInfo value;
    size_t len;
}

class TypeInfo_AssociativeArray : TypeInfo
{
    char[] toString()
    {
    return value.toString() ~ "[" ~ key.toString() ~ "]";
    }

    int opEquals(Object o)
    {   TypeInfo_AssociativeArray c;

    return this is o ||
        ((c = cast(TypeInfo_AssociativeArray)o) !is null &&
         this.key == c.key &&
         this.value == c.value);
    }

    // BUG: need to add the rest of the functions

    size_t tsize()
    {
    return (char[int]).sizeof;
    }

    TypeInfo next() { return value; }
    uint flags() { return 1; }

    TypeInfo value;
    TypeInfo key;
}

class TypeInfo_Function : TypeInfo
{
    char[] toString()
    {
    return next.toString() ~ "()";
    }

    int opEquals(Object o)
    {   TypeInfo_Function c;

    return this is o ||
        ((c = cast(TypeInfo_Function)o) !is null &&
         this.next == c.next);
    }

    // BUG: need to add the rest of the functions

    size_t tsize()
    {
    return 0;   // no size for functions
    }

    TypeInfo next;
}

class TypeInfo_Delegate : TypeInfo
{
    char[] toString()
    {
    return next.toString() ~ " delegate()";
    }

    int opEquals(Object o)
    {   TypeInfo_Delegate c;

    return this is o ||
        ((c = cast(TypeInfo_Delegate)o) !is null &&
         this.next == c.next);
    }

    // BUG: need to add the rest of the functions

    size_t tsize()
    {   alias int delegate() dg;
    return dg.sizeof;
    }

    uint flags() { return 1; }

    TypeInfo next;
}

/+

class TypeInfo_Class : TypeInfo
{
    char[] toString() { return info.name; }

    int opEquals(Object o)
    {   TypeInfo_Class c;

    return this is o ||
        ((c = cast(TypeInfo_Class)o) !is null &&
         this.info.name == c.classinfo.name);
    }

    hash_t getHash(void *p)
    {
    Object o = *cast(Object*)p;
    assert(o);
    return o.toHash();
    }

    int equals(void *p1, void *p2)
    {
    Object o1 = *cast(Object*)p1;
    Object o2 = *cast(Object*)p2;

    return (o1 is o2) || (o1 && o1.opEquals(o2));
    }

    int compare(void *p1, void *p2)
    {
    Object o1 = *cast(Object*)p1;
    Object o2 = *cast(Object*)p2;
    int c = 0;

    // Regard null references as always being "less than"
    if (o1 !is o2)
    {
        if (o1)
        {   if (!o2)
            c = 1;
        else
            c = o1.opCmp(o2);
        }
        else
        c = -1;
    }
    return c;
    }

    size_t tsize()
    {
    return Object.sizeof;
    }

    uint flags() { return 1; }

    OffsetTypeInfo[] offTi()
    {
    return (info.flags & 4) ? info.offTi : null;
    }

    ClassInfo info;
}

class TypeInfo_Interface : TypeInfo
{
    char[] toString() { return info.name; }

    int opEquals(Object o)
    {   TypeInfo_Interface c;

    return this is o ||
        ((c = cast(TypeInfo_Interface)o) !is null &&
         this.info.name == c.classinfo.name);
    }

    hash_t getHash(void *p)
    {
    Interface* pi = **cast(Interface ***)*cast(void**)p;
    Object o = cast(Object)(*cast(void**)p - pi.offset);
    assert(o);
    return o.toHash();
    }

    int equals(void *p1, void *p2)
    {
    Interface* pi = **cast(Interface ***)*cast(void**)p1;
    Object o1 = cast(Object)(*cast(void**)p1 - pi.offset);
    pi = **cast(Interface ***)*cast(void**)p2;
    Object o2 = cast(Object)(*cast(void**)p2 - pi.offset);

    return o1 == o2 || (o1 && o1.opCmp(o2) == 0);
    }

    int compare(void *p1, void *p2)
    {
    Interface* pi = **cast(Interface ***)*cast(void**)p1;
    Object o1 = cast(Object)(*cast(void**)p1 - pi.offset);
    pi = **cast(Interface ***)*cast(void**)p2;
    Object o2 = cast(Object)(*cast(void**)p2 - pi.offset);
    int c = 0;

    // Regard null references as always being "less than"
    if (o1 != o2)
    {
        if (o1)
        {   if (!o2)
            c = 1;
        else
            c = o1.opCmp(o2);
        }
        else
        c = -1;
    }
    return c;
    }

    size_t tsize()
    {
    return Object.sizeof;
    }

    uint flags() { return 1; }

    ClassInfo info;
}

+/

class TypeInfo_Struct : TypeInfo
{
    char[] toString() { return name; }

    int opEquals(Object o)
    {   TypeInfo_Struct s;

    return this is o ||
        ((s = cast(TypeInfo_Struct)o) !is null &&
         this.name == s.name &&
         this.init.length == s.init.length);
    }

    hash_t getHash(void *p)
    {   hash_t h;

    assert(p);
    if (xtoHash)
    {   //printf("getHash() using xtoHash\n");
        h = (*xtoHash)(p);
    }
    else
    {
        //printf("getHash() using default hash\n");
        // A sorry hash algorithm.
        // Should use the one for strings.
        // BUG: relies on the GC not moving objects
        for (size_t i = 0; i < init.length; i++)
        {   h = h * 9 + *cast(ubyte*)p;
        p++;
        }
    }
    return h;
    }

    int equals(void *p2, void *p1)
    {   int c;

    if (p1 == p2)
        c = 1;
    else if (!p1 || !p2)
        c = 0;
    else if (xopEquals)
        c = (*xopEquals)(p1, p2);
    else
        // BUG: relies on the GC not moving objects
        c = (memcmp(p1, p2, init.length) == 0);
    return c;
    }

    int compare(void *p1, void *p2)
    {
    int c = 0;

    // Regard null references as always being "less than"
    if (p1 != p2)
    {
        if (p1)
        {   if (!p2)
            c = 1;
        else if (xopCmp)
            c = (*xopCmp)(p1, p2);
        else
            // BUG: relies on the GC not moving objects
            c = memcmp(p1, p2, init.length);
        }
        else
        c = -1;
    }
    return c;
    }

    size_t tsize()
    {
    return init.length;
    }

    void[] init() { return m_init; }

    uint flags() { return m_flags; }

    char[] name;
    void[] m_init;  // initializer; init.ptr == null if 0 initialize

    hash_t function(void*) xtoHash;
    int function(void*,void*) xopEquals;
    int function(void*,void*) xopCmp;
    char[] function(void*) xtoString;

    uint m_flags;
}

/+

class TypeInfo_Tuple : TypeInfo
{
    TypeInfo[] elements;

    char[] toString()
    {
    char[] s;
    s = "(";
    foreach (i, element; elements)
    {
        if (i)
        s ~= ',';
        s ~= element.toString();
    }
    s ~= ")";
        return s;
    }

    int opEquals(Object o)
    {
    if (this is o)
        return 1;

    auto t = cast(TypeInfo_Tuple)o;
    if (t && elements.length == t.elements.length)
    {
        for (size_t i = 0; i < elements.length; i++)
        {
        if (elements[i] != t.elements[i])
            return 0;
        }
        return 1;
    }
    return 0;
    }

    hash_t getHash(void *p)
    {
        assert(0);
    }

    int equals(void *p1, void *p2)
    {
        assert(0);
    }

    int compare(void *p1, void *p2)
    {
        assert(0);
    }

    size_t tsize()
    {
        assert(0);
    }

    void swap(void *p1, void *p2)
    {
        assert(0);
    }
}

+/

class TypeInfo_Const : TypeInfo
{
    char[] toString() { return "const " ~ base.toString(); }

    int opEquals(Object o) { return base.opEquals(o); }
    hash_t getHash(void *p) { return base.getHash(p); }
    int equals(void *p1, void *p2) { return base.equals(p1, p2); }
    int compare(void *p1, void *p2) { return base.compare(p1, p2); }
    size_t tsize() { return base.tsize(); }
    void swap(void *p1, void *p2) { return base.swap(p1, p2); }

    TypeInfo next() { return base.next(); }
    uint flags() { return base.flags(); }
    void[] init() { return base.init(); }

    TypeInfo base;
}

class TypeInfo_Invariant : TypeInfo_Const
{
    char[] toString() { return "invariant " ~ base.toString(); }
}

/**
 * All recoverable exceptions should be derived from class Exception.
 */
class Exception : Object
{
    char[] msg;

    /**
     * Constructor; msg is a descriptive message for the exception.
     */
    this(char[] msg)
    {
    this.msg = msg;
    }

    char[] toString() { return msg; }
}

/**
 * All irrecoverable exceptions should be derived from class Error.
 */
class Error : Exception
{
    Error next;

    /**
     * Constructor; msg is a descriptive message for the exception.
     */
    this(char[] msg)
    {
    super(msg);
    }

    this(char[] msg, Error next)
    {
    super(msg);
    this.next = next;
    }
}

//extern (C) int nullext = 0;

