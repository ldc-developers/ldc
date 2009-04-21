/**
 * Part of the D programming language runtime library.
 * Forms the symbols available to all D programs. Includes
 * Object, which is the root of the class object hierarchy.
 *
 * This module is implicitly imported.
 * Macros:
 *      WIKI = Object
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

/*
 *  Modified by Sean Kelly <sean@f4.ca> for use with Tango.
 *  Modified by Tomas Lindquist Olsen <tomas@famolsen.dk> for use with LDC.
 */

module object;

//debug=PRINTF

private
{
    import tango.stdc.string; // : memcmp, memcpy, memmove;
    import tango.stdc.stdlib; // : calloc, realloc, free;
    import util.string;
    import tango.stdc.stdio;  // : printf, snprintf;
    import tango.core.Version;

    extern (C) void onOutOfMemoryError();
    extern (C) Object _d_allocclass(ClassInfo ci);
}

// NOTE: For some reason, this declaration method doesn't work
//       in this particular file (and this file only).  It must
//       be a DMD thing.
//alias typeof(int.sizeof)                    size_t;
//alias typeof(cast(void*)0 - cast(void*)0)   ptrdiff_t;

version( LLVM64 )
{
    alias ulong size_t;
    alias long  ptrdiff_t;
}
else
{
    alias uint  size_t;
    alias int   ptrdiff_t;
}

alias size_t hash_t;

/**
 * All D class objects inherit from Object.
 */
class Object
{
    /**
     * Convert Object to a human readable string.
     */
    char[] toString()
    {
        return this.classinfo.name;
    }

    /**
     * Compute hash function for Object.
     */
    hash_t toHash()
    {
        // BUG: this prevents a compacting GC from working, needs to be fixed
        return cast(hash_t)cast(void*)this;
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
        //return cast(int)cast(void*)this - cast(int)cast(void*)o;

        //throw new Exception("need opCmp for class " ~ this.classinfo.name);
        return this !is o;
    }

    /**
     * Returns !=0 if this object does have the same contents as obj.
     */
    int opEquals(Object o)
    {
        return cast(int)(this is o);
    }

    interface Monitor
    {
        void lock();
        void unlock();
    }
}

/**
 * Information about an interface.
 * When an object is accessed via an interface, an Interface* appears as the
 * first entry in its vtbl.
 */
struct Interface
{
    ClassInfo   classinfo;  /// .classinfo for this interface (not for containing class)
    void*[]     vtbl;
    ptrdiff_t   offset;     /// offset to Interface 'this' from Object 'this'
}

/**
 * Runtime type information about a class. Can be retrieved for any class type
 * or instance by using the .classinfo property.
 * A pointer to this appears as the first entry in the class's vtbl[].
 */
class ClassInfo : Object
{
    byte[]      init;           /** class static initializer
                                 * (init.length gives size in bytes of class)
                                 */
    char[]      name;           /// class name
    void*[]     vtbl;           /// virtual function pointer table
    Interface[] interfaces;     /// interfaces this class implements
    ClassInfo   base;           /// base class
    void*       destructor;
    void function(Object) classInvariant;
    uint        flags;
    //  1:                      // IUnknown
    //  2:                      // has no possible pointers into GC memory
    //  4:                      // has offTi[] member
    //  8:                      // has constructors
    void*       deallocator;
    OffsetTypeInfo[] offTi;
    void* defaultConstructor;   // default Constructor

    /**
     * Search all modules for ClassInfo corresponding to classname.
     * Returns: null if not found
     */
    static ClassInfo find(char[] classname)
    {
        foreach (m; ModuleInfo)
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

    /**
     * Create instance of Object represented by 'this'.
     */
    Object create()
    {
        if (flags & 8 && !defaultConstructor)
            return null;

        Object o = _d_allocclass(this);
        // initialize it
        (cast(byte*) o)[0 .. init.length] = init[];

        if (flags & 8 && defaultConstructor)
        {
            auto ctor = cast(Object function(Object))defaultConstructor;
            return ctor(o);
        }
        return o;
    }
}

/**
 * Array of pairs giving the offset and type information for each
 * member in an aggregate.
 */
struct OffsetTypeInfo
{
    size_t   offset;    /// Offset of member from start of object
    TypeInfo ti;        /// TypeInfo for this member
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
        return stringCompare(this.toString(), ti.toString());
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
    hash_t getHash(void *p) { return cast(hash_t)p; }

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
        return cast(hash_t)*cast(void**)p;
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

class TypeInfo_StaticArray : TypeInfo
{
    char[] toString()
    {
        char [10] tmp = void;
        return value.toString() ~ "[" ~ intToUtf8(tmp, len) ~ "]";
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
        else
            tmp = pbuffer = (new void[sz]).ptr;

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
        return next.toString() ~ "[" ~ key.toString() ~ "]";
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
        return 0;       // no size for functions
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
        return o ? o.toHash() : 0;
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
        {   debug(PRINTF) printf("getHash() using xtoHash\n");
            h = (*xtoHash)(p);
        }
        else
        {
            debug(PRINTF) printf("getHash() using default hash\n");
            // A sorry hash algorithm.
            // Should use the one for strings.
            // BUG: relies on the GC not moving objects
            for (size_t i = 0; i < m_init.length; i++)
            {   h = h * 9 + *cast(ubyte*)p;
                p++;
            }
        }
        return h;
    }

    int equals(void *p1, void *p2)
    {   int c;

        if (p1 == p2)
            c = 1;
        else if (!p1 || !p2)
            c = 0;
        else if (xopEquals)
            c = (*xopEquals)(p1, p2);
        else
            // BUG: relies on the GC not moving objects
            c = (memcmp(p1, p2, m_init.length) == 0);
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
                    // the x86 D calling conv requires the this arg to be last here
                    version(X86)
                        c = (*xopCmp)(p2, p1);
                    else
                        c = (*xopCmp)(p1, p2);
                else
                    // BUG: relies on the GC not moving objects
                    c = memcmp(p1, p2, m_init.length);
            }
            else
                c = -1;
        }
        return c;
    }

    size_t tsize()
    {
        return m_init.length;
    }

    void[] init() { return m_init; }

    uint flags() { return m_flags; }

    char[] name;
    void[] m_init;      // initializer; never null

    hash_t function(void*)    xtoHash;
    int function(void*,void*) xopEquals;
    int function(void*,void*) xopCmp;
    char[] function(void*)    xtoString;

    uint m_flags;
}

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


////////////////////////////////////////////////////////////////////////////////
// Exception
////////////////////////////////////////////////////////////////////////////////


class Exception : Object
{
    static if (Tango.Minor > 998) {
        struct FrameInfo{
            long line;
            size_t iframe;
            ptrdiff_t offsetSymb;
            size_t baseSymb;
            ptrdiff_t offsetImg;
            size_t baseImg;
            size_t address;
            char[] file;
            char[] func;
            char[] extra;
            bool exactAddress;
            bool internalFunction;
            void writeOut(void delegate(char[])sink){
                char[25] buf;
                if (func.length) {
                    sink(func);
                } else {
                    sink("???");
                }
                auto len=sprintf(buf.ptr,"@%zx",baseSymb);
                sink(buf[0..len]);
                len=sprintf(buf.ptr,"%+td ",offsetSymb);
                sink(buf[0..len]);
                if (extra.length){
                    sink(extra);
                    sink(" ");
                }
                sink(file);
                len=sprintf(buf.ptr,":%ld ",line);
                sink(buf[0..len]);
                len=sprintf(buf.ptr,"%zx",baseImg);
                sink(buf[0..len]);
                len=sprintf(buf.ptr,"%+td ",offsetImg);
                sink(buf[0..len]);
                len=sprintf(buf.ptr,"[%zx]",address);
                sink(buf[0..len]);
            }
            void clear(){
                line=0;
                iframe=-1;
                offsetImg=0;
                baseImg=0;
                offsetSymb=0;
                baseSymb=0;
                address=0;
                exactAddress=true;
                internalFunction=false;
                file=null;
                func=null;
                extra=null;
            }
        }
        interface TraceInfo
        {
            int opApply( int delegate( ref FrameInfo fInfo ) );
            void writeOut(void delegate(char[])sink);
        }
    } else static if (Tango.Minor == 998) {
        struct FrameInfo{
            long line;
            ptrdiff_t iframe;
            ptrdiff_t offset;
            size_t address;
            char[] file;
            char[] func;
            char[256] charBuf;
            void writeOut(void delegate(char[])sink){
                char[25] buf;
                sink(func);
                auto len = snprintf(buf.ptr,buf.length,"@%zx ",address);
                sink(buf[0..len]);
                len = snprintf(buf.ptr,buf.length," %+td ",address);
                sink(buf[0..len]);
                if (file.length != 0 || line) {
                    sink(file);
                    len = snprintf(buf.ptr,buf.length,":%ld",line);
                    sink(buf[0..len]);
                }
            }
        }
        interface TraceInfo
        {
            int opApply( int delegate( ref FrameInfo fInfo ) );
        }
    } else {
        static assert(0, "Don't know FrameInfo, TraceInfo definition for Tango < 0.99.8");
    }

    char[]      msg;
    char[]      file;
    size_t      line;
    TraceInfo   info;
    Exception   next;

    this( char[] msg, char[] file, long line, Exception next, TraceInfo info )
    {
        // main constructor, breakpoint this if you want...
        this.msg = msg;
        this.next = next;
        this.file = file;
        this.line = cast(size_t)line;
        this.info = info;
    }

    this( char[] msg, Exception next=null )
    {
        this(msg,null,0,next,rt_createTraceContext(null));
    }

    this( char[] msg, char[] file, long line, Exception next=null )
    {
        this(msg,file,line,next,rt_createTraceContext(null));
    }

    char[] toString()
    {
        return msg;
    }
    
    void writeOut(void delegate(char[])sink){
        if (file.length != 0 || line)
        {
            char[25]buf;
            sink(this.classinfo.name);
            sink("@");
            sink(file);
            sink("(");
            auto len = snprintf(buf.ptr,buf.length,"%ld",line);
            sink(buf[0..len]);
            sink("): ");
            sink(toString());
            sink("\n");
        }
        else
        {
           sink(this.classinfo.name);
           sink(": ");
           sink(toString);
           sink("\n");
        }
        if (info)
        {
            sink("----------------\n");
            foreach (ref t; info){
                t.writeOut(sink);
                sink("\n");
            }
        }
        if (next){
            sink("\n");
            next.writeOut(sink);
        }
    }
}


alias Exception.TraceInfo function( void* ptr = null ) TraceHandler;
private TraceHandler traceHandler = null;


/**
 * Overrides the default trace hander with a user-supplied version.
 *
 * Params:
 *  h = The new trace handler.  Set to null to use the default handler.
 */
extern (C) void  rt_setTraceHandler( TraceHandler h )
{
    traceHandler = h;
}


/**
 * This function will be called when an Exception is constructed.  The
 * user-supplied trace handler will be called if one has been supplied,
 * otherwise no trace will be generated.
 *
 * Params:
 *  ptr = A pointer to the location from which to generate the trace, or null
 *        if the trace should be generated from within the trace handler
 *        itself.
 *
 * Returns:
 *  An object describing the current calling context or null if no handler is
 *  supplied.
 */
extern(C) Exception.TraceInfo rt_createTraceContext( void* ptr )
{
    if( traceHandler is null )
        return null;
    return traceHandler( ptr );
}


////////////////////////////////////////////////////////////////////////////////
// ModuleInfo
////////////////////////////////////////////////////////////////////////////////


enum
{
    MIctorstart  = 1,   // we've started constructing it
    MIctordone   = 2,   // finished construction
    MIstandalone = 4,   // module ctor does not depend on other module
                        // ctors being done first
    MIhasictor   = 8,   // has ictor member
}


class ModuleInfo
{
    char[]          name;
    ModuleInfo[]    importedModules;
    ClassInfo[]     localClasses;
    uint            flags;

    void function() ctor;       // module static constructor (order dependent)
    void function() dtor;       // module static destructor
    void function() unitTest;   // module unit tests

    void* xgetMembers;          // module getMembers() function

    void function() ictor;      // module static constructor (order independent)

    static int opApply( int delegate( inout ModuleInfo ) dg )
    {
        int ret = 0;

        foreach( m; _moduleinfo_array )
        {
            ret = dg( m );
            if( ret )
                break;
        }
        return ret;
    }
}


// this gets initialized in _moduleCtor()
extern (C) ModuleInfo[] _moduleinfo_array;

// This linked list is created by a compiler generated function inserted
// into the .ctor list by the compiler.
struct ModuleReference
{
    ModuleReference* next;
    ModuleInfo       mod;
}
extern (C) ModuleReference* _Dmodule_ref;   // start of linked list

// this list is built from the linked list above
ModuleInfo[] _moduleinfo_dtors;
uint         _moduleinfo_dtors_i;

/**
 * Initialize the modules.
 */

extern (C) void _moduleCtor()
{
    debug(PRINTF) printf("_moduleCtor()\n");

    int len = 0;
    ModuleReference *mr;

    for (mr = _Dmodule_ref; mr; mr = mr.next)
        len++;
    _moduleinfo_array = new ModuleInfo[len];
    len = 0;
    for (mr = _Dmodule_ref; mr; mr = mr.next)
    {   _moduleinfo_array[len] = mr.mod;
        len++;
    }

    _moduleinfo_dtors = new ModuleInfo[_moduleinfo_array.length];
    debug(PRINTF) printf("_moduleinfo_dtors = x%x\n", cast(void *)_moduleinfo_dtors);
    _moduleIndependentCtors();
    _moduleCtor2(null, _moduleinfo_array, 0);
}

extern (C) void _moduleIndependentCtors()
{
    debug(PRINTF) printf("_moduleIndependentCtors()\n");
    foreach (m; _moduleinfo_array)
    {
        if (m && m.flags & MIhasictor && m.ictor)
        {
            (*m.ictor)();
        }
    }
    debug(PRINTF) printf("_moduleIndependentCtors() DONE\n");
}

void _moduleCtor2(ModuleInfo from, ModuleInfo[] mi, int skip)
{
    debug(PRINTF) printf("_moduleCtor2(): %d modules\n", mi.length);
    for (uint i = 0; i < mi.length; i++)
    {
        ModuleInfo m = mi[i];

        debug(PRINTF) printf("\tmodule[%d] = '%p'\n", i, m);
        if (!m)
            continue;
        debug(PRINTF) printf("\tmodule[%d] = '%.*s'\n", i, m.name.length, m.name.ptr);
        if (m.flags & MIctordone)
            continue;
        debug(PRINTF) printf("\tmodule[%d] = '%.*s', m = x%x\n", i, m.name.length, m.name.ptr, m);

        if (m.ctor || m.dtor)
        {
            if (m.flags & MIctorstart)
            {   if (skip || m.flags & MIstandalone)
                    continue;
                assert(from !is null);
                throw new Exception( "Cyclic dependency in module " ~ from.name ~ " for import " ~ m.name);
            }

            m.flags |= MIctorstart;
            _moduleCtor2(m, m.importedModules, 0);
            if (m.ctor)
                (*m.ctor)();
            m.flags &= ~MIctorstart;
            m.flags |= MIctordone;

            // Now that construction is done, register the destructor
            //printf("\tadding module dtor x%x\n", m);
            assert(_moduleinfo_dtors_i < _moduleinfo_dtors.length);
            _moduleinfo_dtors[_moduleinfo_dtors_i++] = m;
        }
        else
        {
            m.flags |= MIctordone;
            _moduleCtor2(m, m.importedModules, 1);
        }
    }
    debug(PRINTF) printf("_moduleCtor2() DONE\n");
}

/**
 * Destruct the modules.
 */

// Starting the name with "_STD" means under linux a pointer to the
// function gets put in the .dtors segment.

extern (C) void _moduleDtor()
{
    debug(PRINTF) printf("_moduleDtor(): %d modules\n", _moduleinfo_dtors_i);

    for (uint i = _moduleinfo_dtors_i; i-- != 0;)
    {
        ModuleInfo m = _moduleinfo_dtors[i];

        debug(PRINTF) printf("\tmodule[%d] = '%.*s', x%x\n", i, m.name, m);
        if (m.dtor)
        {
            (*m.dtor)();
        }
    }
    debug(PRINTF) printf("_moduleDtor() done\n");
}

////////////////////////////////////////////////////////////////////////////////
// Monitor
////////////////////////////////////////////////////////////////////////////////

alias Object.Monitor        IMonitor;
alias void delegate(Object) DEvent;

// NOTE: The dtor callback feature is only supported for monitors that are not
//       supplied by the user.  The assumption is that any object with a user-
//       supplied monitor may have special storage or lifetime requirements and
//       that as a result, storing references to local objects within Monitor
//       may not be safe or desirable.  Thus, devt is only valid if impl is
//       null.
struct Monitor
{
    IMonitor impl;
    /* internal */
    DEvent[] devt;
    /* stuff */
}

Monitor* getMonitor(Object h)
{
    return cast(Monitor*) (cast(void**) h)[1];
}

void setMonitor(Object h, Monitor* m)
{
    (cast(void**) h)[1] = m;
}

extern (C) void _d_monitor_create(Object);
extern (C) void _d_monitor_destroy(Object);
extern (C) void _d_monitor_lock(Object);
extern (C) int  _d_monitor_unlock(Object);

extern (C) void _d_monitordelete(Object h, bool det)
{
    Monitor* m = getMonitor(h);

    if (m !is null)
    {
        IMonitor i = m.impl;
        if (i is null)
        {
            _d_monitor_devt(m, h);
            _d_monitor_destroy(h);
            setMonitor(h, null);
            return;
        }
        if (det && (cast(void*) i) !is (cast(void*) h))
            delete i;
        setMonitor(h, null);
    }
}

extern (C) void _d_monitorenter(Object h)
{
    Monitor* m = getMonitor(h);

    if (m is null)
    {
        _d_monitor_create(h);
        m = getMonitor(h);
    }

    IMonitor i = m.impl;

    if (i is null)
    {
        _d_monitor_lock(h);
        return;
    }
    i.lock();
}

extern (C) void _d_monitorexit(Object h)
{
    Monitor* m = getMonitor(h);
    IMonitor i = m.impl;

    if (i is null)
    {
        _d_monitor_unlock(h);
        return;
    }
    i.unlock();
}

extern (C) void _d_monitor_devt(Monitor* m, Object h)
{
    if (m.devt.length)
    {
        DEvent[] devt;

        synchronized (h)
        {
            devt = m.devt;
            m.devt = null;
        }
        foreach (v; devt)
        {
            if (v)
                v(h);
        }
        free(devt.ptr);
    }
}

extern (C) void rt_attachDisposeEvent(Object h, DEvent e)
{
    synchronized (h)
    {
        Monitor* m = getMonitor(h);
        assert(m.impl is null);

        foreach (inout v; m.devt)
        {
            if (v is null || v == e)
            {
                v = e;
                return;
            }
        }

        auto len = m.devt.length + 4; // grow by 4 elements
        auto pos = m.devt.length;     // insert position
        auto p = realloc(m.devt.ptr, DEvent.sizeof * len);
        if (!p)
            onOutOfMemoryError();
        m.devt = (cast(DEvent*)p)[0 .. len];
        m.devt[pos+1 .. len] = null;
        m.devt[pos] = e;
    }
}

extern (C) void rt_detachDisposeEvent(Object h, DEvent e)
{
    synchronized (h)
    {
        Monitor* m = getMonitor(h);
        assert(m.impl is null);

        foreach (p, v; m.devt)
        {
            if (v == e)
            {
                memmove(&m.devt[p],
                        &m.devt[p+1],
                        (m.devt.length - p - 1) * DEvent.sizeof);
                m.devt[$ - 1] = null;
                return;
            }
        }
    }
}
