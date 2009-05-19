// This is a modification of tango/object.di that includes
// aliases for string, wstring, dstring and Exception.
// This is because dstress expects phobos, which provides
// these aliases.

module object;

public import std.compat;

alias typeof(int.sizeof)                    size_t;
alias typeof(cast(void*)0 - cast(void*)0)   ptrdiff_t;

alias size_t hash_t;
alias int equals_t;

class Object
{
    char[] toString();
    hash_t toHash();
    int    opCmp(Object o);
    equals_t    opEquals(Object o);

    interface Monitor
    {
        void lock();
        void unlock();
    }
}

struct Interface
{
    ClassInfo   classinfo;
    void*[]     vtbl;
    ptrdiff_t   offset;   // offset to Interface 'this' from Object 'this'
}

class ClassInfo : Object
{
    byte[]      init;   // class static initializer
    char[]      name;   // class name
    void*[]     vtbl;   // virtual function pointer table
    Interface[] interfaces;
    ClassInfo   base;
    void*       destructor;
    void(*classInvariant)(Object);
    uint        flags;
    // 1:       // IUnknown
    // 2:       // has no possible pointers into GC memory
    // 4:       // has offTi[] member
    // 8:       // has constructors
    // 32:      // has typeinfo    
    void*       deallocator;
    OffsetTypeInfo[] offTi;
    void*       defaultConstructor;
    TypeInfo typeinfo;

    static ClassInfo find(char[] classname);
    Object create();
}

struct OffsetTypeInfo
{
    size_t   offset;
    TypeInfo ti;
}

class TypeInfo
{
    hash_t   getHash(void *p);
    equals_t equals(void *p1, void *p2);
    int      compare(void *p1, void *p2);
    size_t   tsize();
    void     swap(void *p1, void *p2);
    TypeInfo next();
    void[]   init();
    uint     flags();
    // 1:    // has possible pointers into GC memory
    OffsetTypeInfo[] offTi();
}

class TypeInfo_Typedef : TypeInfo
{
    TypeInfo base;
    char[]   name;
    void[]   m_init;
}

class TypeInfo_Enum : TypeInfo_Typedef
{
}

class TypeInfo_Pointer : TypeInfo
{
    TypeInfo m_next;
}

class TypeInfo_Array : TypeInfo
{
    TypeInfo value;
}

class TypeInfo_StaticArray : TypeInfo
{
    TypeInfo value;
    size_t   len;
}

class TypeInfo_AssociativeArray : TypeInfo
{
    TypeInfo value;
    TypeInfo key;
}

class TypeInfo_Function : TypeInfo
{
    TypeInfo next;
}

class TypeInfo_Delegate : TypeInfo
{
    TypeInfo next;
}

class TypeInfo_Class : TypeInfo
{
    ClassInfo info;
}

class TypeInfo_Interface : TypeInfo
{
    ClassInfo info;
}

class TypeInfo_Struct : TypeInfo
{
    char[] name;
    void[] m_init;

    uint function(void*)      xtoHash;
    int function(void*,void*) xopEquals;
    int function(void*,void*) xopCmp;
    char[] function(void*)    xtoString;

    uint m_flags;
}

class TypeInfo_Tuple : TypeInfo
{
    TypeInfo[]  elements;
}

class ModuleInfo
{
    char[]          name;
    ModuleInfo[]    importedModules;
    ClassInfo[]     localClasses;
    uint            flags;

    void function() ctor;
    void function() dtor;
    void function() unitTest;

    void* xgetMembers;
    void function() ictor;

    static int opApply( int delegate( inout ModuleInfo ) );
}

class Exception : Object
{
    struct FrameInfo{
        long line;
        ptrdiff_t iframe;
        ptrdiff_t offset;
        size_t address;
        char[] file;
        char[] func;
        char[256] charBuf;
        void writeOut(void delegate(char[])sink);
    }
    interface TraceInfo
    {
        int opApply( int delegate( ref FrameInfo fInfo) );
    }

    char[]      msg;
    char[]      file;
    size_t      line;  // long would be better
    TraceInfo   info;
    Exception   next;

    this(char[] msg, char[] file, long line, Exception next, TraceInfo info );
    this(char[] msg, Exception next = null);
    this(char[] msg, char[] file, long line, Exception next = null);
    char[] toString();
    void writeOut(void delegate(char[]) sink);
}
