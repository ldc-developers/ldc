// A copy of the object.di from Tango, with std.compat publically included to
// to make sure the Phobos compatibilty aliases are always available (DStress
// depends on Phobos).
module object;

public import std.compat;

/// unsigned integer type of the size of a pointer
alias typeof(int.sizeof)                    size_t;
/// signed integer type of the size of a pointer
alias typeof(cast(void*)0 - cast(void*)0)   ptrdiff_t;

/// type of hashes used in associative arrays
alias size_t hash_t;
/// type returned by equality comparisons
alias int equals_t;

/// root class for all objects in D
class Object
{
    void dispose();
    /// returns a string representation of the object (for debugging purposes)
    char[] toString();
    /// returns a hash
    hash_t toHash();
    /// compares two objects, returns a number ret, such (a op b) is rewritten as (a.opCmp(b) op 0)
    /// thus if a>b a.opCmp(b)>0
    int    opCmp(Object o);
    /// returns 0 if this==o
    equals_t    opEquals(Object o);

    interface Monitor
    {
        void lock();
        void unlock();
    }
}

/// interface, if COM objects (IUnknown) they might not be casted to Object
struct Interface
{
    /// class info of the interface
    ClassInfo   classinfo;
    void*[]     vtbl;
    /// offset to Interface 'this' from Object 'this'
    ptrdiff_t   offset;
}

struct PointerMap
{
    // Conservative pointer mask (one word, scan it, we don't know if it's
    // really a pointer)
    size_t[] bits = [1, 1, 0];

    size_t size();
    bool mustScanWordAt(size_t offset);
    bool isPointerAt(size_t offset);
    bool canUpdatePointers();
}

struct PointerMapBuilder
{
    private size_t[] m_bits = null;
    private size_t m_size = 0;

    void size(size_t bytes);
    void mustScanWordAt(size_t offset);
    void inlineAt(size_t offset, PointerMap pm);
    PointerMap convertToPointerMap();
}

/// class information
class ClassInfo : Object
{
    byte[]      init;   // class static initializer
    char[]      name;   /// class name
    void*[]     vtbl;   // virtual function pointer table
    Interface[] interfaces; /// implemented interfaces
    ClassInfo   base; /// base class
    void*       destructor; /// compiler dependent storage of destructor function pointer
    void*       classInvariant; /// compiler dependent storage of classInvariant function pointer
    /// flags
    /// 1: IUnknown
    /// 2: has no possible pointers into GC memory
    /// 4: has offTi[] member
    /// 8: has constructors
    //  32: has typeinfo
    uint        flags;
    void*       deallocator;
    OffsetTypeInfo[] offTi; /// offsets of its members (not supported by all compilers)
    void*       defaultConstructor; /// compiler dependent storage of constructor function pointer
    /// TypeInfo information about this class
    TypeInfo typeinfo;
    version (D_HavePointerMap) {
        PointerMap pointermap;
    }
    /// finds the classinfo of the class with the given name
    static ClassInfo find(char[] classname);
    /// creates an instance of this class (works only if there is a constructor without arguments)
    Object create();
}

/// offset of the different fields (at the moment works only with ldc)
struct OffsetTypeInfo
{
    size_t   offset;
    TypeInfo ti;
}

/// information on a type
class TypeInfo
{
    /// returns the hash of the type of this TypeInfo at p
    hash_t   getHash(void *p);
    /// returns 0 if the types of this TypeInfo stored at p1 and p2 are different
    equals_t      equals(void *p1, void *p2);
    /// compares the types of this TypeInfo stored at p1 and p2
    int      compare(void *p1, void *p2);
     /// Return alignment of type
    size_t   talign() { return tsize(); }
    /// returns the size of a type with the current TypeInfo
    size_t   tsize();
    /// swaps the two types stored at p1 and p2
    void     swap(void *p1, void *p2);
    /// "next" TypeInfo (for an array its elements, for a pointer what it is pointed to,...)
    TypeInfo next();
    void[]   init();
    /// flags, 1: has possible pointers into GC memory
    uint     flags();
    PointerMap pointermap();
    /// offsets of the various elements
    OffsetTypeInfo[] offTi();

    /** Return internal info on arguments fitting into 8byte.
       * See X86-64 ABI 3.2.3
     */
    version (X86_64) int argTypes(out TypeInfo arg1, out TypeInfo arg2);
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
    /// typeinfo of the elements, might be null for basic arrays, it is safer to use next()
    TypeInfo value;

    //ensure derived array TypeInfos use correct method
    //if this declaration is forgotten, e.g. TypeInfo_Ai will have a wrong vtbl entry
    PointerMap pointermap();
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

    hash_t function()   xtoHash;
    int function(void*) xopEquals;
    int function(void*) xopCmp;
    char[] function()   xtoString;

    uint m_flags;

    version (D_HavePointerMap) {
        PointerMap m_pointermap;
    }
}

class TypeInfo_Tuple : TypeInfo
{
    TypeInfo[]  elements;
}

/// information about a module (can be used for example to get its unittests)
class ModuleInfo
{
    /// name of the module
    char[]          name;
    ///
    ModuleInfo[]    importedModules;
    ///
    ClassInfo[]     localClasses;
    uint            flags;

    void function() ctor;
    void function() dtor;
    /// unit tests of the module
    void function() unitTest;

    version(GNU){}
    else{
        void* xgetMembers;
        void function() ictor;
    }

    /// loops on all the modules loaded
    static int opApply( int delegate( ref ModuleInfo ) );
}

/// base class for all exceptions/errors
/// it is a good practice to pass line and file to the exception, which can be obtained with
/// __FILE__ and __LINE__, and then passed to the exception constructor
class Exception : Object
{
    /// Information about a frame in the stack
    struct FrameInfo{
        /// line number in the source of the most likely start adress (0 if not available)
        long line;
        /// number of the stack frame (starting at 0 for the top frame)
        ptrdiff_t iframe;
        /// offset from baseSymb: within the function, or from the closest symbol
        ptrdiff_t offsetSymb;
        /// adress of the symbol in this execution
        size_t baseSymb;
        /// offset within the image (from this you can use better methods to get line number
        /// a posteriory)
        ptrdiff_t offsetImg;
        /// base adress of the image (will be dependent on randomization schemes)
        size_t baseImg;
        /// adress of the function, or at which the ipc will return
        /// (which most likely is the one after the adress where it started)
        /// this is the raw adress returned by the backtracing function
        size_t address;
        /// file (image) of the current adress
        char[] file;
        /// name of the function, if possible demangled
        char[] func;
        /// extra information (for example calling arguments)
        char[] extra;
        /// if the address is exact or it is the return address
        bool exactAddress;
        /// if this function is an internal functions (for example the backtracing function itself)
        /// if true by default the frame is not printed
        bool internalFunction;
        alias void function(FrameInfo*,void delegate(char[])) FramePrintHandler;
        /// the default printing function
        static FramePrintHandler defaultFramePrintingFunction;
        /// writes out the current frame info
        void writeOut(void delegate(char[])sink);
        /// clears the frame information stored
        void clear();
    }
    /// trace information has the following interface
    interface TraceInfo
    {
        int opApply( int delegate( ref FrameInfo fInfo) );
        void writeOut(void delegate(char[])sink);
    }
    /// message of the exception
    char[]      msg;
    /// file name
    char[]      file;
    /// line number
    size_t      line;  // long would be better to be consistent
    /// trace of where the exception was raised
    TraceInfo   info;
    /// next exception (if an exception made an other exception raise)
    Exception   next;

    /// designated constructor (breakpoint this if you want to catch all explict Exception creations,
    /// special exception just allocate and init the structure directly)
    this(char[] msg, char[] file, long line, Exception next, TraceInfo info );
    this(char[] msg, Exception next=null);
    this(char[] msg, char[] file, long line, Exception next = null);
    /// returns the message of the exception, should not be used (because it should not allocate,
    /// and thus only a small message is returned)
    char[] toString();
    /// writes out the message of the exception, by default writes toString
    /// override this is you have a better message for the exception
    void writeOutMsg(void delegate(char[]) sink);
    /// writes out the exception message, file, line number, stacktrace (if available) and any
    /// subexceptions
    void writeOut(void delegate(char[]) sink);
}
