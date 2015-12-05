/**
 * This module implements the runtime-part of LDC exceptions
 * on Windows win32.
 */
module ldc.eh.win32;

version(CRuntime_Microsoft):
version(Win32):

import ldc.eh.common;
import core.sys.windows.windows;
import core.exception : onOutOfMemoryError, OutOfMemoryError;
import core.stdc.stdlib : malloc, free;
import core.stdc.string : memcpy;

// pointers are image relative for Win64 versions
version(Win64)
    alias ImgPtr(T) = uint; // offset into image
else
    alias ImgPtr(T) = T;

alias PMFN = ImgPtr!(void function(void*));

struct TypeDescriptor(int N)
{
    version(_RTTI)
        const void * pVFTable;  // Field overloaded by RTTI
    else
        uint hash;  // Hash value computed from type's decorated name

    void * spare;   // reserved, possible for RTTI
    char[N+1] name; // variable size, zero terminated
}

struct PMD
{
    int mdisp;      // Offset of intended data within base
    int pdisp;      // Displacement to virtual base pointer
    int vdisp;      // Index within vbTable to offset of base
}

struct CatchableType
{
    uint  properties;       // Catchable Type properties (Bit field)
    ImgPtr!(TypeDescriptor!1*) pType;   // Pointer to TypeDescriptor
    PMD   thisDisplacement; // Pointer to instance of catch type within thrown object.
    int   sizeOrOffset;     // Size of simple-type object or offset into buffer of 'this' pointer for catch object
    PMFN  copyFunction;     // Copy constructor or CC-closure
}

enum CT_IsSimpleType    = 0x00000001;  // type is a simple type (includes pointers)
enum CT_ByReferenceOnly = 0x00000002;  // type must be caught by reference
enum CT_HasVirtualBase  = 0x00000004;  // type is a class with virtual bases
enum CT_IsWinRTHandle   = 0x00000008;  // type is a winrt handle
enum CT_IsStdBadAlloc   = 0x00000010;  // type is a a std::bad_alloc

struct CatchableTypeArray
{
    int	nCatchableTypes;
    ImgPtr!(CatchableType*)[2] arrayOfCatchableTypes;
}

struct _ThrowInfo
{
    uint    attributes;     // Throw Info attributes (Bit field)
    PMFN    pmfnUnwind;     // Destructor to call when exception has been handled or aborted.
    PMFN    pForwardCompat; // pointer to Forward compatibility frame handler
    ImgPtr!(CatchableTypeArray*) pCatchableTypeArray; // pointer to CatchableTypeArray
}

enum TI_IsConst     = 0x00000001;   // thrown object has const qualifier
enum TI_IsVolatile  = 0x00000002;   // thrown object has volatile qualifier
enum TI_IsUnaligned = 0x00000004;   // thrown object has unaligned qualifier
enum TI_IsPure      = 0x00000008;   // object thrown from a pure module
enum TI_IsWinRT     = 0x00000010;   // object thrown is a WinRT Exception

extern(Windows) void RaiseException(DWORD dwExceptionCode,
                                    DWORD dwExceptionFlags,
                                    DWORD nNumberOfArguments,
                                    ULONG_PTR* lpArguments);

enum int STATUS_MSC_EXCEPTION = 0xe0000000 | ('m' << 16) | ('s' << 8) | ('c' << 0);

enum EXCEPTION_NONCONTINUABLE     = 0x01;
enum EXCEPTION_UNWINDING          = 0x02;

enum EH_MAGIC_NUMBER1             = 0x19930520;

extern(C) void _d_throw_exception(Object e)
{
    if (e is null)
        fatalerror("Cannot throw null exception");
    auto ti = typeid(e);
    if (ti is null)
        fatalerror("Cannot throw corrupt exception object with null classinfo");

    if (exceptionStack.length > 0)
    {
        // we expect that the terminate handler will be called, so hook
        // it to avoid it actually terminating
        if (!old_terminate_handler)
            old_terminate_handler = set_terminate(&msvc_eh_terminate);
    }
    exceptionStack.push(cast(Throwable) e);

    ULONG_PTR[3] ExceptionInformation;
    ExceptionInformation[0] = EH_MAGIC_NUMBER1;
    ExceptionInformation[1] = cast(ULONG_PTR) cast(void*) &e;
    ExceptionInformation[2] = cast(ULONG_PTR) getThrowInfo(ti);

    RaiseException(STATUS_MSC_EXCEPTION, EXCEPTION_NONCONTINUABLE, 3, ExceptionInformation.ptr);
}

///////////////////////////////////////////////////////////////

import rt.util.container.hashtab;
import core.sync.mutex;

__gshared HashTab!(TypeInfo_Class, _ThrowInfo) throwInfoHashtab;
__gshared HashTab!(TypeInfo_Class, CatchableType) catchableHashtab;
__gshared Mutex throwInfoMutex;

// create and cache throwinfo for ti
_ThrowInfo* getThrowInfo(TypeInfo_Class ti)
{
    throwInfoMutex.lock();
    if (auto p = ti in throwInfoHashtab)
    {
        throwInfoMutex.unlock();
        return p;
    }

    size_t classes = 0;
    for (TypeInfo_Class tic = ti; tic; tic = tic.base)
        classes++;

    size_t sz = int.sizeof + classes * ImgPtr!(CatchableType*).sizeof;
    auto cta = cast(CatchableTypeArray*) malloc(sz);
    if (!cta)
        onOutOfMemoryError();
    cta.nCatchableTypes = classes;

    size_t c = 0;
    for (TypeInfo_Class tic = ti; tic; tic = tic.base)
        cta.arrayOfCatchableTypes.ptr[c++] = getCatchableType(tic);

    _ThrowInfo tinf = { 0, null, null, cta };
    throwInfoHashtab[ti] = tinf;
    auto pti = ti in throwInfoHashtab;
    throwInfoMutex.unlock();
    return pti;
}

CatchableType* getCatchableType(TypeInfo_Class ti)
{
    if (auto p = ti in catchableHashtab)
        return p;

    size_t sz = TypeDescriptor!1.sizeof + ti.name.length;
    auto td = cast(TypeDescriptor!1*) malloc(sz);
    if (!td)
        onOutOfMemoryError();

    td.hash = 0;
    td.spare = null;
    td.name.ptr[0] = 'D';
    memcpy(td.name.ptr + 1, ti.name.ptr, ti.name.length);
    td.name.ptr[ti.name.length + 1] = 0;

    CatchableType ct = { CT_IsSimpleType, td, { 0, -1, 0 }, 4, null };
    catchableHashtab[ti] = ct;
    return ti in catchableHashtab;
}

///////////////////////////////////////////////////////////////
extern(C) Object _d_eh_enter_catch(void* ptr)
{
    if (!ptr)
        return null; // null for "catch all" in scope(failure), will rethrow
    Throwable e = *(cast(Throwable*) ptr);

    while(exceptionStack.length > 0)
    {
        Throwable t = exceptionStack.pop();
        if (t is e)
            break;

        auto err = cast(Error) t;
        if (err && !cast(Error)e)
        {
            // there is an Error in flight, but we caught an Exception
            // so we convert it and rethrow the Error
            err.bypassedException = e;
            throw err;
        }
        t.next = e.next;
        e.next = t;
    }

    return e;
}

alias terminate_handler = void function();

extern(C) void** __current_exception();
extern(C) void** __current_exception_context();
extern(C) int* __processing_throw();

extern(C) terminate_handler set_terminate(terminate_handler new_handler);

terminate_handler old_terminate_handler; // explicitely per thread

ExceptionStack exceptionStack;

struct ExceptionStack
{
nothrow:
    ~this()
    {
        if (_p)
            free(_p);
    }

    void push(Throwable e)
    {
        if (_length == _cap)
            grow();
        _p[_length++] = e;
    }

    Throwable pop()
    {
        return _p[--_length];
    }

    ref inout(Throwable) opIndex(size_t idx) inout
    {
        return _p[idx];
    }

    @property size_t length() const { return _length; }
    @property bool empty() const { return !length; }

private:
    void grow()
    {
        // alloc from GC? add array as a GC range?
        immutable ncap = _cap ? 2 * _cap : 64;
        auto p = cast(Throwable*)malloc(ncap * Throwable.sizeof);
        if (p is null)
            onOutOfMemoryError();
        p[0 .. _length] = _p[0 .. _length];
        free(_p);
        _p = p;
        _cap = ncap;
    }

    size_t _length;
    Throwable* _p;
    size_t _cap;
}

// helper to access TLS from naked asm
int tlsUncaughtExceptions() nothrow
{
    return exceptionStack.length;
}

auto tlsOldTerminateHandler() nothrow
{
    return old_terminate_handler;
}

void msvc_eh_terminate() nothrow
{
    asm nothrow {
        naked;
        call tlsUncaughtExceptions;
        cmp EAX, 0;
        je L_term;

        // hacking into the call chain to return EXCEPTION_EXECUTE_HANDLER
        //  as the return value of __FrameUnwindFilter so that
        // __FrameUnwindToState continues with the next unwind block

        // restore ptd->__ProcessingThrow
        push EAX;
        call __processing_throw;
        pop [EAX];

        // undo one level of exception frames from terminate()
        mov EAX,FS:[0];
        mov EAX,[EAX];
        mov FS:[0], EAX;

        // assume standard stack frames for callers
        mov EAX,EBP;   // frame pointer of terminate()
        mov EAX,[EAX]; // frame pointer of __FrameUnwindFilter
        mov ESP,EAX;   // restore stack
        pop EBP;       // and frame pointer
        mov EAX, 1;    // return EXCEPTION_EXECUTE_HANDLER
        ret;

    L_term:
        call tlsOldTerminateHandler;
        cmp EAX, 0;
        je L_ret;
        jmp EAX;
    L_ret:
        ret;
    }
}

///////////////////////////////////////////////////////////////
void msvc_eh_init()
{
    throwInfoMutex = new Mutex;

    // preallocate type descriptors likely to be needed
    getThrowInfo(typeid(Exception));
    // better not have to allocate when this is thrown:
    getThrowInfo(typeid(OutOfMemoryError));
}

shared static this()
{
    // should be called from rt_init
    msvc_eh_init();
}
