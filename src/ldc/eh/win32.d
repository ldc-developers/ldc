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
import rt.util.container.common : xmalloc;

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
extern(C) Object _d_eh_enter_catch(void* ptr, ClassInfo catchType)
{
    assert(ptr);

    // is this a thrown D exception?
    auto e = *(cast(Throwable*) ptr);
    size_t pos = exceptionStack.find(e);
    if (pos >= exceptionStack.length())
        return null;

    auto caught = e;
    // append inner unhandled thrown exceptions
    for (size_t p = pos + 1; p < exceptionStack.length(); p++)
        e = chainExceptions(e, exceptionStack[p]);
    exceptionStack.shrink(pos);

    // given the bad semantics of Errors, we are fine with passing
    //  the test suite with slightly inaccurate behaviour by just
    //  rethrowing a collateral Error here, though it might need to
    //  be caught by a catch handler in an inner scope
    if (e !is caught)
    {
        if (_d_isbaseof(typeid(e), catchType))
            *cast(Throwable*) ptr = e; // the current catch can also catch this Error
        else
            _d_throw_exception(e);
    }
    return e;
}

Throwable chainExceptions(Throwable e, Throwable t)
{
    if (!cast(Error) e)
        if (auto err = cast(Error) t)
        {
            err.bypassedException = e;
            return err;
        }

    auto pChain = &e.next;
    while (*pChain)
        pChain = &(pChain.next);
    *pChain = t;
    return e;
}

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

    void shrink(size_t sz)
    {
        while (_length > sz)
            _p[--_length] = null;
    }

    ref inout(Throwable) opIndex(size_t idx) inout
    {
        return _p[idx];
    }

    size_t find(Throwable e)
    {
        for (size_t i = _length; i > 0; )
            if (exceptionStack[--i] is e)
                return i;
        return ~0;
    }

    @property size_t length() const { return _length; }
    @property bool empty() const { return !length; }

    void swap(ref ExceptionStack other)
    {
        static void swapField(T)(ref T a, ref T b) { T o = b; b = a; a = o; }
        swapField(_length, other._length);
        swapField(_p,      other._p);
        swapField(_cap,    other._cap);
    }

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

///////////////////////////////////////////////////////////////
alias terminate_handler = void function();
extern(C) terminate_handler set_terminate(terminate_handler new_handler);
terminate_handler old_terminate_handler; // explicitely per thread

// helper to access TLS from naked asm
size_t tlsUncaughtExceptions() nothrow
{
    return exceptionStack.length;
}

auto tlsOldTerminateHandler() nothrow
{
    return old_terminate_handler;
}

void msvc_eh_terminate() nothrow
{
    version(Win32)
    {
        asm nothrow
        {
            naked;
            call tlsUncaughtExceptions;
            cmp EAX, 1;
            jle L_term;

            // hacking into the call chain to return EXCEPTION_EXECUTE_HANDLER
            //  as the return value of __FrameUnwindFilter so that
            // __FrameUnwindToState continues with the next unwind block

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
}

///////////////////////////////////////////////////////////////
extern(C) void** __current_exception() nothrow;
extern(C) void** __current_exception_context() nothrow;
extern(C) int* __processing_throw() nothrow;

struct FiberContext
{
    ExceptionStack exceptionStack;
    void* currentException;
    void* currentExceptionContext;
    int processingContext;
}

FiberContext* fiberContext;

extern(C) void* _d_eh_swapContext(FiberContext* newContext) nothrow
{
    import core.stdc.string : memset;
    if (!fiberContext)
    {
        fiberContext = cast(FiberContext*) xmalloc(FiberContext.sizeof);
        memset(fiberContext, 0, FiberContext.sizeof);
    }
    fiberContext.exceptionStack.swap(exceptionStack);
    fiberContext.currentException = *__current_exception();
    fiberContext.currentExceptionContext = *__current_exception_context();
    fiberContext.processingContext = *__processing_throw();

    if (newContext)
    {
        exceptionStack.swap(newContext.exceptionStack);
        *__current_exception() = newContext.currentException;
        *__current_exception_context() = newContext.currentExceptionContext;
        *__processing_throw() = newContext.processingContext;
    }
    else
    {
        exceptionStack = ExceptionStack();
        *__current_exception() = null;
        *__current_exception_context() = null;
        *__processing_throw() = 0;
    }

    FiberContext* old = fiberContext;
    fiberContext = newContext;
    return old;
}

static ~this()
{
    import core.stdc.stdlib : free;
    if (fiberContext)
    {
        destroy(*fiberContext);
        free(fiberContext);
    }
}

///////////////////////////////////////////////////////////////
extern(C) bool _d_enter_cleanup(void* ptr)
{
    // currently just used to avoid that a cleanup handler that can
    // be inferred to not return, is removed by the LLVM optimizer
    //
    // TODO: setup an exception handler here (ptr passes the address
    // of a 40 byte stack area in a parent fuction scope) to deal with
    // unhandled exceptions during unwinding.
    return true;
}

extern(C) void _d_leave_cleanup(void* ptr)
{
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
