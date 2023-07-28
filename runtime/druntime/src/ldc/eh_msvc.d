/**
 * This module implements the runtime-part of LDC exceptions
 * on Windows, based on the MSVC++ runtime.
 */
module ldc.eh_msvc;

version (CRuntime_Microsoft):

import core.sys.windows.windows;
import core.exception : onOutOfMemoryError, OutOfMemoryError;
import core.internal.container.common : xmalloc;
import core.stdc.stdlib : malloc, free, abort;
import core.stdc.string : memcpy;
import ldc.attributes;
import ldc.llvmasm;

// pointers are image relative for Win64 versions
version (Win64)
    struct ImgPtr(T) { uint offset; } // offset into image
else
    alias ImgPtr(T) = T*;

alias PMFN = ImgPtr!(void function(void*));

struct TypeDescriptor
{
    version (_RTTI)
        const void * pVFTable;  // Field overloaded by RTTI
    else
        uint hash;  // Hash value computed from type's decorated name

    void * spare;   // reserved, possible for RTTI
    char[1] name;   // variable size, zero terminated
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
    ImgPtr!TypeDescriptor pType;   // Pointer to TypeDescriptor
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
    ImgPtr!CatchableType[1] arrayOfCatchableTypes; // variable size
}

struct _ThrowInfo
{
    uint    attributes;     // Throw Info attributes (Bit field)
    PMFN    pmfnUnwind;     // Destructor to call when exception has been handled or aborted.
    PMFN    pForwardCompat; // pointer to Forward compatibility frame handler
    ImgPtr!CatchableTypeArray pCatchableTypeArray; // pointer to CatchableTypeArray
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

struct CxxExceptionInfo
{
    size_t Magic;
    Throwable* pThrowable; // null for rethrow
    _ThrowInfo* ThrowInfo;
    version (Win64) void* ImgBase;
}

// D runtime function
extern(C) int _d_isbaseof(ClassInfo oc, ClassInfo c);

// error and exit
extern(C) void fatalerror(const(char)* format, ...)
{
    import core.stdc.stdarg;
    import core.stdc.stdio;

    va_list args;
    va_start(args, format);
    fprintf(stderr, "Fatal error in EH code: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    abort();
}

extern(C) void _d_createTrace(Throwable t, void* context);

extern(C) void _d_throw_exception(Throwable throwable)
{
    if (throwable is null)
        fatalerror("Cannot throw null exception");
    auto ti = typeid(throwable);
    if (ti is null)
        fatalerror("Cannot throw corrupt exception object with null classinfo");

    /* Increment reference count if `o` is a refcounted Throwable
     */
    auto refcount = throwable.refcount();
    if (refcount)       // non-zero means it's refcounted
        throwable.refcount() = refcount + 1;

    if (exceptionStack.length > 0)
    {
        // we expect that the terminate handler will be called, so hook
        // it to avoid it actually terminating
        if (!old_terminate_handler)
            old_terminate_handler = set_terminate(&msvc_eh_terminate);
    }

    exceptionStack.push(throwable);

    _d_createTrace(throwable, null);

    CxxExceptionInfo info;
    info.Magic = EH_MAGIC_NUMBER1;
    info.pThrowable = &throwable;
    info.ThrowInfo = getThrowInfo(ti).toPointer;
    version (Win64) info.ImgBase = ehHeap.base;

    RaiseException(STATUS_MSC_EXCEPTION, EXCEPTION_NONCONTINUABLE,
                   info.sizeof / size_t.sizeof, cast(ULONG_PTR*)&info);
}

///////////////////////////////////////////////////////////////

import core.internal.container.hashtab;
import core.sync.mutex;

__gshared HashTab!(TypeInfo_Class, ImgPtr!_ThrowInfo) throwInfoHashtab;
__gshared HashTab!(TypeInfo_Class, ImgPtr!CatchableType) catchableHashtab;
__gshared Mutex throwInfoMutex;

// create and cache throwinfo for ti
ImgPtr!_ThrowInfo getThrowInfo(TypeInfo_Class ti)
{
    throwInfoMutex.lock();
    if (auto p = ti in throwInfoHashtab)
    {
        throwInfoMutex.unlock();
        return *p;
    }

    int classes = 0;
    for (TypeInfo_Class tic = ti; tic; tic = tic.base)
        classes++;

    size_t sz = int.sizeof + classes * ImgPtr!(CatchableType).sizeof;
    ImgPtr!CatchableTypeArray cta = eh_malloc!CatchableTypeArray(sz);
    toPointer(cta).nCatchableTypes = classes;

    size_t c = 0;
    for (TypeInfo_Class tic = ti; tic; tic = tic.base)
        cta.toPointer.arrayOfCatchableTypes.ptr[c++] = getCatchableType(tic);

    auto tinf = eh_malloc!_ThrowInfo();
    *(tinf.toPointer) = _ThrowInfo(0, PMFN(), PMFN(), cta);
    throwInfoHashtab[ti] = tinf;
    throwInfoMutex.unlock();
    return tinf;
}

ImgPtr!CatchableType getCatchableType(TypeInfo_Class ti)
{
    if (auto p = ti in catchableHashtab)
        return *p;

    const sz = TypeDescriptor.sizeof + ti.name.length + 1;
    auto td = eh_malloc!TypeDescriptor(sz);
    auto ptd = td.toPointer;

    ptd.hash = 0;
    ptd.spare = null;
    ptd.name.ptr[0] = 'D';
    memcpy(ptd.name.ptr + 1, ti.name.ptr, ti.name.length);
    ptd.name.ptr[ti.name.length + 1] = 0;

    auto ct = eh_malloc!CatchableType();
    ct.toPointer[0] = CatchableType(CT_IsSimpleType, td, PMD(0, -1, 0), size_t.sizeof, PMFN());
    catchableHashtab[ti] = ct;
    return ct;
}

///////////////////////////////////////////////////////////////
extern(C) Throwable _d_eh_enter_catch(void* ptr, ClassInfo catchType)
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

    return Throwable.chainTogether(e, t);
}

ExceptionStack exceptionStack;

static ~this()
{
    // destructors not automatically run on globals
    exceptionStack.destroy();
}

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
        auto p = cast(Throwable*)xmalloc(ncap * Throwable.sizeof);
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
size_t tlsUncaughtExceptions() nothrow @assumeUsed
{
    return exceptionStack.length;
}

auto tlsOldTerminateHandler() nothrow @assumeUsed
{
    return old_terminate_handler;
}

void msvc_eh_terminate() nothrow @naked
{
    version (Win32)
    {
        __asm(
           `call __D3ldc7eh_msvc21tlsUncaughtExceptionsFNbZk
            cmp $$1, %eax
            jle L_term

            // hacking into the call chain to return EXCEPTION_EXECUTE_HANDLER
            //  as the return value of __FrameUnwindFilter so that
            // __FrameUnwindToState continues with the next unwind block

            // undo one level of exception frames from terminate()
            mov %fs:(0), %eax
            mov (%eax), %eax
            mov %eax, %fs:(0)

            // assume standard stack frames for callers
            mov %ebp, %eax   // frame pointer of terminate()
            mov (%eax), %eax // frame pointer of __FrameUnwindFilter
            mov %eax, %esp   // restore stack
            pop %ebp         // and frame pointer
            mov $$1, %eax    // return EXCEPTION_EXECUTE_HANDLER
            ret

        L_term:
            call __D3ldc7eh_msvc22tlsOldTerminateHandlerFNbNiNfZPFZv
            cmp $$0, %eax
            je L_ret
            jmp *%eax
        L_ret:
            ret`,
            "~{memory},~{flags},~{ebp},~{esp},~{eax}"
        );
    }
    else
    {
        __asm(
           `push %rbx                      // align stack for better debuggability
            call _D3ldc7eh_msvc21tlsUncaughtExceptionsFNbZm
            cmp $$1, %rax
            jle L_term

            // update stack and IP so we just continue in __FrameUnwindHandler
            // NOTE: these checks can fail if you have breakpoints set at
            //       the respective code locations
            mov 8(%rsp), %rax              // get return address
            cmpb $$0xEB, (%rax)            // jmp?
            jne noJump
            movsbq 1(%rax), %rdx           // follow jmp
            lea 2(%rax,%rdx), %rax
        noJump:
            cmpb $$0xE8, (%rax)            // call abort?
            jne L_term
            add $$5, %rax
            mov (%rax), %edx
            mov $$0xFFFFFF, %rbx
            and %rbx, %rdx
            cmp $$0xC48348, %rdx           // add ESP,nn  (debug UCRT libs)
            je L_addESP_found
            cmp $$0x90, %dl                // nop; (release libs)
            jne L_term

        L_release_ucrt:
            mov 8(%rsp), %rdx
            cmpw $$0xD3FF, -2(%rdx)        // call ebx?
            sete %bl                       // if not, it's UCRT 10.0.14393.0
            movzbq %bl, %rbx
            mov $$0x28, %rdx               // release build of vcruntimelib
            jmp L_retTerminate

        L_addESP_found:
            xor %rbx, %rbx                 // debug version: RBX not pushed inside terminate()
            movzbq 3(%rax), %rdx           // read nn

            cmpb $$0xC3, 4(%rax)           // ret?
            jne L_term

        L_retTerminate:
            lea 0x10(%rsp,%rdx), %rdx      // RSP before returning from terminate()

            mov (%rdx), %rax               // return address inside __FrameUnwindHandler

            or %rbx, %rdx                  // RDX aligned, save RBX == 0 for UCRT 10.0.14393.0, 1 otherwise

            cmpb $$0xEB, -19(%rax)         // skip back to default jump inside "switch" (libvcruntimed.lib)
            je L_switchFound

            cmpb $$0xEB, -20(%rax)         // skip back to default jump inside "switch" (vcruntime140d.dll)
            je L_switchFound2

            mov $$0xC48348C0333048FF, %rbx // dec [rax+30h]; xor eax,eax; add rsp,nn (libvcruntime.lib)
            cmp -0x18(%rax), %rbx
            je L_retFound

            cmp 0x29(%rax), %rbx           // dec [rax+30h]; xor eax,eax; add rsp,nn (vcruntime140.dll)
            je L_retVC14_11

            cmp 0x11(%rax), %rbx           // dec [rax+30h]; xor eax,eax; add rsp,nn (vcruntime140.dll, 14.16.x.x)
            je L_retVC14_16

            cmp 0x1B(%rax), %rbx           // dec [rax+30h]; xor eax,eax; add rsp,nn (vcruntime140.dll 14.14.x.y)
            je L_retVC14_14
            
            mov $$0x30245C8B483048FF, %rbx // dec [rax+30h]; mov rbx,qword ptr [rsp+30h]
            cmp -0x2b(%rax), %rbx          // (libcmt.lib, 14.23.x.x)
            je L_retVC14_23_libcmt
            cmp 0x11(%rax), %rbx           // (vcruntime140.lib, 14.23.x.x)
            je L_retVC14_23_msvcrt

            mov $$0xccc348c48348c033, %rbx // xor eax,eax; add rsp,48h; ret; int 3
            cmp 0x2d(%rax), %rbx           // (libcmtd.lib, 14.23.x.x)
            je L_retVC14_23_libcmtd

            jmp L_term

        L_retVC14_23_msvcrt:               // vcruntime140.dll 14.23.28105
            lea 0x1b(%rax), %rax
            mov 0x38(%rdx), %rbx           // restore RBX from stack
            jmp L_rbxRestored

        L_retVC14_23_libcmt:               // libcmt.lib 14.23.28105
            lea -0x21(%rax), %rax
            mov 0x38(%rdx), %rbx           // restore RBX from stack
            jmp L_rbxRestored

        L_retVC14_23_libcmtd:              // libcmtd.lib/vcruntime140d.dll 14.23.28105
            lea 0x2f(%rax), %rax
            jmp L_rbxRestored              // rbx not saved

        L_retVC14_14:                      // (vcruntime140.dll 14.14.x.y)
            lea 0x20(%rax), %rax
            jmp L_retContinue
        L_retVC14_16:                      // vcruntime140 14.16.27012.6
            lea 0x16(%rax), %rax
            jmp L_retContinue
        L_retVC14_11:                      // vcruntime140 14.11.25415.0 or earlier
            lea 0x2E(%rax), %rax
        L_retContinue:                     // vcruntime140 14.00.23026.0 or later?
            cmpw $$0x8348, (%rax)          // add rsp,nn?
            je L_xorSkipped

            inc %rax                       // vcruntime140 earlier than 14.00.23026.0?
            jmp L_xorSkipped

        L_retFound:
            lea -19(%rax), %rax
            jmp L_xorSkipped

        L_switchFound2:
            dec %rax
        L_switchFound:
            movsbq -18(%rax), %rbx         // follow jump
            lea -17(%rax,%rbx), %rax

            cmpw $$0xC033, (%rax)          // xor EAX,EAX?
            jne L_term

            add $$2, %rax
        L_xorSkipped:
            mov %rdx, %rbx                 // extract UCRT marker from EDX
            and $$~1, %rdx
            and $$1, %rbx

            cmovnz -8(%rdx), %rbx          // restore RBX (pushed inside terminate())
            cmovz (%rsp), %rbx             // RBX not changed in terminate inside UCRT 10.0.14393.0

        L_rbxRestored:
            lea 8(%rdx), %rsp
            push %rax                      // new return after setting return value in __frameUnwindHandler

            call __processing_throw
            movq $$1, (%rax)

            //add $$0x68, %rsp             // TODO: needs to be verified for different CRT builds
            mov $$1, %rax                  // return EXCEPTION_EXECUTE_HANDLER
            ret

        L_term:
            call _D3ldc7eh_msvc22tlsOldTerminateHandlerFNbNiNfZPFZv
            pop %rbx
            cmp $$0, %rax
            je L_ret
            jmp *%rax
        L_ret:
            ret`,
            "~{memory},~{flags},~{rbp},~{rsp},~{rax},~{rbx},~{rdx}"
        );
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

    version (Win64) ehHeap.initialize(0x10000);

    // preallocate type descriptors likely to be needed
    getThrowInfo(typeid(Exception));
    // better not have to allocate when this is thrown:
    getThrowInfo(typeid(OutOfMemoryError));
}

///////////////////////////////////////////////////////////////
version (Win32)
{
    ImgPtr!T eh_malloc(T)(size_t size = T.sizeof)
    {
        return cast(T*) xmalloc(size);
    }

    T* toPointer(T)(T* imgPtr)
    {
        return imgPtr;
    }
}
else
{
    /**
    * Heap dedicated for CatchableTypeArray/CatchableType/TypeDescriptor
    * structs of cached _ThrowInfos.
    * The heap is used to keep these structs tightly together, as they are
    * referenced via 32-bit offsets from a common base. We simply use the
    * heap's start as base (instead of the actual image base), and malloc()
    * returns an offset.
    * The allocated structs are all cached and never released, so this heap
    * can only grow. The offsets remain constant after a grow, so it's only
    * the base which may change.
    */
    struct EHHeap
    {
        void* base;
        size_t capacity;
        size_t length;

        void initialize(size_t initialCapacity)
        {
            base = xmalloc(initialCapacity);
            capacity = initialCapacity;
            length = size_t.sizeof; // don't use offset 0, it has a special meaning
        }

        size_t malloc(size_t size)
        {
            auto offset = length;
            enum alignmentMask = size_t.sizeof - 1;
            auto newLength = (length + size + alignmentMask) & ~alignmentMask;
            auto newCapacity = capacity;
            while (newLength > newCapacity)
                newCapacity *= 2;
            if (newCapacity != capacity)
            {
                auto newBase = xmalloc(newCapacity);
                newBase[0 .. length] = base[0 .. length];
                // old base just leaks, could be used by exceptions still in flight
                base = newBase;
                capacity = newCapacity;
            }
            length = newLength;
            return offset;
        }
    }

    __gshared EHHeap ehHeap;

    ImgPtr!T eh_malloc(T)(size_t size = T.sizeof)
    {
        return ImgPtr!T(cast(uint) ehHeap.malloc(size));
    }

    // NB: The returned pointer may be invalidated by a consequent grow of ehHeap!
    T* toPointer(T)(ImgPtr!T imgPtr)
    {
        return cast(T*) (ehHeap.base + imgPtr.offset);
    }
}
