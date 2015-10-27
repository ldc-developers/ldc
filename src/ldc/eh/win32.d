/**
 * This module implements the runtime-part of LDC exceptions
 * on Windows win32.
 */
module ldc.eh.win32;

version(CRuntime_Microsoft):
version(Win32):

import ldc.eh.common;
import core.sys.windows.windows;
import core.exception : onOutOfMemoryError;
import core.stdc.stdlib : malloc;
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

__gshared TypeDescriptor!16 tdObject    = { 0, null, "_D6object6Object\0" };
__gshared TypeDescriptor!20 tdThrowable = { 0, null, "_D6object9Throwable\0" };
__gshared TypeDescriptor!12 tdException = { 0, null, "_D9Exception\0" };
version(none) {
__gshared CatchableType  ctThrowable = { CT_IsSimpleType, cast(TypeDescriptor!1*) &tdThrowable, { 0, -1, 0 }, 4, null };
__gshared CatchableType  ctException = { CT_IsSimpleType, cast(TypeDescriptor!1*) &tdException, { 0, -1, 0 }, 4, null };
__gshared CatchableTypeArray ctArray = { 2, [ &ctThrowable, &ctException ] };
__gshared _ThrowInfo objectThrowInfo = { 0, null, null, &ctArray };
}

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

    ULONG_PTR[3] ExceptionInformation;
    ExceptionInformation[0] = EH_MAGIC_NUMBER1;
    ExceptionInformation[1] = cast(ULONG_PTR) cast(void*) &e;
    ExceptionInformation[2] = cast(ULONG_PTR) getThrowInfo(ti);

    RaiseException(STATUS_MSC_EXCEPTION, EXCEPTION_NONCONTINUABLE, 3, ExceptionInformation.ptr);
}

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

    static size_t mangledNameLength(string s)
    {
        size_t len = 2 + s.length; // "_D" + identifier + 1 digit per dot
        for (size_t q = 0; q < s.length; )
        {
            size_t p = q;
            while (p < s.length && s.ptr[p] != '.')
                p++;
            for (size_t r = p - q; r >= 10; r /= 10) // add digits for length >= 10
                len++;
            q = p + 1;
        }
        return len;
    }

    static void mangleName(string s, char* buf)
    {
        *buf++ = '_';
        *buf++ = 'D';
        for (size_t q = 0; q < s.length; )
        {
            size_t p = q;
            while (p < s.length && s.ptr[p] != '.')
                p++;
            size_t digits = 10;
            size_t len = p - q;
            for ( ; len >= digits; digits *= 10) {}
            for (digits /= 10; digits > 1; digits /= 10)
            {
                size_t dig = len / digits;
                *buf++ = cast(char)('0' + dig);
                len -= dig * digits;
            }
            *buf++ = cast(char)('0' + len);
            memcpy(buf, s.ptr + q, p - q);
            buf += p - q;
            q = p + 1;
        }
        *buf = 0;
    }

    size_t mangledLength = mangledNameLength(ti.name) + 1;
    size_t sz = TypeDescriptor!1.sizeof + mangledLength;
    auto td = cast(TypeDescriptor!1*) malloc(sz);
    if (!td)
        onOutOfMemoryError();

    td.hash = 0;
    td.spare = null;
    mangleName(ti.name, td.name.ptr);

    CatchableType ct = { CT_IsSimpleType, td, { 0, -1, 0 }, 4, null };
    catchableHashtab[ti] = ct;
    return ti in catchableHashtab;
}

void msvc_eh_init()
{
    throwInfoMutex = new Mutex;

    // Exception has a special mangling that's not reflected in the name

    CatchableType  ctObject = { CT_IsSimpleType, cast(TypeDescriptor!1*) &tdObject, { 0, -1, 0 }, 4, null };
    catchableHashtab[typeid(Object)] = ctObject;
    CatchableType  ctThrowable = { CT_IsSimpleType, cast(TypeDescriptor!1*) &tdThrowable, { 0, -1, 0 }, 4, null };
    catchableHashtab[typeid(Throwable)] = ctThrowable;
    CatchableType  ctException = { CT_IsSimpleType, cast(TypeDescriptor!1*) &tdException, { 0, -1, 0 }, 4, null };
    catchableHashtab[typeid(Exception)] = ctException;
}

shared static this()
{
    // should be called from rt_init
    msvc_eh_init();
}
