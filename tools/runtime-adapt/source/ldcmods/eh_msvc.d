//===-- tools/runtime-adapt/source/ldcmods/eh_msvc.d --------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ldc/eh_msvc.d ← gen/runtime.cpp (_d_eh_enter_catch / personality).
//
//===----------------------------------------------------------------------===//

module ldcmods.eh_msvc;

import compilerparse;
import ldcmods.common;

bool wantEhMsvc(const CompilerModel)
{
    return true;
}

string renderEhMsvc(const CompilerModel, string tag)
{
    return banner(tag) ~ q"D
module ldc.eh_msvc;

version (CRuntime_Microsoft):

// MSVC C++ EH ABI types (gen/runtime.cpp _d_eh_enter_catch / personality).
version (Win64)
    struct ImgPtr(T) { uint offset; }
else
    alias ImgPtr(T) = T*;

alias PMFN = ImgPtr!(void function(void*));

struct TypeDescriptor
{
    version (_RTTI)
        const void* pVFTable;
    else
        uint hash;
    void* spare;
    char[1] name;
}

struct PMD
{
    int mdisp;
    int pdisp;
    int vdisp;
}

struct CatchableType
{
    uint properties;
    ImgPtr!TypeDescriptor pType;
    PMD thisDisplacement;
    int sizeOrOffset;
    PMFN copyFunction;
}

struct CatchableTypeArray
{
    int nCatchableTypes;
    ImgPtr!CatchableType[1] arrayOfCatchableTypes;
}

struct _ThrowInfo
{
    uint attributes;
    PMFN pmfnUnwind;
    PMFN pForwardCompat;
    ImgPtr!CatchableTypeArray pCatchableTypeArray;
}

struct CxxExceptionInfo
{
    size_t Magic;
    Throwable* pThrowable;
    _ThrowInfo* ThrowInfo;
    version (Win64) void* ImgBase;
}

struct ExceptionStack
{
nothrow:
    size_t _length;
    Throwable* _p;
    size_t _cap;
    @property size_t length() const { return _length; }
    @property bool empty() const { return !_length; }
}

struct FiberContext
{
    ExceptionStack exceptionStack;
    void* currentException;
    void* currentExceptionContext;
    int processingContext;
}

extern (C) int _d_isbaseof(ClassInfo oc, ClassInfo c);
extern (C) Throwable _d_eh_enter_catch(void* exception, ClassInfo catchType = null);
extern (C) int _d_eh_personality();
extern (C) void _d_eh_resume_unwind(void* ptr);
extern (C) void _d_throw_exception(Throwable throwable);
extern (C) void* _d_eh_swapContext(FiberContext* newContext) nothrow;
D";
}
