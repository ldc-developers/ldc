
/* Compiler implementation of the D programming language
 * Copyright (C) 2017-2018 by The D Language Foundation, All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/dlang/dmd/blob/master/src/id.h
 */

#ifndef DMD_ID_H
#define DMD_ID_H

#ifdef __DMC__
#pragma once
#endif /* __DMC__ */

struct Id
{
public:
    static void initialize();

#if IN_LLVM
    static Identifier *___in;
    static Identifier *__int;
    static Identifier *___out;
    static Identifier *__LOCAL_SIZE;
    static Identifier *dollar;
    static Identifier *ptr;
    static Identifier *offset;
    static Identifier *offsetof;
    static Identifier *__c_long;
    static Identifier *__c_ulong;
    static Identifier *__c_longlong;
    static Identifier *__c_ulonglong;
    static Identifier *__c_long_double;
    static Identifier *__switch;
    static Identifier *crt_constructor;
    static Identifier *crt_destructor;
    static Identifier *lib;
    static Identifier *ldc;
    static Identifier *dcompute;
    static Identifier *dcPointer;
    static Identifier *object;
    static Identifier *ensure;
    static Identifier *require;
    static Identifier *xopEquals;
    static Identifier *xopCmp;
    static Identifier *xtoHash;
    static Identifier *empty;
    static Identifier *ctfe;
    static Identifier *_arguments;
    static Identifier *_argptr;
    static Identifier *LDC_intrinsic;
    static Identifier *LDC_global_crt_ctor;
    static Identifier *LDC_global_crt_dtor;
    static Identifier *LDC_no_typeinfo;
    static Identifier *LDC_no_moduleinfo;
    static Identifier *LDC_alloca;
    static Identifier *LDC_va_start;
    static Identifier *LDC_va_copy;
    static Identifier *LDC_va_end;
    static Identifier *LDC_va_arg;
    static Identifier *LDC_fence;
    static Identifier *LDC_atomic_load;
    static Identifier *LDC_atomic_store;
    static Identifier *LDC_atomic_cmp_xchg;
    static Identifier *LDC_atomic_rmw;
    static Identifier *LDC_verbose;
    static Identifier *LDC_inline_asm;
    static Identifier *LDC_inline_ir;
    static Identifier *LDC_extern_weak;
    static Identifier *LDC_profile_instr;
    static Identifier *dcReflect;
    static Identifier *criticalenter;
    static Identifier *criticalexit;
    static Identifier *attributes;
    static Identifier *udaSection;
    static Identifier *udaOptStrategy;
    static Identifier *udaTarget;
    static Identifier *udaAssumeUsed;
    static Identifier *udaWeak;
    static Identifier *udaAllocSize;
    static Identifier *udaLLVMAttr;
    static Identifier *udaLLVMFastMathFlag;
    static Identifier *udaKernel;
    static Identifier *udaCompute;
    static Identifier *udaDynamicCompile;
    static Identifier *udaDynamicCompileConst;
#endif
};

#endif /* DMD_ID_H */
