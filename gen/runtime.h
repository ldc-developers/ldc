//===-- gen/runtime.h - D runtime function handlers -------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Code for handling the compiler support functions from the D runtime library.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_RUNTIME_H_
#define LDC_GEN_RUNTIME_H_

namespace llvm
{
    class Function;
    class GlobalVariable;
    class Module;
}

// D runtime support helpers

bool LLVM_D_InitRuntime();
void LLVM_D_FreeRuntime();

llvm::Function* LLVM_D_GetRuntimeFunction(llvm::Module* target, const char* name);

llvm::GlobalVariable* LLVM_D_GetRuntimeGlobal(llvm::Module* target, const char* name);

#if DMDV1
#define _d_allocclass "_d_allocclass"
#define _adEq "_adEq"
#define _adCmp "_adCmp"
#else
#define _d_allocclass "_d_newclass"
#define _adEq "_adEq2"
#define _adCmp "_adCmp2"
#endif

#endif // LDC_GEN_RUNTIME_H_
