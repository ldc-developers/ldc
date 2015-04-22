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

#ifndef LDC_GEN_RUNTIME_H
#define LDC_GEN_RUNTIME_H

namespace llvm
{
    class Function;
    class GlobalVariable;
    class Module;
}

struct Loc;

// D runtime support helpers

bool LLVM_D_InitRuntime();
void LLVM_D_FreeRuntime();

llvm::Function* LLVM_D_GetRuntimeFunction(const Loc &loc, llvm::Module* target, const char* name);

llvm::GlobalVariable* LLVM_D_GetRuntimeGlobal(const Loc &loc, llvm::Module* target, const char* name);

#define _adEq "_adEq2"
#define _adCmp "_adCmp2"

#endif // LDC_GEN_RUNTIME_H
