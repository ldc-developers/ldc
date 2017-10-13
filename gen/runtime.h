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

namespace llvm {
class Function;
class GlobalVariable;
class Module;
}

struct Loc;

// D runtime support helpers
bool initRuntime();
void freeRuntime();

llvm::Function *getRuntimeFunction(const Loc &loc, llvm::Module &target,
                                          const char *name);

llvm::GlobalVariable *
getRuntimeGlobal(const Loc &loc, llvm::Module &target, const char *name);
llvm::Function *getCAssertFunction(const Loc &loc, llvm::Module &target);

#endif // LDC_GEN_RUNTIME_H
