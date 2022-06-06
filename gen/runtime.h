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

#pragma once

namespace llvm {
class Function;
class GlobalVariable;
class Module;
}

struct Loc;
class FuncDeclaration;
class Type;

// D runtime support helpers
bool initRuntime();
void freeRuntime();

llvm::Function *getRuntimeFunction(const Loc &loc, llvm::Module &target,
                                   const char *name);

llvm::Function *getCAssertFunction(const Loc &loc, llvm::Module &target);

llvm::Function *getUnwindResumeFunction(const Loc &loc, llvm::Module &target);

void emitInstrumentationFnEnter(FuncDeclaration *decl);
void emitInstrumentationFnLeave(FuncDeclaration *decl);

Type *getObjectType();
Type *getTypeInfoType();
Type *getEnumTypeInfoType();
Type *getPointerTypeInfoType();
Type *getArrayTypeInfoType();
Type *getStaticArrayTypeInfoType();
Type *getAssociativeArrayTypeInfoType();
Type *getVectorTypeInfoType();
Type *getFunctionTypeInfoType();
Type *getDelegateTypeInfoType();
Type *getClassInfoType();
Type *getInterfaceTypeInfoType();
Type *getStructTypeInfoType();
Type *getTupleTypeInfoType();
Type *getConstTypeInfoType();
Type *getInvariantTypeInfoType();
Type *getSharedTypeInfoType();
Type *getInoutTypeInfoType();
Type *getThrowableType();
Type *getCppTypeInfoPtrType();
Type *getModuleInfoType();
