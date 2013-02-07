//===-- gen/classes.h - D class code generation -----------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Functions for generating LLVM types and init/TypeInfo/etc. values from D
// class declarations and handling class instance values.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_CLASSES_H
#define LDC_GEN_CLASSES_H

#include "gen/structs.h"

struct ClassDeclaration;
struct CtorDeclaration;
struct FuncDeclaration;
struct NewExp;
struct TypeClass;

/// Resolves the llvm type for a class declaration
void DtoResolveClass(ClassDeclaration* cd);

/// Provides the llvm declaration for a class declaration
void DtoDeclareClass(ClassDeclaration* cd);

/// Constructs the constant initializer for a class declaration
void DtoConstInitClass(ClassDeclaration* cd);

/// Provides the llvm definition for a class declaration
void DtoDefineClass(ClassDeclaration* cd);

/// Builds the initializer of cd's ClassInfo.
/// FIXME: this should be put into IrStruct and eventually IrClass.
llvm::Constant* DtoDefineClassInfo(ClassDeclaration* cd);

DValue* DtoNewClass(Loc loc, TypeClass* type, NewExp* newexp);
void DtoInitClass(TypeClass* tc, llvm::Value* dst);
DValue* DtoCallClassCtor(TypeClass* type, CtorDeclaration* ctor, Array* arguments, llvm::Value* mem);
void DtoFinalizeClass(llvm::Value* inst);

DValue* DtoCastClass(DValue* val, Type* to);
DValue* DtoDynamicCastObject(DValue* val, Type* to);

DValue* DtoCastInterfaceToObject(DValue* val, Type* to);
DValue* DtoDynamicCastInterface(DValue* val, Type* to);

llvm::Value* DtoIndexClass(llvm::Value* src, ClassDeclaration* sd, VarDeclaration* vd);

llvm::Value* DtoVirtualFunctionPointer(DValue* inst, FuncDeclaration* fdecl, char* name);

#endif
