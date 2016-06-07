//===-- gen/typinf.h - TypeInfo declaration codegen -------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Codegen for the TypeInfo types/constants required by the D run-time type
// information system.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_TYPEINF_H
#define LDC_GEN_TYPEINF_H

struct IRState;
struct Scope;
class Type;
class TypeInfoDeclaration;

void DtoResolveTypeInfo(TypeInfoDeclaration *tid);
TypeInfoDeclaration *getOrCreateTypeInfoDeclaration(Type *t, Scope *sc);
void TypeInfoDeclaration_codegen(TypeInfoDeclaration *decl, IRState *p);
void TypeInfoClassDeclaration_codegen(TypeInfoDeclaration *decl, IRState *p);
Type *getTypeInfoType(Type *t, Scope *sc);

#endif
