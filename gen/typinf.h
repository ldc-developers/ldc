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

#pragma once

struct Scope;
struct Loc;
class Type;
class TypeInfoDeclaration;

namespace llvm {
class GlobalVariable;
}

void DtoResolveTypeInfo(TypeInfoDeclaration *tid);
TypeInfoDeclaration *getOrCreateTypeInfoDeclaration(const Loc &loc, Type *t,
                                                    Scope *sc);
void TypeInfoDeclaration_codegen(TypeInfoDeclaration *decl);

// Adds some metadata for use by optimization passes.
void emitTypeInfoMetadata(llvm::GlobalVariable *typeinfoGlobal, Type *forType);

// defined in dmd/typinf.d:
bool isSpeculativeType(Type *t);
