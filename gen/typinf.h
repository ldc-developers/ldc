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

struct Loc;
class Type;
class TypeInfoDeclaration;

namespace llvm {
class GlobalVariable;
}

TypeInfoDeclaration *getOrCreateTypeInfoDeclaration(const Loc &loc,
                                                    Type *forType);
llvm::GlobalVariable *DtoResolveTypeInfo(TypeInfoDeclaration *tid);

// Adds some metadata for use by optimization passes.
void emitTypeInfoMetadata(llvm::GlobalVariable *typeinfoGlobal, Type *forType);
