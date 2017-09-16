//===-- gen/typinf.d - TypeInfo declaration codegen ---------------*- D -*-===//
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

module gen.typinf;

import ddmd.mtype;
import ddmd.dscope;

//class TypeInfoDeclaration;

//extern (C++) void DtoResolveTypeInfo(TypeInfoDeclaration *tid);
//extern (C++) TypeInfoDeclaration *getOrCreateTypeInfoDeclaration(Type *t, Scope *sc);
//extern (C++) void TypeInfoDeclaration_codegen(TypeInfoDeclaration *decl, IRState *p);
extern (C++) Type getTypeInfoType(Type t, Scope* sc);

