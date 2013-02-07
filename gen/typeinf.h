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

struct TypeInfoDeclaration;
void DtoResolveTypeInfo(TypeInfoDeclaration* tid);
void DtoDeclareTypeInfo(TypeInfoDeclaration* tid);
void DtoConstInitTypeInfo(TypeInfoDeclaration* tid);

#endif
