//===-- gen/structs.h - D struct codegen ------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Functions for D struct codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMD_GEN_STRUCTS_H
#define LLVMD_GEN_STRUCTS_H

struct StructInitializer;

/**
 * Sets up codegen metadata and emits global data (.init, etc.), if needed.
 *
 * Has no effect on already resolved struct declarations.
 */
void DtoResolveStruct(StructDeclaration* sd);

/// Build constant struct initializer.
LLConstant* DtoConstStructInitializer(StructInitializer* si);

/// Build values for a struct literal.
std::vector<LLConstant*> DtoStructLiteralValues(const StructDeclaration* sd,
                                                const std::vector<LLConstant*>& inits);

/// Returns a boolean=true if the two structs are equal.
LLValue* DtoStructEquals(TOK op, DValue* lhs, DValue* rhs);

/// index a struct one level
LLValue* DtoIndexStruct(LLValue* src, StructDeclaration* sd, VarDeclaration* vd);

/// Return the type returned by DtoUnpaddedStruct called on a value of the
/// specified type.
/// Union types will get expanded into a struct, with a type for each member.
LLType* DtoUnpaddedStructType(Type* dty);

/// Return the struct value represented by v without the padding fields.
/// Unions will be expanded, with a value for each member.
/// Note: v must be a pointer to a struct, but the return value will be a
///       first-class struct value.
LLValue* DtoUnpaddedStruct(Type* dty, LLValue* v);

/// Undo the transformation performed by DtoUnpaddedStruct, writing to lval.
void DtoPaddedStruct(Type* dty, LLValue* v, LLValue* lval);

#endif
