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

#ifndef LDC_GEN_STRUCTS_H
#define LDC_GEN_STRUCTS_H

#include "lexer.h"
#include <vector>

struct DValue;
struct StructDeclaration;
struct StructInitializer;
struct Type;
struct VarDeclaration;
namespace llvm
{
    class Constant;
    class Type;
    class Value;
}

/**
 * Sets up codegen metadata and emits global data (.init, etc.), if needed.
 *
 * Has no effect on already resolved struct declarations.
 */
void DtoResolveStruct(StructDeclaration* sd);

/// Build constant struct initializer.
llvm::Constant* DtoConstStructInitializer(StructInitializer* si);

/// Returns a boolean=true if the two structs are equal.
llvm::Value* DtoStructEquals(TOK op, DValue* lhs, DValue* rhs);

/// index a struct one level
llvm::Value* DtoIndexStruct(llvm::Value* src, StructDeclaration* sd, VarDeclaration* vd);

/// Return the type returned by DtoUnpaddedStruct called on a value of the
/// specified type.
/// Union types will get expanded into a struct, with a type for each member.
llvm::Type* DtoUnpaddedStructType(Type* dty);

/// Return the struct value represented by v without the padding fields.
/// Unions will be expanded, with a value for each member.
/// Note: v must be a pointer to a struct, but the return value will be a
///       first-class struct value.
llvm::Value* DtoUnpaddedStruct(Type* dty, llvm::Value* v);

/// Undo the transformation performed by DtoUnpaddedStruct, writing to lval.
void DtoPaddedStruct(Type* dty, llvm::Value* v, llvm::Value* lval);

#endif
