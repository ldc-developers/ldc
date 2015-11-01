//===-- gen/nested.h - Nested context handling ------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Functions for creating nested contexts for nested D types/functions and
// extracting the values from them.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_NESTED_H
#define LDC_GEN_NESTED_H

#include "declaration.h"
#include "mtype.h"
#include "gen/dvalue.h"
#include "gen/llvm.h"

///////////////////////////////////////////////////////////
// Nested variable and context helpers
///////////////////////////////////////////////////////////

/// Creates the context value for a nested function.
void DtoCreateNestedContext(FuncDeclaration *fd);

/// Resolves the nested context for classes and structs with arbitrary nesting.
void DtoResolveNestedContext(Loc &loc, AggregateDeclaration *decl,
                             LLValue *value);

/// Gets the context value for a call to a nested function or creating a nested
/// class or struct with arbitrary nesting.
llvm::Value *DtoNestedContext(Loc &loc, Dsymbol *sym);

/// Gets the DValue of a nested variable with arbitrary nesting.
DValue *DtoNestedVariable(Loc &loc, Type *astype, VarDeclaration *vd,
                          bool byref = false);

#endif
