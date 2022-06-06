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

#pragma once

#include "dmd/declaration.h"
#include "dmd/mtype.h"
#include "gen/dvalue.h"
#include "gen/llvm.h"

///////////////////////////////////////////////////////////
// Nested variable and context helpers
///////////////////////////////////////////////////////////

class FuncGenState;

/// Creates the nested struct alloca for the current function (if there are any
/// nested references to its variables).
void DtoCreateNestedContext(FuncGenState &funcGen);

/// Resolves the nested context for classes and structs with arbitrary nesting.
void DtoResolveNestedContext(const Loc &loc, AggregateDeclaration *decl,
                             LLValue *value);

/// Gets the context value for a call to a nested function or creating a nested
/// class or struct with arbitrary nesting.
llvm::Value *DtoNestedContext(const Loc &loc, Dsymbol *sym);

/// Gets the DValue of a nested variable with arbitrary nesting.
DValue *DtoNestedVariable(const Loc &loc, Type *astype, VarDeclaration *vd,
                          bool byref = false);
