//===-- gen/inlinemlir.cpp - Inline MLIR implementation-----------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation for the LDC-specific MLIR inline IR feature.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "dmd/arraytypes.h"

class DValue;
class FuncDeclaration;
struct Loc;

namespace llvm {
class Function;
class Value;
}

/// Check LDC_inline_mlir pragma declaration is valid
/// Will call fatal() in case of errors
void DtoCheckInlineMLIRPragma(Identifier *ident, Dsymbol *s);

DValue *DtoInlineMLIR(Loc loc, FuncDeclaration *fdecl, Expressions *arguments, llvm::Value *sretPointer = nullptr);
