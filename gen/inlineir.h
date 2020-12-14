//===-- gen/irstate.h - Inline IR implementation-----------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation for the LDC-specific LLVM inline IR feature.
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

/// Check LDC_inline_ir pragma declaration is valid
/// Will call fatal() in case of errors
void DtoCheckInlineIRPragma(Identifier *ident, Dsymbol *s);

DValue *DtoInlineIRExpr(const Loc &loc, FuncDeclaration *fdecl,
                        Expressions *arguments,
                        llvm::Value *sretPointer = nullptr);
