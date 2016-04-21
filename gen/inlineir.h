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

#ifndef LDC_GEN_INLINEIR_H
#define LDC_GEN_INLINEIR_H

#include "ddmd/arraytypes.h"

class DValue;
class FuncDeclaration;
struct Loc;

namespace llvm {
class Function;
}

DValue *DtoInlineIRExpr(Loc &loc, FuncDeclaration *fdecl,
                        Expressions *arguments);

#endif
