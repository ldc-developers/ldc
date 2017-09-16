//===-- gen/aa.h - Associative array codegen helpers ------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Helpers for generating calls to associative array runtime functions.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_AA_H
#define LDC_GEN_AA_H

#include "tokens.h"

enum TOK;
class DValue;
class DLValue;
struct Loc;
class Type;
namespace llvm {
class Value;
}

DLValue *DtoAAIndex(Loc &loc, Type *type, DValue *aa, DValue *key, bool lvalue);
DValue *DtoAAIn(Loc &loc, Type *type, DValue *aa, DValue *key);
DValue *DtoAARemove(Loc &loc, DValue *aa, DValue *key);
llvm::Value *DtoAAEquals(Loc &loc, TOK op, DValue *l, DValue *r);

#endif // LDC_GEN_AA_H
