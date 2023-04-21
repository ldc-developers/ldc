//===-- gen/binops.h - Binary numeric operations ----------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/globals.h"
#include "dmd/tokens.h"

class Expression;
class Type;
struct Loc;

namespace llvm {
class Value;
}

class DValue;

// lhs + rhs
DValue *binAdd(const Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs - rhs
DValue *binMin(const Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs * rhs
DValue *binMul(const Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs / rhs
DValue *binDiv(const Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs % rhs
DValue *binMod(const Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);

// lhs & rhs
DValue *binAnd(const Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs | rhs
DValue *binOr(const Loc &loc, Type *type, DValue *lhs, Expression *rhs,
              bool loadLhsAfterRhs = false);
// lhs ^ rhs
DValue *binXor(const Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs << rhs
DValue *binShl(const Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs >> rhs
DValue *binShr(const Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs >>> rhs
DValue *binUshr(const Loc &loc, Type *type, DValue *lhs, Expression *rhs,
                bool loadLhsAfterRhs = false);

llvm::Value *DtoBinNumericEquals(const Loc &loc, DValue *lhs, DValue *rhs,
                                 EXP op);
llvm::Value *DtoBinFloatsEquals(const Loc &loc, DValue *lhs, DValue *rhs,
                                EXP op);
llvm::Value *mergeVectorEquals(llvm::Value *resultsVector, EXP op);
