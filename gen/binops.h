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
DValue *binAdd(Loc loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs - rhs
DValue *binMin(Loc loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs * rhs
DValue *binMul(Loc loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs / rhs
DValue *binDiv(Loc loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs % rhs
DValue *binMod(Loc loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);

// lhs & rhs
DValue *binAnd(Loc loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs | rhs
DValue *binOr(Loc loc, Type *type, DValue *lhs, Expression *rhs,
              bool loadLhsAfterRhs = false);
// lhs ^ rhs
DValue *binXor(Loc loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs << rhs
DValue *binShl(Loc loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs >> rhs
DValue *binShr(Loc loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs = false);
// lhs >>> rhs
DValue *binUshr(Loc loc, Type *type, DValue *lhs, Expression *rhs,
                bool loadLhsAfterRhs = false);

llvm::Value *DtoBinNumericEquals(Loc loc, DValue *lhs, DValue *rhs,
                                 EXP op);
llvm::Value *DtoBinFloatsEquals(Loc loc, DValue *lhs, DValue *rhs,
                                EXP op);
llvm::Value *mergeVectorEquals(llvm::Value *resultsVector, EXP op);
