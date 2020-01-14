//===-- binops.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/binops.h"

#include "dmd/declaration.h"
#include "dmd/expression.h"
#include "gen/complex.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"

//////////////////////////////////////////////////////////////////////////////

dinteger_t undoStrideMul(Loc &loc, Type *t, dinteger_t offset) {
  assert(t->ty == Tpointer);
  d_uns64 elemSize = t->nextOf()->size(loc);
  assert((offset % elemSize) == 0 &&
         "Expected offset by an integer amount of elements");

  return offset / elemSize;
}

//////////////////////////////////////////////////////////////////////////////

namespace {
struct RVals {
  DRValue *lhs, *rhs;
};

RVals evalSides(DValue *lhs, Expression *rhs, bool loadLhsAfterRhs) {
  RVals rvals;

  if (!loadLhsAfterRhs) {
    rvals.lhs = lhs->getRVal();
    rvals.rhs = toElem(rhs)->getRVal();
  } else {
    rvals.rhs = toElem(rhs)->getRVal();
    rvals.lhs = lhs->getRVal();
  }

  return rvals;
}

/// Tries to remove a MulExp by a constant value of baseSize from e. Returns
/// NULL if not possible.
Expression *extractNoStrideInc(Expression *e, d_uns64 baseSize, bool &negate) {
  MulExp *mul;
  while (true) {
    if (auto ne = e->isNegExp()) {
      negate = !negate;
      e = ne->e1;
      continue;
    }

    if (auto me = e->isMulExp()) {
      mul = me;
      break;
    }

    return nullptr;
  }

  if (!mul->e2->isConst()) {
    return nullptr;
  }
  dinteger_t stride = mul->e2->toInteger();

  if (stride != baseSize) {
    return nullptr;
  }

  return mul->e1;
}

DValue *emitPointerOffset(Loc loc, DValue *base, Expression *offset,
                          bool negateOffset, Type *resultType,
                          bool loadLhsAfterRhs) {
  // The operand emitted by the frontend is in units of bytes, and not
  // pointer elements. We try to undo this before resorting to
  // temporarily bitcasting the pointer to i8.

  LLValue *llBase = nullptr;
  LLValue *llOffset = nullptr;
  LLValue *llResult = nullptr;

  if (offset->isConst()) {
    llBase = DtoRVal(base);
    dinteger_t byteOffset = offset->toInteger();
    if (byteOffset == 0) {
      llResult = llBase;
    } else {
      llOffset = DtoConstSize_t(undoStrideMul(loc, base->type, byteOffset));
    }
  } else {
    Expression *noStrideInc = extractNoStrideInc(
        offset, base->type->nextOf()->size(loc), negateOffset);
    auto rvals =
        evalSides(base, noStrideInc ? noStrideInc : offset, loadLhsAfterRhs);
    llBase = DtoRVal(rvals.lhs);
    llOffset = DtoRVal(rvals.rhs);
    if (!noStrideInc) // byte offset => cast base to i8*
      llBase = DtoBitCast(llBase, getVoidPtrType());
  }

  if (!llResult) {
    if (negateOffset)
      llOffset = gIR->ir->CreateNeg(llOffset);
    llResult = DtoGEP1(llBase, llOffset);
  }

  return new DImValue(resultType, DtoBitCast(llResult, DtoType(resultType)));
}

// LDC issue #2537 / DMD issue #18317: associative arrays can be
// added/subtracted via `typeof(null)` (implicitly cast to the AA type).
// If the specified type is an AA type, this function makes sure one operand is
// a null constant and returns the other operand (AA) as new DImValue.
// Returns null if type is not an AA.
DValue *isAssociativeArrayAndNull(Type *type, LLValue *lhs, LLValue *rhs) {
  if (type->ty != Taarray)
    return nullptr;

  if (auto constantL = isaConstant(lhs)) {
    if (constantL->isNullValue())
      return new DImValue(type, rhs);
  };
  if (auto constantR = isaConstant(rhs)) {
    if (constantR->isNullValue())
      return new DImValue(type, lhs);
  }

  llvm_unreachable(
      "associative array addition/subtraction without null operand");
}
}

//////////////////////////////////////////////////////////////////////////////

DValue *binAdd(Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs) {
  Type *lhsType = lhs->type->toBasetype();
  Type *rhsType = rhs->type->toBasetype();

  if (lhsType != rhsType && lhsType->ty == Tpointer && rhsType->isintegral()) {
    Logger::println("Adding integer to pointer");
    return emitPointerOffset(loc, lhs, rhs, false, type, loadLhsAfterRhs);
  }

  auto rvals = evalSides(lhs, rhs, loadLhsAfterRhs);

  if (type->ty == Tnull)
    return DtoNullValue(type, loc);
  if (type->iscomplex())
    return DtoComplexAdd(loc, type, rvals.lhs, rvals.rhs);

  LLValue *l = DtoRVal(DtoCast(loc, rvals.lhs, type));
  LLValue *r = DtoRVal(DtoCast(loc, rvals.rhs, type));

  if (auto aa = isAssociativeArrayAndNull(type, l, r))
    return aa;

  LLValue *res = (type->isfloating() ? gIR->ir->CreateFAdd(l, r)
                                     : gIR->ir->CreateAdd(l, r));

  return new DImValue(type, res);
}

//////////////////////////////////////////////////////////////////////////////

DValue *binMin(Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs) {
  Type *lhsType = lhs->type->toBasetype();
  Type *rhsType = rhs->type->toBasetype();

  if (lhsType != rhsType && lhsType->ty == Tpointer && rhsType->isintegral()) {
    Logger::println("Subtracting integer from pointer");
    return emitPointerOffset(loc, lhs, rhs, true, type, loadLhsAfterRhs);
  }

  auto rvals = evalSides(lhs, rhs, loadLhsAfterRhs);

  if (lhsType->ty == Tpointer && rhsType->ty == Tpointer) {
    LLValue *l = DtoRVal(rvals.lhs);
    LLValue *r = DtoRVal(rvals.rhs);
    LLType *llSizeT = DtoSize_t();
    l = gIR->ir->CreatePtrToInt(l, llSizeT);
    r = gIR->ir->CreatePtrToInt(r, llSizeT);
    LLValue *diff = gIR->ir->CreateSub(l, r);
    LLType *llType = DtoType(type);
    if (diff->getType() != llType)
      diff = gIR->ir->CreateIntToPtr(diff, llType);
    return new DImValue(type, diff);
  }

  if (type->ty == Tnull)
    return DtoNullValue(type, loc);
  if (type->iscomplex())
    return DtoComplexMin(loc, type, rvals.lhs, rvals.rhs);

  LLValue *l = DtoRVal(DtoCast(loc, rvals.lhs, type));
  LLValue *r = DtoRVal(DtoCast(loc, rvals.rhs, type));

  if (auto aa = isAssociativeArrayAndNull(type, l, r))
    return aa;

  LLValue *res = (type->isfloating() ? gIR->ir->CreateFSub(l, r)
                                     : gIR->ir->CreateSub(l, r));

  return new DImValue(type, res);
}

//////////////////////////////////////////////////////////////////////////////

DValue *binMul(Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs) {
  auto rvals = evalSides(lhs, rhs, loadLhsAfterRhs);

  if (type->iscomplex())
    return DtoComplexMul(loc, type, rvals.lhs, rvals.rhs);

  LLValue *l = DtoRVal(DtoCast(loc, rvals.lhs, type));
  LLValue *r = DtoRVal(DtoCast(loc, rvals.rhs, type));
  LLValue *res = (type->isfloating() ? gIR->ir->CreateFMul(l, r)
                                     : gIR->ir->CreateMul(l, r));

  return new DImValue(type, res);
}

//////////////////////////////////////////////////////////////////////////////

DValue *binDiv(Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs) {
  auto rvals = evalSides(lhs, rhs, loadLhsAfterRhs);

  if (type->iscomplex())
    return DtoComplexDiv(loc, type, rvals.lhs, rvals.rhs);

  LLValue *l = DtoRVal(DtoCast(loc, rvals.lhs, type));
  LLValue *r = DtoRVal(DtoCast(loc, rvals.rhs, type));
  LLValue *res;
  if (type->isfloating()) {
    res = gIR->ir->CreateFDiv(l, r);
  } else if (!isLLVMUnsigned(type)) {
    res = gIR->ir->CreateSDiv(l, r);
  } else {
    res = gIR->ir->CreateUDiv(l, r);
  }

  return new DImValue(type, res);
}

//////////////////////////////////////////////////////////////////////////////

DValue *binMod(Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs) {
  auto rvals = evalSides(lhs, rhs, loadLhsAfterRhs);

  if (type->iscomplex())
    return DtoComplexMod(loc, type, rvals.lhs, rvals.rhs);

  LLValue *l = DtoRVal(DtoCast(loc, rvals.lhs, type));
  LLValue *r = DtoRVal(DtoCast(loc, rvals.rhs, type));
  LLValue *res;
  if (type->isfloating()) {
    res = gIR->ir->CreateFRem(l, r);
  } else if (!isLLVMUnsigned(type)) {
    res = gIR->ir->CreateSRem(l, r);
  } else {
    res = gIR->ir->CreateURem(l, r);
  }

  return new DImValue(type, res);
}

//////////////////////////////////////////////////////////////////////////////

namespace {
DValue *binBitwise(llvm::Instruction::BinaryOps binOp, Loc &loc, Type *type,
                   DValue *lhs, Expression *rhs, bool loadLhsAfterRhs) {
  auto rvals = evalSides(lhs, rhs, loadLhsAfterRhs);

  LLValue *l = DtoRVal(DtoCast(loc, rvals.lhs, type));
  LLValue *r = DtoRVal(DtoCast(loc, rvals.rhs, type));
  LLValue *res = llvm::BinaryOperator::Create(binOp, l, r, "", gIR->scopebb());

  return new DImValue(type, res);
}
}

DValue *binAnd(Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs) {
  return binBitwise(llvm::Instruction::And, loc, type, lhs, rhs,
                    loadLhsAfterRhs);
}

DValue *binOr(Loc &loc, Type *type, DValue *lhs, Expression *rhs,
              bool loadLhsAfterRhs) {
  return binBitwise(llvm::Instruction::Or, loc, type, lhs, rhs,
                    loadLhsAfterRhs);
}

DValue *binXor(Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs) {
  return binBitwise(llvm::Instruction::Xor, loc, type, lhs, rhs,
                    loadLhsAfterRhs);
}

DValue *binShl(Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs) {
  return binBitwise(llvm::Instruction::Shl, loc, type, lhs, rhs,
                    loadLhsAfterRhs);
}

DValue *binShr(Loc &loc, Type *type, DValue *lhs, Expression *rhs,
               bool loadLhsAfterRhs) {
  auto op = (isLLVMUnsigned(type) ? llvm::Instruction::LShr
                                  : llvm::Instruction::AShr);
  return binBitwise(op, loc, type, lhs, rhs, loadLhsAfterRhs);
}

DValue *binUshr(Loc &loc, Type *type, DValue *lhs, Expression *rhs,
                bool loadLhsAfterRhs) {
  return binBitwise(llvm::Instruction::LShr, loc, type, lhs, rhs,
                    loadLhsAfterRhs);
}

//////////////////////////////////////////////////////////////////////////////

LLValue *DtoBinNumericEquals(Loc &loc, DValue *lhs, DValue *rhs, TOK op) {
  assert(op == TOKequal || op == TOKnotequal || op == TOKidentity ||
         op == TOKnotidentity);
  Type *t = lhs->type->toBasetype();
  assert(t->isfloating());
  Logger::println("numeric equality");

  LLValue *res = nullptr;
  if (t->iscomplex()) {
    Logger::println("complex");
    res = DtoComplexEquals(loc, op, lhs, rhs);
  } else if (t->isfloating()) {
    Logger::println("floating");
    res = DtoBinFloatsEquals(loc, lhs, rhs, op);
  }

  assert(res);
  return res;
}

//////////////////////////////////////////////////////////////////////////////

LLValue *DtoBinFloatsEquals(Loc &loc, DValue *lhs, DValue *rhs, TOK op) {
  LLValue *res = nullptr;
  if (op == TOKequal || op == TOKnotequal) {
    LLValue *l = DtoRVal(lhs);
    LLValue *r = DtoRVal(rhs);
    res = (op == TOKequal ? gIR->ir->CreateFCmpOEQ(l, r)
                          : gIR->ir->CreateFCmpUNE(l, r));
    if (lhs->type->toBasetype()->ty == Tvector) {
      res = mergeVectorEquals(res, op);
    }
  } else {
    const auto cmpop =
        op == TOKidentity ? llvm::ICmpInst::ICMP_EQ : llvm::ICmpInst::ICMP_NE;
    LLValue *sz = DtoConstSize_t(getTypeStoreSize(DtoType(lhs->type)));
    LLValue *val = DtoMemCmp(makeLValue(loc, lhs), makeLValue(loc, rhs), sz);
    res = gIR->ir->CreateICmp(cmpop, val,
                              LLConstantInt::get(val->getType(), 0, false));
  }
  assert(res);
  return res;
}

//////////////////////////////////////////////////////////////////////////////

LLValue *mergeVectorEquals(LLValue *resultsVector, TOK op) {
  // `resultsVector` is a vector of i1 values, the pair-wise results.
  // Bitcast to an integer and check the bits via additional integer
  // comparison.
  const auto sizeInBits = getTypeBitSize(resultsVector->getType());
  LLType *integerType = LLType::getIntNTy(gIR->context(), sizeInBits);
  LLValue *v = DtoBitCast(resultsVector, integerType);

  if (op == TOKequal) {
    // all pairs must be equal for the vectors to be equal
    LLConstant *allEqual = LLConstant::getAllOnesValue(integerType);
    return gIR->ir->CreateICmpEQ(v, allEqual);
  } else if (op == TOKnotequal) {
    // any not-equal pair suffices for the vectors to be not-equal
    LLConstant *noneNotEqual = LLConstantInt::get(integerType, 0);
    return gIR->ir->CreateICmpNE(v, noneNotEqual);
  }

  llvm_unreachable("Unsupported operator.");
  return nullptr;
}
