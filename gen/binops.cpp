//===-- binops.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#include "declaration.h"
#include "gen/complex.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"

//////////////////////////////////////////////////////////////////////////////

DImValue *DtoBinAdd(DRValue *lhs, DRValue *rhs) {
  Type *t = lhs->type;
  LLValue *l = DtoRVal(lhs);
  LLValue *r = DtoRVal(rhs);

  LLValue *res;
  if (t->isfloating()) {
    res = gIR->ir->CreateFAdd(l, r);
  } else {
    res = gIR->ir->CreateAdd(l, r);
  }

  return new DImValue(t, res);
}

//////////////////////////////////////////////////////////////////////////////

DImValue *DtoBinSub(DRValue *lhs, DRValue *rhs) {
  Type *t = lhs->type;
  LLValue *l = DtoRVal(lhs);
  LLValue *r = DtoRVal(rhs);

  LLValue *res;
  if (t->isfloating()) {
    res = gIR->ir->CreateFSub(l, r);
  } else {
    res = gIR->ir->CreateSub(l, r);
  }

  return new DImValue(t, res);
}

//////////////////////////////////////////////////////////////////////////////

DImValue *DtoBinMul(Type *targettype, DRValue *lhs, DRValue *rhs) {
  Type *t = lhs->type;
  LLValue *l = DtoRVal(lhs);
  LLValue *r = DtoRVal(rhs);

  LLValue *res;
  if (t->isfloating()) {
    res = gIR->ir->CreateFMul(l, r);
  } else {
    res = gIR->ir->CreateMul(l, r);
  }

  return new DImValue(targettype, res);
}

//////////////////////////////////////////////////////////////////////////////

DImValue *DtoBinDiv(Type *targettype, DRValue *lhs, DRValue *rhs) {
  Type *t = lhs->type;
  LLValue *l = DtoRVal(lhs);
  LLValue *r = DtoRVal(rhs);

  LLValue *res;
  if (t->isfloating()) {
    res = gIR->ir->CreateFDiv(l, r);
  } else if (!isLLVMUnsigned(t)) {
    res = gIR->ir->CreateSDiv(l, r);
  } else {
    res = gIR->ir->CreateUDiv(l, r);
  }

  return new DImValue(targettype, res);
}

//////////////////////////////////////////////////////////////////////////////

DImValue *DtoBinRem(Type *targettype, DRValue *lhs, DRValue *rhs) {
  Type *t = lhs->type;
  LLValue *l = DtoRVal(lhs);
  LLValue *r = DtoRVal(rhs);

  LLValue *res;
  if (t->isfloating()) {
    res = gIR->ir->CreateFRem(l, r);
  } else if (!isLLVMUnsigned(t)) {
    res = gIR->ir->CreateSRem(l, r);
  } else {
    res = gIR->ir->CreateURem(l, r);
  }

  return new DImValue(targettype, res);
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
  if (op == TOKequal) {
    res = gIR->ir->CreateFCmpOEQ(DtoRVal(lhs), DtoRVal(rhs));
  } else if (op == TOKnotequal) {
    res = gIR->ir->CreateFCmpUNE(DtoRVal(lhs), DtoRVal(rhs));
  } else {
    llvm::ICmpInst::Predicate cmpop;
    if (op == TOKidentity) {
      cmpop = llvm::ICmpInst::ICMP_EQ;
    } else {
      cmpop = llvm::ICmpInst::ICMP_NE;
    }

    LLValue *sz = DtoConstSize_t(getTypeStoreSize(DtoType(lhs->type)));
    LLValue *val = DtoMemCmp(makeLValue(loc, lhs), makeLValue(loc, rhs), sz);
    res = gIR->ir->CreateICmp(cmpop, val,
                              LLConstantInt::get(val->getType(), 0, false));
  }
  assert(res);
  return res;
}
