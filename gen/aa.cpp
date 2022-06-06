//===-- aa.cpp ------------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/aa.h"

#include "dmd/aggregate.h"
#include "dmd/declaration.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "gen/arrays.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irmodule.h"

// returns the keytype typeinfo
static LLConstant *to_keyti(const Loc &loc, DValue *aa, LLType *targetType) {
  // keyti param
  assert(aa->type->toBasetype()->ty == TY::Taarray);
  TypeAArray *aatype = static_cast<TypeAArray *>(aa->type->toBasetype());
  LLConstant *ti = DtoTypeInfoOf(loc, aatype->index, /*base=*/false);
  return DtoBitCast(ti, targetType);
}

////////////////////////////////////////////////////////////////////////////////

DLValue *DtoAAIndex(const Loc &loc, Type *type, DValue *aa, DValue *key,
                    bool lvalue) {
  // D2:
  // call:
  // extern(C) void* _aaGetY(AA* aa, TypeInfo aati, size_t valuesize, void*
  // pkey)
  // or
  // extern(C) void* _aaInX(AA aa*, TypeInfo keyti, void* pkey)

  // first get the runtime function
  llvm::Function *func =
      getRuntimeFunction(loc, gIR->module, lvalue ? "_aaGetY" : "_aaInX");
  LLFunctionType *funcTy = func->getFunctionType();

  // aa param
  LLValue *aaval = lvalue ? DtoLVal(aa) : DtoRVal(aa);
  aaval = DtoBitCast(aaval, funcTy->getParamType(0));

  // pkey param
  LLValue *pkey = makeLValue(loc, key);
  pkey = DtoBitCast(pkey, funcTy->getParamType(lvalue ? 3 : 2));

  // call runtime
  LLValue *ret;
  if (lvalue) {
    LLValue *rawAATI =
        DtoTypeInfoOf(loc, aa->type->unSharedOf()->mutableOf(), /*base=*/false);
    LLValue *castedAATI = DtoBitCast(rawAATI, funcTy->getParamType(1));
    LLValue *valsize = DtoConstSize_t(getTypeAllocSize(DtoType(type)));
    ret = gIR->CreateCallOrInvoke(func, aaval, castedAATI, valsize, pkey,
                                  "aa.index");
  } else {
    LLValue *keyti = to_keyti(loc, aa, funcTy->getParamType(1));
    ret = gIR->CreateCallOrInvoke(func, aaval, keyti, pkey, "aa.index");
  }

  // cast return value
  LLType *targettype = DtoPtrToType(type);
  if (ret->getType() != targettype) {
    ret = DtoBitCast(ret, targettype);
  }

  // Only check bounds for rvalues ('aa[key]').
  // Lvalue use ('aa[key] = value') auto-adds an element.
  if (!lvalue && gIR->emitArrayBoundsChecks()) {
    llvm::BasicBlock *okbb = gIR->insertBB("aaboundsok");
    llvm::BasicBlock *failbb = gIR->insertBBAfter(okbb, "aaboundscheckfail");

    LLValue *nullaa = LLConstant::getNullValue(ret->getType());
    LLValue *cond = gIR->ir->CreateICmpNE(nullaa, ret, "aaboundscheck");
    gIR->ir->CreateCondBr(cond, okbb, failbb);

    // set up failbb to call the array bounds error runtime function
    gIR->ir->SetInsertPoint(failbb);
    emitRangeError(gIR, loc);

    // if ok, proceed in okbb
    gIR->ir->SetInsertPoint(okbb);
  }
  return new DLValue(type, ret);
}

////////////////////////////////////////////////////////////////////////////////

DValue *DtoAAIn(const Loc &loc, Type *type, DValue *aa, DValue *key) {
  // D1:
  // call:
  // extern(C) void* _aaIn(AA aa*, TypeInfo keyti, void* pkey)

  // D2:
  // call:
  // extern(C) void* _aaInX(AA aa*, TypeInfo keyti, void* pkey)

  // first get the runtime function
  llvm::Function *func = getRuntimeFunction(loc, gIR->module, "_aaInX");
  LLFunctionType *funcTy = func->getFunctionType();

  IF_LOG Logger::cout() << "_aaIn = " << *func << '\n';

  // aa param
  LLValue *aaval = DtoRVal(aa);
  IF_LOG {
    Logger::cout() << "aaval: " << *aaval << '\n';
    Logger::cout() << "totype: " << *funcTy->getParamType(0) << '\n';
  }
  aaval = DtoBitCast(aaval, funcTy->getParamType(0));

  // keyti param
  LLValue *keyti = to_keyti(loc, aa, funcTy->getParamType(1));

  // pkey param
  LLValue *pkey = makeLValue(loc, key);
  pkey = DtoBitCast(pkey, getVoidPtrType());

  // call runtime
  LLValue *ret = gIR->CreateCallOrInvoke(func, aaval, keyti, pkey, "aa.in");

  // cast return value
  LLType *targettype = DtoType(type);
  if (ret->getType() != targettype) {
    ret = DtoBitCast(ret, targettype);
  }

  return new DImValue(type, ret);
}

////////////////////////////////////////////////////////////////////////////////

DValue *DtoAARemove(const Loc &loc, DValue *aa, DValue *key) {
  // D1:
  // call:
  // extern(C) void _aaDel(AA aa, TypeInfo keyti, void* pkey)

  // D2:
  // call:
  // extern(C) bool _aaDelX(AA aa, TypeInfo keyti, void* pkey)

  // first get the runtime function
  llvm::Function *func = getRuntimeFunction(loc, gIR->module, "_aaDelX");
  LLFunctionType *funcTy = func->getFunctionType();

  IF_LOG Logger::cout() << "_aaDel = " << *func << '\n';

  // aa param
  LLValue *aaval = DtoRVal(aa);
  IF_LOG {
    Logger::cout() << "aaval: " << *aaval << '\n';
    Logger::cout() << "totype: " << *funcTy->getParamType(0) << '\n';
  }
  aaval = DtoBitCast(aaval, funcTy->getParamType(0));

  // keyti param
  LLValue *keyti = to_keyti(loc, aa, funcTy->getParamType(1));

  // pkey param
  LLValue *pkey = makeLValue(loc, key);
  pkey = DtoBitCast(pkey, funcTy->getParamType(2));

  // call runtime
  LLValue *res = gIR->CreateCallOrInvoke(func, aaval, keyti, pkey);

  return new DImValue(Type::tbool, res);
}

////////////////////////////////////////////////////////////////////////////////

LLValue *DtoAAEquals(const Loc &loc, EXP op, DValue *l, DValue *r) {
  Type *t = l->type->toBasetype();
  assert(t == r->type->toBasetype() &&
         "aa equality is only defined for aas of same type");
  llvm::Function *func = getRuntimeFunction(loc, gIR->module, "_aaEqual");
  LLFunctionType *funcTy = func->getFunctionType();

  LLValue *aaval = DtoBitCast(DtoRVal(l), funcTy->getParamType(1));
  LLValue *abval = DtoBitCast(DtoRVal(r), funcTy->getParamType(2));
  LLValue *aaTypeInfo = DtoTypeInfoOf(loc, t);
  LLValue *res =
      gIR->CreateCallOrInvoke(func, aaTypeInfo, aaval, abval, "aaEqRes");

  const auto predicate = eqTokToICmpPred(op, /* invert = */ true);
  res = gIR->ir->CreateICmp(predicate, res, DtoConstInt(0));

  return res;
}
