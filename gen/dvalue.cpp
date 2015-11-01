//===-- dvalue.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dvalue.h"
#include "declaration.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

static bool checkVarValueType(LLType *t, bool extraDeref) {
  if (extraDeref) {
    llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(t);
    if (!pt) {
      return false;
    }

    t = pt->getElementType();
  }

  llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(t);
  if (!pt) {
    return false;
  }

  // bools should not be stored as i1 any longer.
  if (pt->getElementType() == llvm::Type::getInt1Ty(gIR->context())) {
    return false;
  }

  return true;
}

DVarValue::DVarValue(Type *t, VarDeclaration *vd, LLValue *llvmValue)
    : DValue(t), var(vd), val(llvmValue) {
  assert(checkVarValueType(llvmValue->getType(), isSpecialRefVar(vd)));
}

DVarValue::DVarValue(Type *t, LLValue *llvmValue)
    : DValue(t), var(nullptr), val(llvmValue) {
  assert(checkVarValueType(llvmValue->getType(), false));
}

LLValue *DVarValue::getLVal() {
  assert(val);
  if (var && isSpecialRefVar(var)) {
    return DtoLoad(val);
  }
  return val;
}

LLValue *DVarValue::getRVal() {
  assert(val);

  llvm::Value *storage = val;
  if (var && isSpecialRefVar(var)) {
    storage = DtoLoad(storage);
  }

  if (DtoIsPassedByRef(type->toBasetype())) {
    return storage;
  }

  llvm::Value *rawValue = DtoLoad(storage);

  if (type->toBasetype()->ty == Tbool) {
    assert(rawValue->getType() == llvm::Type::getInt8Ty(gIR->context()));
    return gIR->ir->CreateTrunc(rawValue,
                                llvm::Type::getInt1Ty(gIR->context()));
  }

  return rawValue;
}

LLValue *DVarValue::getRefStorage() {
  assert(val);
  assert(isSpecialRefVar(var));
  return val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

LLValue *DSliceValue::getRVal() {
  assert(len);
  assert(ptr);
  return DtoAggrPair(len, ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

DFuncValue::DFuncValue(Type *t, FuncDeclaration *fd, llvm::Value *v,
                       llvm::Value *vt)
    : DValue(t), func(fd), val(v), vthis(vt) {}

DFuncValue::DFuncValue(FuncDeclaration *fd, LLValue *v, LLValue *vt)
    : DValue(fd->type), func(fd), val(v), vthis(vt) {}

LLValue *DFuncValue::getRVal() {
  assert(val);
  return val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

LLValue *DConstValue::getRVal() {
  assert(c);
  return c;
}
