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

namespace {
bool isDefinedInFuncEntryBB(llvm::Value *v) {
  auto instr = llvm::dyn_cast<llvm::Instruction>(v);
  if (!instr) {
    // Global, constant, ...
    return true;
  }

  auto bb = instr->getParent();
  if (bb != &(bb->getParent()->getEntryBlock())) {
    return false;
  }

  // An invoke instruction in the entry BB does not necessarily dominate the
  // rest of the function because of the failure path.
  return !llvm::isa<llvm::InvokeInst>(instr);
}
}

////////////////////////////////////////////////////////////////////////////////

bool DImValue::definedInFuncEntryBB() { return isDefinedInFuncEntryBB(val); }

////////////////////////////////////////////////////////////////////////////////

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

DVarValue::DVarValue(Type *t, LLValue *llvmValue, bool isSpecialRefVar)
    : DValue(t), val(llvmValue), isSpecialRefVar(isSpecialRefVar) {
  assert(llvmValue && "Unexpected null llvm::Value.");
  assert(checkVarValueType(llvmValue->getType(), isSpecialRefVar));
}

LLValue *DVarValue::getLVal() { return isSpecialRefVar ? DtoLoad(val) : val; }

LLValue *DVarValue::getRVal() {
  assert(val);

  llvm::Value *storage = val;
  if (isSpecialRefVar) {
    storage = DtoLoad(storage);
  }

  if (DtoIsInMemoryOnly(type->toBasetype())) {
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
  assert(isSpecialRefVar);
  return val;
}

bool DVarValue::definedInFuncEntryBB() { return isDefinedInFuncEntryBB(val); }

////////////////////////////////////////////////////////////////////////////////

LLValue *DSliceValue::getRVal() {
  assert(len);
  assert(ptr);
  return DtoAggrPair(len, ptr);
}

bool DSliceValue::definedInFuncEntryBB() {
  return isDefinedInFuncEntryBB(len) && isDefinedInFuncEntryBB(ptr);
}

////////////////////////////////////////////////////////////////////////////////

DFuncValue::DFuncValue(Type *t, FuncDeclaration *fd, llvm::Value *v,
                       llvm::Value *vt)
    : DValue(t), func(fd), val(v), vthis(vt) {}

DFuncValue::DFuncValue(FuncDeclaration *fd, LLValue *v, LLValue *vt)
    : DValue(fd->type), func(fd), val(v), vthis(vt) {}

LLValue *DFuncValue::getRVal() {
  assert(val);
  return val;
}

bool DFuncValue::definedInFuncEntryBB() {
  if (!isDefinedInFuncEntryBB(val)) {
    return false;
  }

  if (vthis && !isDefinedInFuncEntryBB(vthis)) {
    return false;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////

LLValue *DConstValue::getRVal() {
  assert(c);
  return c;
}
