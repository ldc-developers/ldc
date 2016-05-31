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
bool isDefinedInFuncEntryBB(LLValue *v) {
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

bool DValue::definedInFuncEntryBB() { return isDefinedInFuncEntryBB(val); }

////////////////////////////////////////////////////////////////////////////////

DConstValue::DConstValue(Type *t, llvm::Constant *con) : DValue(t, con) {}

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

DVarValue::DVarValue(Type *t, LLValue *v, bool isSpecialRefVar)
    : DValue(t, v), isSpecialRefVar(isSpecialRefVar) {
  assert(v && "Unexpected null llvm::Value.");
  assert(checkVarValueType(v->getType(), isSpecialRefVar));
}

LLValue *DVarValue::getLVal() { return isSpecialRefVar ? DtoLoad(val) : val; }

LLValue *DVarValue::getRVal() {
  LLValue *storage = val;
  if (isSpecialRefVar) {
    storage = DtoLoad(storage);
  }

  if (DtoIsInMemoryOnly(type->toBasetype())) {
    return storage;
  }

  LLValue *rawValue = DtoLoad(storage);

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

////////////////////////////////////////////////////////////////////////////////

DSliceValue::DSliceValue(Type *t, LLValue *length, LLValue *ptr)
    : DValue(t, DtoAggrPair(length, ptr)) {}

LLValue *DSliceValue::getLength() { return DtoExtractValue(val, 0, ".len"); }

LLValue *DSliceValue::getPtr() { return DtoExtractValue(val, 1, ".ptr"); }

////////////////////////////////////////////////////////////////////////////////

DFuncValue::DFuncValue(Type *t, FuncDeclaration *fd, LLValue *v, LLValue *vt)
    : DValue(t, v), func(fd), vthis(vt) {}

DFuncValue::DFuncValue(FuncDeclaration *fd, LLValue *v, LLValue *vt)
    : DValue(fd->type, v), func(fd), vthis(vt) {}

bool DFuncValue::definedInFuncEntryBB() {
  return isDefinedInFuncEntryBB(val) &&
         (!vthis || isDefinedInFuncEntryBB(vthis));
}
