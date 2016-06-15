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

LLValue *DtoLVal(DValue *v) {
  auto lval = v->isLVal();
  assert(lval);
  return lval->getLVal()->val;
}

////////////////////////////////////////////////////////////////////////////////

DValue::DValue(Type *t, LLValue *v) : type(t), val(v) {
  assert(type);
  assert(val);
}

bool DValue::definedInFuncEntryBB() { return isDefinedInFuncEntryBB(val); }

////////////////////////////////////////////////////////////////////////////////

DRValue::DRValue(Type *t, LLValue *v) : DValue(t, v) {
  assert(!DtoIsInMemoryOnly(t) &&
         "Cannot represent memory-only type as DRValue");
}

////////////////////////////////////////////////////////////////////////////////

DConstValue::DConstValue(Type *t, LLConstant *con) : DRValue(t, con) {}

////////////////////////////////////////////////////////////////////////////////

DSliceValue::DSliceValue(Type *t, LLValue *length, LLValue *ptr)
    : DRValue(t, DtoAggrPair(length, ptr)) {}

LLValue *DSliceValue::getLength() { return DtoExtractValue(val, 0, ".len"); }

LLValue *DSliceValue::getPtr() { return DtoExtractValue(val, 1, ".ptr"); }

////////////////////////////////////////////////////////////////////////////////

DFuncValue::DFuncValue(Type *t, FuncDeclaration *fd, LLValue *v, LLValue *vt)
    : DRValue(t, v), func(fd), vthis(vt) {}

DFuncValue::DFuncValue(FuncDeclaration *fd, LLValue *v, LLValue *vt)
    : DRValue(fd->type, v), func(fd), vthis(vt) {}

bool DFuncValue::definedInFuncEntryBB() {
  return isDefinedInFuncEntryBB(val) &&
         (!vthis || isDefinedInFuncEntryBB(vthis));
}

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

DLValue::DLValue(Type *t, LLValue *v) : DValue(t, v) {
  assert(checkVarValueType(v->getType(), false));
}

DRValue *DLValue::getRVal() {
  if (DtoIsInMemoryOnly(type)) {
    llvm_unreachable("getRVal() for memory-only type");
    return nullptr;
  }

  LLValue *rval = DtoLoad(val);
  if (type->toBasetype()->ty == Tbool) {
    assert(rval->getType() == llvm::Type::getInt8Ty(gIR->context()));
    rval = gIR->ir->CreateTrunc(rval, llvm::Type::getInt1Ty(gIR->context()));
  }

  return new DImValue(type, rval);
}

////////////////////////////////////////////////////////////////////////////////

DSpecialRefValue::DSpecialRefValue(Type *t, LLValue *v) : DLValue(v, t) {
  assert(checkVarValueType(v->getType(), true));
}

DRValue *DSpecialRefValue::getRVal() {
  return DLValue(type, DtoLoad(val)).getRVal();
}

DLValue *DSpecialRefValue::getLVal() { return new DLValue(type, DtoLoad(val)); }
