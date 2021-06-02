//===-- dvalue.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dvalue.h"

#include "dmd/declaration.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "gen/tollvm.h"
#include "llvm/IR/MDBuilder.h"

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

DImValue::DImValue(Type *t, llvm::Value *v) : DRValue(t, v) {
  // TODO: get rid of Tfunction exception
  // v may be an addrspace qualified pointer so strip it before doing a pointer
  // equality check.
  assert(t->toBasetype()->ty == Tfunction ||
         stripAddrSpaces(v->getType()) == DtoType(t));
  assert(t->toBasetype()->ty != Tarray && "use DSliceValue for dynamic arrays");
}

////////////////////////////////////////////////////////////////////////////////

DConstValue::DConstValue(Type *t, LLConstant *con) : DRValue(t, con) {
  assert(con->getType() == DtoType(t));
}

////////////////////////////////////////////////////////////////////////////////

DSliceValue::DSliceValue(Type *t, LLValue *pair, LLValue *length, LLValue *ptr)
    : DRValue(t, pair), length(length), ptr(ptr) {
  assert(t->toBasetype()->ty == Tarray);
  // v may be an addrspace qualified pointer so strip it before doing a pointer
  // equality check.
  assert(stripAddrSpaces(pair->getType()) == DtoType(t));
}

DSliceValue::DSliceValue(Type *t, LLValue *pair)
    : DSliceValue(t, pair, nullptr, nullptr) {}

DSliceValue::DSliceValue(Type *t, LLValue *length, LLValue *ptr)
    : DSliceValue(t, DtoAggrPair(length, ptr), length, ptr) {}

LLValue *DSliceValue::getLength() {
  return length ? length : DtoExtractValue(val, 0, ".len");
}

LLValue *DSliceValue::getPtr() {
  return ptr ? ptr : DtoExtractValue(val, 1, ".ptr");
}

////////////////////////////////////////////////////////////////////////////////

DFuncValue::DFuncValue(Type *t, FuncDeclaration *fd, LLValue *v, LLValue *vt)
    : DRValue(t, v), func(fd), vthis(vt) {}

DFuncValue::DFuncValue(FuncDeclaration *fd, LLValue *v, LLValue *vt)
    : DFuncValue(fd->type, fd, v, vt) {}

bool DFuncValue::definedInFuncEntryBB() {
  return isDefinedInFuncEntryBB(val) &&
         (!vthis || isDefinedInFuncEntryBB(vthis));
}

////////////////////////////////////////////////////////////////////////////////

DLValue::DLValue(Type *t, LLValue *v) : DValue(t, v) {
  // v may be an addrspace qualified pointer so strip it before doing a pointer
  // equality check.
  assert(t->toBasetype()->ty == Ttuple ||
         stripAddrSpaces(v->getType()) == DtoPtrToType(t));
}

DRValue *DLValue::getRVal() {
  if (DtoIsInMemoryOnly(type)) {
    llvm_unreachable("getRVal() for memory-only type");
    return nullptr;
  }

  LLValue *rval = DtoLoad(val, "", DtoAlignment(type));

  const auto ty = type->toBasetype()->ty;
  if (ty == Tbool) {
    assert(rval->getType() == llvm::Type::getInt8Ty(gIR->context()));

    if (isOptimizationEnabled()) {
      // attach range metadata for i8 being loaded: [0, 2)
      llvm::MDBuilder mdBuilder(gIR->context());
      llvm::cast<llvm::LoadInst>(rval)->setMetadata(
          llvm::LLVMContext::MD_range,
          mdBuilder.createRange(llvm::APInt(8, 0), llvm::APInt(8, 2)));
    }

    // truncate to i1
    rval = gIR->ir->CreateTrunc(rval, llvm::Type::getInt1Ty(gIR->context()));
  } else if (ty == Tarray) {
    return new DSliceValue(type, rval);
  }

  return new DImValue(type, rval);
}

////////////////////////////////////////////////////////////////////////////////

DSpecialRefValue::DSpecialRefValue(Type *t, LLValue *v) : DLValue(v, t) {
  assert(v->getType() == DtoPtrToType(t)->getPointerTo());
}

DRValue *DSpecialRefValue::getRVal() {
  return DLValue(type, DtoLoad(val)).getRVal();
}

DLValue *DSpecialRefValue::getLVal() { return new DLValue(type, DtoLoad(val)); }
