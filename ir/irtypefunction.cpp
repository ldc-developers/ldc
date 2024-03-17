//===-- irtypefunction.cpp ------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irtypefunction.h"

#include "dmd/mtype.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "llvm/IR/DerivedTypes.h"

using namespace dmd;

IrTypeFunction::IrTypeFunction(Type *dt, llvm::Type *lt, IrFuncTy irFty_)
    : IrType(dt, lt), irFty(std::move(irFty_)) {}

IrTypeFunction *IrTypeFunction::get(Type *dt) {
  TypeFunction *tf = dt->isTypeFunction();
  assert(tf);

  auto &ctype = getIrType(tf);
  assert(!ctype);

  IrFuncTy irFty(tf);
  llvm::Type *lt = DtoFunctionType(tf, irFty, nullptr, nullptr);

  // Could have already built the type as part of a struct forward reference,
  // just as for pointers and arrays.
  if (!ctype) {
    ctype = new IrTypeFunction(dt, lt, irFty);
  }

  return ctype->isFunction();
}

//////////////////////////////////////////////////////////////////////////////

IrTypeDelegate::IrTypeDelegate(Type *dt, llvm::Type *lt, IrFuncTy irFty_)
    : IrType(dt, lt), irFty(std::move(irFty_)) {}

IrTypeDelegate *IrTypeDelegate::get(Type *t) {
  assert(t->ty == TY::Tdelegate);
  TypeFunction *tf = t->nextOf()->isTypeFunction();
  assert(tf);

  auto &ctype = getIrType(t);
  assert(!ctype);

  IrFuncTy irFty(tf);
  llvm::Type *ltf =
      DtoFunctionType(tf, irFty, nullptr, pointerTo(Type::tvoid));
  llvm::Type *fptr = ltf->getPointerTo(gDataLayout->getProgramAddressSpace());
  llvm::Type *types[] = {getVoidPtrType(), fptr};
  LLStructType *lt = LLStructType::get(gIR->context(), types, false);

  // Could have already built the type as part of a struct forward reference,
  // just as for pointers and arrays.
  if (!ctype) {
    ctype = new IrTypeDelegate(t, lt, irFty);
  }

  return ctype->isDelegate();
}
