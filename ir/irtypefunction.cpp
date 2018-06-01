//===-- irtypefunction.cpp ------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DerivedTypes.h"
#include "mtype.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/functions.h"

#include "ir/irtypefunction.h"

IrTypeFunction::IrTypeFunction(Type *dt, llvm::Type *lt, IrFuncTy irFty_)
    : IrType(dt, lt), irFty(std::move(irFty_)) {}

IrTypeFunction *IrTypeFunction::get(Type *dt) {
  assert(!dt->ctype);
  assert(dt->ty == Tfunction);

  TypeFunction *tf = static_cast<TypeFunction *>(dt);

  IrFuncTy irFty(tf);
  llvm::Type *lt = DtoFunctionType(tf, irFty, nullptr, nullptr);

  // Could have already built the type as part of a struct forward reference,
  // just as for pointers and arrays.
  if (!dt->ctype) {
    dt->ctype = new IrTypeFunction(dt, lt, irFty);
  }
  return dt->ctype->isFunction();
}

//////////////////////////////////////////////////////////////////////////////

IrTypeDelegate::IrTypeDelegate(Type *dt, llvm::Type *lt, IrFuncTy irFty_)
    : IrType(dt, lt), irFty(std::move(irFty_)) {}

IrTypeDelegate *IrTypeDelegate::get(Type *t) {
  assert(!t->ctype);
  assert(t->ty == Tdelegate);
  assert(t->nextOf()->ty == Tfunction);

  TypeFunction *tf = static_cast<TypeFunction *>(t->nextOf());

  IrFuncTy irFty(tf);
  llvm::Type *ltf =
      DtoFunctionType(tf, irFty, nullptr, Type::tvoid->pointerTo());
  llvm::Type *types[] = {getVoidPtrType(), getPtrToType(ltf)};
  LLStructType *lt = LLStructType::get(gIR->context(), types, false);

  // Could have already built the type as part of a struct forward reference,
  // just as for pointers and arrays.
  if (!t->ctype) {
    t->ctype = new IrTypeDelegate(t, lt, irFty);
  }
  return t->ctype->isDelegate();
}
