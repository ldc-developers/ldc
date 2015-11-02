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

  IrFuncTy irFty;
  llvm::Type *lt = DtoFunctionType(dt, irFty, nullptr, nullptr);

  auto result = new IrTypeFunction(dt, lt, irFty);
  dt->ctype = result;
  return result;
}

//////////////////////////////////////////////////////////////////////////////

IrTypeDelegate::IrTypeDelegate(Type *dt, llvm::Type *lt, IrFuncTy irFty_)
    : IrType(dt, lt), irFty(std::move(irFty_)) {}

IrTypeDelegate *IrTypeDelegate::get(Type *t) {
  assert(!t->ctype);
  assert(t->ty == Tdelegate);
  assert(t->nextOf()->ty == Tfunction);

  IrFuncTy irFty;
  llvm::Type *ltf =
      DtoFunctionType(t->nextOf(), irFty, nullptr, Type::tvoid->pointerTo());
  llvm::Type *types[] = {getVoidPtrType(), getPtrToType(ltf)};
  LLStructType *lt = LLStructType::get(gIR->context(), types, false);

  auto result = new IrTypeDelegate(t, lt, irFty);
  t->ctype = result;
  return result;
}
