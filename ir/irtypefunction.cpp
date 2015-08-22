//===-- irtypefunction.cpp ------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_LLVM_VER >= 303
#include "llvm/IR/DerivedTypes.h"
#else
#include "llvm/DerivedTypes.h"
#endif
#include "mtype.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/functions.h"

#include "ir/irtypefunction.h"

IrTypeFunction::IrTypeFunction(Type* dt, llvm::Type* lt, const IrFuncTy& irFty_)
:   IrType(dt, lt), irFty(irFty_)
{
}

IrTypeFunction* IrTypeFunction::get(Type* dt, Type* nestedContextOverride)
{
    assert(!dt->ctype);
    assert(dt->ty == Tfunction);

    IrFuncTy irFty;
    llvm::Type* lt = DtoFunctionType(dt, irFty, NULL, nestedContextOverride);

    IrTypeFunction* result = new IrTypeFunction(dt, lt, irFty);
    dt->ctype = result;
    return result;
}

//////////////////////////////////////////////////////////////////////////////

IrTypeDelegate::IrTypeDelegate(Type* dt, llvm::Type* lt, const IrFuncTy& irFty_)
:   IrType(dt, lt), irFty(irFty_)
{
}

IrTypeDelegate* IrTypeDelegate::get(Type* t)
{
    assert(!t->ctype);
    assert(t->ty == Tdelegate);
    assert(t->nextOf()->ty == Tfunction);

    IrFuncTy irFty;
    llvm::Type* ltf = DtoFunctionType(t->nextOf(), irFty, NULL,
        Type::tvoid->pointerTo());
    llvm::Type *types[] = { getVoidPtrType(), getPtrToType(ltf) };
    LLStructType* lt = LLStructType::get(gIR->context(), types, false);

    IrTypeDelegate* result = new IrTypeDelegate(t, lt, irFty);
    t->ctype = result;
    return result;
}
