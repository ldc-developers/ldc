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

IrTypeFunction::IrTypeFunction(Type* dt, LLType* lt, const IrFuncTy &irFty_)
:   IrType(dt, lt), irFty(irFty_)
{
}

IrTypeFunction* IrTypeFunction::get(Type* dt, Type* nestedContextOverride)
{
    assert(!dt->ctype);
    assert(dt->ty == Tfunction);

    IrFuncTy irFty;
    TypeFunction* tf = static_cast<TypeFunction*>(dt);
    llvm::Type* lt = DtoFunctionType(tf, irFty, NULL, nestedContextOverride);

    if (!dt->ctype)
        dt->ctype = new IrTypeFunction(dt, lt, irFty);
    return dt->ctype->isFunction();
}

//////////////////////////////////////////////////////////////////////////////

IrTypeDelegate::IrTypeDelegate(Type * dt, LLType* lt, const IrFuncTy &irFty_)
:   IrType(dt, lt), irFty(irFty_)
{
}

IrTypeDelegate* IrTypeDelegate::get(Type* t)
{
    assert(!t->ctype);
    assert(t->ty == Tdelegate);
    assert(t->nextOf()->ty == Tfunction);

    TypeDelegate *dt = (TypeDelegate*)t;

    if (!dt->ctype)
    {
        TypeFunction* tf = static_cast<TypeFunction*>(dt->nextOf());
        IrFuncTy irFty;
        llvm::Type* ltf = DtoFunctionType(tf, irFty, NULL, Type::tvoid->pointerTo());

        llvm::Type *types[] = { getVoidPtrType(), 
                                getPtrToType(ltf) };
        LLStructType* lt = LLStructType::get(gIR->context(), types, false);
        dt->ctype = new IrTypeDelegate(dt, lt, irFty);
    }

    return dt->ctype->isDelegate();
}
