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

IrTypeFunction::IrTypeFunction(Type* dt, LLType* lt)
:   IrType(dt, lt)
{
}

IrTypeFunction* IrTypeFunction::get(Type* dt, Type* nestedContextOverride)
{
    assert(!dt->irtype);
    assert(dt->ty == Tfunction);

    TypeFunction* tf = static_cast<TypeFunction*>(dt);
    llvm::Type* lt = DtoFunctionType(tf, tf->irFty, NULL, nestedContextOverride);

    if (!dt->irtype)
        dt->irtype = new IrTypeFunction(dt, lt);
    return dt->irtype->isFunction();
}

//////////////////////////////////////////////////////////////////////////////

IrTypeDelegate::IrTypeDelegate(Type * dt, LLType* lt)
:   IrType(dt, lt)
{
}

IrTypeDelegate* IrTypeDelegate::get(Type* t)
{
    assert(!t->irtype);
    assert(t->ty == Tdelegate);
    assert(t->nextOf()->ty == Tfunction);

    TypeDelegate *dt = (TypeDelegate*)t;

    if (!dt->irtype)
    {
        TypeFunction* tf = static_cast<TypeFunction*>(dt->nextOf());
        llvm::Type* ltf = DtoFunctionType(tf, dt->irFty, NULL, Type::tvoid->pointerTo());

        llvm::Type *types[] = { getVoidPtrType(), 
                                getPtrToType(ltf) };
        LLStructType* lt = LLStructType::get(gIR->context(), types, false);
        dt->irtype = new IrTypeDelegate(dt, lt);
    }

    return dt->irtype->isDelegate();
}
