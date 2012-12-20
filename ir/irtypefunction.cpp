//===-- irtypefunction.cpp ------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DerivedTypes.h"
#include "mtype.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/functions.h"

#include "ir/irtypefunction.h"

IrTypeFunction::IrTypeFunction(Type* dt, LLType* lt)
:   IrType(dt, lt)
{
}

IrTypeFunction* IrTypeFunction::get(Type* dt)
{
    assert(dt->ty == Tfunction);

    // We can't get cycles here, but we can end up building the type as part of
    // a class vtbl, ...
    llvm::Type* lt;
    TypeFunction* tf = static_cast<TypeFunction*>(dt);
    if (tf->funcdecl)
        lt = DtoFunctionType(tf->funcdecl);
    else
        lt = DtoFunctionType(tf,NULL,NULL);

    if (!dt->irtype)
        dt->irtype = new IrTypeFunction(dt, lt);
    return dt->irtype->isFunction();
}

//////////////////////////////////////////////////////////////////////////////

IrTypeDelegate::IrTypeDelegate(Type * dt, LLType* lt)
:   IrType(dt, lt)
{
}

IrTypeDelegate* IrTypeDelegate::get(Type* dt)
{
    assert(dt->ty == Tdelegate);

    // We can't get cycles here, but we can end up building the type as part of
    // a class vtbl, ...
    LLType* func = DtoFunctionType(dt->nextOf(), NULL, Type::tvoid->pointerTo());
    if (!dt->irtype)
    {
        llvm::SmallVector<LLType*, 2> types;
        types.push_back(getVoidPtrType());
        types.push_back(getPtrToType(func));
        LLStructType* lt = LLStructType::get(gIR->context(), types);
        dt->irtype = new IrTypeDelegate(dt, lt);
    }

    return dt->irtype->isDelegate();
}
