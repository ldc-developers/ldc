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

    // We can't get cycles here, but we can end up building the type as part of
    // a class vtbl, ...
    llvm::Type* lt;
    TypeFunction* tf = static_cast<TypeFunction*>(dt);
    if (tf->funcdecl)
        lt = DtoFunctionType(tf->funcdecl);
    else
        lt = DtoFunctionType(tf, NULL, nestedContextOverride);

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
    assert(!dt->irtype);
    assert(dt->ty == Tdelegate);

    // We can't get cycles here, but we could end up building the type as part
    // of a class vtbl, ...
    if (!dt->nextOf()->irtype)
    {
        // Build the underlying function type. Be sure to set irtype here, so
        // the nested context arg doesn't disappear if DtoType is ever called
        // on dt->nextOf().
        IrTypeFunction::get(dt->nextOf(), Type::tvoid->pointerTo());
    }
    if (!dt->irtype)
    {
        assert(static_cast<TypeFunction*>(dt->nextOf())->fty.arg_nest &&
            "Underlying function type should have nested context arg, "
            "picked up random pre-existing type?"
        );

        llvm::Type *types[] = { getVoidPtrType(), 
                                getPtrToType(dt->nextOf()->irtype->getLLType()) };
        LLStructType* lt = LLStructType::get(gIR->context(), types, false);
        dt->irtype = new IrTypeDelegate(dt, lt);
    }

    return dt->irtype->isDelegate();
}
