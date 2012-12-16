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

IrTypeFunction::IrTypeFunction(Type* dt)
:   IrType(dt, func2llvm(dt))
{
    irfty = NULL;
}

llvm::Type * IrTypeFunction::buildType()
{
    return type;
}

llvm::Type* IrTypeFunction::func2llvm(Type* dt)
{
    llvm::Type* T;
    TypeFunction* tf = static_cast<TypeFunction*>(dt);
    if (tf->funcdecl)
        T = DtoFunctionType(tf->funcdecl);
    else
        T = DtoFunctionType(tf,NULL,NULL);
    return T;
}

//////////////////////////////////////////////////////////////////////////////

IrTypeDelegate::IrTypeDelegate(Type * dt)
:   IrType(dt, delegate2llvm(dt))
{
}

llvm::Type* IrTypeDelegate::buildType()
{
    return type;
}

llvm::Type* IrTypeDelegate::delegate2llvm(Type* dt)
{
    assert(dt->ty == Tdelegate);
    LLType* func = DtoFunctionType(dt->nextOf(), NULL, Type::tvoid->pointerTo());
    llvm::SmallVector<LLType*, 2> types;
    types.push_back(getVoidPtrType());
    types.push_back(getPtrToType(func));
    LLStructType* dgtype = LLStructType::get(gIR->context(), types);
    return dgtype;
}
