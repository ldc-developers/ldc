#include "llvm/DerivedTypes.h"
#include "mtype.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/functions.h"

#include "ir/irtypefunction.h"

IrTypeFunction::IrTypeFunction(Type * dt)
:   IrType(dt, llvm::OpaqueType::get())
{
    irfty = NULL;
}

const llvm::Type * IrTypeFunction::buildType()
{
    const llvm::Type* T;
    TypeFunction* tf = (TypeFunction*)dtype;
    if (tf->funcdecl)
        T = DtoFunctionType(tf->funcdecl);
    else
        T = DtoFunctionType(tf,NULL,NULL);

    llvm::cast<llvm::OpaqueType>(pa.get())->refineAbstractTypeTo(T);
    return pa.get();
}

//////////////////////////////////////////////////////////////////////////////

IrTypeDelegate::IrTypeDelegate(Type * dt)
:   IrType(dt, llvm::OpaqueType::get())
{
}

const llvm::Type * IrTypeDelegate::buildType()
{
    assert(dtype->ty == Tdelegate);
    const LLType* i8ptr = getVoidPtrType();
    const LLType* func = DtoFunctionType(dtype->nextOf(), NULL, Type::tvoid->pointerTo());
    const LLType* funcptr = getPtrToType(func);
    const LLStructType* dgtype = LLStructType::get(gIR->context(), i8ptr, funcptr, NULL);
    gIR->module->addTypeName(dtype->toChars(), dgtype);

    llvm::cast<llvm::OpaqueType>(pa.get())->refineAbstractTypeTo(dgtype);
    return pa.get();
}
