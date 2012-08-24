#include "llvm/Target/TargetData.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/functions.h"

#include "ir/ir.h"
#include "ir/irfunction.h"


unsigned GetTypeAlignment(Ir* ir, Type* t)
{
    return gTargetData->getABITypeAlignment(DtoType(t));
}

unsigned GetPointerSize(Ir* ir)
{
    return gTargetData->getPointerSize();
}

unsigned GetTypeStoreSize(Ir* ir, Type* t)
{
    return gTargetData->getTypeStoreSize(DtoType(t));
}

unsigned GetTypeAllocSize(Ir* ir, Type* t)
{
    return gTargetData->getTypeAllocSize(DtoType(t));
}

Ir::Ir()
: irs(NULL)
{
}

void Ir::addFunctionBody(IrFunction * f)
{
    functionbodies.push_back(f);
}

void Ir::emitFunctionBodies()
{
    while (!functionbodies.empty())
    {
        IrFunction* irf = functionbodies.front();
        functionbodies.pop_front();
        DtoDefineFunction(irf->decl);
    }
}
