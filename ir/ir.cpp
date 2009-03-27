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
