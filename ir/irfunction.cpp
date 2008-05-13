#include "gen/tollvm.h"
#include "ir/irfunction.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrFunction::IrFunction(FuncDeclaration* fd)
{
    decl = fd;

    Type* t = DtoDType(fd->type);
    assert(t->ty == Tfunction);
    type = (TypeFunction*)t;
    func = NULL;
    allocapoint = NULL;

    queued = false;
    defined = false;

    retArg = NULL;
    thisVar = NULL;
    nestedVar = NULL;
    _arguments = NULL;
    _argptr = NULL;
    dwarfSubProg = NULL;

    srcfileArg = NULL;
    inVolatile = false;
}

IrFunction::~IrFunction()
{
}
