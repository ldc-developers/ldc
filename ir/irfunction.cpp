#include "gen/tollvm.h"
#include "ir/irfunction.h"

IrFinally::IrFinally()
{
    bb = 0;
    retbb = 0;
}

IrFinally::IrFinally(llvm::BasicBlock* b, llvm::BasicBlock* rb)
{
    bb = b;
    retbb = rb;
}

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
    finallyretval = NULL;

    queued = false;
    defined = false;

    retArg = NULL;
    thisVar = NULL;
    nestedVar = NULL;
    _arguments = NULL;
    _argptr = NULL;
    dwarfSubProg = NULL;
}

IrFunction::~IrFunction()
{
}
