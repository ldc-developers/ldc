#include "gen/llvm.h"

#include "mtype.h"
#include "declaration.h"

#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/irstate.h"
#include "gen/dvalue.h"
#include "gen/functions.h"

#include "gen/logger.h"

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoCallDFunc(FuncDeclaration* fdecl, Array* arguments, TypeClass* type, LLValue* thismem)
{
    Logger::println("Calling function: %s", fdecl->toPrettyChars());
    LOG_SCOPE;

    assert(fdecl);
    DtoForceDeclareDsymbol(fdecl);
    llvm::Function* fn = fdecl->ir.irFunc->func;
    TypeFunction* tf = (TypeFunction*)DtoDType(fdecl->type);

    llvm::PAListPtr palist;

    int thisOffset = 0;
    if (type || thismem)
    {
        assert(type && thismem);
        thisOffset = 1;
    }

    std::vector<LLValue*> args;
    if (thisOffset)
        args.push_back(thismem);
    for (size_t i=0; i<arguments->dim; ++i)
    {
        Expression* ex = (Expression*)arguments->data[i];
        Argument* fnarg = Argument::getNth(tf->parameters, i);
        DValue* argval = DtoArgument(fnarg, ex);
        LLValue* a = argval->getRVal();
        const LLType* aty = fn->getFunctionType()->getParamType(i+thisOffset);
        if (a->getType() != aty)
        {
            Logger::cout() << "expected: " << *aty << '\n';
            Logger::cout() << "got:      " << *a->getType() << '\n';
            a = DtoBitCast(a, aty);
        }
        args.push_back(a);
        if (fnarg && fnarg->llvmByVal)
            palist = palist.addAttr(i+thisOffset+1, llvm::ParamAttr::ByVal); // return,this,args...
    }

    CallOrInvoke* call = gIR->CreateCallOrInvoke(fn, args.begin(), args.end(), "tmp");
    call->setCallingConv(DtoCallingConv(LINKd));
    call->setParamAttrs(palist);

    return new DImValue(type, call->get(), false);
}
