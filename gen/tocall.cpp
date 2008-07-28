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

TypeFunction* DtoTypeFunction(Type* type)
{
    TypeFunction* tf = 0;
    type = type->toBasetype();
    if (type->ty == Tfunction)
    {
         tf = (TypeFunction*)type;
    }
    else if (type->ty == Tdelegate)
    {
        assert(type->next->ty == Tfunction);
        tf = (TypeFunction*)type->next;
    }
    return tf;
}

//////////////////////////////////////////////////////////////////////////////////////////

unsigned DtoCallingConv(LINK l)
{
    if (l == LINKc || l == LINKcpp)
        return llvm::CallingConv::C;
    else if (l == LINKd || l == LINKdefault)
        return llvm::CallingConv::Fast;
    else if (l == LINKwindows)
        return llvm::CallingConv::X86_StdCall;
    else
        assert(0 && "Unsupported calling convention");
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoVaArg(Loc& loc, Type* type, Expression* valistArg)
{
    DValue* expelem = valistArg->toElem(gIR);
    const LLType* llt = DtoType(type);
    if (DtoIsPassedByRef(type))
        llt = getPtrToType(llt);
    // issue a warning for broken va_arg instruction.
    if (global.params.cpu != ARCHx86)
        warning("%s: va_arg for C variadic functions is probably broken for anything but x86", loc.toChars());
    // done
    return new DImValue(type, gIR->ir->CreateVAArg(expelem->getLVal(), llt, "tmp"));
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoCallableValue(DValue* fn)
{
    Type* type = fn->getType()->toBasetype();
    if (type->ty == Tfunction)
    {
        return fn->getRVal();
    }
    else if (type->ty == Tdelegate)
    {
        LLValue* dg = fn->getRVal();
        LLValue* funcptr = DtoGEPi(dg, 0, 1);
        return DtoLoad(funcptr);
    }
    else
    {
        assert(0 && "not a callable type");
        return NULL;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

const LLFunctionType* DtoExtractFunctionType(const LLType* type)
{
    if (const LLFunctionType* fty = isaFunction(type))
        return fty;
    else if (const LLPointerType* pty = isaPointer(type))
    {
        if (const LLFunctionType* fty = isaFunction(pty->getElementType()))
            return fty;
    }
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoBuildDVarArgList(std::vector<LLValue*>& args, llvm::PAListPtr& palist, TypeFunction* tf, Expressions* arguments, size_t argidx)
{
    Logger::println("doing d-style variadic arguments");

    std::vector<const LLType*> vtypes;

    // number of non variadic args
    int begin = tf->parameters->dim;
    Logger::println("num non vararg params = %d", begin);

    // build struct with argument types (non variadic args)
    for (int i=begin; i<arguments->dim; i++)
    {
        Expression* argexp = (Expression*)arguments->data[i];
        vtypes.push_back(DtoType(argexp->type));
        size_t sz = getABITypeSize(vtypes.back());
        if (sz < PTRSIZE)
            vtypes.back() = DtoSize_t();
    }
    const LLStructType* vtype = LLStructType::get(vtypes);
    Logger::cout() << "d-variadic argument struct type:\n" << *vtype << '\n';
    LLValue* mem = new llvm::AllocaInst(vtype,"_argptr_storage",gIR->topallocapoint());

    // store arguments in the struct
    for (int i=begin,k=0; i<arguments->dim; i++,k++)
    {
        Expression* argexp = (Expression*)arguments->data[i];
        if (global.params.llvmAnnotate)
            DtoAnnotation(argexp->toChars());
        LLValue* argdst = DtoGEPi(mem,0,k);
        argdst = DtoBitCast(argdst, getPtrToType(DtoType(argexp->type)));
        DtoVariadicArgument(argexp, argdst);
    }

    // build type info array
    assert(Type::typeinfo->ir.irStruct->constInit);
    const LLType* typeinfotype = DtoType(Type::typeinfo->type);
    const LLArrayType* typeinfoarraytype = LLArrayType::get(typeinfotype,vtype->getNumElements());

    llvm::GlobalVariable* typeinfomem =
        new llvm::GlobalVariable(typeinfoarraytype, true, llvm::GlobalValue::InternalLinkage, NULL, "._arguments.storage", gIR->module);
    Logger::cout() << "_arguments storage: " << *typeinfomem << '\n';

    std::vector<LLConstant*> vtypeinfos;
    for (int i=begin,k=0; i<arguments->dim; i++,k++)
    {
        Expression* argexp = (Expression*)arguments->data[i];
        vtypeinfos.push_back(DtoTypeInfoOf(argexp->type));
    }

    // apply initializer
    LLConstant* tiinits = llvm::ConstantArray::get(typeinfoarraytype, vtypeinfos);
    typeinfomem->setInitializer(tiinits);

    // put data in d-array
    std::vector<LLConstant*> pinits;
    pinits.push_back(DtoConstSize_t(vtype->getNumElements()));
    pinits.push_back(llvm::ConstantExpr::getBitCast(typeinfomem, getPtrToType(typeinfotype)));
    const LLType* tiarrty = DtoType(Type::typeinfo->type->arrayOf());
    tiinits = llvm::ConstantStruct::get(pinits);
    LLValue* typeinfoarrayparam = new llvm::GlobalVariable(tiarrty,
        true, llvm::GlobalValue::InternalLinkage, tiinits, "._arguments.array", gIR->module);

    // specify arguments
    args.push_back(typeinfoarrayparam);
    ++argidx;
    args.push_back(gIR->ir->CreateBitCast(mem, getPtrToType(LLType::Int8Ty), "tmp"));
    ++argidx;

    // pass non variadic args
    for (int i=0; i<begin; i++)
    {
        Argument* fnarg = Argument::getNth(tf->parameters, i);
        DValue* argval = DtoArgument(fnarg, (Expression*)arguments->data[i]);
        args.push_back(argval->getRVal());

        if (fnarg->llvmByVal)
            palist = palist.addAttr(argidx, llvm::ParamAttr::ByVal);

        ++argidx;
    }
}


DValue* DtoCallFunction(Type* resulttype, DValue* fnval, Expressions* arguments)
{
    // the callee D type
    Type* calleeType = fnval->getType();

    // get func value if any
    DFuncValue* dfnval = fnval->isFunc();

    // handle special va_copy / va_end intrinsics
    bool va_intrinsic = (dfnval && dfnval->func && (dfnval->func->llvmInternal == LLVMva_intrinsic));

    // get function type info
    TypeFunction* tf = DtoTypeFunction(calleeType);
    assert(tf);

    // misc
    bool retinptr = tf->llvmRetInPtr;
    bool usesthis = tf->llvmUsesThis;
    bool delegatecall = (calleeType->toBasetype()->ty == Tdelegate);
    bool nestedcall = (dfnval && dfnval->func && dfnval->func->isNested());
    bool dvarargs = (tf->linkage == LINKd && tf->varargs == 1);

    unsigned callconv = DtoCallingConv(tf->linkage);

    // get callee llvm value
    LLValue* callable = DtoCallableValue(fnval);
    const LLFunctionType* callableTy = DtoExtractFunctionType(callable->getType());
    assert(callableTy);

    // get llvm argument iterator, for types
    LLFunctionType::param_iterator argbegin = callableTy->param_begin();
    LLFunctionType::param_iterator argiter = argbegin;

    // handle implicit arguments
    std::vector<LLValue*> args;

    // return in hidden ptr is first
    if (retinptr)
    {
        LLValue* retvar = new llvm::AllocaInst(argiter->get()->getContainedType(0), ".rettmp", gIR->topallocapoint());
        ++argiter;
        args.push_back(retvar);
    }

    // then comes the 'this' argument
    if (dfnval && dfnval->vthis)
    {
        LLValue* thisarg = DtoBitCast(dfnval->vthis, argiter->get());
        ++argiter;
        args.push_back(thisarg);
    }
    // or a delegate context arg
    else if (delegatecall)
    {
        LLValue* ctxarg = DtoLoad(DtoGEPi(fnval->getRVal(), 0,0));
        assert(ctxarg->getType() == argiter->get());
        ++argiter;
        args.push_back(ctxarg);
    }
    // or a nested function context arg
    else if (nestedcall)
    {
        LLValue* contextptr = DtoNestedContext(dfnval->func->toParent2()->isFuncDeclaration());
        if (!contextptr)
            contextptr = getNullPtr(getVoidPtrType());
        else
            contextptr = DtoBitCast(contextptr, getVoidPtrType());
        ++argiter;
        args.push_back(contextptr);
    }

    // handle the rest of the arguments based on param passing style
    llvm::PAListPtr palist;

    // variadic instrinsics need some custom casts
    if (va_intrinsic)
    {
        size_t n = arguments->dim;
        for (int i=0; i<n; i++)
        {
            Expression* exp = (Expression*)arguments->data[i];
            DValue* expelem = exp->toElem(gIR);
            // cast to va_list*
            LLValue* val = DtoBitCast(expelem->getLVal(), getVoidPtrType());
            ++argiter;
            args.push_back(val);
        }
    }

    // d style varargs needs a few more hidden arguments as well as special passing
    else if (dvarargs)
    {
        DtoBuildDVarArgList(args, palist, tf, arguments, argiter-argbegin+1);
    }

    // otherwise we're looking at a normal function call
    else
    {
        Logger::println("doing normal arguments");
        for (int i=0; i<arguments->dim; i++) {
            int j = argiter-argbegin;
            Argument* fnarg = Argument::getNth(tf->parameters, i);
            DValue* argval = DtoArgument(fnarg, (Expression*)arguments->data[i]);
            LLValue* arg = argval->getRVal();
            if (fnarg && arg->getType() != callableTy->getParamType(j))
                arg = DtoBitCast(arg, callableTy->getParamType(j));
            if (fnarg && fnarg->llvmByVal)
                palist = palist.addAttr(j+1, llvm::ParamAttr::ByVal);
            ++argiter;
            args.push_back(arg);
        }
    }

    #if 0
    Logger::println("%d params passed", n);
    for (int i=0; i<args.size(); ++i) {
        assert(args[i]);
        Logger::cout() << "arg["<<i<<"] = " << *args[i] << '\n';
    }
    #endif

    // void returns cannot not be named
    const char* varname = "";
    if (callableTy->getReturnType() != LLType::VoidTy)
        varname = "tmp";

    //Logger::cout() << "Calling: " << *funcval << '\n';

    // call the function
    CallOrInvoke* call = gIR->CreateCallOrInvoke(callable, args.begin(), args.end(), varname);

    // get return value
    LLValue* retllval = (retinptr) ? args[0] : call->get();

    // if the type of retllval is abstract, refine to concrete
    if (retllval->getType()->isAbstract())
        retllval = DtoBitCast(retllval, getPtrToType(DtoType(resulttype)), "retval");

    // set calling convention
    if (dfnval && dfnval->func)
    {
        int li = dfnval->func->llvmInternal;
        if (li != LLVMintrinsic && li != LLVMva_start && li != LLVMva_intrinsic)
        {
            call->setCallingConv(callconv);
        }
    }
    else
    {
        call->setCallingConv(callconv);
    }

    // param attrs
    call->setParamAttrs(palist);

    return new DImValue(resulttype, retllval, false);
}













