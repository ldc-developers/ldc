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

TypeFunction* DtoTypeFunction(DValue* fnval)
{
    Type* type = fnval->getType()->toBasetype();
    if (type->ty == Tfunction)
    {
         return (TypeFunction*)type;
    }
    else if (type->ty == Tdelegate)
    {
        Type* next = type->nextOf();
        assert(next->ty == Tfunction);
        return (TypeFunction*)next;
    }

    assert(0 && "cant get TypeFunction* from non lazy/function/delegate");
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

unsigned DtoCallingConv(LINK l)
{
    if (l == LINKc || l == LINKcpp || l == LINKintrinsic)
        return llvm::CallingConv::C;
    else if (l == LINKd || l == LINKdefault)
    {
        //TODO: StdCall is not a good base on Windows due to extra name mangling
        // applied there
        if (global.params.cpu == ARCHx86)
            return (global.params.os != OSWindows) ? llvm::CallingConv::X86_StdCall : llvm::CallingConv::C;
        else
            return llvm::CallingConv::Fast;
    }
    // on the other hand, here, it's exactly what we want!!! TODO: right?
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
        if (fn->isLVal())
        {
            LLValue* dg = fn->getLVal();
            LLValue* funcptr = DtoGEPi(dg, 0, 1);
            return DtoLoad(funcptr);
        }
        else
        {
            LLValue* dg = fn->getRVal();
            assert(isaStruct(dg));
            return gIR->ir->CreateExtractValue(dg, 1, ".funcptr");
        }
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

void DtoBuildDVarArgList(std::vector<LLValue*>& args, std::vector<llvm::AttributeWithIndex>& attrs, TypeFunction* tf, Expressions* arguments, size_t argidx)
{
    Logger::println("doing d-style variadic arguments");

    std::vector<const LLType*> vtypes;

    // number of non variadic args
    int begin = tf->parameters->dim;
    Logger::println("num non vararg params = %d", begin);

    // get n args in arguments list
    size_t n_arguments = arguments ? arguments->dim : 0;

    // build struct with argument types (non variadic args)
    for (int i=begin; i<n_arguments; i++)
    {
        Expression* argexp = (Expression*)arguments->data[i];
        vtypes.push_back(DtoType(argexp->type));
        size_t sz = getABITypeSize(vtypes.back());
        if (sz < PTRSIZE)
            vtypes.back() = DtoSize_t();
    }
    const LLStructType* vtype = LLStructType::get(vtypes);

    if (Logger::enabled())
        Logger::cout() << "d-variadic argument struct type:\n" << *vtype << '\n';

    LLValue* mem = DtoAlloca(vtype,"_argptr_storage");

    // store arguments in the struct
    for (int i=begin,k=0; i<n_arguments; i++,k++)
    {
        Expression* argexp = (Expression*)arguments->data[i];
        if (global.params.llvmAnnotate)
            DtoAnnotation(argexp->toChars());
        LLValue* argdst = DtoGEPi(mem,0,k);
        argdst = DtoBitCast(argdst, getPtrToType(DtoType(argexp->type)));
        DtoVariadicArgument(argexp, argdst);
    }

    // build type info array
    const LLType* typeinfotype = DtoType(Type::typeinfo->type);
    const LLArrayType* typeinfoarraytype = LLArrayType::get(typeinfotype,vtype->getNumElements());

    llvm::GlobalVariable* typeinfomem =
        new llvm::GlobalVariable(typeinfoarraytype, true, llvm::GlobalValue::InternalLinkage, NULL, "._arguments.storage", gIR->module);
    if (Logger::enabled())
        Logger::cout() << "_arguments storage: " << *typeinfomem << '\n';

    std::vector<LLConstant*> vtypeinfos;
    for (int i=begin,k=0; i<n_arguments; i++,k++)
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
    args.push_back(DtoLoad(typeinfoarrayparam));
    ++argidx;
    args.push_back(gIR->ir->CreateBitCast(mem, getPtrToType(LLType::Int8Ty), "tmp"));
    ++argidx;

    // pass non variadic args
    for (int i=0; i<begin; i++)
    {
        Argument* fnarg = Argument::getNth(tf->parameters, i);
        DValue* argval = DtoArgument(fnarg, (Expression*)arguments->data[i]);
        args.push_back(argval->getRVal());

        if (fnarg->llvmAttrs)
        {
            llvm::AttributeWithIndex Attr;
            Attr.Index = argidx;
            Attr.Attrs = fnarg->llvmAttrs;
            attrs.push_back(Attr);
        }

        ++argidx;
    }
}


DValue* DtoCallFunction(Loc& loc, Type* resulttype, DValue* fnval, Expressions* arguments)
{
    // the callee D type
    Type* calleeType = fnval->getType();

    // if the type has not yet been processed, do so now
    if (calleeType->ir.type == NULL)
        DtoType(calleeType);

    // get func value if any
    DFuncValue* dfnval = fnval->isFunc();

    // handle special vararg intrinsics
    bool va_intrinsic = (dfnval && dfnval->func && dfnval->func->isVaIntrinsic());

    // get function type info
    TypeFunction* tf = DtoTypeFunction(fnval);

    // misc
    bool retinptr = tf->retInPtr;
    bool thiscall = tf->usesThis;
    bool delegatecall = (calleeType->toBasetype()->ty == Tdelegate);
    bool nestedcall = tf->usesNest;
    bool dvarargs = (tf->linkage == LINKd && tf->varargs == 1);

    unsigned callconv = DtoCallingConv(tf->linkage);

    // get callee llvm value
    LLValue* callable = DtoCallableValue(fnval);
    const LLFunctionType* callableTy = DtoExtractFunctionType(callable->getType());
    assert(callableTy);

    if (Logger::enabled())
    {
        Logger::cout() << "callable: " << *callable << '\n';
    }

    // get n arguments
    size_t n_arguments = arguments ? arguments->dim : 0;

    // get llvm argument iterator, for types
    LLFunctionType::param_iterator argbegin = callableTy->param_begin();
    LLFunctionType::param_iterator argiter = argbegin;

    // parameter attributes
    std::vector<llvm::AttributeWithIndex> attrs;
    llvm::AttributeWithIndex Attr;

    // return attrs
    if (tf->retAttrs)
    {
        Attr.Index = 0;
        Attr.Attrs = tf->retAttrs;
        attrs.push_back(Attr);
    }

    // handle implicit arguments
    std::vector<LLValue*> args;

    // return in hidden ptr is first
    if (retinptr)
    {
        LLValue* retvar = DtoAlloca(argiter->get()->getContainedType(0), ".rettmp");
        ++argiter;
        args.push_back(retvar);

        // add attrs for hidden ptr
        Attr.Index = 1;
        Attr.Attrs = llvm::Attribute::StructRet;
        attrs.push_back(Attr);
    }

    // then comes a context argument...
    if(thiscall || delegatecall || nestedcall)
    {
        // ... which can be a 'this' argument
        if (thiscall && dfnval && dfnval->vthis)
        {
            LLValue* thisarg = DtoBitCast(dfnval->vthis, argiter->get());
            ++argiter;
            args.push_back(thisarg);
        }
        // ... or a delegate context arg
        else if (delegatecall)
        {
            LLValue* ctxarg;
            if (fnval->isLVal())
            {
                ctxarg = DtoLoad(DtoGEPi(fnval->getLVal(), 0,0));
            }
            else
            {
                ctxarg = gIR->ir->CreateExtractValue(fnval->getRVal(), 0, ".ptr");
            }
            assert(ctxarg->getType() == argiter->get());
            ++argiter;
            args.push_back(ctxarg);
        }
        // ... or a nested function context arg
        else if (nestedcall)
        {
            LLValue* contextptr = DtoNestedContext(loc, dfnval->func);
            contextptr = DtoBitCast(contextptr, getVoidPtrType());
            ++argiter;
            args.push_back(contextptr);
        }
        else
        {
            error(loc, "Context argument required but none given");
            fatal();
        }

        // add attributes for context argument
        if (tf->thisAttrs)
        {
            Attr.Index = retinptr ? 2 : 1;
            Attr.Attrs = tf->thisAttrs;
            attrs.push_back(Attr);
        }
    }

    // handle the rest of the arguments based on param passing style

    // variadic instrinsics need some custom casts
    if (va_intrinsic)
    {
        for (int i=0; i<n_arguments; i++)
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
        DtoBuildDVarArgList(args, attrs, tf, arguments, argiter-argbegin+1);
    }

    // otherwise we're looking at a normal function call
    // or a C style vararg call
    else
    {
        Logger::println("doing normal arguments");

        size_t n = Argument::dim(tf->parameters);

        LLSmallVector<unsigned, 10> attrptr(n, 0);

        // do formal params
        int beg = argiter-argbegin;
        for (int i=0; i<n; i++)
        {
            Argument* fnarg = Argument::getNth(tf->parameters, i);
            assert(fnarg);
            DValue* argval = DtoArgument(fnarg, (Expression*)arguments->data[i]);
            LLValue* arg = argval->getRVal();

            int j = tf->reverseParams ? beg + n - i - 1 : beg + i;

            // parameter type mismatch, this is hard to get rid of
            if (arg->getType() != callableTy->getParamType(j))
            {
            #if 0
                if (Logger::enabled())
                {
                    Logger::cout() << "arg:     " << *arg << '\n';
                    Logger::cout() << "expects: " << *callableTy->getParamType(j) << '\n';
                }
            #endif
                arg = DtoBitCast(arg, callableTy->getParamType(j));
            }

            // param attrs
            attrptr[i] = fnarg->llvmAttrs;

            ++argiter;
            args.push_back(arg);
        }

        // reverse the relevant params as well as the param attrs
        if (tf->reverseParams)
        {
            std::reverse(args.begin() + tf->reverseIndex, args.end());
            std::reverse(attrptr.begin(), attrptr.end());
        }

        // add attributes
        for (int i = 0; i < n; i++)
        {
            if (attrptr[i])
            {
                Attr.Index = beg + i + 1;
                Attr.Attrs = attrptr[i];
                attrs.push_back(Attr);
            }
        }

        // do C varargs
        if (n_arguments > n)
        {
            for (int i=n; i<n_arguments; i++)
            {
                Argument* fnarg = Argument::getNth(tf->parameters, i);
                DValue* argval = DtoArgument(fnarg, (Expression*)arguments->data[i]);
                LLValue* arg = argval->getRVal();

                // FIXME: do we need any param attrs here ?

                ++argiter;
                args.push_back(arg);
            }
        }
    }

    #if 1
    Logger::println("%lu params passed", args.size());
    for (int i=0; i<args.size(); ++i) {
        assert(args[i]);
        Logger::cout() << "arg["<<i<<"] = " << *args[i] << '\n';
    }
    #endif

    // void returns cannot not be named
    const char* varname = "";
    if (callableTy->getReturnType() != LLType::VoidTy)
        varname = "tmp";

    Logger::cout() << "Calling: " << *callable << '\n';

    // call the function
    CallOrInvoke* call = gIR->CreateCallOrInvoke(callable, args.begin(), args.end(), varname);

    // get return value
    LLValue* retllval = (retinptr) ? args[0] : call->get();

    // repaint the type if necessary
    if (resulttype)
    {
        Type* rbase = resulttype->toBasetype();
        Type* nextbase = tf->nextOf()->toBasetype();
    #if DMDV2
        rbase = rbase->mutableOf();
        nextbase = nextbase->mutableOf();
    #endif
        if (!rbase->equals(nextbase))
        {
            Logger::println("repainting return value from '%s' to '%s'", tf->nextOf()->toChars(), rbase->toChars());
            switch(rbase->ty)
            {
            case Tarray:
                retllval = DtoAggrPaint(retllval, DtoType(rbase));
                break;

            case Tclass:
            case Taarray:
            case Tpointer:
                retllval = DtoBitCast(retllval, DtoType(rbase));
                break;

            default:
                assert(0 && "unhandled repainting of return value");
            }
            if (Logger::enabled())
                Logger::cout() << "final return value: " << *retllval << '\n';
        }
    }

    // set calling convention and parameter attributes
    llvm::AttrListPtr attrlist = llvm::AttrListPtr::get(attrs.begin(), attrs.end());
    if (dfnval && dfnval->func)
    {
        LLFunction* llfunc = llvm::dyn_cast<LLFunction>(dfnval->val);
        if (llfunc && llfunc->isIntrinsic()) // override intrinsic attrs
            attrlist = llvm::Intrinsic::getAttributes((llvm::Intrinsic::ID)llfunc->getIntrinsicID());
        else
            call->setCallingConv(callconv);
    }
    else
        call->setCallingConv(callconv);
    call->setAttributes(attrlist);

    return new DImValue(resulttype, retllval);
}
