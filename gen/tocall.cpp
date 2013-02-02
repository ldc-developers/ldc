//===-- tocall.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"

#include "mtype.h"
#include "declaration.h"
#include "id.h"

#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/irstate.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/abi.h"
#include "gen/nested.h"

#include "gen/logger.h"

//////////////////////////////////////////////////////////////////////////////////////////

TypeFunction* DtoTypeFunction(DValue* fnval)
{
    Type* type = fnval->getType()->toBasetype();
    if (type->ty == Tfunction)
    {
         return static_cast<TypeFunction*>(type);
    }
    else if (type->ty == Tdelegate)
    {
        // FIXME: There is really no reason why the function type should be
        // unmerged at this stage, but the frontend still seems to produce such
        // cases; for example for the uint(uint) next type of the return type of
        // (&zero)(), leading to a crash in DtoCallFunction:
        // ---
        // void test8198() {
        //   uint delegate(uint) zero() { return null; }
        //   auto a = (&zero)()(0);
        // }
        // ---
        // Calling merge() here works around the symptoms, but does not fix the
        // root cause.

        Type* next = type->nextOf()->merge();
        assert(next->ty == Tfunction);
        return static_cast<TypeFunction*>(next);
    }

    assert(0 && "cant get TypeFunction* from non lazy/function/delegate");
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::CallingConv::ID DtoCallingConv(Loc loc, LINK l)
{
    if (l == LINKc || l == LINKcpp || l == LINKintrinsic)
        return llvm::CallingConv::C;
    else if (l == LINKd || l == LINKdefault)
    {
        //TODO: StdCall is not a good base on Windows due to extra name mangling
        // applied there
        if (global.params.cpu == ARCHx86 || global.params.cpu == ARCHx86_64)
            return (global.params.os != OSWindows) ? llvm::CallingConv::X86_StdCall : llvm::CallingConv::C;
        else
            return llvm::CallingConv::Fast;
    }
    // on the other hand, here, it's exactly what we want!!! TODO: right?
    // On Windows 64bit, there is only one calling convention!
    else if (l == LINKwindows)
        return global.params.cpu == ARCHx86_64 ? llvm::CallingConv::C : llvm::CallingConv::X86_StdCall;
    else if (l == LINKpascal)
        return llvm::CallingConv::X86_StdCall;
    else
    {
        error(loc, "unsupported calling convention");
        fatal();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoVaArg(Loc& loc, Type* type, Expression* valistArg)
{
    DValue* expelem = valistArg->toElem(gIR);
    LLType* llt = DtoType(type);
    if (DtoIsPassedByRef(type))
        llt = getPtrToType(llt);
    // issue a warning for broken va_arg instruction.
    if (global.params.cpu != ARCHx86)
        warning(Loc(), "%s: va_arg for C variadic functions is probably broken for anything but x86", loc.toChars());
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

LLFunctionType* DtoExtractFunctionType(LLType* type)
{
    if (LLFunctionType* fty = isaFunction(type))
        return fty;
    else if (LLPointerType* pty = isaPointer(type))
    {
        if (LLFunctionType* fty = isaFunction(pty->getElementType()))
            return fty;
    }
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

static LLValue *fixArgument(DValue *argval, TypeFunction* tf, LLType *callableArgType, size_t argIndex)
{
#if 0
    if (Logger::enabled()) {
        Logger::cout() << "Argument before ABI: " << *argval->getRVal() << '\n';
        Logger::cout() << "Argument type before ABI: " << *DtoType(argval->getType()) << '\n';
    }
#endif

    // give the ABI a say
    LLValue* arg = tf->fty.putParam(argval->getType(), argIndex, argval);

#if 0
    if (Logger::enabled()) {
        Logger::cout() << "Argument after ABI: " << *arg << '\n';
        Logger::cout() << "Argument type after ABI: " << *arg->getType() << '\n';
    }
#endif

    // Hack around LDC assuming structs and static arrays are in memory:
    // If the function wants a struct, and the argument value is a
    // pointer to a struct, load from it before passing it in.
    int ty = argval->getType()->toBasetype()->ty;
    if (isaPointer(arg) && !isaPointer(callableArgType) &&
#if DMDV2
        (ty == Tstruct || ty == Tsarray))
#else
        ty == Tstruct)
#endif
    {
        Logger::println("Loading struct type for function argument");
        arg = DtoLoad(arg);
    }

    // parameter type mismatch, this is hard to get rid of
    if (arg->getType() != callableArgType)
    {
    #if 1
        if (Logger::enabled())
        {
            Logger::cout() << "arg:     " << *arg << '\n';
            Logger::cout() << "of type: " << *arg->getType() << '\n';
            Logger::cout() << "expects: " << *callableArgType << '\n';
        }
    #endif
        if (isaStruct(arg))
            arg = DtoAggrPaint(arg, callableArgType);
        else
            arg = DtoBitCast(arg, callableArgType);
    }
    return arg;
}

//////////////////////////////////////////////////////////////////////////////////////////

#if LDC_LLVM_VER >= 303
static inline void addToAttributes(llvm::AttributeSet &Attrs,
                                   unsigned Idx, llvm::Attribute Attr)
{
    llvm::AttrBuilder Builder(Attr);
    Attrs = Attrs.addAttributes(gIR->context(), Idx,
                                llvm::AttributeSet::get(gIR->context(), Idx, Builder));
}
#else
static inline void addToAttributes(std::vector<llvm::AttributeWithIndex> &attrs,
                                   unsigned Idx, llvm::Attributes Attr)
{
    attrs.push_back(llvm::AttributeWithIndex::get(Idx, Attr));
}
#endif


//////////////////////////////////////////////////////////////////////////////////////////

void DtoBuildDVarArgList(std::vector<LLValue*>& args,
#if LDC_LLVM_VER >= 303
                         llvm::AttributeSet &attrs,
#else
                         std::vector<llvm::AttributeWithIndex> &attrs,
#endif
                         TypeFunction* tf, Expressions* arguments,
                         size_t argidx,
                         LLFunctionType* callableTy)
{
    Logger::println("doing d-style variadic arguments");
    LOG_SCOPE

    std::vector<LLType*> vtypes;

    // number of non variadic args
    int begin = Parameter::dim(tf->parameters);
    Logger::println("num non vararg params = %d", begin);

    // get n args in arguments list
    size_t n_arguments = arguments ? arguments->dim : 0;

    // build struct with argument types (non variadic args)
    for (size_t i=begin; i<n_arguments; i++)
    {
        Expression* argexp = static_cast<Expression*>(arguments->data[i]);
        assert(argexp->type->ty != Ttuple);
        vtypes.push_back(DtoType(argexp->type));
        size_t sz = getTypePaddedSize(vtypes.back());
        size_t asz = (sz + PTRSIZE - 1) & ~(PTRSIZE -1);
        if (sz != asz)
        {
            if (sz < PTRSIZE)
            {
                vtypes.back() = DtoSize_t();
            }
            else
            {
                // ok then... so we build some type that is big enough
                // and aligned to PTRSIZE
                std::vector<LLType*> gah;
                gah.reserve(asz/PTRSIZE);
                size_t gah_sz = 0;
                while (gah_sz < asz)
                {
                    gah.push_back(DtoSize_t());
                    gah_sz += PTRSIZE;
                }
                vtypes.back() = LLStructType::get(gIR->context(), gah, true);
            }
        }
    }
    LLStructType* vtype = LLStructType::get(gIR->context(), vtypes);

    if (Logger::enabled())
        Logger::cout() << "d-variadic argument struct type:\n" << *vtype << '\n';

    LLValue* mem = DtoRawAlloca(vtype, 0, "_argptr_storage");

    // store arguments in the struct
    for (size_t i=begin,k=0; i<n_arguments; i++,k++)
    {
        Expression* argexp = static_cast<Expression*>(arguments->data[i]);
        LLValue* argdst = DtoGEPi(mem,0,k);
        argdst = DtoBitCast(argdst, getPtrToType(DtoType(argexp->type)));
        DtoVariadicArgument(argexp, argdst);
    }

    // build type info array
    LLType* typeinfotype = DtoType(Type::typeinfo->type);
    LLArrayType* typeinfoarraytype = LLArrayType::get(typeinfotype,vtype->getNumElements());

    llvm::GlobalVariable* typeinfomem =
        new llvm::GlobalVariable(*gIR->module, typeinfoarraytype, true, llvm::GlobalValue::InternalLinkage, NULL, "._arguments.storage");
    if (Logger::enabled())
        Logger::cout() << "_arguments storage: " << *typeinfomem << '\n';

    std::vector<LLConstant*> vtypeinfos;
    for (size_t i=begin; i<n_arguments; i++)
    {
        Expression* argexp = static_cast<Expression*>(arguments->data[i]);
        vtypeinfos.push_back(DtoTypeInfoOf(argexp->type));
    }

    // apply initializer
    LLConstant* tiinits = LLConstantArray::get(typeinfoarraytype, vtypeinfos);
    typeinfomem->setInitializer(tiinits);

    // put data in d-array
    std::vector<LLConstant*> pinits;
    pinits.push_back(DtoConstSize_t(vtype->getNumElements()));
    pinits.push_back(llvm::ConstantExpr::getBitCast(typeinfomem, getPtrToType(typeinfotype)));
    LLType* tiarrty = DtoType(Type::typeinfo->type->arrayOf());
    tiinits = LLConstantStruct::get(isaStruct(tiarrty), pinits);
    LLValue* typeinfoarrayparam = new llvm::GlobalVariable(*gIR->module, tiarrty,
        true, llvm::GlobalValue::InternalLinkage, tiinits, "._arguments.array");

    // specify arguments
    args.push_back(DtoLoad(typeinfoarrayparam));
    if (HAS_ATTRIBUTES(tf->fty.arg_arguments->attrs)) {
        addToAttributes(attrs, argidx, tf->fty.arg_arguments->attrs);
    }
    ++argidx;

    args.push_back(gIR->ir->CreateBitCast(mem, getPtrToType(LLType::getInt8Ty(gIR->context())), "tmp"));
    if (HAS_ATTRIBUTES(tf->fty.arg_argptr->attrs)) {
        addToAttributes(attrs, argidx, tf->fty.arg_argptr->attrs);
    }

    // pass non variadic args
    for (int i=0; i<begin; i++)
    {
        Parameter* fnarg = Parameter::getNth(tf->parameters, i);
        DValue* argval = DtoArgument(fnarg, static_cast<Expression*>(arguments->data[i]));
        args.push_back(fixArgument(argval, tf, callableTy->getParamType(argidx++), i));

        if (HAS_ATTRIBUTES(tf->fty.args[i]->attrs))
        {
            addToAttributes(attrs, argidx, tf->fty.args[i]->attrs);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

// FIXME: this function is a mess !

DValue* DtoCallFunction(Loc& loc, Type* resulttype, DValue* fnval, Expressions* arguments)
{
    if (Logger::enabled()) {
        Logger::println("DtoCallFunction()");
    }
    LOG_SCOPE

    // the callee D type
    Type* calleeType = fnval->getType();

    // make sure the callee type has been processed
    DtoType(calleeType);

    // get func value if any
    DFuncValue* dfnval = fnval->isFunc();

    // handle special vararg intrinsics
    bool va_intrinsic = (dfnval && dfnval->func && dfnval->func->isVaIntrinsic());

    // get function type info
    TypeFunction* tf = DtoTypeFunction(fnval);

    // misc
    bool retinptr = tf->fty.arg_sret;
    bool thiscall = tf->fty.arg_this;
    bool delegatecall = (calleeType->toBasetype()->ty == Tdelegate);
    bool nestedcall = tf->fty.arg_nest;
    bool dvarargs = (tf->linkage == LINKd && tf->varargs == 1);

    llvm::CallingConv::ID callconv = DtoCallingConv(loc, tf->linkage);

    // get callee llvm value
    LLValue* callable = DtoCallableValue(fnval);
    LLFunctionType* callableTy = DtoExtractFunctionType(callable->getType());
    assert(callableTy);

//     if (Logger::enabled())
//         Logger::cout() << "callable: " << *callable << '\n';

    // get n arguments
    size_t n_arguments = arguments ? arguments->dim : 0;

    // get llvm argument iterator, for types
    LLFunctionType::param_iterator argbegin = callableTy->param_begin();
    LLFunctionType::param_iterator argiter = argbegin;

    // parameter attributes
#if LDC_LLVM_VER >= 303
    llvm::AttributeSet attrs;
#else
    std::vector<llvm::AttributeWithIndex> attrs;
#endif

    // return attrs
    if (HAS_ATTRIBUTES(tf->fty.ret->attrs))
    {
        addToAttributes(attrs, 0, tf->fty.ret->attrs);
    }

    // handle implicit arguments
    std::vector<LLValue*> args;
    args.reserve(tf->fty.args.size());

    // return in hidden ptr is first
    if (retinptr)
    {
        LLValue* retvar = DtoRawAlloca((*argiter)->getContainedType(0), resulttype->alignsize(), ".rettmp");
        ++argiter;
        args.push_back(retvar);

        // add attrs for hidden ptr
#if LDC_LLVM_VER >= 303
        const unsigned Index = 1;
        llvm::AttrBuilder builder(tf->fty.arg_sret->attrs);
        assert((builder.contains(llvm::Attribute::StructRet) || builder.contains(llvm::Attribute::InReg))
            && "Sret arg not sret or inreg?");
        llvm::AttributeSet as = llvm::AttributeSet::get(gIR->context(), Index, builder);
        attrs = attrs.addAttributes(gIR->context(), Index, as);
#else
        llvm::AttributeWithIndex Attr;
        Attr.Index = 1;
        Attr.Attrs = tf->fty.arg_sret->attrs;
#if LDC_LLVM_VER == 302
        assert((Attr.Attrs.hasAttribute(llvm::Attributes::StructRet) || Attr.Attrs.hasAttribute(llvm::Attributes::InReg))
            && "Sret arg not sret or inreg?");
#else
        assert((Attr.Attrs & (llvm::Attribute::StructRet | llvm::Attribute::InReg))
            && "Sret arg not sret or inreg?");
#endif
        attrs.push_back(Attr);
#endif
    }

    // then comes a context argument...
    if(thiscall || delegatecall || nestedcall)
    {
#if DMDV2
        if (dfnval && (dfnval->func->ident == Id::ensure || dfnval->func->ident == Id::require)) {
            // ... which can be the this "context" argument for a contract
            // invocation (in D2, we do not generate a full nested contexts
            // for __require/__ensure as the needed parameters are passed
            // explicitly, while in D1, the normal nested function handling
            // mechanisms are used)
            LLValue* thisarg = DtoBitCast(DtoLoad(gIR->func()->thisArg), getVoidPtrType());
            ++argiter;
            args.push_back(thisarg);
        }
        else
#endif
        if (thiscall && dfnval && dfnval->vthis)
        {
            // ... or a normal 'this' argument
            LLValue* thisarg = DtoBitCast(dfnval->vthis, *argiter);
            ++argiter;
            args.push_back(thisarg);
        }
        else if (delegatecall)
        {
            // ... or a delegate context arg
            LLValue* ctxarg;
            if (fnval->isLVal())
            {
                ctxarg = DtoLoad(DtoGEPi(fnval->getLVal(), 0,0));
            }
            else
            {
                ctxarg = gIR->ir->CreateExtractValue(fnval->getRVal(), 0, ".ptr");
            }
            ctxarg = DtoBitCast(ctxarg, *argiter);
            ++argiter;
            args.push_back(ctxarg);
        }
        else if (nestedcall)
        {
            // ... or a nested function context arg
            if (dfnval) {
                LLValue* contextptr = DtoNestedContext(loc, dfnval->func);
                contextptr = DtoBitCast(contextptr, getVoidPtrType());
                args.push_back(contextptr);
            } else {
                args.push_back(llvm::UndefValue::get(getVoidPtrType()));
            }
            ++argiter;
        }
        else
        {
            error(loc, "Context argument required but none given");
            fatal();
        }

        // add attributes for context argument
        if (tf->fty.arg_this && HAS_ATTRIBUTES(tf->fty.arg_this->attrs))
        {
            addToAttributes(attrs, retinptr ? 2 : 1, tf->fty.arg_this->attrs);
        }
        else if (tf->fty.arg_nest && HAS_ATTRIBUTES(tf->fty.arg_nest->attrs))
        {
            addToAttributes(attrs, retinptr ? 2 : 1, tf->fty.arg_nest->attrs);
        }
    }

    // handle the rest of the arguments based on param passing style

    // variadic intrinsics need some custom casts
    if (va_intrinsic)
    {
        for (size_t i=0; i<n_arguments; i++)
        {
            Expression* exp = static_cast<Expression*>(arguments->data[i]);
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
        DtoBuildDVarArgList(args, attrs, tf, arguments, argiter-argbegin+1, callableTy);
    }

    // otherwise we're looking at a normal function call
    // or a C style vararg call
    else
    {
        Logger::println("doing normal arguments");
        if (Logger::enabled()) {
            Logger::println("Arguments so far: (%d)", static_cast<int>(args.size()));
            Logger::indent();
            for (size_t i = 0; i < args.size(); i++) {
                Logger::cout() << *args[i] << '\n';
            }
            Logger::undent();
            Logger::cout() << "Function type: " << tf->toChars() << '\n';
            //Logger::cout() << "LLVM functype: " << *callable->getType() << '\n';
        }

        size_t n = Parameter::dim(tf->parameters);
        std::vector<DValue*> argvals;
        if (dfnval && dfnval->func->isArrayOp) {
            // slightly different approach for array operators
            for (int i=n-1; i>=0; --i) {
                Parameter* fnarg = Parameter::getNth(tf->parameters, i);
                assert(fnarg);
                DValue* argval = DtoArgument(fnarg, static_cast<Expression*>(arguments->data[i]));
                argvals.insert(argvals.begin(), argval);
            }
        } else {
            for (size_t i=0; i<n; ++i) {
                Parameter* fnarg = Parameter::getNth(tf->parameters, i);
                assert(fnarg);
                DValue* argval = DtoArgument(fnarg, static_cast<Expression*>(arguments->data[i]));
                argvals.push_back(argval);
            }
        }

#if LDC_LLVM_VER == 302
        LLSmallVector<llvm::Attributes, 10> attrptr(n, llvm::Attributes());
#elif LDC_LLVM_VER < 302
        LLSmallVector<llvm::Attributes, 10> attrptr(n, llvm::Attribute::None);
#endif
        // do formal params
        int beg = argiter-argbegin;
        for (size_t i=0; i<n; i++)
        {
            DValue* argval = argvals.at(i);

            int j = tf->fty.reverseParams ? beg + n - i - 1 : beg + i;
            LLValue *arg = fixArgument(argval, tf, callableTy->getParamType(j), i);
            args.push_back(arg);

#if LDC_LLVM_VER >= 303
            addToAttributes(attrs, beg + 1 + (tf->fty.reverseParams ? n-i-1: i), tf->fty.args[i]->attrs);
#else
            attrptr[i] = tf->fty.args[i]->attrs;
#endif
            ++argiter;
        }

        // reverse the relevant params as well as the param attrs
        if (tf->fty.reverseParams)
        {
            std::reverse(args.begin() + beg, args.end());
#if LDC_LLVM_VER < 303
            std::reverse(attrptr.begin(), attrptr.end());
#endif
        }

#if LDC_LLVM_VER < 303
        // add attributes
        for (size_t i = 0; i < n; i++)
        {
            if (HAS_ATTRIBUTES(attrptr[i]))
            {
                addToAttributes(attrs, beg + i + 1, attrptr[i]);
            }
        }
#endif
        // do C varargs
        if (n_arguments > n)
        {
            for (size_t i=n; i<n_arguments; i++)
            {
                Parameter* fnarg = Parameter::getNth(tf->parameters, i);
                DValue* argval = DtoArgument(fnarg, static_cast<Expression*>(arguments->data[i]));
                LLValue* arg = argval->getRVal();

                // FIXME: do we need any param attrs here ?

                ++argiter;
                args.push_back(arg);
            }
        }
    }

#if 0
    if (Logger::enabled())
    {
        Logger::println("%lu params passed", args.size());
        for (int i=0; i<args.size(); ++i) {
            assert(args[i]);
            Logger::cout() << "arg["<<i<<"] = " << *args[i] << '\n';
        }
    }
#endif

    // void returns cannot not be named
    const char* varname = "";
    if (callableTy->getReturnType() != LLType::getVoidTy(gIR->context()))
        varname = "tmp";

#if 0
    if (Logger::enabled())
        Logger::cout() << "Calling: " << *callable << '\n';
#endif

    // call the function
    LLCallSite call = gIR->CreateCallOrInvoke(callable, args, varname);

    // get return value
    LLValue* retllval = (retinptr) ? args[0] : call.getInstruction();

    // Ignore ABI for intrinsics
    if (tf->linkage != LINKintrinsic && !retinptr)
    {
        // do abi specific return value fixups
        DImValue dretval(tf->next, retllval);
        retllval = tf->fty.getRet(tf->next, &dretval);
    }

    // Hack around LDC assuming structs and static arrays are in memory:
    // If the function returns a struct or a static array, and the return
    // value is not a pointer to a struct or a static array, store it to
    // a stack slot before continuing.
    int ty = tf->next->toBasetype()->ty;
    if ((ty == Tstruct && !isaPointer(retllval))
#if DMDV2
        || (ty == Tsarray && isaArray(retllval))
#endif
        )
    {
        Logger::println("Storing return value to stack slot");
        LLValue* mem = DtoRawAlloca(retllval->getType(), 0);
        DtoStore(retllval, mem);
        retllval = mem;
    }

    // repaint the type if necessary
    if (resulttype)
    {
        Type* rbase = stripModifiers(resulttype->toBasetype());
        Type* nextbase = stripModifiers(tf->nextOf()->toBasetype());
        if (!rbase->equals(nextbase))
        {
            Logger::println("repainting return value from '%s' to '%s'", tf->nextOf()->toChars(), rbase->toChars());
            switch(rbase->ty)
            {
            case Tarray:
            #if DMDV2
                if (tf->isref)
                    retllval = DtoBitCast(retllval, DtoType(rbase->pointerTo()));
                else
            #endif
                retllval = DtoAggrPaint(retllval, DtoType(rbase));
                break;

            case Tsarray:
                // nothing ?
                break;

            case Tclass:
            case Taarray:
            case Tpointer:
            #if DMDV2
                if (tf->isref)
                    retllval = DtoBitCast(retllval, DtoType(rbase->pointerTo()));
                else
            #endif
                retllval = DtoBitCast(retllval, DtoType(rbase));
                break;

#if DMDV2
            case Tstruct:
                if (nextbase->ty == Taarray && !tf->isref)
                {
                    // In the D2 frontend, the associative array type and its
                    // object.AssociativeArray representation are used
                    // interchangably in some places. However, AAs are returned
                    // by value and not in an sret argument, so if the struct
                    // type will be used, give the return value storage here
                    // so that we get the right amount of indirections.
                    LLValue* tmp = DtoAlloca(rbase, ".aalvauetmp");
                    LLValue* val = DtoInsertValue(
                        llvm::UndefValue::get(DtoType(rbase)), retllval, 0);
                    DtoStore(val, tmp);
                    retllval = tmp;
                    retinptr = true;
                    break;
                }
                // Fall through.
#endif

            default:
                // Unfortunately, DMD has quirks resp. bugs with regard to name
                // mangling: For voldemort-type functions which return a nested
                // struct, the mangled name of the return type changes during
                // semantic analysis.
                //
                // (When the function deco is first computed as part of
                // determining the return type deco, its return type part is
                // left off to avoid cycles. If mangle/toDecoBuffer is then
                // called again for the type, it will pick up the previous
                // result and return the full deco string for the nested struct
                // type, consisting of both the full mangled function name, and
                // the struct identifier.)
                //
                // Thus, the type merging in stripModifiers does not work
                // reliably, and the equality check above can fail even if the
                // types only differ in a qualifier.
                //
                // Because a proper fix for this in the frontend is hard, we
                // just carry on and hope that the frontend didn't mess up,
                // i.e. that the LLVM types really match up.
                //
                // An example situation where this case occurs is:
                // ---
                // auto iota() {
                //     static struct Result {
                //         this(int) {}
                //         inout(Result) test() inout { return cast(inout)Result(0); }
                //     }
                //     return Result.init;
                // }
                // void main() { auto r = iota(); }
                // ---
                Logger::println("Unknown return mismatch type, ignoring.");
                break;
            }
            if (Logger::enabled())
                Logger::cout() << "final return value: " << *retllval << '\n';
        }
    }

    // set calling convention and parameter attributes
#if LDC_LLVM_VER >= 303
	llvm::AttributeSet attrlist = attrs;
#elif LDC_LLVM_VER == 302
	llvm::AttrListPtr attrlist = llvm::AttrListPtr::get(gIR->context(),
        llvm::ArrayRef<llvm::AttributeWithIndex>(attrs));
#else
    llvm::AttrListPtr attrlist = llvm::AttrListPtr::get(attrs.begin(), attrs.end());
#endif
    if (dfnval && dfnval->func)
    {
        LLFunction* llfunc = llvm::dyn_cast<LLFunction>(dfnval->val);
        if (llfunc && llfunc->isIntrinsic()) // override intrinsic attrs
#if LDC_LLVM_VER >= 302
            attrlist = llvm::Intrinsic::getAttributes(gIR->context(), static_cast<llvm::Intrinsic::ID>(llfunc->getIntrinsicID()));
#else
            attrlist = llvm::Intrinsic::getAttributes(static_cast<llvm::Intrinsic::ID>(llfunc->getIntrinsicID()));
#endif
        else
            call.setCallingConv(callconv);
    }
    else
        call.setCallingConv(callconv);
    call.setAttributes(attrlist);

    // if we are returning through a pointer arg
    // or if we are returning a reference
    // make sure we provide a lvalue back!
    if (retinptr
#if DMDV2
        || tf->isref
#endif
        )
        return new DVarValue(resulttype, retllval);

    return new DImValue(resulttype, retllval);
}
