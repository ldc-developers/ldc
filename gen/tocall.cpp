//===-- tocall.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "declaration.h"
#include "id.h"
#include "mtype.h"
#include "target.h"
#include "pragma.h"
#include "gen/abi.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/nested.h"
#include "gen/tollvm.h"

//////////////////////////////////////////////////////////////////////////////////////////

IrFuncTy &DtoIrTypeFunction(DValue* fnval)
{
    if (DFuncValue* dfnval = fnval->isFunc())
    {
        if (dfnval->func)
            return dfnval->func->irFty;
    }

    Type* type = stripModifiers(fnval->getType()->toBasetype());
    if (type->ty == Tfunction)
        return static_cast<TypeFunction*>(type)->irFty;
    else if (type->ty == Tdelegate)
        return static_cast<TypeDelegate*>(type)->irFty;

    llvm_unreachable("Cannot get IrFuncTy from non lazy/function/delegate");
}

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

    llvm_unreachable("Cannot get TypeFunction* from non lazy/function/delegate");
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoVaArg(Loc& loc, Type* type, Expression* valistArg)
{
    DValue* expelem = toElem(valistArg);
    LLType* llt = DtoType(type);
    if (DtoIsPassedByRef(type))
        llt = getPtrToType(llt);
    // issue a warning for broken va_arg instruction.
    if (global.params.targetTriple.getArch() != llvm::Triple::x86
        && global.params.targetTriple.getArch() != llvm::Triple::ppc64
#if LDC_LLVM_VER >= 305
        && global.params.targetTriple.getArch() != llvm::Triple::ppc64le
#endif
        )
        warning(loc, "va_arg for C variadic functions is probably broken for anything but x86 and ppc64");
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

    llvm_unreachable("Not a callable type.");
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

static LLValue *fixArgument(DValue *argval, IrFuncTy &irFty, LLType *callableArgType, size_t argIndex)
{
#if 0
    IF_LOG {
        Logger::cout() << "Argument before ABI: " << *argval->getRVal() << '\n';
        Logger::cout() << "Argument type before ABI: " << *DtoType(argval->getType()) << '\n';
    }
#endif

    // give the ABI a say
    LLValue* arg = irFty.putParam(argval->getType(), argIndex, argval);

#if 0
    IF_LOG {
        Logger::cout() << "Argument after ABI: " << *arg << '\n';
        Logger::cout() << "Argument type after ABI: " << *arg->getType() << '\n';
    }
#endif

    // Hack around LDC assuming structs and static arrays are in memory:
    // If the function wants a struct, and the argument value is a
    // pointer to a struct, load from it before passing it in.
    int ty = argval->getType()->toBasetype()->ty;
    if (isaPointer(arg) && !isaPointer(callableArgType) &&
        (ty == Tstruct || ty == Tsarray))
    {
        Logger::println("Loading struct type for function argument");
        arg = DtoLoad(arg);
    }

    // parameter type mismatch, this is hard to get rid of
    if (arg->getType() != callableArgType)
    {
    #if 1
        IF_LOG {
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
                                   unsigned Idx, llvm::AttrBuilder B)
{
    llvm::AttrBuilder Builder(B);
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
                         TypeFunction* tf, IrFuncTy &irFty,
                         Expressions* arguments, size_t argidx,
                         LLFunctionType* callableTy)
{
    IF_LOG Logger::println("doing d-style variadic arguments");
    LOG_SCOPE

    std::vector<LLType*> vtypes;

    // number of non variadic args
    int begin = Parameter::dim(tf->parameters);
    IF_LOG Logger::println("num non vararg params = %d", begin);

    // get n args in arguments list
    size_t n_arguments = arguments ? arguments->dim : 0;

    // build struct with argument types (non variadic args)
    for (size_t i=begin; i<n_arguments; i++)
    {
        Expression* argexp = static_cast<Expression*>(arguments->data[i]);
        assert(argexp->type->ty != Ttuple);
        vtypes.push_back(DtoType(argexp->type));
        size_t sz = getTypePaddedSize(vtypes.back());
        size_t asz = (sz + Target::ptrsize - 1) & ~(Target::ptrsize -1);
        if (sz != asz)
        {
            if (sz < Target::ptrsize)
            {
                vtypes.back() = DtoSize_t();
            }
            else
            {
                // ok then... so we build some type that is big enough
                // and aligned to Target::ptrsize
                std::vector<LLType*> gah;
                gah.reserve(asz/Target::ptrsize);
                size_t gah_sz = 0;
                while (gah_sz < asz)
                {
                    gah.push_back(DtoSize_t());
                    gah_sz += Target::ptrsize;
                }
                vtypes.back() = LLStructType::get(gIR->context(), gah, true);
            }
        }
    }
    LLStructType* vtype = LLStructType::get(gIR->context(), vtypes);

    IF_LOG Logger::cout() << "d-variadic argument struct type:\n" << *vtype << '\n';

    LLValue* mem = DtoRawAlloca(vtype, 0, "_argptr_storage");

    // store arguments in the struct
    for (size_t i=begin,k=0; i<n_arguments; i++,k++)
    {
        Expression* argexp = static_cast<Expression*>(arguments->data[i]);
        LLValue* argdst = DtoGEPi(mem,0,k);
        argdst = DtoBitCast(argdst, getPtrToType(i1ToI8(DtoType(argexp->type))));
        DtoVariadicArgument(argexp, argdst);
    }

    // build type info array
    LLType* typeinfotype = DtoType(Type::dtypeinfo->type);
    LLArrayType* typeinfoarraytype = LLArrayType::get(typeinfotype,vtype->getNumElements());

    llvm::GlobalVariable* typeinfomem =
        new llvm::GlobalVariable(*gIR->module, typeinfoarraytype, true, llvm::GlobalValue::InternalLinkage, NULL, "._arguments.storage");
    IF_LOG Logger::cout() << "_arguments storage: " << *typeinfomem << '\n';

    std::vector<LLConstant*> vtypeinfos;
    vtypeinfos.reserve(n_arguments);
    for (size_t i=begin; i<n_arguments; i++)
    {
        Expression* argexp = static_cast<Expression*>(arguments->data[i]);
        vtypeinfos.push_back(DtoTypeInfoOf(argexp->type));
    }

    // apply initializer
    LLConstant* tiinits = LLConstantArray::get(typeinfoarraytype, vtypeinfos);
    typeinfomem->setInitializer(tiinits);

    // put data in d-array
    LLConstant* pinits[] = {
        DtoConstSize_t(vtype->getNumElements()),
        llvm::ConstantExpr::getBitCast(typeinfomem, getPtrToType(typeinfotype))
    };
    LLType* tiarrty = DtoType(Type::dtypeinfo->type->arrayOf());
    tiinits = LLConstantStruct::get(isaStruct(tiarrty), llvm::ArrayRef<LLConstant*>(pinits));
    LLValue* typeinfoarrayparam = new llvm::GlobalVariable(*gIR->module, tiarrty,
        true, llvm::GlobalValue::InternalLinkage, tiinits, "._arguments.array");

    // specify arguments
    args.push_back(DtoLoad(typeinfoarrayparam));
    if (HAS_ATTRIBUTES(irFty.arg_arguments->attrs)) {
        addToAttributes(attrs, argidx, irFty.arg_arguments->attrs);
    }
    ++argidx;

    args.push_back(gIR->ir->CreateBitCast(mem, getPtrToType(LLType::getInt8Ty(gIR->context())), "tmp"));
    if (HAS_ATTRIBUTES(irFty.arg_argptr->attrs)) {
        addToAttributes(attrs, argidx, irFty.arg_argptr->attrs);
    }

    // pass non variadic args
    for (int i=0; i<begin; i++)
    {
        Parameter* fnarg = Parameter::getNth(tf->parameters, i);
        DValue* argval = DtoArgument(fnarg, static_cast<Expression*>(arguments->data[i]));
        args.push_back(fixArgument(argval, irFty, callableTy->getParamType(argidx++), i));

        if (HAS_ATTRIBUTES(irFty.args[i]->attrs))
        {
            addToAttributes(attrs, argidx, irFty.args[i]->attrs);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

// FIXME: this function is a mess !

DValue* DtoCallFunction(Loc& loc, Type* resulttype, DValue* fnval, Expressions* arguments, llvm::Value *retvar)
{
    IF_LOG Logger::println("DtoCallFunction()");
    LOG_SCOPE

    // the callee D type
    Type* calleeType = fnval->getType();

    // make sure the callee type has been processed
    DtoType(calleeType);

    // get func value if any
    DFuncValue* dfnval = fnval->isFunc();

    // handle intrinsics
    bool intrinsic = (dfnval && dfnval->func && dfnval->func->llvmInternal == LLVMintrinsic);

    // handle special vararg intrinsics
    bool va_intrinsic = (dfnval && dfnval->func && dfnval->func->isVaIntrinsic());

    // get function type info
    IrFuncTy &irFty = DtoIrTypeFunction(fnval);
    TypeFunction* tf = DtoTypeFunction(fnval);

    // misc
    bool retinptr = irFty.arg_sret;
    bool thiscall = irFty.arg_this;
    bool delegatecall = (calleeType->toBasetype()->ty == Tdelegate);
    bool nestedcall = irFty.arg_nest;
    bool dvarargs = (tf->linkage == LINKd && tf->varargs == 1);

    llvm::CallingConv::ID callconv = gABI->callingConv(tf->linkage);

    // get callee llvm value
    LLValue* callable = DtoCallableValue(fnval);
    LLFunctionType* callableTy = DtoExtractFunctionType(callable->getType());
    assert(callableTy);

//     IF_LOG Logger::cout() << "callable: " << *callable << '\n';

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
    if (HAS_ATTRIBUTES(irFty.ret->attrs))
    {
        addToAttributes(attrs, 0, irFty.ret->attrs);
    }

    // handle implicit arguments
    std::vector<LLValue*> args;
    args.reserve(irFty.args.size());

    // return in hidden ptr is first
    if (retinptr)
    {
        if (!retvar)
            retvar = DtoRawAlloca((*argiter)->getContainedType(0), resulttype->alignsize(), ".rettmp");
        ++argiter;
        args.push_back(retvar);

        // add attrs for hidden ptr
#if LDC_LLVM_VER >= 303
        const unsigned Index = 1;
        llvm::AttrBuilder builder(irFty.arg_sret->attrs);
        assert((builder.contains(llvm::Attribute::StructRet) || builder.contains(llvm::Attribute::InReg))
            && "Sret arg not sret or inreg?");
        llvm::AttributeSet as = llvm::AttributeSet::get(gIR->context(), Index, builder);
        attrs = attrs.addAttributes(gIR->context(), Index, as);
#else
        llvm::AttributeWithIndex Attr;
        Attr.Index = 1;
        Attr.Attrs = irFty.arg_sret->attrs;
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
        if (irFty.arg_this && HAS_ATTRIBUTES(irFty.arg_this->attrs))
        {
            addToAttributes(attrs, retinptr ? 2 : 1, irFty.arg_this->attrs);
        }
        else if (irFty.arg_nest && HAS_ATTRIBUTES(irFty.arg_nest->attrs))
        {
            addToAttributes(attrs, retinptr ? 2 : 1, irFty.arg_nest->attrs);
        }
    }

    // handle the rest of the arguments based on param passing style

    // variadic intrinsics need some custom casts
    if (va_intrinsic)
    {
        for (size_t i=0; i<n_arguments; i++)
        {
            Expression* exp = static_cast<Expression*>(arguments->data[i]);
            DValue* expelem = toElem(exp);
            // cast to va_list*
            LLValue* val = DtoBitCast(expelem->getLVal(), getVoidPtrType());
            ++argiter;
            args.push_back(val);
        }
    }

    // d style varargs needs a few more hidden arguments as well as special passing
    else if (dvarargs)
    {
        DtoBuildDVarArgList(args, attrs, tf, irFty, arguments, argiter-argbegin+1, callableTy);
    }

    // otherwise we're looking at a normal function call
    // or a C style vararg call
    else
    {
        Logger::println("doing normal arguments");
        IF_LOG {
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
        argvals.reserve(n);
        if (dfnval && dfnval->func->isArrayOp) {
            // For array ops, the druntime implementation signatures are crafted
            // specifically such that the evaluation order is as expected with
            // the strange DMD reverse parameter passing order. Thus, we need
            // to actually build the arguments right-to-left for them.
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

            int j = irFty.reverseParams ? beg + n - i - 1 : beg + i;
            LLValue *arg = fixArgument(argval, irFty, callableTy->getParamType(j), i);
            args.push_back(arg);

#if LDC_LLVM_VER >= 303
            addToAttributes(attrs, beg + 1 + (irFty.reverseParams ? n-i-1: i), irFty.args[i]->attrs);
#else
            attrptr[i] = irFty.args[i]->attrs;
#endif
            ++argiter;
        }

        // reverse the relevant params as well as the param attrs
        if (irFty.reverseParams)
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
    IF_LOG {
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
    IF_LOG Logger::cout() << "Calling: " << *callable << '\n';
#endif

    // call the function
    LLCallSite call = gIR->CreateCallOrInvoke(callable, args, varname);

    // get return value
    LLValue* retllval = (retinptr) ? args[0] : call.getInstruction();

    // Ignore ABI for intrinsics
    if (!intrinsic && !retinptr)
    {
        // do abi specific return value fixups
        DImValue dretval(tf->next, retllval);
        retllval = irFty.getRet(tf->next, &dretval);
    }

    // Hack around LDC assuming structs and static arrays are in memory:
    // If the function returns a struct or a static array, and the return
    // value is not a pointer to a struct or a static array, store it to
    // a stack slot before continuing.
    int ty = tf->next->toBasetype()->ty;
    if ((ty == Tstruct && !isaPointer(retllval))
        || (ty == Tsarray && isaArray(retllval))
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
            IF_LOG Logger::println("repainting return value from '%s' to '%s'", tf->nextOf()->toChars(), rbase->toChars());
            switch(rbase->ty)
            {
            case Tarray:
                if (tf->isref)
                    retllval = DtoBitCast(retllval, DtoType(rbase->pointerTo()));
                else
                retllval = DtoAggrPaint(retllval, DtoType(rbase));
                break;

            case Tsarray:
                // nothing ?
                break;

            case Tclass:
            case Taarray:
            case Tpointer:
                if (tf->isref)
                    retllval = DtoBitCast(retllval, DtoType(rbase->pointerTo()));
                else
                retllval = DtoBitCast(retllval, DtoType(rbase));
                break;

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
            IF_LOG Logger::cout() << "final return value: " << *retllval << '\n';
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
        || tf->isref
        )
        return new DVarValue(resulttype, retllval);

    return new DImValue(resulttype, retllval);
}
