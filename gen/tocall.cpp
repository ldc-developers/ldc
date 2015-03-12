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
#include "ir/irtype.h"

//////////////////////////////////////////////////////////////////////////////////////////

IrFuncTy &DtoIrTypeFunction(DValue* fnval)
{
    if (DFuncValue* dfnval = fnval->isFunc())
    {
        if (dfnval->func)
            return getIrFunc(dfnval->func)->irFty;
    }

    Type* type = stripModifiers(fnval->getType()->toBasetype());
    DtoType(type);
    assert(type->ctype);
    return type->ctype->getIrFuncTy();
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
            return DtoLoad(funcptr, ".funcptr");
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

static void addExplicitArguments(std::vector<LLValue*>& args, AttrSet& attrs,
    IrFuncTy& irFty, LLFunctionType* callableTy, const std::vector<DValue*>& argvals, int numFormalParams)
{
    const int numImplicitArgs = args.size();
    const int numExplicitArgs = argvals.size();

    args.resize(numImplicitArgs + numExplicitArgs, static_cast<LLValue*>(0));

    // construct and initialize an IrFuncTyArg object for each vararg
    std::vector<IrFuncTyArg*> optionalIrArgs;
    for (int i = numFormalParams; i < numExplicitArgs; i++) {
        Type* argType = argvals[i]->getType();
        bool passByVal = gABI->passByVal(argType);

        AttrBuilder initialAttrs;
        if (passByVal)
            initialAttrs.add(LDC_ATTRIBUTE(ByVal));
        else
            initialAttrs.add(DtoShouldExtend(argType));

        optionalIrArgs.push_back(new IrFuncTyArg(argType, passByVal, initialAttrs));
    }

    // let the ABI rewrite the IrFuncTyArg objects
    gABI->rewriteVarargs(irFty, optionalIrArgs);

    for (int i = 0; i < numExplicitArgs; i++)
    {
        int j = numImplicitArgs + (irFty.reverseParams ? numExplicitArgs - i - 1 : i);

        DValue* argval = argvals[i];
        Type* argType = argval->getType();

        const bool isVararg = (i >= numFormalParams);
        IrFuncTyArg* irArg = NULL;
        LLValue* arg = NULL;

        if (!isVararg)
        {
            irArg = irFty.args[i];
            arg = irFty.putParam(argType, i, argval);
        }
        else
        {
            irArg = optionalIrArgs[i - numFormalParams];
            arg = irFty.putParam(argType, *irArg, argval);
        }

        LLType* callableArgType = (isVararg ? NULL : callableTy->getParamType(j));

        // Hack around LDC assuming structs and static arrays are in memory:
        // If the function wants a struct, and the argument value is a
        // pointer to a struct, load from it before passing it in.
        if (isaPointer(arg) && DtoIsPassedByRef(argType) &&
            ( (!isVararg && !isaPointer(callableArgType)) ||
              (isVararg && !irArg->byref && !irArg->isByVal()) ) )
        {
            Logger::println("Loading struct type for function argument");
            arg = DtoLoad(arg);
        }

        // parameter type mismatch, this is hard to get rid of
        if (!isVararg && arg->getType() != callableArgType)
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

        args[j] = arg;
        attrs.add(j + 1, irArg->attrs);

        if (isVararg)
            delete irArg;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

static LLValue* getTypeinfoArrayArgumentForDVarArg(Expressions* arguments, int begin)
{
    IF_LOG Logger::println("doing d-style variadic arguments");
    LOG_SCOPE

    // number of non variadic args
    IF_LOG Logger::println("num non vararg params = %d", begin);

    // get n args in arguments list
    size_t n_arguments = arguments ? arguments->dim : 0;

    const size_t numVariadicArgs = n_arguments - begin;

    // build type info array
    LLType* typeinfotype = DtoType(Type::dtypeinfo->type);
    LLArrayType* typeinfoarraytype = LLArrayType::get(typeinfotype, numVariadicArgs);

    llvm::GlobalVariable* typeinfomem =
        new llvm::GlobalVariable(*gIR->module, typeinfoarraytype, true, llvm::GlobalValue::InternalLinkage, NULL, "._arguments.storage");
    IF_LOG Logger::cout() << "_arguments storage: " << *typeinfomem << '\n';

    std::vector<LLConstant*> vtypeinfos;
    vtypeinfos.reserve(n_arguments);
    for (size_t i=begin; i<n_arguments; i++)
    {
        vtypeinfos.push_back(DtoTypeInfoOf((*arguments)[i]->type));
    }

    // apply initializer
    LLConstant* tiinits = LLConstantArray::get(typeinfoarraytype, vtypeinfos);
    typeinfomem->setInitializer(tiinits);

    // put data in d-array
    LLConstant* pinits[] = {
        DtoConstSize_t(numVariadicArgs),
        llvm::ConstantExpr::getBitCast(typeinfomem, getPtrToType(typeinfotype))
    };
    LLType* tiarrty = DtoType(Type::dtypeinfo->type->arrayOf());
    tiinits = LLConstantStruct::get(isaStruct(tiarrty), llvm::ArrayRef<LLConstant*>(pinits));
    LLValue* typeinfoarrayparam = new llvm::GlobalVariable(*gIR->module, tiarrty,
        true, llvm::GlobalValue::InternalLinkage, tiinits, "._arguments.array");

    return DtoLoad(typeinfoarrayparam);
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

    // get function type info
    IrFuncTy &irFty = DtoIrTypeFunction(fnval);
    TypeFunction* tf = DtoTypeFunction(fnval);

    // misc
    bool retinptr = irFty.arg_sret;
    bool thiscall = irFty.arg_this;
    bool delegatecall = (calleeType->toBasetype()->ty == Tdelegate);
    bool nestedcall = irFty.arg_nest;
    bool dvarargs = irFty.arg_arguments;

    llvm::CallingConv::ID callconv = gABI->callingConv(tf->linkage);

    // get callee llvm value
    LLValue* callable = DtoCallableValue(fnval);
    LLFunctionType* callableTy = DtoExtractFunctionType(callable->getType());
    assert(callableTy);

//     IF_LOG Logger::cout() << "callable: " << *callable << '\n';

    // get number of explicit arguments
    size_t n_arguments = arguments ? arguments->dim : 0;

    // get llvm argument iterator, for types
    LLFunctionType::param_iterator argTypesBegin = callableTy->param_begin();

    // parameter attributes
    AttrSet attrs;

    // return attrs
    attrs.add(0, irFty.ret->attrs);

    // handle implicit arguments
    std::vector<LLValue*> args;
    args.reserve(irFty.args.size());

    // return in hidden ptr is first
    if (retinptr)
    {
        if (!retvar)
            retvar = DtoRawAlloca((*argTypesBegin)->getContainedType(0), resulttype->alignsize(), ".rettmp");
        args.push_back(retvar);

        // add attrs for hidden ptr
        // after adding the argument to args, args.size() is the index for the
        // related attributes since attrs[0] are the return value's attributes
        attrs.add(args.size(), irFty.arg_sret->attrs);

        // verify that sret and/or inreg attributes are set
        const AttrBuilder& sretAttrs = irFty.arg_sret->attrs;
        assert((sretAttrs.contains(LDC_ATTRIBUTE(StructRet)) || sretAttrs.contains(LDC_ATTRIBUTE(InReg)))
            && "Sret arg not sret or inreg?");
    }

    // then comes a context argument...
    if(thiscall || delegatecall || nestedcall)
    {
        LLType* contextArgType = *(argTypesBegin + args.size());

        if (dfnval && (dfnval->func->ident == Id::ensure || dfnval->func->ident == Id::require)) {
            // ... which can be the this "context" argument for a contract
            // invocation (in D2, we do not generate a full nested contexts
            // for __require/__ensure as the needed parameters are passed
            // explicitly, while in D1, the normal nested function handling
            // mechanisms are used)
            LLValue* thisarg = DtoBitCast(DtoLoad(gIR->func()->thisArg), getVoidPtrType());
            args.push_back(thisarg);
        }
        else if (thiscall && dfnval && dfnval->vthis)
        {
            // ... or a normal 'this' argument
            LLValue* thisarg = DtoBitCast(dfnval->vthis, contextArgType);
            args.push_back(thisarg);
        }
        else if (delegatecall)
        {
            // ... or a delegate context arg
            LLValue* ctxarg;
            if (fnval->isLVal()) {
                ctxarg = DtoLoad(DtoGEPi(fnval->getLVal(), 0, 0), ".ptr");
            } else {
                ctxarg = gIR->ir->CreateExtractValue(fnval->getRVal(), 0, ".ptr");
            }
            ctxarg = DtoBitCast(ctxarg, contextArgType);
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
        }
        else
        {
            error(loc, "Context argument required but none given");
            fatal();
        }

        // add attributes for context argument
        if (irFty.arg_this) {
            attrs.add(args.size(), irFty.arg_this->attrs);
        } else if (irFty.arg_nest) {
            attrs.add(args.size(), irFty.arg_nest->attrs);
        }
    }

    const int numFormalParams = Parameter::dim(tf->parameters); // excl. variadics

    // D vararg functions need an additional "TypeInfo[] _arguments" argument
    if (dvarargs) {
        LLValue* argumentsArg = getTypeinfoArrayArgumentForDVarArg(arguments, numFormalParams);
        args.push_back(argumentsArg);
        attrs.add(args.size(), irFty.arg_arguments->attrs);
    }

    // handle explicit arguments

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

    std::vector<DValue*> argvals(n_arguments, static_cast<DValue*>(0));
    if (dfnval && dfnval->func->isArrayOp) {
        // For array ops, the druntime implementation signatures are crafted
        // specifically such that the evaluation order is as expected with
        // the strange DMD reverse parameter passing order. Thus, we need
        // to actually build the arguments right-to-left for them.
        for (int i = numFormalParams - 1; i >= 0; --i) {
            Parameter* fnarg = Parameter::getNth(tf->parameters, i);
            assert(fnarg);
            DValue* argval = DtoArgument(fnarg, (*arguments)[i]);
            argvals[i] = argval;
        }
    } else {
        for (int i = 0; i < numFormalParams; ++i) {
            Parameter* fnarg = Parameter::getNth(tf->parameters, i);
            assert(fnarg);
            DValue* argval = DtoArgument(fnarg, (*arguments)[i]);
            argvals[i] = argval;
        }
    }
    // add varargs
    for (size_t i = numFormalParams; i < n_arguments; ++i)
        argvals[i] = DtoArgument(0, (*arguments)[i]);

    addExplicitArguments(args, attrs, irFty, callableTy, argvals, numFormalParams);

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
//    const char* varname = "";
//    if (callableTy->getReturnType() != LLType::getVoidTy(gIR->context()))
//        varname = retvar->getName().data();

#if 0
    IF_LOG Logger::cout() << "Calling: " << *callable << '\n';
#endif

    // call the function
    LLCallSite call = gIR->CreateCallOrInvoke(callable, args);

    // get return value
    LLValue* retllval = (retinptr) ? args[0] : call.getInstruction();

    // Hack around LDC assuming structs and static arrays are in memory:
    // If the function returns a struct or a static array, and the return
    // value is not a pointer to a struct or a static array, store it to
    // a stack slot before continuing.
    Type* dReturnType = tf->next;
    TY returnTy = dReturnType->toBasetype()->ty;
    bool storeReturnValueOnStack =
        (returnTy == Tstruct && !isaPointer(retllval)) ||
        (returnTy == Tsarray && isaArray(retllval));

    // Ignore ABI for intrinsics
    if (!intrinsic && !retinptr)
    {
        // do abi specific return value fixups
        DImValue dretval(dReturnType, retllval);
        if (storeReturnValueOnStack)
        {
            Logger::println("Storing return value to stack slot");
            LLValue* mem = DtoRawAlloca(DtoType(dReturnType), 0);
            irFty.getRet(dReturnType, &dretval, mem);
            retllval = mem;
            storeReturnValueOnStack = false;
        }
        else
        {
            retllval = irFty.getRet(dReturnType, &dretval);
            storeReturnValueOnStack =
                (returnTy == Tstruct && !isaPointer(retllval)) ||
                (returnTy == Tsarray && isaArray(retllval));
        }
    }

    if (storeReturnValueOnStack)
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
    llvm::AttributeSet attrlist = attrs.toNativeSet();
#else
    llvm::AttrListPtr attrlist = attrs.toNativeSet();
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
