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
#include "ir/irfunction.h"
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
    // Number of arguments added to the LLVM type that are implicit on the
    // frontend side of things (this, context pointers, etc.)
    const size_t implicitLLArgCount = args.size();

    // Number of formal arguments in the LLVM type (i.e. excluding varargs).
    const size_t formalLLArgCount = irFty.args.size();

    // The number of explicit arguments in the D call expression (including
    // varargs), not all of which necessarily generate a LLVM argument.
    const size_t explicitDArgCount = argvals.size();

    // construct and initialize an IrFuncTyArg object for each vararg
    std::vector<IrFuncTyArg*> optionalIrArgs;
    for (size_t i = numFormalParams; i < explicitDArgCount; i++) {
        Type* argType = argvals[i]->getType();
        bool passByVal = gABI->passByVal(argType);

        AttrBuilder initialAttrs;
        if (passByVal)
            initialAttrs.add(LDC_ATTRIBUTE(ByVal));
        else
            initialAttrs.add(DtoShouldExtend(argType));

        optionalIrArgs.push_back(new IrFuncTyArg(argType, passByVal, initialAttrs));
        optionalIrArgs.back()->parametersIdx = i;
    }

    // let the ABI rewrite the IrFuncTyArg objects
    gABI->rewriteVarargs(irFty, optionalIrArgs);

    const size_t explicitLLArgCount = formalLLArgCount + optionalIrArgs.size();
    args.resize(implicitLLArgCount + explicitLLArgCount, static_cast<llvm::Value*>(0));

    // Iterate the explicit arguments from left to right in the D source,
    // which is the reverse of the LLVM order if irFty.reverseParams is true.
    for (size_t i = 0; i < explicitLLArgCount; ++i)
    {
        const bool isVararg = (i >= irFty.args.size());
        IrFuncTyArg* irArg = NULL;
        if (isVararg)
            irArg = optionalIrArgs[i - numFormalParams];
        else
            irArg = irFty.args[i];

        DValue* const argval = argvals[irArg->parametersIdx];
        Type* const argType = argval->getType();

        llvm::Value* llVal = NULL;
        if (isVararg)
            llVal = irFty.putParam(*irArg, argval);
        else
            llVal = irFty.putParam(i, argval);

        const size_t llArgIdx = implicitLLArgCount +
            (irFty.reverseParams ? explicitLLArgCount - i - 1 : i);
        llvm::Type* const callableArgType =
            (isVararg ? NULL : callableTy->getParamType(llArgIdx));

        // Hack around LDC assuming structs and static arrays are in memory:
        // If the function wants a struct, and the argument value is a
        // pointer to a struct, load from it before passing it in.
        if (isaPointer(llVal) && DtoIsPassedByRef(argType) &&
            ((!isVararg && !isaPointer(callableArgType)) ||
             (isVararg && !irArg->byref && !irArg->isByVal())))
        {
            Logger::println("Loading struct type for function argument");
            llVal = DtoLoad(llVal);
        }

        // parameter type mismatch, this is hard to get rid of
        if (!isVararg && llVal->getType() != callableArgType)
        {
            IF_LOG
            {
                Logger::cout() << "arg:     " << *llVal << '\n';
                Logger::cout() << "expects: " << *callableArgType << '\n';
            }
            if (isaStruct(llVal))
                llVal = DtoAggrPaint(llVal, callableArgType);
            else
                llVal = DtoBitCast(llVal, callableArgType);
        }

        args[llArgIdx] = llVal;
        // +1 as index 0 contains the function attributes.
        attrs.add(llArgIdx + 1, irArg->attrs);

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
        new llvm::GlobalVariable(gIR->module, typeinfoarraytype, true, llvm::GlobalValue::InternalLinkage, NULL, "._arguments.storage");
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
    LLValue* typeinfoarrayparam = new llvm::GlobalVariable(gIR->module, tiarrty,
        true, llvm::GlobalValue::InternalLinkage, tiinits, "._arguments.array");

    return DtoLoad(typeinfoarrayparam);
}

//////////////////////////////////////////////////////////////////////////////////////////

bool DtoLowerMagicIntrinsic(IRState* p, FuncDeclaration* fndecl, CallExp *e, DValue*& result)
{
    // va_start instruction
    if (fndecl->llvmInternal == LLVMva_start) {
        if (e->arguments->dim < 1 || e->arguments->dim > 2) {
            e->error("va_start instruction expects 1 (or 2) arguments");
            fatal();
        }
        LLValue* pAp = toElem((*e->arguments)[0])->getLVal(); // va_list*
        // variadic extern(D) function with implicit _argptr?
        if (LLValue* pArgptr = p->func()->_argptr) {
            DtoStore(DtoLoad(pArgptr), pAp); // ap = _argptr
            result = new DImValue(e->type, pAp);
        } else {
            LLValue* vaStartArg = gABI->prepareVaStart(pAp);
            result = new DImValue(e->type, p->ir->CreateCall(
                GET_INTRINSIC_DECL(vastart), vaStartArg, ""));
        }
        return true;
    }

    // va_copy instruction
    if (fndecl->llvmInternal == LLVMva_copy) {
        if (e->arguments->dim != 2) {
            e->error("va_copy instruction expects 2 arguments");
            fatal();
        }
        LLValue* pDest = toElem((*e->arguments)[0])->getLVal(); // va_list*
        LLValue* src   = toElem((*e->arguments)[1])->getRVal(); // va_list
        gABI->vaCopy(pDest, src);
        result = new DVarValue(e->type, pDest);
        return true;
    }

    // va_arg instruction
    if (fndecl->llvmInternal == LLVMva_arg) {
        if (e->arguments->dim != 1) {
            e->error("va_arg instruction expects 1 argument");
            fatal();
        }
        LLValue* pAp = toElem((*e->arguments)[0])->getLVal(); // va_list*
        LLValue* vaArgArg = gABI->prepareVaArg(pAp);
        LLType* llType = DtoType(e->type);
        if (DtoIsPassedByRef(e->type))
            llType = getPtrToType(llType);
        result = new DImValue(e->type, p->ir->CreateVAArg(vaArgArg, llType));
        return true;
    }

    // C alloca
    if (fndecl->llvmInternal == LLVMalloca) {
        if (e->arguments->dim != 1) {
            e->error("alloca expects 1 arguments");
            fatal();
        }
        Expression* exp = (*e->arguments)[0];
        DValue* expv = toElem(exp);
        if (expv->getType()->toBasetype()->ty != Tint32)
            expv = DtoCast(e->loc, expv, Type::tint32);
        result = new DImValue(e->type, p->ir->CreateAlloca(
            LLType::getInt8Ty(p->context()), expv->getRVal(), ".alloca"));
        return true;
    }

    // fence instruction
    if (fndecl->llvmInternal == LLVMfence) {
        if (e->arguments->dim != 1) {
            e->error("fence instruction expects 1 arguments");
            fatal();
        }
        p->ir->CreateFence(llvm::AtomicOrdering((*e->arguments)[0]->toInteger()));
        return true;
    }

    // atomic store instruction
    if (fndecl->llvmInternal == LLVMatomic_store) {
        if (e->arguments->dim != 3) {
            e->error("atomic store instruction expects 3 arguments");
            fatal();
        }
        Expression* exp1 = (*e->arguments)[0];
        Expression* exp2 = (*e->arguments)[1];
        int atomicOrdering = (*e->arguments)[2]->toInteger();
        LLValue* val = toElem(exp1)->getRVal();
        LLValue* ptr = toElem(exp2)->getRVal();

        if (!val->getType()->isIntegerTy()) {
            llvm::PointerType *v = isaPointer(val->getType());
            if (v && v->getContainedType(0)->isStructTy()) {
                switch (const size_t N = getTypeBitSize(v->getContainedType(0))) {
                    case 8:
                    case 16:
                    case 32:
                    case 64:
                    case 128:
                        val = DtoLoad(DtoBitCast(val, llvm::Type::getIntNPtrTy(gIR->context(), static_cast<unsigned>(N))));
                        ptr = DtoBitCast(ptr, llvm::Type::getIntNPtrTy(gIR->context(), static_cast<unsigned>(N)));
                        break;
                    default:
                        goto errorStore;
                }
            }
            else {
errorStore:
                e->error("atomic store only supports integer types, not '%s'", exp1->type->toChars());
                fatal();
            }
        }

        llvm::StoreInst* ret = p->ir->CreateStore(val, ptr);
        ret->setAtomic(llvm::AtomicOrdering(atomicOrdering));
        ret->setAlignment(getTypeAllocSize(val->getType()));
        return true;
    }

    // atomic load instruction
    if (fndecl->llvmInternal == LLVMatomic_load) {
        if (e->arguments->dim != 2) {
            e->error("atomic load instruction expects 2 arguments");
            fatal();
        }

        Expression* exp = (*e->arguments)[0];
        int atomicOrdering = (*e->arguments)[1]->toInteger();

        LLValue* ptr = toElem(exp)->getRVal();
        LLType* ptrTy = ptr->getType()->getContainedType(0);
        Type* retType = exp->type->nextOf();

        if (!ptrTy->isIntegerTy()) {
            if (ptrTy->isStructTy()) {
                switch (const size_t N = getTypeBitSize(ptrTy)) {
                    case 8:
                    case 16:
                    case 32:
                    case 64:
                    case 128:
                        ptr = DtoBitCast(ptr, llvm::Type::getIntNPtrTy(gIR->context(), static_cast<unsigned>(N)));
                        break;
                    default:
                        goto errorLoad;
                }
            }
            else {
errorLoad:
                e->error("atomic load only supports integer types, not '%s'", retType->toChars());
                fatal();
            }
        }

        llvm::LoadInst* load = p->ir->CreateLoad(ptr);
        load->setAlignment(getTypeAllocSize(load->getType()));
        load->setAtomic(llvm::AtomicOrdering(atomicOrdering));
        llvm::Value* val = load;
        if (val->getType() != ptrTy)
            val = DtoAllocaDump(val, retType);
        result = new DImValue(retType, val);
        return true;
    }

    // cmpxchg instruction
    if (fndecl->llvmInternal == LLVMatomic_cmp_xchg) {
        if (e->arguments->dim != 4) {
            e->error("cmpxchg instruction expects 4 arguments");
            fatal();
        }
        Expression* exp1 = (*e->arguments)[0];
        Expression* exp2 = (*e->arguments)[1];
        Expression* exp3 = (*e->arguments)[2];
        int atomicOrdering = (*e->arguments)[3]->toInteger();
        LLValue* ptr = toElem(exp1)->getRVal();
        LLValue* cmp = toElem(exp2)->getRVal();
        LLValue* val = toElem(exp3)->getRVal();
        LLType* retTy = val->getType();

        if (!cmp->getType()->isIntegerTy()) {
            llvm::PointerType *v = isaPointer(cmp->getType());
            if (v && v->getContainedType(0)->isStructTy()) {
                switch (const size_t N = getTypeBitSize(v->getContainedType(0))) {
                    case 8:
                    case 16:
                    case 32:
                    case 64:
                    case 128:
                        ptr = DtoBitCast(ptr, llvm::Type::getIntNPtrTy(gIR->context(), static_cast<unsigned>(N)));
                        cmp = DtoLoad(DtoBitCast(cmp, llvm::Type::getIntNPtrTy(gIR->context(), static_cast<unsigned>(N))));
                        val = DtoLoad(DtoBitCast(val, llvm::Type::getIntNPtrTy(gIR->context(), static_cast<unsigned>(N))));
                        break;
                    default:
                        goto errorCmpxchg;
                }
            }
            else {
errorCmpxchg:
                e->error("cmpxchg only supports integer types, not '%s'", exp2->type->toChars());
                fatal();
            }
        }
#if LDC_LLVM_VER >= 305
        LLValue* ret = p->ir->CreateAtomicCmpXchg(ptr, cmp, val, llvm::AtomicOrdering(atomicOrdering), llvm::AtomicOrdering(atomicOrdering));
        // Use the same quickfix as for dragonegg - see r210956
        ret = p->ir->CreateExtractValue(ret, 0);
#else
        LLValue* ret = p->ir->CreateAtomicCmpXchg(ptr, cmp, val, llvm::AtomicOrdering(atomicOrdering));
#endif
        llvm::Value* retVal = ret;
        if (retVal->getType() != retTy)
            retVal = DtoAllocaDump(retVal, exp3->type);
        result = new DImValue(exp3->type, retVal);
        return true;
    }

    // atomicrmw instruction
    if (fndecl->llvmInternal == LLVMatomic_rmw) {
        if (e->arguments->dim != 3) {
            e->error("atomic_rmw instruction expects 3 arguments");
            fatal();
        }

        static const char *ops[] = {
            "xchg",
            "add",
            "sub",
            "and",
            "nand",
            "or",
            "xor",
            "max",
            "min",
            "umax",
            "umin",
            0
        };

        int op = 0;
        for (; ; ++op) {
            if (ops[op] == 0) {
                e->error("unknown atomic_rmw operation %s", fndecl->intrinsicName.c_str());
                fatal();
            }
            if (fndecl->intrinsicName == ops[op])
                break;
        }

        Expression* exp1 = (*e->arguments)[0];
        Expression* exp2 = (*e->arguments)[1];
        int atomicOrdering = (*e->arguments)[2]->toInteger();
        LLValue* ptr = toElem(exp1)->getRVal();
        LLValue* val = toElem(exp2)->getRVal();
        LLValue* ret = p->ir->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp(op), ptr, val,
                                                llvm::AtomicOrdering(atomicOrdering));
        result = new DImValue(exp2->type, ret);
        return true;
    }

    // bitop
    if (fndecl->llvmInternal == LLVMbitop_bt ||
       fndecl->llvmInternal == LLVMbitop_btr||
       fndecl->llvmInternal == LLVMbitop_btc||
       fndecl->llvmInternal == LLVMbitop_bts)
    {
        if (e->arguments->dim != 2) {
            e->error("bitop intrinsic expects 2 arguments");
            fatal();
        }

        Expression* exp1 = (*e->arguments)[0];
        Expression* exp2 = (*e->arguments)[1];
        LLValue* ptr = toElem(exp1)->getRVal();
        LLValue* bitnum = toElem(exp2)->getRVal();

        unsigned bitmask = DtoSize_t()->getBitWidth() - 1;
        assert(bitmask == 31 || bitmask == 63);
        // auto q = cast(size_t*)ptr + (bitnum >> (64bit ? 6 : 5));
        LLValue* q = DtoBitCast(ptr, DtoSize_t()->getPointerTo());
        q = DtoGEP1(q, p->ir->CreateLShr(bitnum, bitmask == 63 ? 6 : 5), "bitop.q");

        // auto mask = 1 << (bitnum & bitmask);
        LLValue* mask = p->ir->CreateAnd(bitnum, DtoConstSize_t(bitmask), "bitop.tmp");
        mask = p->ir->CreateShl(DtoConstSize_t(1), mask, "bitop.mask");

        // auto result = (*q & mask) ? -1 : 0;
        LLValue* val = p->ir->CreateZExt(DtoLoad(q, "bitop.tmp"), DtoSize_t(), "bitop.val");
        LLValue* ret = p->ir->CreateAnd(val, mask, "bitop.tmp");
        ret = p->ir->CreateICmpNE(ret, DtoConstSize_t(0), "bitop.tmp");
        ret = p->ir->CreateSelect(ret, DtoConstInt(-1), DtoConstInt(0), "bitop.result");

        if (fndecl->llvmInternal != LLVMbitop_bt) {
            llvm::Instruction::BinaryOps op;
            if (fndecl->llvmInternal == LLVMbitop_btc) {
                // *q ^= mask;
                op = llvm::Instruction::Xor;
            } else if (fndecl->llvmInternal == LLVMbitop_btr) {
                // *q &= ~mask;
                mask = p->ir->CreateNot(mask);
                op = llvm::Instruction::And;
            } else if (fndecl->llvmInternal == LLVMbitop_bts) {
                // *q |= mask;
                op = llvm::Instruction::Or;
            } else {
                llvm_unreachable("Unrecognized bitop intrinsic.");
            }

            LLValue *newVal = p->ir->CreateBinOp(op, val, mask, "bitop.new_val");
            newVal = p->ir->CreateTrunc(newVal, DtoSize_t(), "bitop.tmp");
            DtoStore(newVal, q);
        }

        result = new DImValue(e->type, ret);
        return true;
    }

    if (fndecl->llvmInternal == LLVMbitop_vld)
    {
        if (e->arguments->dim != 1) {
            e->error("bitop.vld intrinsic expects 1 argument");
            fatal();
        }
        // TODO: Check types

        Expression* exp1 = (*e->arguments)[0];
        LLValue* ptr = toElem(exp1)->getRVal();
        result = new DImValue(exp1->type, DtoVolatileLoad(ptr));
        return true;
    }

    if (fndecl->llvmInternal == LLVMbitop_vst)
    {
        if (e->arguments->dim != 2) {
            e->error("bitop.vst intrinsic expects 2 arguments");
            fatal();
        }
        // TODO: Check types

        Expression* exp1 = (*e->arguments)[0];
        Expression* exp2 = (*e->arguments)[1];
        LLValue* ptr = toElem(exp1)->getRVal();
        LLValue* val = toElem(exp2)->getRVal();
        DtoVolatileStore(val, ptr);
        return true;
    }

    return false;
}

//////////////////////////////////////////////////////////////////////////////////////////

void addImplicitArguments(std::vector<LLValue*>& args, AttrSet& attrs, Loc& loc, DValue* fnval,
    LLFunctionType* llCalleeType, Expressions* arguments, Type* resulttype, LLValue* retvar)
{
    Type* const calleeType = fnval->getType();
    DFuncValue* const dfnval = fnval->isFunc();
    IrFuncTy& irFty = DtoIrTypeFunction(fnval);
    TypeFunction* const tf = DtoTypeFunction(fnval);

    const bool sret = irFty.arg_sret;
    const bool thiscall = irFty.arg_this;
    const bool delegatecall = (calleeType->toBasetype()->ty == Tdelegate);
    const bool nestedcall = irFty.arg_nest;
    const bool dvarargs = irFty.arg_arguments;
    LLFunctionType::param_iterator argTypesBegin = llCalleeType->param_begin();

    // return in hidden ptr is first
    if (sret)
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
    if (thiscall || delegatecall || nestedcall)
    {
        LLType* contextArgType = *(argTypesBegin + args.size());

        if (dfnval && (dfnval->func->ident == Id::ensure || dfnval->func->ident == Id::require))
        {
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
            if (fnval->isLVal())
                ctxarg = DtoLoad(DtoGEPi(fnval->getLVal(), 0, 0), ".ptr");
            else
                ctxarg = gIR->ir->CreateExtractValue(fnval->getRVal(), 0, ".ptr");
            ctxarg = DtoBitCast(ctxarg, contextArgType);
            args.push_back(ctxarg);
        }
        else if (nestedcall)
        {
            // ... or a nested function context arg
            if (dfnval)
            {
                LLValue* contextptr = DtoNestedContext(loc, dfnval->func);
                contextptr = DtoBitCast(contextptr, getVoidPtrType());
                args.push_back(contextptr);
            }
            else
                args.push_back(llvm::UndefValue::get(getVoidPtrType()));
        }
        else
        {
            error(loc, "Context argument required but none given");
            fatal();
        }

        // add attributes for context argument
        if (irFty.arg_this)
            attrs.add(args.size(), irFty.arg_this->attrs);
        else if (irFty.arg_nest)
            attrs.add(args.size(), irFty.arg_nest->attrs);
    }

    // D vararg functions need an additional "TypeInfo[] _arguments" argument
    if (irFty.arg_arguments) {
        int numFormalParams = Parameter::dim(tf->parameters);
        LLValue* argumentsArg = getTypeinfoArrayArgumentForDVarArg(arguments, numFormalParams);
        args.push_back(argumentsArg);
        attrs.add(args.size(), irFty.arg_arguments->attrs);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

// FIXME: this function is a mess !

DValue* DtoCallFunction(Loc& loc, Type* resulttype, DValue* fnval, Expressions* arguments, llvm::Value* retvar)
{
    IF_LOG Logger::println("DtoCallFunction()");
    LOG_SCOPE

    // make sure the D callee type has been processed
    DtoType(fnval->getType());

    // get func value if any
    DFuncValue* dfnval = fnval->isFunc();

    // get function type info
    IrFuncTy& irFty = DtoIrTypeFunction(fnval);
    TypeFunction* const tf = DtoTypeFunction(fnval);
    Type* const returntype = tf->next;
    const TY returnTy = returntype->toBasetype()->ty;

    if (resulttype == NULL)
        resulttype = returntype;

    // get callee llvm value
    LLValue* const callable = DtoCallableValue(fnval);
    LLFunctionType* const callableTy = DtoExtractFunctionType(callable->getType());
    assert(callableTy);
    const llvm::CallingConv::ID callconv = gABI->callingConv(callableTy, tf->linkage);

//     IF_LOG Logger::cout() << "callable: " << *callable << '\n';

    // parameter attributes
    AttrSet attrs;

    // return attrs
    attrs.add(0, irFty.ret->attrs);

    std::vector<LLValue*> args;
    args.reserve(irFty.args.size());

    // handle implicit arguments (sret, this/nested context, _arguments)
    addImplicitArguments(args, attrs, loc, fnval, callableTy, arguments, resulttype, retvar);

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

    const int numFormalParams = Parameter::dim(tf->parameters); // excl. variadics
    const size_t n_arguments = arguments ? arguments->dim : 0;  // number of explicit arguments

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

    // call the function
    LLCallSite call = gIR->func()->scopes->callOrInvoke(callable, args);

    // get return value
    bool retinptr = irFty.arg_sret;
    LLValue* retllval = (retinptr) ? args[0] : call.getInstruction();

    // Hack around LDC assuming structs and static arrays are in memory:
    // If the function returns a struct or a static array, and the return
    // value is not a pointer to a struct or a static array, store it to
    // a stack slot before continuing.
    bool storeReturnValueOnStack =
        (returnTy == Tstruct && !isaPointer(retllval)) ||
        (returnTy == Tsarray && isaArray(retllval));

    bool retValIsAlloca = false;

    // Ignore ABI for intrinsics
    const bool intrinsic = (dfnval && dfnval->func && DtoIsIntrinsic(dfnval->func));
    if (!intrinsic && !retinptr)
    {
        // do abi specific return value fixups
        if (storeReturnValueOnStack)
        {
            Logger::println("Storing return value to stack slot");
            LLValue* mem = DtoAlloca(returntype);
            irFty.getRet(returntype, retllval, mem);
            retllval = mem;
            retValIsAlloca = true;
            storeReturnValueOnStack = false;
        }
        else
        {
            retllval = irFty.getRet(returntype, retllval);
            storeReturnValueOnStack =
                (returnTy == Tstruct && !isaPointer(retllval)) ||
                (returnTy == Tsarray && isaArray(retllval));
        }
    }

    if (storeReturnValueOnStack)
    {
        Logger::println("Storing return value to stack slot");
        retllval = DtoAllocaDump(retllval, returntype);
        retValIsAlloca = true;
    }

    // repaint the type if necessary
    Type* rbase = stripModifiers(resulttype->toBasetype(), true);
    Type* nextbase = stripModifiers(returntype->toBasetype(), true);
    if (!rbase->equals(nextbase))
    {
        IF_LOG Logger::println("repainting return value from '%s' to '%s'", returntype->toChars(), rbase->toChars());
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
                LLValue* val = DtoInsertValue(
                    llvm::UndefValue::get(DtoType(rbase)), retllval, 0);
                retllval = DtoAllocaDump(val, rbase, ".aalvaluetmp");
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
        {
#if LDC_LLVM_VER >= 302
            attrlist = llvm::Intrinsic::getAttributes(gIR->context(), static_cast<llvm::Intrinsic::ID>(llfunc->getIntrinsicID()));
#else
            attrlist = llvm::Intrinsic::getAttributes(static_cast<llvm::Intrinsic::ID>(llfunc->getIntrinsicID()));
#endif
        }
        else
        {
            call.setCallingConv(callconv);
        }
    }
    else
    {
        call.setCallingConv(callconv);
    }
    call.setAttributes(attrlist);

    // Special case for struct constructor calls: For temporaries, using the
    // this pointer value returned from the constructor instead of the alloca
    // passed as a parameter (which has the same value anyway) might lead to
    // instruction dominance issues because of the way it interacts with the
    // cleanups (see struct ctor hack in ToElemVisitor::visit(CallExp *)).
    if (dfnval && dfnval->func && dfnval->func->isCtorDeclaration() &&
        dfnval->func->isMember2()->isStructDeclaration())
    {
        return new DVarValue(resulttype, dfnval->vthis);
    }

    // if we are returning through a pointer arg
    // or if we are returning a reference
    // make sure we provide a lvalue back!
    if (retinptr || (tf->isref && returnTy != Tvoid) || retValIsAlloca)
        return new DVarValue(resulttype, retllval);

    return new DImValue(resulttype, retllval);
}
