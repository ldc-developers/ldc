//===-- aa.cpp ------------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"

#include "mtype.h"
#include "module.h"
#include "declaration.h"
#include "aggregate.h"

#include "gen/aa.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/irstate.h"
#include "gen/dvalue.h"
#include "ir/irmodule.h"

#if DMDV2
// returns the keytype typeinfo
static LLValue* to_keyti(DValue* aa)
{
    // keyti param
    assert(aa->type->toBasetype()->ty == Taarray);
    TypeAArray * aatype = static_cast<TypeAArray*>(aa->type->toBasetype());
    return DtoTypeInfoOf(aatype->index, false);
}
#else
// returns the keytype typeinfo
static LLValue* to_keyti(DValue* key)
{
    // keyti param
    Type* keytype = key->getType();
    return DtoTypeInfoOf(keytype, false);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////

DValue* DtoAAIndex(Loc& loc, Type* type, DValue* aa, DValue* key, bool lvalue)
{
    // D1:
    // call:
    // extern(C) void* _aaGet(AA* aa, TypeInfo keyti, size_t valuesize, void* pkey)
    // or
    // extern(C) void* _aaIn(AA aa*, TypeInfo keyti, void* pkey)

    // D2:
    // call:
    // extern(C) void* _aaGetX(AA* aa, TypeInfo keyti, size_t valuesize, void* pkey)
    // or
    // extern(C) void* _aaInX(AA aa*, TypeInfo keyti, void* pkey)

    // first get the runtime function
#if DMDV2
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, lvalue?"_aaGetX":"_aaInX");
#else
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, lvalue?"_aaGet":"_aaIn");
#endif
    LLFunctionType* funcTy = func->getFunctionType();

    // aa param
    LLValue* aaval = lvalue ? aa->getLVal() : aa->getRVal();
    aaval = DtoBitCast(aaval, funcTy->getParamType(0));

    // keyti param
#if DMDV2
    LLValue* keyti = to_keyti(aa);
#else
    LLValue* keyti = to_keyti(key);
#endif
    keyti = DtoBitCast(keyti, funcTy->getParamType(1));

    // pkey param
    LLValue* pkey = makeLValue(loc, key);
    pkey = DtoBitCast(pkey, funcTy->getParamType(lvalue ? 3 : 2));

    // call runtime
    LLValue* ret;
    if (lvalue) {
        // valuesize param
        LLValue* valsize = DtoConstSize_t(getTypePaddedSize(DtoType(type)));
        
        ret = gIR->CreateCallOrInvoke4(func, aaval, keyti, valsize, pkey, "aa.index").getInstruction();
    } else {
        ret = gIR->CreateCallOrInvoke3(func, aaval, keyti, pkey, "aa.index").getInstruction();
    }

    // cast return value
    LLType* targettype = getPtrToType(DtoType(type));
    if (ret->getType() != targettype)
        ret = DtoBitCast(ret, targettype);

    // Only check bounds for rvalues ('aa[key]').
    // Lvalue use ('aa[key] = value') auto-adds an element.
    if (!lvalue && global.params.useArrayBounds) {
        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* failbb = llvm::BasicBlock::Create(gIR->context(), "aaboundscheckfail", gIR->topfunc(), oldend);
        llvm::BasicBlock* okbb = llvm::BasicBlock::Create(gIR->context(), "aaboundsok", gIR->topfunc(), oldend);

        LLValue* nullaa = LLConstant::getNullValue(ret->getType());
        LLValue* cond = gIR->ir->CreateICmpNE(nullaa, ret, "aaboundscheck");
        gIR->ir->CreateCondBr(cond, okbb, failbb);

        // set up failbb to call the array bounds error runtime function

        gIR->scope() = IRScope(failbb, okbb);

        std::vector<LLValue*> args;

#if DMDV2
        // module param
        LLValue *moduleInfoSymbol = gIR->func()->decl->getModule()->moduleInfoSymbol();
        LLType *moduleInfoType = DtoType(Module::moduleinfo->type);
        args.push_back(DtoBitCast(moduleInfoSymbol, getPtrToType(moduleInfoType)));
#else
        // file param
        IrModule* irmod = getIrModule(NULL);
        args.push_back(DtoLoad(irmod->fileName));
#endif

        // line param
        LLConstant* c = DtoConstUint(loc.linnum);
        args.push_back(c);

        // call
        llvm::Function* errorfn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_array_bounds");
        gIR->CreateCallOrInvoke(errorfn, args);

        // the function does not return
        gIR->ir->CreateUnreachable();

        // if ok, proceed in okbb
        gIR->scope() = IRScope(okbb, oldend);
    }
    return new DVarValue(type, ret);
}

/////////////////////////////////////////////////////////////////////////////////////

DValue* DtoAAIn(Loc& loc, Type* type, DValue* aa, DValue* key)
{
    // D1:
    // call:
    // extern(C) void* _aaIn(AA aa*, TypeInfo keyti, void* pkey)

    // D2:
    // call:
    // extern(C) void* _aaInX(AA aa*, TypeInfo keyti, void* pkey)

    // first get the runtime function
#if DMDV2
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_aaInX");
#else
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_aaIn");
#endif
    LLFunctionType* funcTy = func->getFunctionType();

    if (Logger::enabled())
        Logger::cout() << "_aaIn = " << *func << '\n';

    // aa param
    LLValue* aaval = aa->getRVal();
    if (Logger::enabled())
    {
        Logger::cout() << "aaval: " << *aaval << '\n';
        Logger::cout() << "totype: " << *funcTy->getParamType(0) << '\n';
    }
    aaval = DtoBitCast(aaval, funcTy->getParamType(0));

    // keyti param
#if DMDV2
    LLValue* keyti = to_keyti(aa);
#else
    LLValue* keyti = to_keyti(key);
#endif
    keyti = DtoBitCast(keyti, funcTy->getParamType(1));

    // pkey param
    LLValue* pkey = makeLValue(loc, key);
    pkey = DtoBitCast(pkey, getVoidPtrType());

    // call runtime
    LLValue* ret = gIR->CreateCallOrInvoke3(func, aaval, keyti, pkey, "aa.in").getInstruction();

    // cast return value
    LLType* targettype = DtoType(type);
    if (ret->getType() != targettype)
        ret = DtoBitCast(ret, targettype);

    return new DImValue(type, ret);
}

/////////////////////////////////////////////////////////////////////////////////////

DValue *DtoAARemove(Loc& loc, DValue* aa, DValue* key)
{
    // D1:
    // call:
    // extern(C) void _aaDel(AA aa, TypeInfo keyti, void* pkey)

    // D2:
    // call:
    // extern(C) bool _aaDelX(AA aa, TypeInfo keyti, void* pkey)

    // first get the runtime function
#if DMDV2
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_aaDelX");
#else
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_aaDel");
#endif
    LLFunctionType* funcTy = func->getFunctionType();

    if (Logger::enabled())
        Logger::cout() << "_aaDel = " << *func << '\n';

    // aa param
    LLValue* aaval = aa->getRVal();
    if (Logger::enabled())
    {
        Logger::cout() << "aaval: " << *aaval << '\n';
        Logger::cout() << "totype: " << *funcTy->getParamType(0) << '\n';
    }
    aaval = DtoBitCast(aaval, funcTy->getParamType(0));

    // keyti param
#if DMDV2
    LLValue* keyti = to_keyti(aa);
#else
    LLValue* keyti = to_keyti(key);
#endif
    keyti = DtoBitCast(keyti, funcTy->getParamType(1));

    // pkey param
    LLValue* pkey = makeLValue(loc, key);
    pkey = DtoBitCast(pkey, funcTy->getParamType(2));

    // build arg vector
    LLSmallVector<LLValue*, 3> args;
    args.push_back(aaval);
    args.push_back(keyti);
    args.push_back(pkey);

    // call runtime
    LLCallSite call = gIR->CreateCallOrInvoke(func, args);

#if DMDV2
    return new DImValue(Type::tbool, call.getInstruction());
#else
    return NULL;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoAAEquals(Loc& loc, TOK op, DValue* l, DValue* r)
{
    Type* t = l->getType()->toBasetype();
    assert(t == r->getType()->toBasetype() && "aa equality is only defined for aas of same type");
#if DMDV2
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_aaEqual");
    LLFunctionType* funcTy = func->getFunctionType();

    LLValue* aaval = DtoBitCast(l->getRVal(), funcTy->getParamType(1));
    LLValue* abval = DtoBitCast(r->getRVal(), funcTy->getParamType(2));
    LLValue* aaTypeInfo = DtoTypeInfoOf(t);
    LLValue* res = gIR->CreateCallOrInvoke3(func, aaTypeInfo, aaval, abval, "aaEqRes").getInstruction();
#else
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_aaEq");
    LLFunctionType* funcTy = func->getFunctionType();
    
    LLValue* aaval = DtoBitCast(l->getRVal(), funcTy->getParamType(0));
    LLValue* abval = DtoBitCast(r->getRVal(), funcTy->getParamType(1));
    LLValue* aaTypeInfo = DtoTypeInfoOf(t);
    LLValue* res = gIR->CreateCallOrInvoke3(func, aaval, abval, aaTypeInfo, "aaEqRes").getInstruction();
#endif
    res = gIR->ir->CreateICmpNE(res, DtoConstInt(0), "tmp");
    if (op == TOKnotequal)
        res = gIR->ir->CreateNot(res, "tmp");
    return res;
}


