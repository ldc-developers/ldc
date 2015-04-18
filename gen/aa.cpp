//===-- aa.cpp ------------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/aa.h"
#include "aggregate.h"
#include "declaration.h"
#include "module.h"
#include "mtype.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "ir/irmodule.h"

// returns the keytype typeinfo
static LLValue* to_keyti(DValue* aa)
{
    // keyti param
    assert(aa->type->toBasetype()->ty == Taarray);
    TypeAArray * aatype = static_cast<TypeAArray*>(aa->type->toBasetype());
    return DtoTypeInfoOf(aatype->index, false);
}

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
    llvm::Function* func = LLVM_D_GetRuntimeFunction(loc, gIR->module, lvalue?"_aaGetX":"_aaInX");
    LLFunctionType* funcTy = func->getFunctionType();

    // aa param
    LLValue* aaval = lvalue ? aa->getLVal() : aa->getRVal();
    aaval = DtoBitCast(aaval, funcTy->getParamType(0));

    // keyti param
    LLValue* keyti = to_keyti(aa);
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
    LLType* targettype = getPtrToType(i1ToI8(DtoType(type)));
    if (ret->getType() != targettype)
        ret = DtoBitCast(ret, targettype);

    // Only check bounds for rvalues ('aa[key]').
    // Lvalue use ('aa[key] = value') auto-adds an element.
    if (!lvalue && gIR->emitArrayBoundsChecks()) {
        llvm::BasicBlock* oldend = gIR->scopeend();
        llvm::BasicBlock* failbb = llvm::BasicBlock::Create(gIR->context(), "aaboundscheckfail", gIR->topfunc(), oldend);
        llvm::BasicBlock* okbb = llvm::BasicBlock::Create(gIR->context(), "aaboundsok", gIR->topfunc(), oldend);

        LLValue* nullaa = LLConstant::getNullValue(ret->getType());
        LLValue* cond = gIR->ir->CreateICmpNE(nullaa, ret, "aaboundscheck");
        gIR->ir->CreateCondBr(cond, okbb, failbb);

        // set up failbb to call the array bounds error runtime function

        gIR->scope() = IRScope(failbb, okbb);

        LLValue* args[] = {
            // file param
            DtoModuleFileName(gIR->func()->decl->getModule(), loc),
            // line param
            DtoConstUint(loc.linnum)
        };

        // call
        LLVM_D_CallRuntimeFunction(loc, "_d_arraybounds", args);

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
    llvm::Function* func = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_aaInX");
    LLFunctionType* funcTy = func->getFunctionType();

    IF_LOG Logger::cout() << "_aaIn = " << *func << '\n';

    // aa param
    LLValue* aaval = aa->getRVal();
    IF_LOG {
        Logger::cout() << "aaval: " << *aaval << '\n';
        Logger::cout() << "totype: " << *funcTy->getParamType(0) << '\n';
    }
    aaval = DtoBitCast(aaval, funcTy->getParamType(0));

    // keyti param
    LLValue* keyti = to_keyti(aa);
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
    llvm::Function* func = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_aaDelX");
    LLFunctionType* funcTy = func->getFunctionType();

    IF_LOG Logger::cout() << "_aaDel = " << *func << '\n';

    // aa param
    LLValue* aaval = aa->getRVal();
    IF_LOG {
        Logger::cout() << "aaval: " << *aaval << '\n';
        Logger::cout() << "totype: " << *funcTy->getParamType(0) << '\n';
    }
    aaval = DtoBitCast(aaval, funcTy->getParamType(0));

    // keyti param
    LLValue* keyti = to_keyti(aa);
    keyti = DtoBitCast(keyti, funcTy->getParamType(1));

    // pkey param
    LLValue* pkey = makeLValue(loc, key);
    pkey = DtoBitCast(pkey, funcTy->getParamType(2));

    // build arg vector
    LLValue* args[] = { aaval, keyti, pkey };

    // call runtime
    LLCallSite call = gIR->CreateCallOrInvoke(func, args);

    return new DImValue(Type::tbool, call.getInstruction());
}

/////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoAAEquals(Loc& loc, TOK op, DValue* l, DValue* r)
{
    Type* t = l->getType()->toBasetype();
    assert(t == r->getType()->toBasetype() && "aa equality is only defined for aas of same type");
    llvm::Function* func = LLVM_D_GetRuntimeFunction(loc, gIR->module, "_aaEqual");
    LLFunctionType* funcTy = func->getFunctionType();

    LLValue* aaval = DtoBitCast(l->getRVal(), funcTy->getParamType(1));
    LLValue* abval = DtoBitCast(r->getRVal(), funcTy->getParamType(2));
    LLValue* aaTypeInfo = DtoTypeInfoOf(t);
    LLValue* res = gIR->CreateCallOrInvoke3(func, aaTypeInfo, aaval, abval, "aaEqRes").getInstruction();
    res = gIR->ir->CreateICmpNE(res, DtoConstInt(0));
    if (op == TOKnotequal)
        res = gIR->ir->CreateNot(res);
    return res;
}


