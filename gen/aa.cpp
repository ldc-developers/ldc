#include "gen/llvm.h"

#include "mtype.h"
#include "declaration.h"
#include "aggregate.h"

#include "gen/aa.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/irstate.h"
#include "gen/dvalue.h"

// makes sure the key value lives in memory so it can be passed to the runtime functions without problems
// returns the pointer
static LLValue* to_pkey(Loc& loc, DValue* key)
{
    Type* keytype = key->getType();
    bool needmem = !DtoIsPassedByRef(keytype);
    LLValue* pkey;
    if (key->isIm()) {
        pkey = key->getRVal();
    }
    else if (DVarValue* var = key->isVar()) {
        pkey = key->getLVal();
        needmem = false;
    }
    else if (key->isConst()) {
        needmem = true;
        pkey = key->getRVal();
    }
    else {
        LLValue* tmp = DtoAlloca(DtoType(keytype), "aatmpkeystorage");
        DVarValue var(keytype, tmp);
        DtoAssign(loc, &var, key);
        return tmp;
    }

    // give memory
    if (needmem) {
        LLValue* tmp = DtoAlloca(DtoType(keytype), "aatmpkeystorage");
        DtoStore(pkey, tmp);
        pkey = tmp;
    }

    return pkey;
}

// returns the keytype typeinfo
static LLValue* to_keyti(DValue* key)
{
    // keyti param
    Type* keytype = key->getType();
    return DtoTypeInfoOf(keytype, false);
}

/////////////////////////////////////////////////////////////////////////////////////

DValue* DtoAAIndex(Loc& loc, Type* type, DValue* aa, DValue* key, bool lvalue)
{
    // call:
    // extern(C) void* _aaGet(AA* aa, TypeInfo keyti, size_t valuesize, void* pkey)
    // or
    // extern(C) void* _aaGetRvalue(AA aa, TypeInfo keyti, size_t valuesize, void* pkey)

    // first get the runtime function
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, lvalue?"_aaGet":"_aaGetRvalue");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    // aa param
    LLValue* aaval = lvalue ? aa->getLVal() : aa->getRVal();
    aaval = DtoBitCast(aaval, funcTy->getParamType(0));

    // keyti param
    LLValue* keyti = to_keyti(key);
    keyti = DtoBitCast(keyti, funcTy->getParamType(1));

    // valuesize param
    LLValue* valsize = DtoConstSize_t(getTypePaddedSize(DtoType(type)));

    // pkey param
    LLValue* pkey = to_pkey(loc, key);
    pkey = DtoBitCast(pkey, funcTy->getParamType(3));

    // call runtime
    LLValue* ret = gIR->CreateCallOrInvoke4(func, aaval, keyti, valsize, pkey, "aa.index").getInstruction();

    // cast return value
    const LLType* targettype = getPtrToType(DtoType(type));
    if (ret->getType() != targettype)
        ret = DtoBitCast(ret, targettype);

    return new DVarValue(type, ret);
}

/////////////////////////////////////////////////////////////////////////////////////

DValue* DtoAAIn(Loc& loc, Type* type, DValue* aa, DValue* key)
{
    // call:
    // extern(C) void* _aaIn(AA aa*, TypeInfo keyti, void* pkey)

    // first get the runtime function
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_aaIn");
    const llvm::FunctionType* funcTy = func->getFunctionType();

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
    LLValue* keyti = to_keyti(key);
    keyti = DtoBitCast(keyti, funcTy->getParamType(1));

    // pkey param
    LLValue* pkey = to_pkey(loc, key);
    pkey = DtoBitCast(pkey, funcTy->getParamType(2));

    // call runtime
    LLValue* ret = gIR->CreateCallOrInvoke3(func, aaval, keyti, pkey, "aa.in").getInstruction();

    // cast return value
    const LLType* targettype = DtoType(type);
    if (ret->getType() != targettype)
        ret = DtoBitCast(ret, targettype);

    return new DImValue(type, ret);
}

/////////////////////////////////////////////////////////////////////////////////////

void DtoAARemove(Loc& loc, DValue* aa, DValue* key)
{
    // call:
    // extern(C) void _aaDel(AA aa, TypeInfo keyti, void* pkey)

    // first get the runtime function
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_aaDel");
    const llvm::FunctionType* funcTy = func->getFunctionType();

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
    LLValue* keyti = to_keyti(key);
    keyti = DtoBitCast(keyti, funcTy->getParamType(1));

    // pkey param
    LLValue* pkey = to_pkey(loc, key);
    pkey = DtoBitCast(pkey, funcTy->getParamType(2));

    // build arg vector
    LLSmallVector<LLValue*, 3> args;
    args.push_back(aaval);
    args.push_back(keyti);
    args.push_back(pkey);

    // call runtime
    gIR->CreateCallOrInvoke(func, args.begin(), args.end());
}
