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
    else if (key->isThis()) {
        pkey = key->getRVal();
        needmem = true;
    }
    else if (DVarValue* var = key->isVar()) {
        if (var->lval) {
            pkey = key->getLVal();
            needmem = false;
        }
        else {
            pkey = key->getRVal();
        }
    }
    else if (key->isConst()) {
        needmem = true;
        pkey = key->getRVal();
    }
    else {
        LLValue* tmp = new llvm::AllocaInst(DtoType(keytype), "aatmpkeystorage", gIR->topallocapoint());
        DVarValue* var = new DVarValue(keytype, tmp, true);
        DtoAssign(loc, var, key);
        return tmp;
    }

    // give memory
    if (needmem) {
        LLValue* tmp = new llvm::AllocaInst(DtoType(keytype), "aatmpkeystorage", gIR->topallocapoint());
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

DValue* DtoAAIndex(Loc& loc, Type* type, DValue* aa, DValue* key)
{
    // call:
    // extern(C) void* _aaGet(AA* aa, TypeInfo keyti, void* pkey, size_t valuesize)

    // first get the runtime function
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_aaGet");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    // aa param
    LLValue* aaval = aa->getLVal();
    aaval = DtoBitCast(aaval, funcTy->getParamType(0));

    // keyti param
    LLValue* keyti = to_keyti(key);
    keyti = DtoBitCast(keyti, funcTy->getParamType(1));

    // valuesize param
    LLValue* valsize = DtoConstSize_t(getABITypeSize(DtoType(type)));

    // pkey param
    LLValue* pkey = to_pkey(loc, key);
    pkey = DtoBitCast(pkey, funcTy->getParamType(3));

    // call runtime
    LLValue* ret = gIR->CreateCallOrInvoke4(func, aaval, keyti, valsize, pkey, "aa.index")->get();

    // cast return value
    const LLType* targettype = getPtrToType(DtoType(type));
    if (ret->getType() != targettype)
        ret = DtoBitCast(ret, targettype);

    return new DVarValue(type, ret, true);
}

/////////////////////////////////////////////////////////////////////////////////////

DValue* DtoAAIn(Loc& loc, Type* type, DValue* aa, DValue* key)
{
    // call:
    // extern(C) void* _aaIn(AA aa*, TypeInfo keyti, void* pkey)

    // first get the runtime function
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_aaIn");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    Logger::cout() << "_aaIn = " << *func << '\n';

    // aa param
    LLValue* aaval = aa->getRVal();
    Logger::cout() << "aaval: " << *aaval << '\n';
    Logger::cout() << "totype: " << *funcTy->getParamType(0) << '\n';
    aaval = DtoBitCast(aaval, funcTy->getParamType(0));

    // keyti param
    LLValue* keyti = to_keyti(key);
    keyti = DtoBitCast(keyti, funcTy->getParamType(1));

    // pkey param
    LLValue* pkey = to_pkey(loc, key);
    pkey = DtoBitCast(pkey, funcTy->getParamType(2));

    // call runtime
    LLValue* ret = gIR->CreateCallOrInvoke3(func, aaval, keyti, pkey, "aa.in")->get();

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

    Logger::cout() << "_aaDel = " << *func << '\n';

    // aa param
    LLValue* aaval = aa->getRVal();
    Logger::cout() << "aaval: " << *aaval << '\n';
    Logger::cout() << "totype: " << *funcTy->getParamType(0) << '\n';
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
