#include "gen/llvm.h"

#include "mtype.h"
#include "declaration.h"
#include "aggregate.h"

#include "gen/aa.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "gen/logger.h"
#include "gen/irstate.h"
#include "gen/dvalue.h"

// makes sure the key value lives in memory so it can be passed to the runtime functions without problems
// returns the pointer
static llvm::Value* to_pkey(DValue* key)
{
    Type* keytype = key->getType();
    bool needmem = !DtoIsPassedByRef(keytype);
    llvm::Value* pkey;
    if (key->isIm()) {
        pkey = key->getRVal();
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
        llvm::Value* tmp = new llvm::AllocaInst(DtoType(keytype), "aatmpkeystorage", gIR->topallocapoint());
        DVarValue* var = new DVarValue(keytype, tmp, true);
        DtoAssign(var, key);
        return tmp;
    }

    // give memory
    if (needmem) {
        llvm::Value* tmp = new llvm::AllocaInst(DtoType(keytype), "aatmpkeystorage", gIR->topallocapoint());
        DtoStore(pkey, tmp);
        pkey = tmp;
    }

    return pkey;
}

// returns the keytype typeinfo
static llvm::Value* to_keyti(DValue* key)
{
    // keyti param
    Type* keytype = key->getType();
    keytype->getTypeInfo(NULL);
    TypeInfoDeclaration* tid = keytype->getTypeInfoDeclaration();
    assert(tid);
    DtoResolveDsymbol(Type::typeinfo);
    DtoForceDeclareDsymbol(tid);
    assert(gIR->irDsymbol[tid].irGlobal->value);
    return gIR->irDsymbol[tid].irGlobal->value;
}

/////////////////////////////////////////////////////////////////////////////////////

DValue* DtoAAIndex(Type* type, DValue* aa, DValue* key)
{
    // call:
    // extern(C) void* _aaGet(AA* aa, TypeInfo keyti, void* pkey, size_t valuesize)

    // first get the runtime function
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_aaGet");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    // aa param
    llvm::Value* aaval = aa->getLVal();
    aaval = DtoBitCast(aaval, funcTy->getParamType(0));

    // keyti param
    llvm::Value* keyti = to_keyti(key);
    keyti = DtoBitCast(keyti, funcTy->getParamType(1));

    // valuesize param
    llvm::Value* valsize = DtoConstSize_t(getABITypeSize(DtoType(type)));

    // pkey param
    llvm::Value* pkey = to_pkey(key);
    pkey = DtoBitCast(pkey, funcTy->getParamType(3));

    // build arg vector
    std::vector<llvm::Value*> args;
    args.push_back(aaval);
    args.push_back(keyti);
    args.push_back(valsize);
    args.push_back(pkey);

    // call runtime
    llvm::Value* ret = gIR->ir->CreateCall(func, args.begin(), args.end(), "aa.index");

    // cast return value
    const llvm::Type* targettype = getPtrToType(DtoType(type));
    if (ret->getType() != targettype)
        ret = DtoBitCast(ret, targettype);

    return new DVarValue(type, ret, true);
}

/////////////////////////////////////////////////////////////////////////////////////

DValue* DtoAAIn(Type* type, DValue* aa, DValue* key)
{
    // call:
    // extern(C) void* _aaIn(AA aa*, TypeInfo keyti, void* pkey)

    // first get the runtime function
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_aaIn");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    Logger::cout() << "_aaIn = " << *func << '\n';

    // aa param
    llvm::Value* aaval = aa->getRVal();
    Logger::cout() << "aaval: " << *aaval << '\n';
    Logger::cout() << "totype: " << *funcTy->getParamType(0) << '\n';
    aaval = DtoBitCast(aaval, funcTy->getParamType(0));

    // keyti param
    llvm::Value* keyti = to_keyti(key);
    keyti = DtoBitCast(keyti, funcTy->getParamType(1));

    // pkey param
    llvm::Value* pkey = to_pkey(key);
    pkey = DtoBitCast(pkey, funcTy->getParamType(2));

    // build arg vector
    std::vector<llvm::Value*> args;
    args.push_back(aaval);
    args.push_back(keyti);
    args.push_back(pkey);

    // call runtime
    llvm::Value* ret = gIR->ir->CreateCall(func, args.begin(), args.end(), "aa.in");

    // cast return value
    const llvm::Type* targettype = DtoType(type);
    if (ret->getType() != targettype)
        ret = DtoBitCast(ret, targettype);

    return new DImValue(type, ret);
}

/////////////////////////////////////////////////////////////////////////////////////

void DtoAARemove(DValue* aa, DValue* key)
{
    // call:
    // extern(C) void _aaDel(AA aa, TypeInfo keyti, void* pkey)

    // first get the runtime function
    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_aaDel");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    Logger::cout() << "_aaDel = " << *func << '\n';

    // aa param
    llvm::Value* aaval = aa->getRVal();
    Logger::cout() << "aaval: " << *aaval << '\n';
    Logger::cout() << "totype: " << *funcTy->getParamType(0) << '\n';
    aaval = DtoBitCast(aaval, funcTy->getParamType(0));

    // keyti param
    llvm::Value* keyti = to_keyti(key);
    keyti = DtoBitCast(keyti, funcTy->getParamType(1));

    // pkey param
    llvm::Value* pkey = to_pkey(key);
    pkey = DtoBitCast(pkey, funcTy->getParamType(2));

    // build arg vector
    std::vector<llvm::Value*> args;
    args.push_back(aaval);
    args.push_back(keyti);
    args.push_back(pkey);

    // call runtime
    gIR->ir->CreateCall(func, args.begin(), args.end(),"");
}
