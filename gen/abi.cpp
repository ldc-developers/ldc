//===-- abi.cpp -----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/abi.h"
#include "mars.h"
#include "gen/abi-generic.h"
#include "gen/abi-aarch64.h"
#include "gen/abi-mips64.h"
#include "gen/abi-ppc64.h"
#include "gen/abi-win64.h"
#include "gen/abi-x86-64.h"
#include "gen/abi-x86.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irfuncty.h"
#include <algorithm>

//////////////////////////////////////////////////////////////////////////////

void ABIRewrite::getL(Type* dty, LLValue* v, LLValue* lval)
{
    LLValue* rval = get(dty, v);
    assert(rval->getType() == lval->getType()->getContainedType(0));
    DtoStore(rval, lval);
}

//////////////////////////////////////////////////////////////////////////////

LLValue* ABIRewrite::getAddressOf(DValue* v)
{
    Type* dty = v->getType();
    if (DtoIsPassedByRef(dty))
    {
        // v is lowered to a LL pointer to the struct/static array
        return v->getRVal();
    }

    if (v->isLVal())
        return v->getLVal();

    return DtoAllocaDump(v, ".getAddressOf_dump");
}

void ABIRewrite::storeToMemory(LLValue* rval, LLValue* address)
{
    LLType* pointerType = address->getType();
    assert(pointerType->isPointerTy());
    LLType* pointeeType = pointerType->getPointerElementType();

    LLType* rvalType = rval->getType();
    if (rvalType != pointeeType)
    {
        if (getTypeStoreSize(rvalType) > getTypeAllocSize(pointeeType))
        {
            // not enough allocated memory
            LLValue* paddedDump = DtoAllocaDump(rval, 0, ".storeToMemory_paddedDump");
            DtoAggrCopy(address, paddedDump);
            return;
        }

        address = DtoBitCast(address, getPtrToType(rvalType), ".storeToMemory_bitCastAddress");
    }

    DtoStore(rval, address);
}

LLValue* ABIRewrite::loadFromMemory(LLValue* address, LLType* asType, const char* name)
{
    LLType* pointerType = address->getType();
    assert(pointerType->isPointerTy());
    LLType* pointeeType = pointerType->getPointerElementType();

    if (asType == pointeeType)
        return DtoLoad(address, name);

    if (getTypeStoreSize(asType) > getTypeAllocSize(pointeeType))
    {
        // not enough allocated memory
        LLValue* paddedDump = DtoRawAlloca(asType, 0, ".loadFromMemory_paddedDump");
        DtoMemCpy(paddedDump, address, DtoConstSize_t(getTypeAllocSize(pointeeType)));
        return DtoLoad(paddedDump, name);
    }

    address = DtoBitCast(address, getPtrToType(asType), ".loadFromMemory_bitCastAddress");
    return DtoLoad(address, name);
}

//////////////////////////////////////////////////////////////////////////////

void TargetABI::rewriteVarargs(IrFuncTy& fty, std::vector<IrFuncTyArg*>& args)
{
    for (unsigned i = 0; i < args.size(); ++i)
    {
        IrFuncTyArg& arg = *args[i];
        if (!arg.byref) // don't rewrite ByVal arguments
            rewriteArgument(fty, arg);
    }
}

//////////////////////////////////////////////////////////////////////////////

LLValue* TargetABI::prepareVaStart(LLValue* pAp)
{
    // pass a void* pointer to ap to LLVM's va_start intrinsic
    return DtoBitCast(pAp, getVoidPtrType());
}

//////////////////////////////////////////////////////////////////////////////

void TargetABI::vaCopy(LLValue* pDest, LLValue* src)
{
    // simply bitcopy src over dest
    DtoStore(src, pDest);
}

//////////////////////////////////////////////////////////////////////////////

LLValue* TargetABI::prepareVaArg(LLValue* pAp)
{
    // pass a void* pointer to ap to LLVM's va_arg intrinsic
    return DtoBitCast(pAp, getVoidPtrType());
}

//////////////////////////////////////////////////////////////////////////////

Type* TargetABI::vaListType()
{
    // char* is used by default in druntime.
    return Type::tchar->pointerTo();
}

//////////////////////////////////////////////////////////////////////////////

// Some reasonable defaults for when we don't know what ABI to use.
struct UnknownTargetABI : TargetABI
{
    bool returnInArg(TypeFunction* tf)
    {
        if (tf->isref)
            return false;

        // Return structs and static arrays on the stack. The latter is needed
        // because otherwise LLVM tries to actually return the array in a number
        // of physical registers, which leads, depending on the target, to
        // either horrendous codegen or backend crashes.
        Type* rt = tf->next->toBasetype();
        return (rt->ty == Tstruct || rt->ty == Tsarray);
    }

    bool passByVal(Type* t)
    {
        return t->toBasetype()->ty == Tstruct;
    }

    void rewriteFunctionType(TypeFunction* t, IrFuncTy &fty)
    {
        // why?
    }
};

//////////////////////////////////////////////////////////////////////////////

TargetABI * TargetABI::getTarget()
{
    switch (global.params.targetTriple.getArch())
    {
    case llvm::Triple::x86:
        return getX86TargetABI();
    case llvm::Triple::x86_64:
        if (global.params.targetTriple.isOSWindows())
            return getWin64TargetABI();
        else
            return getX86_64TargetABI();
    case llvm::Triple::mips:
    case llvm::Triple::mipsel:
    case llvm::Triple::mips64:
    case llvm::Triple::mips64el:
        return getMIPS64TargetABI(global.params.is64bit);
    case llvm::Triple::ppc64:
    case llvm::Triple::ppc64le:
        return getPPC64TargetABI(global.params.targetTriple.isArch64Bit());
#if LDC_LLVM_VER == 305
    case llvm::Triple::arm64:
    case llvm::Triple::arm64_be:
#endif
    case llvm::Triple::aarch64:
    case llvm::Triple::aarch64_be:
        return getAArch64TargetABI();
    default:
        Logger::cout() << "WARNING: Unknown ABI, guessing...\n";
        return new UnknownTargetABI;
    }
}

//////////////////////////////////////////////////////////////////////////////

// A simple ABI for LLVM intrinsics.
struct IntrinsicABI : TargetABI
{
    RemoveStructPadding remove_padding;

    bool returnInArg(TypeFunction* tf)
    {
        return false;
    }

    bool passByVal(Type* t)
    {
        return false;
    }

    void rewriteArgument(IrFuncTy& fty, IrFuncTyArg& arg)
    {
        Type* ty = arg.type->toBasetype();
        if (ty->ty != Tstruct)
            return;
        // TODO: Check that no unions are passed in or returned.

        LLType* abiTy = DtoUnpaddedStructType(arg.type);

        if (abiTy && abiTy != arg.ltype) {
            arg.ltype = abiTy;
            arg.rewrite = &remove_padding;
        }
    }

    void rewriteFunctionType(TypeFunction* tf, IrFuncTy &fty)
    {
        if (!fty.arg_sret) {
            Type* rt = fty.ret->type->toBasetype();
            if (rt->ty == Tstruct) {
                Logger::println("Intrinsic ABI: Transforming return type");
                rewriteArgument(fty, *fty.ret);
            }
        }

        Logger::println("Intrinsic ABI: Transforming arguments");
        LOG_SCOPE;

        for (IrFuncTy::ArgIter I = fty.args.begin(), E = fty.args.end(); I != E; ++I) {
            IrFuncTyArg& arg = **I;

            IF_LOG Logger::cout() << "Arg: " << arg.type->toChars() << '\n';

            // Arguments that are in memory are of no interest to us.
            if (arg.byref)
                continue;

            rewriteArgument(fty, arg);

            IF_LOG Logger::cout() << "New arg type: " << *arg.ltype << '\n';
        }
    }
};

TargetABI * TargetABI::getIntrinsic()
{
    static IntrinsicABI iabi;
    return &iabi;
}
