//===-- abi.cpp -----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"

#include <algorithm>

#include "mars.h"

#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"
#include "gen/abi.h"
#include "gen/logger.h"
#include "gen/dvalue.h"
#include "gen/abi-generic.h"
#include "gen/abi-x86.h"
#include "gen/abi-x86-64.h"
#include "ir/irfunction.h"
#include "ir/irfuncty.h"

//////////////////////////////////////////////////////////////////////////////

void ABIRewrite::getL(Type* dty, DValue* v, llvm::Value* lval)
{
    LLValue* rval = get(dty, v);
    assert(rval->getType() == lval->getType()->getContainedType(0));
    DtoStore(rval, lval);
}

//////////////////////////////////////////////////////////////////////////////

// Some reasonable defaults for when we don't know what ABI to use.
struct UnknownTargetABI : TargetABI
{
    bool returnInArg(TypeFunction* tf)
    {
#if DMDV2
        if (tf->isref)
            return false;
#endif
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

    void rewriteFunctionType(TypeFunction* t)
    {
        // why?
    }
};

//////////////////////////////////////////////////////////////////////////////

TargetABI * TargetABI::getTarget()
{
    switch(global.params.cpu)
    {
    case ARCHx86:
        return getX86TargetABI();
    case ARCHx86_64:
        return getX86_64TargetABI();
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

    void fixup(IrFuncTyArg& arg) {
        assert(arg.type->ty == Tstruct);
        // TODO: Check that no unions are passed in or returned.

        LLType* abiTy = DtoUnpaddedStructType(arg.type);

        if (abiTy && abiTy != arg.ltype) {
            arg.ltype = abiTy;
            arg.rewrite = &remove_padding;
        }
    }

    void rewriteFunctionType(TypeFunction* tf)
    {
        assert(tf->linkage == LINKintrinsic);

        IrFuncTy& fty = tf->fty;

        if (!fty.arg_sret) {
            Type* rt = fty.ret->type->toBasetype();
            if (rt->ty == Tstruct)  {
                Logger::println("Intrinsic ABI: Transforming return type");
                fixup(*fty.ret);
            }
        }

        Logger::println("Intrinsic ABI: Transforming arguments");
        LOG_SCOPE;

        for (IrFuncTy::ArgIter I = fty.args.begin(), E = fty.args.end(); I != E; ++I) {
            IrFuncTyArg& arg = **I;

            if (Logger::enabled())
                Logger::cout() << "Arg: " << arg.type->toChars() << '\n';

            // Arguments that are in memory are of no interest to us.
            if (arg.byref)
                continue;

            Type* ty = arg.type->toBasetype();
            if (ty->ty == Tstruct)
                fixup(arg);

            if (Logger::enabled())
                Logger::cout() << "New arg type: " << *arg.ltype << '\n';
        }
    }
};

TargetABI * TargetABI::getIntrinsic()
{
    static IntrinsicABI iabi;
    return &iabi;
}
