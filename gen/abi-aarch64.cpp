//===-- abi-aarch64.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The Procedure Call Standard can be found here:
// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf
//
//===----------------------------------------------------------------------===//

#include "gen/abi.h"
#include "gen/abi-generic.h"
#include "gen/abi-aarch64.h"

struct AArch64TargetABI : TargetABI
{
    llvm::CallingConv::ID callingConv(LINK l)
    {
        switch (l)
        {
        case LINKc:
        case LINKcpp:
        case LINKpascal:
        case LINKwindows:
        case LINKd:
        case LINKdefault:
            return llvm::CallingConv::C;
        default:
            llvm_unreachable("Unhandled D linkage type.");
        }
    }

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

// The public getter for abi.cpp
TargetABI* getAArch64TargetABI()
{
    return new AArch64TargetABI();
}