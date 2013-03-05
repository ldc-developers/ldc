//===-- abi-x86-64.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The PowerOpen 64bit ABI can be found here:
// http://refspecs.linuxfoundation.org/ELF/ppc64/PPC-elf64abi-1.9.html
//
//===----------------------------------------------------------------------===//

#include "gen/abi-ppc64.h"
#include "gen/abi.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"

struct PPC64TargetABI : TargetABI {
    llvm::CallingConv::ID callingConv(LINK l)
    {
        switch (l)
        {
        case LINKc:
        case LINKcpp:
        case LINKintrinsic:
        case LINKpascal:
        case LINKwindows:
            return llvm::CallingConv::C;
        case LINKd:
        case LINKdefault:
            return llvm::CallingConv::Fast;
        default:
            llvm_unreachable("Unhandled D linkage type.");
        }
    }

    void newFunctionType(TypeFunction* tf)
    {
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

    void doneWithFunctionType()
    {
    }

    void rewriteFunctionType(TypeFunction* tf)
    {
    }
};

// The public getter for abi.cpp
TargetABI* getPPC64TargetABI()
{
    return new PPC64TargetABI();
}
