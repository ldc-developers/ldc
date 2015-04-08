//===-- abi-ppc64.cpp -----------------------------------------------------===//
//
//                         LDC ? the LLVM D compiler
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

#include "gen/abi.h"
#include "gen/abi-generic.h"
#include "gen/abi-ppc64.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"

struct PPC64TargetABI : TargetABI {
    ExplicitByvalRewrite byvalRewrite;
    IntegerRewrite integerRewrite;
    const bool Is64Bit;

    PPC64TargetABI(const bool Is64Bit) : Is64Bit(Is64Bit)
    { }

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
        TY ty = t->toBasetype()->ty;
        return ty == Tstruct || ty == Tsarray;
    }

    void rewriteFunctionType(TypeFunction* tf, IrFuncTy &fty)
    {
        // EXPLICIT PARAMETERS
        for (IrFuncTy::ArgIter I = fty.args.begin(), E = fty.args.end(); I != E; ++I)
        {
            IrFuncTyArg& arg = **I;

            if (!arg.byref)
                rewriteArgument(fty, arg);
        }
    }

    void rewriteArgument(IrFuncTy& fty, IrFuncTyArg& arg)
    {
        Type* ty = arg.type->toBasetype();

        if (ty->ty == Tstruct || ty->ty == Tsarray)
        {
            if (canRewriteAsInt(ty))
            {
                if (!IntegerRewrite::isObsoleteFor(arg.ltype))
                {
                    arg.rewrite = &integerRewrite;
                    arg.ltype = integerRewrite.type(arg.type, arg.ltype);
                }
            }
            else
            {
                // these types are passed byval:
                // the caller allocates a copy and then passes a pointer to the copy
                arg.rewrite = &byvalRewrite;
                arg.ltype = byvalRewrite.type(arg.type, arg.ltype);

                // the copy is treated as a local variable of the callee
                // hence add the NoAlias and NoCapture attributes
                arg.attrs.clear()
                         .add(LDC_ATTRIBUTE(NoAlias))
                         .add(LDC_ATTRIBUTE(NoCapture));
            }
        }
    }

    // Returns true if the D type can be bit-cast to an integer of the same size.
    bool canRewriteAsInt(Type* t)
    {
        const unsigned size = t->size();
        return size == 1 || size == 2 || size == 4 || (Is64Bit && size == 8);
    }
};

// The public getter for abi.cpp
TargetABI* getPPC64TargetABI(bool Is64Bit)
{
    return new PPC64TargetABI(Is64Bit);
}
