//===-- gen/abi-mips64.cpp - MIPS64 ABI description ------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The MIPS64 N32 and N64 ABI can be found here:
// http://techpubs.sgi.com/library/dynaweb_docs/0640/SGI_Developer/books/Mpro_n32_ABI/sgi_html/index.html
//
//===----------------------------------------------------------------------===//

#include "gen/abi.h"
#include "gen/abi-generic.h"
#include "gen/abi-mips64.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"

struct MIPS64TargetABI : TargetABI {
    ExplicitByvalRewrite byvalRewrite;
    IntegerRewrite integerRewrite;
    const bool Is64Bit;

    MIPS64TargetABI(const bool Is64Bit) : Is64Bit(Is64Bit)
    { }

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
        for (auto arg : fty.args)
        {
            if (!arg->byref)
                rewriteArgument(fty, *arg);
        }
    }

    void rewriteArgument(IrFuncTy& fty, IrFuncTyArg& arg)
    {
        // FIXME
    }

    // Returns true if the D type can be bit-cast to an integer of the same size.
    bool canRewriteAsInt(Type* t)
    {
        const unsigned size = t->size();
        return size == 1 || size == 2 || size == 4 || (Is64Bit && size == 8);
    }
};

// The public getter for abi.cpp
TargetABI* getMIPS64TargetABI(bool Is64Bit)
{
    return new MIPS64TargetABI(Is64Bit);
}
