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
    ByvalRewrite byvalRewrite;
    CompositeToInt compositeToInt;
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
        TY ty = t->toBasetype()->ty;
        return ty == Tstruct || ty == Tsarray;
    }

    void doneWithFunctionType()
    {
    }

    void rewriteFunctionType(TypeFunction* tf, IrFuncTy &fty)
    {
        // EXPLICIT PARAMETERS

        for (IrFuncTy::ArgRIter I = fty.args.rbegin(), E = fty.args.rend(); I != E; ++I)
        {
            IrFuncTyArg& arg = **I;

            if (arg.byref)
                continue;

            Type* ty = arg.type->toBasetype();

            if ((ty->ty == Tstruct || ty->ty == Tsarray))
            {
                if (canRewriteAsInt(ty))
                {
                    arg.rewrite = &compositeToInt;
                    arg.ltype = compositeToInt.type(arg.type, arg.ltype);
                }
                else
                {
                    // these types are passed byval:
                    // the caller allocates a copy and then passes a pointer to the copy
                    arg.rewrite = &byvalRewrite;
                    arg.ltype = byvalRewrite.type(arg.type, arg.ltype);

                    // the copy is treated as a local variable of the callee
                    // hence add the NoAlias and NoCapture attributes
#if LDC_LLVM_VER >= 303
                    arg.attrs.clear();
                    arg.attrs.addAttribute(llvm::Attribute::NoAlias)
                             .addAttribute(llvm::Attribute::NoCapture);
#elif LDC_LLVM_VER == 302
                    arg.attrs = llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::NoAlias)
                                                                                         .addAttribute(llvm::Attributes::NoCapture));
#else
                    arg.attrs = llvm::Attribute::NoAlias | llvm::Attribute::NoCapture;
#endif
                }
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
