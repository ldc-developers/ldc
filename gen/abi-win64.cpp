//===-- abi-win64.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// extern(C) implements the C calling convention for x86-64 on Windows, see
// http://msdn.microsoft.com/en-us/library/7kcdt6fy%28v=vs.110%29.aspx
//
//===----------------------------------------------------------------------===//

#include "mtype.h"
#include "declaration.h"
#include "aggregate.h"

#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/tollvm.h"
#include "gen/logger.h"
#include "gen/dvalue.h"
#include "gen/llvmhelpers.h"
#include "gen/abi.h"
#include "gen/abi-win64.h"
#include "gen/abi-generic.h"
#include "ir/irfunction.h"

#include <cassert>
#include <string>
#include <utility>

struct Win64TargetABI : TargetABI
{
    ExplicitByvalRewrite byvalRewrite;
    IntegerRewrite integerRewrite;

    llvm::CallingConv::ID callingConv(LINK l);

    bool returnInArg(TypeFunction* tf);

    bool passByVal(Type* t);

    void rewriteFunctionType(TypeFunction* tf, IrFuncTy &fty);

    void rewriteArgument(IrFuncTy& fty, IrFuncTyArg& arg);

private:
    // Returns true if the D type is a composite (struct/static array/complex number).
    bool isComposite(Type* t)
    {
        return t->ty == Tstruct || t->ty == Tsarray
            || t->iscomplex(); // treat complex numbers as structs too
    }

    // Returns true if the D type can be bit-cast to an integer of the same size.
    bool canRewriteAsInt(Type* t)
    {
        unsigned size = t->size();
        return size == 1 || size == 2 || size == 4 || size == 8;
    }

    bool realIs80bits()
    {
#if LDC_LLVM_VER >= 305
        return !global.params.targetTriple.isWindowsMSVCEnvironment();
#else
        return true;
#endif
    }

    // Returns true if the D type is passed byval (the callee getting a pointer
    // to a dedicated hidden copy).
    bool isPassedWithByvalSemantics(Type* t)
    {
        return
            // * structs/static arrays/complex numbers which can NOT be rewritten as integers
            (isComposite(t) && !canRewriteAsInt(t)) ||
            // * 80-bit real and ireal
            ((t->ty == Tfloat80 || t->ty == Timaginary80) && realIs80bits());
    }
};


// The public getter for abi.cpp
TargetABI* getWin64TargetABI()
{
    return new Win64TargetABI;
}


llvm::CallingConv::ID Win64TargetABI::callingConv(LINK l)
{
    switch (l)
    {
    case LINKc:
    case LINKcpp:
    case LINKpascal:
    case LINKd:
    case LINKdefault:
    case LINKwindows:
        return llvm::CallingConv::C;
    default:
        llvm_unreachable("Unhandled D linkage type.");
    }
}

bool Win64TargetABI::returnInArg(TypeFunction* tf)
{
    if (tf->isref)
        return false;

    Type* rt = tf->next->toBasetype();

    // * everything <= 64 bits and of a size that is a power of 2
    //   (incl. 2x32-bit cfloat) is returned in a register (RAX, or
    //   XMM0 for single float/ifloat/double/idouble)
    // * all other structs/static arrays/complex numbers and 80-bit
    //   real/ireal are returned via struct-return (sret)
    return isPassedWithByvalSemantics(rt);
}

bool Win64TargetABI::passByVal(Type* t)
{
    t = t->toBasetype();

    // FIXME: LLVM doesn't support ByVal on Win64 yet
    //return isPassedWithByvalSemantics(t);
    return false;
}

void Win64TargetABI::rewriteFunctionType(TypeFunction* tf, IrFuncTy &fty)
{
    // RETURN VALUE
    if (!tf->isref)
    {
        Type* rt = fty.ret->type->toBasetype();
        if (isComposite(rt) && canRewriteAsInt(rt))
        {
            fty.ret->rewrite = &integerRewrite;
            fty.ret->ltype = integerRewrite.type(fty.ret->type, fty.ret->ltype);
        }
    }

    // EXPLICIT PARAMETERS
    for (IrFuncTy::ArgIter I = fty.args.begin(), E = fty.args.end(); I != E; ++I)
    {
        IrFuncTyArg& arg = **I;

        if (arg.byref)
            continue;

        rewriteArgument(fty, arg);
    }

}

void Win64TargetABI::rewriteArgument(IrFuncTy& fty, IrFuncTyArg& arg)
{
    Type* ty = arg.type->toBasetype();

    if (isComposite(ty) && canRewriteAsInt(ty))
    {
        arg.rewrite = &integerRewrite;
        arg.ltype = integerRewrite.type(arg.type, arg.ltype);
    }
    // FIXME: this should actually be handled by LLVM and the ByVal arg attribute
    else if (isPassedWithByvalSemantics(ty))
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
