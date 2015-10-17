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

    bool returnInArg(TypeFunction* tf);

    bool passByVal(Type* t);

    bool passThisBeforeSret(TypeFunction* tf);

    void rewriteFunctionType(TypeFunction* tf, IrFuncTy &fty);

    void rewriteArgument(IrFuncTy& fty, IrFuncTyArg& arg);

private:
    // Returns true if the D type is an aggregate:
    // * struct
    // * static/dynamic array
    // * delegate
    // * complex number
    bool isAggregate(Type* t)
    {
        TY ty = t->ty;
        return ty == Tstruct || ty == Tsarray || /*ty == Tarray ||*/ ty == Tdelegate
            || t->iscomplex();
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
            // * aggregates which can NOT be rewritten as integers
            //   (size > 64 bits or not a power of 2)
            (isAggregate(t) && !canRewriteAsInt(t)) ||
            // * 80-bit real and ireal
            (realIs80bits() && (t->ty == Tfloat80 || t->ty == Timaginary80));
    }
};


// The public getter for abi.cpp
TargetABI* getWin64TargetABI()
{
    return new Win64TargetABI;
}

bool Win64TargetABI::returnInArg(TypeFunction* tf)
{
    if (tf->isref)
        return false;

    Type* rt = tf->next->toBasetype();

    // * let LLVM return 80-bit real/ireal on the x87 stack, for DMD compliance
    if (realIs80bits() && (rt->ty == Tfloat80 || rt->ty == Timaginary80))
        return false;

    // * all POD types <= 64 bits and of a size that is a power of 2
    //   (incl. 2x32-bit cfloat) are returned in a register (RAX, or
    //   XMM0 for single float/ifloat/double/idouble)
    // * all other types are returned via struct-return (sret)
    return (rt->ty == Tstruct && !((TypeStruct*)rt)->sym->isPOD())
        || isPassedWithByvalSemantics(rt);
}

bool Win64TargetABI::passByVal(Type* t)
{
    return false;
}

bool Win64TargetABI::passThisBeforeSret(TypeFunction* tf)
{
    return tf->linkage == LINKcpp;
}

void Win64TargetABI::rewriteFunctionType(TypeFunction* tf, IrFuncTy &fty)
{
    // RETURN VALUE
    if (!fty.ret->byref && fty.ret->type->toBasetype()->ty != Tvoid)
        rewriteArgument(fty, *fty.ret);

    // EXPLICIT PARAMETERS
    for (IrFuncTy::ArgIter I = fty.args.begin(), E = fty.args.end(); I != E; ++I)
    {
        IrFuncTyArg& arg = **I;

        if (!arg.byref)
            rewriteArgument(fty, arg);
    }

    // extern(D): reverse parameter order for non variadics, for DMD-compliance
    if (tf->linkage == LINKd && tf->varargs != 1 && fty.args.size() > 1)
        fty.reverseParams = true;
}

void Win64TargetABI::rewriteArgument(IrFuncTy& fty, IrFuncTyArg& arg)
{
    LLType* originalLType = arg.ltype;
    Type* t = arg.type->toBasetype();

    if (isPassedWithByvalSemantics(t))
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
    else if (isAggregate(t) && canRewriteAsInt(t) && !IntegerRewrite::isObsoleteFor(originalLType))
    {
        arg.rewrite = &integerRewrite;
        arg.ltype = integerRewrite.type(arg.type, arg.ltype);
    }

    IF_LOG if (arg.rewrite)
    {
        Logger::println("Rewriting argument type %s", t->toChars());
        LOG_SCOPE;
        Logger::cout() << *originalLType << " => " << *arg.ltype << '\n';
    }
}
