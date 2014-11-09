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


// Returns true if the D type can be bit-cast to an integer of the same size.
static bool canRewriteAsInt(Type* t)
{
    unsigned size = t->size();
    return size == 1 || size == 2 || size == 4 || size == 8;
}

// Returns true if the D type is a composite (struct or static array).
static bool isComposite(const Type* t)
{
    return t->ty == Tstruct || t->ty == Tsarray;
}

static bool realIs80bits()
{
#if LDC_LLVM_VER >= 305
    return !global.params.targetTriple.isWindowsMSVCEnvironment();
#else
    return true;
#endif
}

// Returns true if the D type is passed byval (the callee getting a pointer
// to a dedicated hidden copy).
static bool isPassedWithByvalSemantics(Type* t)
{
    return
        // * structs and static arrays which can NOT be rewritten as integers
        (isComposite(t) && !canRewriteAsInt(t)) ||
        // * 80-bit real and ireal
        ((t->ty == Tfloat80 || t->ty == Timaginary80) && realIs80bits()) ||
        // * cdouble and creal
        (t->ty == Tcomplex64 || t->ty == Tcomplex80);
}

// FIXME: This should actually be handled by LLVM and the ByVal arg attribute.
struct Win64_byval_rewrite : ABIRewrite
{
    // Get instance from pointer.
    LLValue* get(Type* dty, DValue* v)
    {
        LLValue* ptr = v->getRVal();
        return DtoLoad(ptr); // *ptr
    }

    // Convert the caller's instance to a pointer for the callee.
    // The pointer points to a dedicated copy for the callee which
    // is allocated by the caller.
    LLValue* put(Type* dty, DValue* v)
    {
        /* NOTE: probably not safe
        // optimization: do not copy if parameter is not mutable
        if (!dty->isMutable() && v->isLVal())
            return v->getLVal();
        */

        LLValue* original = v->getRVal();
        LLValue* copy;

        llvm::Type* type = original->getType();
        if (type->isPointerTy())
        {
            type = type->getPointerElementType();
            copy = DtoRawAlloca(type, 16, "copy_for_callee");
            DtoStore(DtoLoad(original), copy); // *copy = *original
        }
        else
        {
            copy = DtoRawAlloca(type, 16, "copy_for_callee");
            DtoStore(original, copy);          // *copy = original
        }

        return copy;
    }

    // T => T*
    LLType* type(Type* dty, LLType* t)
    {
        return getPtrToType(DtoType(dty));
    }
};


struct Win64TargetABI : TargetABI
{
    Win64_byval_rewrite byval_rewrite;
    CompositeToInt compositeToInt;
    CfloatToInt cfloatToInt;

    llvm::CallingConv::ID callingConv(LINK l);

    bool returnInArg(TypeFunction* tf);

    bool passByVal(Type* t);

    void rewriteFunctionType(TypeFunction* tf, IrFuncTy &fty);

    void rewriteArgument(IrFuncTyArg& arg);
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

    // everything <= 64 bits and of a size that is a power of 2
    // is returned in a register (RAX, or XMM0 for single float/
    // double) - except for cfloat
    // 80-bit real/ireal is returned on top of the x87 stack: ST(0)
    // complex numbers are returned in XMM0 & XMM1 (cfloat, cdouble)
    // or ST(1) & ST(0) (80-bit creal)
    // all other structs and static arrays are returned by struct-return (sret)
    return isComposite(rt) && !canRewriteAsInt(rt);
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
            fty.ret->rewrite = &compositeToInt;
            fty.ret->ltype = compositeToInt.type(fty.ret->type, fty.ret->ltype);
        }
    }

    // EXPLICIT PARAMETERS
    for (IrFuncTy::ArgRIter I = fty.args.rbegin(), E = fty.args.rend(); I != E; ++I)
    {
        IrFuncTyArg& arg = **I;

        if (arg.byref)
            continue;

        rewriteArgument(arg);
    }
}

void Win64TargetABI::rewriteArgument(IrFuncTyArg& arg)
{
    Type* ty = arg.type->toBasetype();

    if (ty->ty == Tcomplex32)
    {
        // {float,float} cannot be bit-cast to int64 (using CompositeToInt)
        // FIXME: is there a way to force a bit-cast?
        arg.rewrite = &cfloatToInt;
        arg.ltype = cfloatToInt.type(arg.type, arg.ltype);
    }
    else if (isComposite(ty) && canRewriteAsInt(ty))
    {
        arg.rewrite = &compositeToInt;
        arg.ltype = compositeToInt.type(arg.type, arg.ltype);
    }
    // FIXME: this should actually be handled by LLVM and the ByVal arg attribute
    else if (isPassedWithByvalSemantics(ty))
    {
        // these types are passed byval:
        // the caller allocates a copy and then passes a pointer to the copy
        // FIXME: use tightly packed struct for creal like DMD?
        arg.rewrite = &byval_rewrite;
        arg.ltype = byval_rewrite.type(arg.type, arg.ltype);

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
