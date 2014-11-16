//===-- abi-x86-64.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// BIG RED TODO NOTE: On x86_64, the C ABI should also be used for extern(D)
// functions, as mandated by the language standard and required for DMD
// compatibility. The below description and implementation dates back to the
// time where x86_64 was still an exotic target for D. Also, the frontend
// toArgTypes() machinery should be used for doing the type classification to
// reduce code duplication and make sure the va_arg implementation is always
// up to date with the code we emit.
//
//===----------------------------------------------------------------------===//
//
// extern(C) implements the C calling convention for x86-64, as found in
// http://www.x86-64.org/documentation/abi-0.99.pdf
//
// Note:
//   Where a discrepancy was found between llvm-gcc and the ABI documentation,
//   llvm-gcc behavior was used for compatibility (after it was verified that
//   regular gcc has the same behavior).
//
// LLVM gets it right for most types, but complex numbers and structs need some
// help. To make sure it gets those right we essentially bitcast small structs
// to a type to which LLVM assigns the appropriate registers, and pass that
// instead. Structs that are required to be passed in memory are explicitly
// marked with the ByVal attribute to ensure no part of them ends up in
// registers when only a subset of the desired registers are available.
//
// We don't perform the same transformation for D-specific types that contain
// multiple parts, such as dynamic arrays and delegates. They're passed as if
// the parts were passed as separate parameters. This helps make things like
// printf("%.*s", o.toString()) work as expected; if we didn't do this that
// wouldn't work if there were 4 other integer/pointer arguments before the
// toString() call because the string got bumped to memory with one integer
// register still free. Keeping it untransformed puts the length in a register
// and the pointer in memory, as printf expects it.
//
//===----------------------------------------------------------------------===//

#include "gen/abi-x86-64.h"
#include "aggregate.h"
#include "declaration.h"
#include "mtype.h"
#include "gen/abi-generic.h"
#include "gen/abi-x86-64.h"
#include "gen/abi.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include <cassert>
#include <map>
#include <string>
#include <utility>

TypeTuple* toArgTypes(Type* t); // in dmd2/argtypes.c

namespace {
    /**
     * This function helps filter out things that look like structs to C,
     * but should be passed to C in separate arguments anyway.
     *
     * (e.g. dynamic arrays are passed as separate length and ptr. This
     * is both less work and makes printf("%.*s", o.toString()) work)
     */
    inline bool keepUnchanged(Type* t) {
        switch (t->ty) {
            case Tarray:    // dynamic array
            case Taarray:   // assoc array
            case Tdelegate:
                return true;

            default:
                return false;
        }
    }

    /**
     * Structs (and cfloats) may be rewritten to exploit registers.
     * This function returns the rewritten type, or null if no transformation is needed.
     */
    LLType* getAbiType(Type* ty) {
        ty = ty->toBasetype();

        // First, check if there's any need of a transformation:

        if (keepUnchanged(ty))
            return 0;

        // Only consider rewriting cfloats and structs
        if (!(ty->ty == Tcomplex32 || ty->ty == Tstruct))
            return 0; // Nothing to do

        // Empty structs should also be handled correctly by LLVM
        if (ty->size() == 0)
            return 0;

        TypeTuple* argTypes = toArgTypes(ty);
        if (argTypes->arguments->empty()) // cannot be passed in registers
            return 0;

        // Okay, we may need to transform. Figure out a canonical type:

        LLType* abiTy = 0;
        if (argTypes->arguments->size() == 1) { // single part
            abiTy = DtoType((*argTypes->arguments->begin())->type);
        } else {                                // multiple parts => LLVM struct
            std::vector<LLType*> parts;
            for (Array<Parameter*>::iterator I = argTypes->arguments->begin(), E = argTypes->arguments->end(); I != E; ++I)
                parts.push_back(DtoType((*I)->type));
            abiTy = LLStructType::get(gIR->context(), parts);
        }

        //IF_LOG Logger::cout() << "getAbiType(" << ty->toChars() << "): " << *abiTy << '\n';

        return abiTy;
    }
}

/**
 * This type performs the actual struct/cfloat rewriting by simply storing to
 * memory so that it's then readable as the other type (i.e., bit-casting).
 */
struct X86_64_C_struct_rewrite : ABIRewrite {
    // Get struct from ABI-mangled representation
    LLValue* get(Type* dty, DValue* v)
    {
        LLValue* lval;
        if (v->isLVal()) {
            lval = v->getLVal();
        } else {
            // No memory location, create one.
            LLValue* rval = v->getRVal();
            lval = DtoRawAlloca(rval->getType(), 0);
            DtoStore(rval, lval);
        }

        LLType* pTy = getPtrToType(DtoType(dty));
        return DtoLoad(DtoBitCast(lval, pTy), "get-result");
    }

    // Get struct from ABI-mangled representation, and store in the provided location.
    void getL(Type* dty, DValue* v, llvm::Value* lval) {
        LLValue* rval = v->getRVal();
        LLType* pTy = getPtrToType(rval->getType());
        DtoStore(rval, DtoBitCast(lval, pTy));
    }

    // Turn a struct into an ABI-mangled representation
    LLValue* put(Type* dty, DValue* v)
    {
        LLValue* lval;
        if (v->isLVal()) {
            lval = v->getLVal();
        } else {
            // No memory location, create one.
            LLValue* rval = v->getRVal();
            lval = DtoRawAlloca(rval->getType(), 0);
            DtoStore(rval, lval);
        }

        LLType* abiTy = getAbiType(dty);
        assert(abiTy && "Why are we rewriting a non-rewritten type?");

        LLType* pTy = getPtrToType(abiTy);
        return DtoLoad(DtoBitCast(lval, pTy), "put-result");
    }

    /// should return the transformed type for this rewrite
    LLType* type(Type* dty, LLType* t)
    {
        return getAbiType(dty);
    }
};


struct X86_64TargetABI : TargetABI {
    X86_64_C_struct_rewrite struct_rewrite;
    CompositeToInt compositeToInt;

    llvm::CallingConv::ID callingConv(LINK l);

    bool returnInArg(TypeFunction* tf);

    bool passByVal(Type* t);

    void rewriteFunctionType(TypeFunction* tf, IrFuncTy &fty);

    void rewriteArgument(IrFuncTyArg& arg);

private:
    // Rewrite structs and static arrays <= 64 bit and of a size that is a power of 2
    // to an integer of the same size.
    bool canRewriteAsInt(Type* t) {
        t = t->toBasetype();
        unsigned size = t->size();
        return (t->ty == Tstruct || t->ty == Tsarray)
            && (size == 1 || size == 2 || size == 4 || size == 8);
    }
};


// The public getter for abi.cpp
TargetABI* getX86_64TargetABI() {
    return new X86_64TargetABI;
}


llvm::CallingConv::ID X86_64TargetABI::callingConv(LINK l)
{
    switch (l)
    {
    case LINKc:
    case LINKcpp:
    case LINKd:
    case LINKdefault:
        return llvm::CallingConv::C;
    case LINKpascal:
    case LINKwindows: // Doesn't really make sense, user should use Win64 target.
        return llvm::CallingConv::X86_StdCall;
    default:
        llvm_unreachable("Unhandled D linkage type.");
    }
}

bool X86_64TargetABI::returnInArg(TypeFunction* tf) {
    if (tf->isref)
        return false;

    Type* rt = tf->next->toBasetype();
    return rt->ty != Tvoid && passByVal(rt);
}

bool X86_64TargetABI::passByVal(Type* t) {
    t = t->toBasetype();

    if (t->size() == 0 || keepUnchanged(t) || canRewriteAsInt(t))
        return false;

    TypeTuple* argTypes = toArgTypes(t);
    if (!argTypes) {
        IF_LOG Logger::cout() << "X86_64TargetABI::passByVal(): no argTypes for " << t->toChars() << "!\n";
        return false; // TODO: verify
    }

    bool onStack = argTypes->arguments->empty(); // empty => cannot be passed in registers

    //if (onStack)
    //    IF_LOG Logger::cout() << "Passed byval: " << t->toChars() << '\n';

    return onStack;
}

void X86_64TargetABI::rewriteArgument(IrFuncTyArg& arg) {
    Type* t = arg.type->toBasetype();

    if (canRewriteAsInt(t)) {
        arg.rewrite = &compositeToInt;
        arg.ltype = compositeToInt.type(arg.type, arg.ltype);
        return;
    }

    LLType* abiTy = getAbiType(t);
    if (abiTy && abiTy != arg.ltype) {
        assert(t->ty == Tcomplex32 || t->ty == Tstruct);

        arg.rewrite = &struct_rewrite;
        arg.ltype = abiTy;

        IF_LOG Logger::cout() << "Rewriting argument: " << t->toChars() << " => " << *abiTy << '\n';
    }
}

void X86_64TargetABI::rewriteFunctionType(TypeFunction* tf, IrFuncTy &fty) {
    // RETURN VALUE
    if (!tf->isref) {
        Logger::println("x86-64 ABI: Transforming return type");
        rewriteArgument(*fty.ret);
    }

    // EXPLICIT PARAMETERS
    Logger::println("x86-64 ABI: Transforming arguments");
    LOG_SCOPE;

    for (IrFuncTy::ArgIter I = fty.args.begin(), E = fty.args.end(); I != E; ++I) {
        IrFuncTyArg& arg = **I;

        IF_LOG Logger::cout() << "Arg: " << arg.type->toChars() << '\n';

        // Arguments that are in memory are of no interest to us.
        if (arg.byref)
            continue;

        rewriteArgument(arg);
        IF_LOG Logger::cout() << "New arg type: " << *arg.ltype << '\n';
    }

    // extern(D): reverse parameter order for non variadics, for DMD-compliance
    if (tf->linkage == LINKd && tf->varargs != 1 && fty.args.size() > 1)
        fty.reverseParams = true;
}

bool isSystemVAMD64Target() {
    return global.params.targetTriple.getArch() == llvm::Triple::x86_64
        && !global.params.targetTriple.isOSWindows();
}

LLType* getSystemVAMD64NativeValistType() {
    LLType* uintType = LLType::getInt32Ty(gIR->context());
    LLType* voidPointerType = getVoidPtrType();

    std::vector<LLType*> parts;       // struct __va_list {
    parts.push_back(uintType);        //   uint gp_offset;
    parts.push_back(uintType);        //   uint fp_offset;
    parts.push_back(voidPointerType); //   void* overflow_arg_area;
    parts.push_back(voidPointerType); //   void* reg_save_area; }

    return LLStructType::get(gIR->context(), parts, "__va_list");
}
