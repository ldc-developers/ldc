//===-- abi-x86-64.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
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
// LLVM gets it right for most types, but complex numbers, structs and static
// arrays need some help. To make sure it gets those right we essentially
// bitcast these types to a type to which LLVM assigns the appropriate
// registers (using DMD's toArgTypes() machinery), and pass that instead.
// Structs that are required to be passed in memory are marked with the ByVal
// attribute to ensure no part of them ends up in registers when only a subset
// of the desired registers are available.
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
  namespace dmd_abi {
    // Structs, static arrays and cfloats may be rewritten to exploit registers.
    // This function returns the rewritten type, or null if no transformation is needed.
    LLType* getAbiType(Type* ty) {
        // First, check if there's any need of a transformation:
        if (!(ty->ty == Tcomplex32 || ty->ty == Tstruct || ty->ty == Tsarray))
            return NULL; // Nothing to do

        // Okay, we may need to transform. Figure out a canonical type:

        TypeTuple* argTypes = toArgTypes(ty);
        if (!argTypes || argTypes->arguments->empty())
            return NULL; // don't rewrite

        LLType* abiTy = NULL;
        if (argTypes->arguments->size() == 1) {
            abiTy = DtoType((*argTypes->arguments->begin())->type);
            // don't rewrite to a single bit (assertions in tollvm.cpp), choose a byte instead
            abiTy = i1ToI8(abiTy);
        } else {
            std::vector<LLType*> parts;
            for (Array<Parameter*>::iterator I = argTypes->arguments->begin(), E = argTypes->arguments->end(); I != E; ++I) {
                LLType* partType = DtoType((*I)->type);
                // round up the DMD argtype for an eightbyte of a struct to a corresponding 64-bit type
                // this makes sure that 64 bits of the chosen register are used and thus
                // makes sure all potential padding bytes of a struct are copied
                if (partType->isIntegerTy())
                    partType = LLType::getInt64Ty(gIR->context());
                else if (partType->isFloatTy())
                    partType = LLType::getDoubleTy(gIR->context());
                parts.push_back(partType);
            }
            abiTy = LLStructType::get(gIR->context(), parts);
        }

        return abiTy;
    }

    bool passByVal(Type* ty) {
        TypeTuple* argTypes = toArgTypes(ty);
        if (!argTypes)
            return false;

        return argTypes->arguments->empty(); // empty => cannot be passed in registers
    }
  } // namespace dmd_abi

    // Checks two LLVM types for memory-layout equivalency.
    // A pointer type is equivalent to any other pointer type.
    bool typesAreEquivalent(LLType* a, LLType* b) {
        if (a == b)
            return true;
        if (!a || !b)
            return false;

        LLStructType* structA;
        while ((structA = isaStruct(a)) && structA->getNumElements() == 1)
            a = structA->getElementType(0);

        LLStructType* structB;
        while ((structB = isaStruct(b)) && structB->getNumElements() == 1)
            b = structB->getElementType(0);

        if (a == b || (a->isPointerTy() && b->isPointerTy()))
            return true;

        if (!(structA && structB) ||
            structA->isPacked() != structB->isPacked() ||
            structA->getNumElements() != structB->getNumElements()) {
            return false;
        }

        for (unsigned i = 0; i < structA->getNumElements(); ++i) {
            if (!typesAreEquivalent(structA->getElementType(i), structB->getElementType(i)))
                return false;
        }

        return true;
    }

    LLType* getAbiType(Type* ty) {
        return dmd_abi::getAbiType(ty->toBasetype());
    }

    struct RegCount {
        char int_regs, sse_regs;

        RegCount() : int_regs(6), sse_regs(8) {}

        explicit RegCount(LLType* ty) : int_regs(0), sse_regs(0) {
            if (LLStructType* structTy = isaStruct(ty)) {
                for (unsigned i = 0; i < structTy->getNumElements(); ++i)
                {
                    RegCount elementRegCount(structTy->getElementType(i));
                    int_regs += elementRegCount.int_regs;
                    sse_regs += elementRegCount.sse_regs;
                }

                assert(int_regs + sse_regs <= 2);
            } else { // not a struct
                if (ty->isIntegerTy() || ty->isPointerTy()) {
                    ++int_regs;
                } else if (ty->isFloatingPointTy() || ty->isVectorTy()) {
                    // X87 reals are passed on the stack
                    if (!ty->isX86_FP80Ty())
                        ++sse_regs;
                } else {
                    unsigned sizeInBits = gDataLayout->getTypeSizeInBits(ty);
                    IF_LOG Logger::cout() << "SysV RegCount: assuming 1 GP register for type " << *ty
                        << " (" << sizeInBits << " bits)\n";
                    assert(sizeInBits > 0 && sizeInBits <= 64);
                    ++int_regs;
                }
            }
        }

        enum SubtractionResult {
            ArgumentFitsIn,
            ArgumentWouldFitInPartially,
            ArgumentDoesntFitIn
        };

        SubtractionResult trySubtract(const IrFuncTyArg& arg) {
            const RegCount wanted(arg.ltype);

            const bool anyRegAvailable = (wanted.int_regs > 0 && int_regs > 0) ||
                                         (wanted.sse_regs > 0 && sse_regs > 0);
            if (!anyRegAvailable)
                return ArgumentDoesntFitIn;

            if (int_regs < wanted.int_regs || sse_regs < wanted.sse_regs)
                return ArgumentWouldFitInPartially;

            int_regs -= wanted.int_regs;
            sse_regs -= wanted.sse_regs;

            return ArgumentFitsIn;
        }
    };
}

/**
 * This type performs the actual struct/cfloat rewriting by simply storing to
 * memory so that it's then readable as the other type (i.e., bit-casting).
 */
struct X86_64_C_struct_rewrite : ABIRewrite {
    LLValue* get(Type* dty, DValue* v)
    {
        LLValue* address = storeToMemory(v->getRVal(), 0, ".X86_64_C_struct_rewrite_dump");
        LLType* type = DtoType(dty);
        return loadFromMemory(address, type, ".X86_64_C_struct_rewrite_getResult");
    }

    void getL(Type* dty, DValue* v, LLValue* lval) {
        storeToMemory(v->getRVal(), lval);
    }

    LLValue* put(Type* dty, DValue* v) {
        assert(dty == v->getType());
        LLValue* address = getAddressOf(v);

        LLType* abiTy = getAbiType(dty);
        assert(abiTy && "Why are we rewriting a non-rewritten type?");

        return loadFromMemory(address, abiTy, ".X86_64_C_struct_rewrite_putResult");
    }

    LLType* type(Type* dty, LLType* t) {
        return getAbiType(dty);
    }
};

/**
 * This type is used to force LLVM to pass a LL struct in memory,
 * on the function arguments stack. We need this to prevent LLVM
 * from passing a LL struct partially in registers, partially in
 * memory.
 * This is achieved by passing a pointer to the struct and using
 * the ByVal LLVM attribute.
 */
struct ImplicitByvalRewrite : ABIRewrite {
    LLValue* get(Type* dty, DValue* v) {
        LLValue* pointer = v->getRVal();
        return DtoLoad(pointer, ".ImplicitByvalRewrite_getResult");
    }

    void getL(Type* dty, DValue* v, LLValue* lval) {
        LLValue* pointer = v->getRVal();
        DtoAggrCopy(lval, pointer);
    }

    LLValue* put(Type* dty, DValue* v) {
        assert(dty == v->getType());
        return getAddressOf(v);
    }

    LLType* type(Type* dty, LLType* t) {
        return getPtrToType(DtoType(dty));
    }
};

struct X86_64TargetABI : TargetABI {
    X86_64_C_struct_rewrite struct_rewrite;
    ImplicitByvalRewrite byvalRewrite;

    llvm::CallingConv::ID callingConv(LINK l);

    bool returnInArg(TypeFunction* tf);

    bool passByVal(Type* t);

    void rewriteFunctionType(TypeFunction* tf, IrFuncTy& fty);
    void rewriteVarargs(IrFuncTy& fty, std::vector<IrFuncTyArg*>& args);
    void rewriteArgument(IrFuncTy& fty, IrFuncTyArg& arg);
    void rewriteArgument(IrFuncTyArg& arg, RegCount& regCount);

    LLValue* prepareVaStart(LLValue* pAp);

    void vaCopy(LLValue* pDest, LLValue* src);

    LLValue* prepareVaArg(LLValue* pAp);

private:
    LLType* getValistType();
    RegCount& getRegCount(IrFuncTy& fty) { return reinterpret_cast<RegCount&>(fty.tag); }
};


// The public getter for abi.cpp
TargetABI* getX86_64TargetABI() {
    return new X86_64TargetABI;
}


llvm::CallingConv::ID X86_64TargetABI::callingConv(LINK l) {
    switch (l) {
        case LINKc:
        case LINKcpp:
        case LINKpascal:
        case LINKwindows: // Doesn't really make sense, user should use Win64 target.
        case LINKd:
        case LINKdefault:
            return llvm::CallingConv::C;
        default:
            llvm_unreachable("Unhandled D linkage type.");
    }
}

bool X86_64TargetABI::returnInArg(TypeFunction* tf) {
    if (tf->isref)
        return false;

    Type* rt = tf->next;
    return passByVal(rt);
}

bool X86_64TargetABI::passByVal(Type* t) {
    return dmd_abi::passByVal(t->toBasetype());
}

void X86_64TargetABI::rewriteArgument(IrFuncTy& fty, IrFuncTyArg& arg) {
    llvm_unreachable("Please use the other overload explicitly.");
}

void X86_64TargetABI::rewriteArgument(IrFuncTyArg& arg, RegCount& regCount) {
    LLType* originalLType = arg.ltype;
    Type* t = arg.type->toBasetype();

    LLType* abiTy = getAbiType(t);
    if (abiTy && !typesAreEquivalent(abiTy, originalLType)) {
        IF_LOG {
            Logger::println("Rewriting argument type %s", t->toChars());
            LOG_SCOPE;
            Logger::cout() << *originalLType << " => " << *abiTy << '\n';
        }

        arg.rewrite = &struct_rewrite;
        arg.ltype = abiTy;
    }

    if (regCount.trySubtract(arg) == RegCount::ArgumentWouldFitInPartially) {
        // pass LL structs implicitly ByVal, otherwise LLVM passes
        // them partially in registers, partially in memory
        assert(originalLType->isStructTy());
        IF_LOG Logger::cout() << "Passing implicitly ByVal: " << arg.type->toChars() << " (" << *originalLType << ")\n";
        arg.rewrite = &byvalRewrite;
        arg.ltype = originalLType->getPointerTo();
        arg.attrs.add(LDC_ATTRIBUTE(ByVal));
    }
}

void X86_64TargetABI::rewriteFunctionType(TypeFunction* tf, IrFuncTy &fty) {
    RegCount& regCount = getRegCount(fty);
    regCount = RegCount(); // initialize

    // RETURN VALUE
    if (!tf->isref && !fty.arg_sret && tf->next->toBasetype()->ty != Tvoid) {
        Logger::println("x86-64 ABI: Transforming return type");
        LOG_SCOPE;
        RegCount dummy;
        rewriteArgument(*fty.ret, dummy);
    }

    // IMPLICIT PARAMETERS
    if (fty.arg_sret)
        regCount.int_regs--;
    if (fty.arg_this || fty.arg_nest)
        regCount.int_regs--;
    if (fty.arg_arguments)
        regCount.int_regs -= 2; // dynamic array

    // EXPLICIT PARAMETERS
    Logger::println("x86-64 ABI: Transforming argument types");
    LOG_SCOPE;

    // extern(D): reverse parameter order for non variadics, for DMD-compliance
    if (tf->linkage == LINKd && tf->varargs != 1 && fty.args.size() > 1)
        fty.reverseParams = true;

    int begin = 0, end = fty.args.size(), step = 1;
    if (fty.reverseParams) {
        begin = end - 1;
        end = -1;
        step = -1;
    }
    for (int i = begin; i != end; i += step) {
        IrFuncTyArg& arg = *fty.args[i];

        if (arg.byref) {
            if (!arg.isByVal() && regCount.int_regs > 0)
                regCount.int_regs--;

            continue;
        }

        rewriteArgument(arg, regCount);
    }

    // regCount (fty.tag) is now in the state after all implicit & formal args,
    // ready to serve as initial state for each vararg call site, see below
}

void X86_64TargetABI::rewriteVarargs(IrFuncTy& fty, std::vector<IrFuncTyArg*>& args)
{
    // use a dedicated RegCount copy for each call site and initialize it with fty.tag
    RegCount regCount = getRegCount(fty);

    for (unsigned i = 0; i < args.size(); ++i)
    {
        IrFuncTyArg& arg = *args[i];
        if (!arg.byref) // don't rewrite ByVal arguments
            rewriteArgument(arg, regCount);
    }
}


/**
 * The System V AMD64 ABI uses a special native va_list type - a 24-bytes struct passed by
 * reference.
 * In druntime, the struct is defined as core.stdc.stdarg.__va_list; the actually used
 * core.stdc.stdarg.va_list type is a raw char* pointer though to achieve byref semantics.
 * This requires a little bit of compiler magic in the following implementations.
 */

LLType* X86_64TargetABI::getValistType() {
    LLType* uintType = LLType::getInt32Ty(gIR->context());
    LLType* voidPointerType = getVoidPtrType();

    std::vector<LLType*> parts;       // struct __va_list {
    parts.push_back(uintType);        //   uint gp_offset;
    parts.push_back(uintType);        //   uint fp_offset;
    parts.push_back(voidPointerType); //   void* overflow_arg_area;
    parts.push_back(voidPointerType); //   void* reg_save_area; }

    return LLStructType::get(gIR->context(), parts);
}

LLValue* X86_64TargetABI::prepareVaStart(LLValue* pAp) {
    // Since the user only created a char* pointer (ap) on the stack before invoking va_start,
    // we first need to allocate the actual __va_list struct and set 'ap' to its address.
    LLValue* valistmem = DtoRawAlloca(getValistType(), 0, "__va_list_mem");
    valistmem = DtoBitCast(valistmem, getVoidPtrType());
    DtoStore(valistmem, pAp); // ap = (void*)__va_list_mem

    // pass a void* pointer to the actual struct to LLVM's va_start intrinsic
    return valistmem;
}

void X86_64TargetABI::vaCopy(LLValue* pDest, LLValue* src) {
    // Analog to va_start, we need to allocate a __va_list struct on the stack first
    // and set the passed 'dest' char* pointer to its address.
    LLValue* valistmem = DtoRawAlloca(getValistType(), 0, "__va_list_mem");
    DtoStore(DtoBitCast(valistmem, getVoidPtrType()), pDest);

    // Now bitcopy the source struct over the destination struct.
    src = DtoBitCast(src, valistmem->getType());
    DtoStore(DtoLoad(src), valistmem); // *(__va_list*)dest = *(__va_list*)src
}

LLValue* X86_64TargetABI::prepareVaArg(LLValue* pAp)
{
    // pass a void* pointer to the actual __va_list struct to LLVM's va_arg intrinsic
    return DtoLoad(pAp);
}
