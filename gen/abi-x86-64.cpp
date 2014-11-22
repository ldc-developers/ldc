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

    namespace ldc {
    enum ArgClass {
        Integer, Sse, SseUp, X87, X87Up, ComplexX87, NoClass, Memory
    };

    struct Classification {
        bool isMemory;
        ArgClass classes[2];

        Classification() : isMemory(false) {
            classes[0] = NoClass;
            classes[1] = NoClass;
        }

        void addField(unsigned offset, ArgClass cl) {
            if (isMemory)
                return;

            // Note that we don't need to bother checking if it crosses 8 bytes.
            // We don't get here with unaligned fields, and anything that can be
            // big enough to cross 8 bytes (cdoubles, reals, structs and arrays)
            // is special-cased in classifyType()
            int idx = (offset < 8 ? 0 : 1);

            ArgClass nw = merge(classes[idx], cl);
            if (nw != classes[idx]) {
                classes[idx] = nw;

                if (nw == Memory) {
                    classes[1-idx] = Memory;
                    isMemory = true;
                }
            }
        }

    private:
        ArgClass merge(ArgClass accum, ArgClass cl) {
            if (accum == cl)
                return accum;
            if (accum == NoClass)
                return cl;
            if (cl == NoClass)
                return accum;
            if (accum == Memory || cl == Memory)
                return Memory;
            if (accum == Integer || cl == Integer)
                return Integer;
            if (accum == X87 || accum == X87Up || accum == ComplexX87 ||
                cl == X87 || cl == X87Up || cl == ComplexX87)
                return Memory;
            return Sse;
        }
    };

    void classifyType(Classification& accum, Type* ty, d_uns64 offset) {
        IF_LOG Logger::cout() << "Classifying " << ty->toChars() << " @ " << offset << '\n';

        ty = ty->toBasetype();

        if (ty->size() == 0) {
            return;
        } else if (ty->isintegral() || ty->ty == Tpointer || ty->ty == Tvoid) {
            accum.addField(offset, Integer);
        } else if (ty->ty == Tfloat80 || ty->ty == Timaginary80) {
            accum.addField(offset, X87);
            accum.addField(offset+8, X87Up);
        } else if (ty->ty == Tcomplex80) {
            accum.addField(offset, ComplexX87);
            // make sure other half knows about it too:
            accum.addField(offset+16, ComplexX87);
        } else if (ty->ty == Tcomplex64) {
            accum.addField(offset, Sse);
            accum.addField(offset+8, Sse);
        } else if (ty->ty == Tcomplex32) {
            accum.addField(offset, Sse);
            accum.addField(offset+4, Sse);
        } else if (ty->isfloating()) {
            accum.addField(offset, Sse);
        } else if (ty->size() > 16 || hasUnalignedFields(ty)) {
            // This isn't creal, yet is > 16 bytes, so pass in memory.
            // Must be after creal case but before arrays and structs,
            // the other types that can get bigger than 16 bytes
            accum.addField(offset, Memory);
        } else if (ty->ty == Tsarray) {
            Type* eltType = ty->nextOf();
            d_uns64 eltsize = eltType->size();
            if (eltsize > 0) {
                d_uns64 dim = ty->size() / eltsize;
                assert(dim <= 16
                        && "Array of non-empty type <= 16 bytes but > 16 elements?");
                for (d_uns64 i = 0; i < dim; i++) {
                    classifyType(accum, eltType, offset);
                    offset += eltsize;
                }
            }
        } else if (ty->ty == Tstruct) {
            VarDeclarations& fields = static_cast<TypeStruct*>(ty)->sym->fields;
            if (fields.dim == 0) { // no fields, but size > 0? treat as void
                classifyType(accum, Type::tvoid, offset);
            } else for (size_t i = 0; i < fields.dim; i++) {
                classifyType(accum, fields[i]->type, offset + fields[i]->offset);
            }
        } else {
            IF_LOG Logger::cout() << "x86-64 ABI: Implicitly handled type: "
                               << ty->toChars() << '\n';
            // arrays, delegates, etc. (pointer-sized fields, <= 16 bytes)
            assert((offset == 0 || offset == 8)
                    && "must be aligned and doesn't fit otherwise");
            assert(ty->size() % 8 == 0 && "Not a multiple of pointer size?");

            accum.addField(offset, Integer);
            if (ty->size() > 8)
                accum.addField(offset+8, Integer);
        }
    }

    Classification classify(Type* ty) {
        typedef std::map<Type*, Classification> ClassMap;
        static ClassMap cache;

        ClassMap::iterator it = cache.find(ty);
        if (it != cache.end()) {
            return it->second;
        } else {
            Classification cl;
            classifyType(cl, ty, 0);
            cache[ty] = cl;
            return cl;
        }
    }

    /// Returns the type to pass as, or null if no transformation is needed.
    LLType* getAbiType(Type* ty) {
        ty = ty->toBasetype();

        // First, check if there's any need of a transformation:

        if (keepUnchanged(ty))
            return 0;

        if (ty->ty != Tcomplex32 && ty->ty != Tstruct)
            return 0; // Nothing to do,

        Classification cl = classify(ty);
        assert(!cl.isMemory);

        if (cl.classes[0] == NoClass) {
            assert(cl.classes[1] == NoClass && "Non-empty struct with empty first half?");
            return 0; // Empty structs should also be handled correctly by LLVM
        }

        // Okay, we may need to transform. Figure out a canonical type:

        std::vector<LLType*> parts;

        unsigned size = ty->size();

        switch (cl.classes[0]) {
            case Integer: {
                unsigned bits = (size >= 8 ? 64 : (size * 8));
                parts.push_back(LLIntegerType::get(gIR->context(), bits));
                break;
            }

            case Sse:
                parts.push_back(size <= 4 ? LLType::getFloatTy(gIR->context()) : LLType::getDoubleTy(gIR->context()));
                break;

            case X87:
                assert(cl.classes[1] == X87Up && "Upper half of real not X87Up?");
                /// The type only contains a single real/ireal field,
                /// so just use that type.
                return const_cast<LLType*>(LLType::getX86_FP80Ty(gIR->context()));

            default:
                llvm_unreachable("Unanticipated argument class.");
        }

        switch(cl.classes[1]) {
            case NoClass:
                assert(parts.size() == 1);
                // No need to use a single-element struct type.
                // Just use the element type instead.
                return const_cast<LLType*>(parts[0]);
                break;

            case Integer: {
                assert(size > 8);
                unsigned bits = (size - 8) * 8;
                parts.push_back(LLIntegerType::get(gIR->context(), bits));
                break;
            }
            case Sse:
                parts.push_back(size <= 12 ? LLType::getFloatTy(gIR->context()) : LLType::getDoubleTy(gIR->context()));
                break;

            case X87Up:
                if(cl.classes[0] == X87) {
                    // This won't happen: it was short-circuited while
                    // processing the first half.
                } else {
                    // I can't find this anywhere in the ABI documentation,
                    // but this is what gcc does (both regular and llvm-gcc).
                    // (This triggers for types like union { real r; byte b; })
                    parts.push_back(LLType::getDoubleTy(gIR->context()));
                }
                break;

            default:
                llvm_unreachable("Unanticipated argument class for second half.");
        }
        return LLStructType::get(gIR->context(), parts);
    }
    } // ldc namespace

    /**
     * Structs (and cfloats) may be rewritten to exploit registers.
     * This function returns the rewritten type, or null if no transformation is needed.
     */
    LLType* getAbiType_argTypes(Type* ty) {
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
            // don't rewrite to a single bit (assertions in tollvm.cpp)
            if (abiTy == LLType::getInt1Ty(gIR->context()))
                return 0;
        } else {                                // multiple parts => LLVM struct
            std::vector<LLType*> parts;
            for (Array<Parameter*>::iterator I = argTypes->arguments->begin(), E = argTypes->arguments->end(); I != E; ++I)
                parts.push_back(DtoType((*I)->type));
            abiTy = LLStructType::get(gIR->context(), parts);
        }

        //IF_LOG Logger::cout() << "getAbiType(" << ty->toChars() << "): " << *abiTy << '\n';

        return abiTy;
    }

    // Temporary implementation validating the new toArgTypes()-based version
    // against the previous LDC-specific version.
    LLType* getAbiType(Type* ty) {
        LLType* argTypesType = getAbiType_argTypes(ty);
        IF_LOG Logger::println("ldc::getAbiType(%s)...", ty->toChars());
        LLType* ldcType = ldc::getAbiType(ty);

        IF_LOG if (argTypesType != ldcType) {
            Logger::print("getAbiType(%s) mismatch: ", ty->toChars());
            if (!argTypesType)
                Logger::print("(null)");
            else
                Logger::cout() << *argTypesType;
            Logger::print(" (toArgTypes) vs. ");
            if (!ldcType)
                Logger::print("(null)");
            else
                Logger::cout() << *ldcType;
            Logger::println(" (LDC)");
        }
        //assert(argTypesType == ldcType && "getAbiType() mismatch between toArgTypes() and LDC!");

        return argTypesType;
    }

    // Returns true if the previous LDC-specific version classifies the type
    // as being passed on the stack.
    bool ldcWouldPassByVal(Type* ty) {
        IF_LOG Logger::println("ldc::classify(%s)...", ty->toChars());
        return ldc::classify(ty).isMemory;
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
        LLValue* rval = v->getRVal();
        LLValue* lval;
        // already lowered to a pointer to the struct/static array?
        if (rval->getType()->isPointerTy()) {
            lval = rval;
        } else if (v->isLVal()) {
            lval = v->getLVal();
        } else {
            // No memory location, create one.
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

    llvm::CallingConv::ID callingConv(LINK l);

    bool returnInArg(TypeFunction* tf);

    bool passByVal(Type* t);

    void rewriteFunctionType(TypeFunction* tf, IrFuncTy &fty);

    void rewriteArgument(IrFuncTyArg& arg);

    LLValue* prepareVaStart(LLValue* pAp);

    void vaCopy(LLValue* pDest, LLValue* src);

    LLValue* prepareVaArg(LLValue* pAp);

private:
    LLType* getValistType();
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

    if (t->size() == 0 || keepUnchanged(t))
        return false;

    bool byval = false;

    TypeTuple* argTypes = toArgTypes(t);
    if (!argTypes) {
        IF_LOG Logger::println("passByVal(%s): toArgTypes() returned null!", t->toChars());
    } else
        byval = argTypes->arguments->empty(); // empty => cannot be passed in registers

    bool ldcResult = ldcWouldPassByVal(t);
    IF_LOG if (byval != ldcResult) {
        Logger::println("passByVal(%s) mismatch: %s (toArgTypes) vs. %s (LDC)",
            t->toChars(), byval ? "true" : "false", ldcResult ? "true" : "false");
    }
    //assert(byval == ldcResult && "passByVal() mismatch between toArgTypes() and LDC!");

    return byval;
}

void X86_64TargetABI::rewriteArgument(IrFuncTyArg& arg) {
    Type* t = arg.type->toBasetype();

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
