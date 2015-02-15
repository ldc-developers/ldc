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

//#define VALIDATE_AGAINST_OLD_LDC_VERSION

TypeTuple* toArgTypes(Type* t); // in dmd2/argtypes.c

namespace {
#ifdef VALIDATE_AGAINST_OLD_LDC_VERSION
  namespace ldc_abi {
    /**
     * This function helps filter out things that look like structs to C,
     * but should be passed to C in separate arguments anyway.
     *
     * (e.g. dynamic arrays are passed as separate length and ptr. This
     * is both less work and makes printf("%.*s", o.toString()) work)
     */
    inline bool keepUnchanged(Type* t) {
        if (t->size() == 0)
            return true;

        switch (t->ty) {
            case Tarray:    // dynamic array
            case Taarray:   // assoc array
            case Tdelegate:
            case Tvoid:
            case Tnull:
                return true;

            default:
                return false;
        }
    }

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

        explicit Classification(Type* ty) : isMemory(false) {
            classes[0] = NoClass;
            classes[1] = NoClass;

            classify(ty, 0);

            finalize();

            static const char* classNames[] = { "Integer", "Sse", "SseUp", "X87", "X87Up", "ComplexX87", "NoClass", "Memory" };
            IF_LOG Logger::println("classify(%s): <%s, %s>", ty->toChars(), classNames[classes[0]], classNames[classes[1]]);
        }

    private:
        static ArgClass merge(ArgClass accum, ArgClass cl) {
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

        void addField(unsigned offset, ArgClass cl) {
            if (isMemory)
                return;

            // Note that we don't need to bother checking if it crosses 8 bytes.
            // We don't get here with unaligned fields, and anything that can be
            // big enough to cross 8 bytes is special-cased in classify()
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

        void classify(Type* ty, d_uns64 offset) {
            IF_LOG Logger::cout() << "Classifying " << ty->toChars() << " @ " << offset << '\n';

            ty = ty->toBasetype();
            const d_uns64 size = ty->size();

            if (ty->ty == Tvoid) {
                if (size > 0) {
                    // treat as byte
                    assert(size == 1);
                    addField(offset, Integer);
                }
            } else if (ty->isintegral() || ty->ty == Tpointer) {
                addField(offset, Integer);
            } else if (ty->ty == Tfloat80 || ty->ty == Timaginary80) {
                addField(offset, X87);
                addField(offset+8, X87Up);
            } else if (ty->ty == Tcomplex80) {
                addField(offset, ComplexX87);
                // make sure other half knows about it too:
                addField(offset+16, ComplexX87);
            } else if (ty->ty == Tcomplex64) {
                addField(offset, Sse);
                addField(offset+8, Sse);
            } else if (ty->ty == Tcomplex32) {
                addField(offset, Sse);
                addField(offset+4, Sse);
            } else if (ty->isfloating()) {
                addField(offset, Sse);
            } else if (size > 16 || hasUnalignedFields(ty)) {
                // This isn't creal, yet is > 16 bytes, so pass in memory.
                // Must be after creal case but before arrays and structs,
                // the other types that can get bigger than 16 bytes
                addField(offset, Memory);
            } else if (ty->ty == Tsarray) {
                // treat a static array as struct
                Type* eltType = ty->nextOf();
                d_uns64 eltsize = eltType->size();
                if (size == 0 || eltsize == 0) {
                    // no and/or empty elements? treat as void
                    classify(Type::tvoid, offset);
                } else {
                    d_uns64 dim = size / eltsize;
                    assert(dim <= 16
                            && "Array of non-empty type <= 16 bytes but > 16 elements?");
                    for (d_uns64 i = 0; i < dim; i++) {
                        classify(eltType, offset + i*eltsize);
                    }
                }
            } else if (ty->ty == Tstruct) {
                VarDeclarations& fields = static_cast<TypeStruct*>(ty)->sym->fields;
                if (fields.dim == 0) {
                    // no fields? treat as void
                    classify(Type::tvoid, offset);
                } else for (size_t i = 0; i < fields.dim; i++) {
                    classify(fields[i]->type, offset + fields[i]->offset);
                }
            } else {
                IF_LOG Logger::println("x86-64 ABI: Implicitly handled type: %s", ty->toChars());
                // class references, dynamic and associative arrays, delegates, typeof(null)...
                // simply assume 1 or 2 pointer-sized fields passed in GP registers
                assert((size == 8 || size == 16)
                        && "Not a multiple of pointer size?");
                assert((offset % 8 == 0 && offset + size <= 16)
                        && "Must be aligned and fit into 16 bytes");

                addField(offset, Integer);
                if (size > 8)
                    addField(offset+8, Integer);
            }
        }

        void finalize() {
            if (isMemory)
                return;

            if (classes[1] == X87Up && classes[0] != X87) {
                classes[0] = Memory;
                classes[1] = Memory;
                isMemory = true;
            }
        }
    };

    Classification classify(Type* ty) {
        typedef std::map<Type*, Classification> ClassMap;
        static ClassMap cache;

        ClassMap::iterator it = cache.find(ty);
        if (it != cache.end()) {
            return it->second;
        } else {
            Classification cl(ty);
            cache[ty] = cl;
            return cl;
        }
    }

    LLType* getFittingIntegerType(unsigned size) {
        assert(size <= 8);

        if (size == 0)
            size = 1;
        else if (size == 3)
            size = 4;
        else if (size > 4 && size < 8)
            size = 8;

        return LLIntegerType::get(gIR->context(), size * 8);
    }

    // Returns the type to pass as, or null if no transformation is needed.
    LLType* getAbiType(Type* ty) {
        // First, check if there's any need of a transformation:

        if (!(ty->ty == Tcomplex32 || ty->ty == Tstruct || ty->ty == Tsarray))
            return 0; // Nothing to do

        const Classification cl = classify(ty);
        assert(!cl.isMemory);

        if (cl.classes[0] == NoClass) {
            assert(cl.classes[1] == NoClass && "Non-empty struct with empty first half?");
            return 0; // Empty structs should also be handled correctly by LLVM
        }

        // Okay, we may need to transform. Figure out a canonical type:

        std::vector<LLType*> parts;

        const unsigned size = ty->size();

        switch (cl.classes[0]) {
            case Integer:
                parts.push_back(getFittingIntegerType(std::min(size, 8u)));
                break;

            case Sse:
                parts.push_back(size <= 4 ? LLType::getFloatTy(gIR->context()) : LLType::getDoubleTy(gIR->context()));
                break;

            case X87:
                assert(cl.classes[1] == X87Up && "Upper half of real not X87Up?");
                // The type only contains a single real/ireal field,
                // so just use that type.
                return LLType::getX86_FP80Ty(gIR->context());

            case ComplexX87:
                assert(cl.classes[1] == ComplexX87 && "Upper half of creal not ComplexX87?");
                parts.resize(2, LLType::getX86_FP80Ty(gIR->context()));
                return LLStructType::get(gIR->context(), parts);

            default:
                llvm_unreachable("Unanticipated argument class.");
        }

        if (cl.classes[1] == NoClass) {
            assert(size <= 8);
            // No need to use a single-element struct type.
            // Just use the element type instead.
            return parts[0];
        }

        assert(size > 8);
        switch (cl.classes[1]) {
            case Integer:
                parts.push_back(getFittingIntegerType(size - 8));
                break;

            case Sse:
                parts.push_back(size - 8 <= 4 ? LLType::getFloatTy(gIR->context()) : LLType::getDoubleTy(gIR->context()));
                break;

            default:
                llvm_unreachable("Unanticipated argument class for second half.");
        }

        return LLStructType::get(gIR->context(), parts);
    }

    bool passByVal(Type* ty) {
        if (keepUnchanged(ty))
            return false;

        Classification cl = classify(ty);
        return cl.isMemory;
    }
  } // namespace ldc_abi
#endif

  namespace dmd_abi {
    // Structs, static arrays and cfloats may be rewritten to exploit registers.
    // This function returns the rewritten type, or null if no transformation is needed.
    LLType* getAbiType(Type* ty) {
        // First, check if there's any need of a transformation:
        if (!(ty->ty == Tcomplex32 || ty->ty == Tstruct || ty->ty == Tsarray))
            return 0; // Nothing to do

        // Okay, we may need to transform. Figure out a canonical type:

        TypeTuple* argTypes = toArgTypes(ty);
        if (!argTypes || argTypes->arguments->empty())
            return 0;  // don't rewrite

        LLType* abiTy = 0;
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

    // Temporary implementation validating the new toArgTypes()-based version against the LDC implementation.
    LLType* getAbiType(Type* ty) {
        ty = ty->toBasetype();

        LLType* dmdType = dmd_abi::getAbiType(ty);

#ifdef VALIDATE_AGAINST_OLD_LDC_VERSION
        LLType* ldcType = ldc_abi::getAbiType(ty);

        IF_LOG if (dmdType != ldcType) {
            Logger::print("getAbiType(%s) mismatch: ", ty->toChars());
            if (!dmdType)
                Logger::print("<null>");
            else
                Logger::cout() << *dmdType;
            Logger::print(" (DMD) vs. ");
            if (!ldcType)
                Logger::print("<null>");
            else
                Logger::cout() << *ldcType;
            Logger::println(" (LDC)");
        }

        //assert(dmdType == ldcType && "getAbiType() mismatch between DMD and LDC!");
#endif

        return dmdType;
    }

    bool typesAreEquivalent(LLType* a, LLType* b)
    {
        if (a == b)
            return true;

        LLStructType* structA;
        while ((structA = isaStruct(a)) && structA->getNumElements() == 1)
            a = structA->getElementType(0);

        LLStructType* structB;
        while ((structB = isaStruct(b)) && structB->getNumElements() == 1)
            b = structB->getElementType(0);

        return a == b
            || (structA && structB && structA->isLayoutIdentical(structB));
    }

    struct RegCount {
        char int_regs, sse_regs;

        RegCount() : int_regs(6), sse_regs(8) {}

        explicit RegCount(LLType* ty) : int_regs(0), sse_regs(0) {
            std::vector<LLType*> types;

            if (ty->isStructTy()) {
                unsigned numElements = ty->getStructNumElements();
                assert(numElements == 1 || numElements == 2);
                for (unsigned i = 0; i < numElements; ++i)
                    types.push_back(ty->getStructElementType(i));
            } else {
                types.push_back(ty);
            }

            for (unsigned i = 0; i < types.size(); ++i) {
                ty = types[i];
                if (ty->isIntegerTy() || ty->isPointerTy()) {
                    ++int_regs;
                } else if (ty->isFloatingPointTy() || ty->isVectorTy()) {
                    // X87 reals are passed on the stack
                    if (!ty->isX86_FP80Ty())
                        ++sse_regs;
                } else {
                    unsigned sizeInBits = ty->getPrimitiveSizeInBits();
                    IF_LOG Logger::cout() << "SysV RegCount: assuming 1 GP register for type " << *ty
                        << " (" << sizeInBits << " bits)\n";
                    assert(sizeInBits <= 64);
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

            TY ty = arg.type->toBasetype()->ty;
            // TODO: check what is really allowed to be passed partially
            const bool allowPartialPassing = (/* ty == Tarray || ty == Taarray || */ ty == Tdelegate);
            if (!allowPartialPassing && (int_regs < wanted.int_regs || sse_regs < wanted.sse_regs))
                return ArgumentWouldFitInPartially;

            int_regs = std::max(0, int_regs - wanted.int_regs);
            sse_regs = std::max(0, sse_regs - wanted.sse_regs);

            return ArgumentFitsIn;
        }
    };
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
    void getL(Type* dty, DValue* v, LLValue* lval) {
        LLValue* rval = v->getRVal();
        LLType* pTy = getPtrToType(rval->getType());
        DtoStore(rval, DtoBitCast(lval, pTy));
    }

    // Turn a struct into an ABI-mangled representation
    LLValue* put(Type* dty, DValue* v) {
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

    // Return the transformed type for this rewrite
    LLType* type(Type* dty, LLType* t) {
        return getAbiType(dty);
    }
};

/**
 * This type is used to force LLVM to pass a struct in memory.
 * This is achieved by passing the struct's address and using
 * the ByVal LLVM attribute.
 * We need this to prevent LLVM from passing a struct partially
 * in registers, partially in memory.
 */
struct ExplicitByvalRewrite : ABIRewrite {
    LLValue* get(Type* dty, DValue* v) {
        LLValue* ptr = v->getRVal();
        return DtoLoad(ptr);
    }

    void getL(Type* dty, DValue* v, LLValue* lval) {
        LLValue* ptr = v->getRVal();
        DtoAggrCopy(lval, ptr);
    }

    LLValue* put(Type* dty, DValue* v) {
        if (v->isLVal())
            return v->getLVal();

        LLValue* rval = v->getRVal();
        LLValue* address = DtoRawAlloca(rval->getType(), 0, ".explicit_byval_rewrite");
        DtoStore(rval, address);
        return address;
    }

    LLType* type(Type* dty, LLType* t) {
        return getPtrToType(DtoType(dty));
    }
};

struct X86_64TargetABI : TargetABI {
    X86_64_C_struct_rewrite struct_rewrite;
    ExplicitByvalRewrite explicitByvalRewrite;

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
    t = t->toBasetype();

    bool dmdResult = dmd_abi::passByVal(t);

#ifdef VALIDATE_AGAINST_OLD_LDC_VERSION
    bool ldcResult = ldc_abi::passByVal(t);

    IF_LOG if (dmdResult != ldcResult) {
        Logger::println("passByVal(%s) mismatch: %s (DMD) vs. %s (LDC)", t->toChars(),
            dmdResult ? "true" : "false", ldcResult ? "true" : "false");
    }

    //assert(dmdResult == ldcResult && "passByVal() mismatch between DMD and LDC!");
#endif

    return dmdResult;
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
        // pass structs explicitly byval, otherwise LLVM passes
        // them partially in registers, partially on the stack
        assert(originalLType->isStructTy());
        IF_LOG Logger::cout() << "Passing explicitly ByVal: " << arg.type->toChars() << " (" << *originalLType << ")\n";
        arg.rewrite = &explicitByvalRewrite;
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
