/* TargetABI implementation for x86-64.
 * Written for LDC by Frits van Bommel in 2009.
 * 
 * extern(D) follows no particular external ABI, but tries to be smart about
 * passing structs and returning them. It should probably be reviewed if the
 * way LLVM implements fastcc on this platform ever changes.
 * (Specifically, the number of return registers of various types is hardcoded)
 * 
 * 
 * extern(C) implements the C calling convention for x86-64, as found in
 * http://www.x86-64.org/documentation/abi-0.99.pdf
 * 
 * Note:
 *   Where a discrepancy was found between llvm-gcc and the ABI documentation,
 *   llvm-gcc behavior was used for compatibility (after it was verified that
 *   regular gcc has the same behavior).
 * 
 * LLVM gets it right for most types, but complex numbers and structs need some
 * help. To make sure it gets those right we essentially bitcast small structs
 * to a type to which LLVM assigns the appropriate registers, and pass that
 * instead. Structs that are required to be passed in memory are explicitly
 * marked with the ByVal attribute to ensure no part of them ends up in
 * registers when only a subset of the desired registers are available.
 * 
 * We don't perform the same transformation for D-specific types that contain
 * multiple parts, such as dynamic arrays and delegates. They're passed as if
 * the parts were passed as separate parameters. This helps make things like
 * printf("%.*s", o.toString()) work as expected; if we didn't do this that
 * wouldn't work if there were 4 other integer/pointer arguments before the
 * toString() call because the string got bumped to memory with one integer
 * register still free. Keeping it untransformed puts the length in a register
 * and the pointer in memory, as printf expects it.
 */

#include "mtype.h"
#include "declaration.h"
#include "aggregate.h"

#include "gen/llvm.h"
#include "gen/tollvm.h"
#include "gen/logger.h"
#include "gen/dvalue.h"
#include "gen/llvmhelpers.h"
#include "gen/abi.h"
#include "gen/abi-x86-64.h"
#include "gen/abi-generic.h"
#include "ir/irfunction.h"

#include <cassert>
#include <map>
#include <string>
#include <utility>

// Implementation details for extern(C)
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
        if (Logger::enabled())
            Logger::cout() << "Classifying " << ty->toChars() << " @ " << offset << '\n';
        
        ty = ty->toBasetype();
        
        if (ty->isintegral() || ty->ty == Tpointer) {
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
            Array* fields = &((TypeStruct*) ty)->sym->fields;
            for (size_t i = 0; i < fields->dim; i++) {
                VarDeclaration* field = (VarDeclaration*) fields->data[i];
                classifyType(accum, field->type, offset + field->offset);
            }
        } else {
            if (Logger::enabled())
                Logger::cout() << "x86-64 ABI: Implicitly handled type: "
                               << ty->toChars() << '\n';
            // arrays, delegates, etc. (pointer-sized fields, <= 16 bytes)
            assert(offset == 0 || offset == 8 
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
        
        std::vector<const LLType*> parts;
        
        unsigned size = ty->size();
        
        switch (cl.classes[0]) {
            case Integer: {
                unsigned bits = (size >= 8 ? 64 : (size * 8));
                parts.push_back(LLIntegerType::get(bits));
                break;
            }
            
            case Sse:
                parts.push_back(size <= 4 ? LLType::FloatTy : LLType::DoubleTy);
                break;
            
            case X87:
                assert(cl.classes[1] == X87Up && "Upper half of real not X87Up?");
                /// The type only contains a single real/ireal field,
                /// so just use that type.
                return const_cast<LLType*>(LLType::X86_FP80Ty);
            
            default:
                assert(0 && "Unanticipated argument class");
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
                parts.push_back(LLIntegerType::get(bits));
                break;
            }
            case Sse:
                parts.push_back(size <= 12 ? LLType::FloatTy : LLType::DoubleTy);
                break;
            
            case X87Up:
                if(cl.classes[0] == X87) {
                    // This won't happen: it was short-circuited while
                    // processing the first half.
                } else {                    
                    // I can't find this anywhere in the ABI documentation,
                    // but this is what gcc does (both regular and llvm-gcc).
                    // (This triggers for types like union { real r; byte b; })
                    parts.push_back(LLType::DoubleTy);
                }
                break;
            
            default:
                assert(0 && "Unanticipated argument class for second half");
        }
        return LLStructType::get(parts);
    }
}


// Implementation details for extern(D)
namespace x86_64_D_cc {
    struct DRegCount {
        unsigned ints;
        unsigned sse;
        unsigned x87;
        
        DRegCount(unsigned ints_, unsigned sse_, unsigned x87_)
        : ints(ints_), sse(sse_), x87(x87_) {}
    };
    
    // Count the number of registers needed for a simple type.
    // (Not a struct or static array)
    DRegCount regsNeededForSimpleType(Type* t) {
        DRegCount r(0, 0, 0);
        switch(t->ty) {
            case Tstruct:
            case Tsarray:
                assert(0 && "Not a simple type!");
                // Return huge numbers if assertions are disabled, so it'll always get
                // bumped to memory.
                r.ints = r.sse = r.x87 = (unsigned)-1;
                break;
            
            // Floats, doubles and such are passed in SSE registers
            case Tfloat32:
            case Tfloat64:
            case Timaginary32:
            case Timaginary64:
                r.sse = 1;
                break;
            
            case Tcomplex32:
            case Tcomplex64:
                r.sse = 2;
                break;
            
            // Reals, ireals and creals are passed in x87 registers
            case Tfloat80:
            case Timaginary80:
                r.x87 = 1;
                break;
            
            case Tcomplex80:
                r.x87 = 2;
                break;
            
            // Anything else is passed in one or two integer registers,
            // depending on its size.
            default: {
                int needed = (t->size() + 7) / 8;
                assert(needed <= 2);
                r.ints = needed;
                break;
            }
        }
        return r;
    }
    
    // Returns true if it's possible (and a good idea) to pass the struct in the
    // specified number of registers.
    // (May return false if it's a bad idea to pass the type in registers for
    // reasons other than it not fitting)
    // Note that if true is returned, 'left' is also modified to contain the
    // number of registers left. This property is used in the recursive case.
    // If false is returned, 'left' is garbage.
    bool shouldPassStructInRegs(TypeStruct* t, DRegCount& left) {
        // If it has unaligned fields, there's probably a reason for it,
        // so keep it in memory.
        if (hasUnalignedFields(t))
            return false;
        
        Array* fields = &t->sym->fields;
        d_uns64 nextbyte = 0;
        for (d_uns64 i = 0; i < fields->dim; i++) {
            VarDeclaration* field = (VarDeclaration*) fields->data[i];
            
            // This depends on ascending order of field offsets in structs
            // without overlapping fields.
            if (field->offset < nextbyte) {
                // Don't return unions (or structs containing them) in registers.
                return false;
            }
            nextbyte = field->offset + field->type->size();
            
            switch (field->type->ty) {
                case Tstruct:
                    if (!shouldPassStructInRegs((TypeStruct*) field->type, left))
                        return false;
                    break;
                
                case Tsarray:
                    // Don't return static arrays in registers
                    // (indexing registers doesn't work well)
                    return false;
                
                default: {
                    DRegCount needed = regsNeededForSimpleType(field->type);
                    if (needed.ints > left.ints || needed.sse > left.sse || needed.x87 > left.x87)
                        return false;
                    left.ints -= needed.ints;
                    left.sse -= needed.sse;
                    left.x87 -= needed.x87;
                    break;
                }
            }
        }
        return true;
    }
    
    // Returns true if the struct fits in return registers in the x86-64 fastcc
    // calling convention.
    bool retStructInRegs(TypeStruct* st) {
        // 'fastcc' allows returns in up to two registers of each kind:
        DRegCount state(2, 2, 2);
        return shouldPassStructInRegs(st, state);
    }
    
    // Heuristic for determining whether to pass a struct type directly or
    // bump it to memory.
    bool passStructTypeDirectly(TypeStruct* st) {
        // If the type fits in a reasonable number of registers,
        // pass it directly.
        // This does not necessarily mean it will actually be passed in
        // registers. For example, x87 registers are never actually used for
        // parameters.
        DRegCount state(2, 2, 2);
        return shouldPassStructInRegs(st, state);
        
        // This doesn't work well: Since the register count can differ depending
        // on backend options, there's no way to be exact anyway.
        /*
        // Regular fastcc:      6 int, 8 sse, 0 x87
        // fastcc + tailcall:   5 int, 8 sse, 0 x87
        RegCount state(5, 8, 0);
        */
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/// Just store to memory and it's readable as the other type.
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
        
        const LLType* pTy = getPtrToType(DtoType(dty));
        return DtoLoad(DtoBitCast(lval, pTy), "get-result");
    }
    
    // Get struct from ABI-mangled representation, and store in the provided location.
    void getL(Type* dty, DValue* v, llvm::Value* lval) {
        LLValue* rval = v->getRVal();
        const LLType* pTy = getPtrToType(rval->getType());
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
        
        const LLType* pTy = getPtrToType(abiTy);
        return DtoLoad(DtoBitCast(lval, pTy), "put-result");
    }
    
    /// should return the transformed type for this rewrite
    const LLType* type(Type* dty, const LLType* t)
    {
        return getAbiType(dty);
    }
};


struct RegCount {
    unsigned char int_regs, sse_regs;
};


struct X86_64TargetABI : TargetABI {
    X86_64_C_struct_rewrite struct_rewrite;
    RemoveStructPadding remove_padding;
    
    void newFunctionType(TypeFunction* tf) {
        funcTypeStack.push_back(FuncTypeData(tf->linkage));
    }
    
    bool returnInArg(TypeFunction* tf);
    
    bool passByVal(Type* t);
    
    void rewriteFunctionType(TypeFunction* tf);
    
    void doneWithFunctionType() {
        funcTypeStack.pop_back();
    }
    
private:
    struct FuncTypeData {
        LINK linkage;       // Linkage of the function type currently under construction
        RegCount state;     // bookkeeping for extern(C) parameter registers
        
        FuncTypeData(LINK linkage_)
        : linkage(linkage_)
        {
            state.int_regs = 6;
            state.sse_regs = 8;
        }
    };
    std::vector<FuncTypeData> funcTypeStack;
    
    LINK linkage() {
        assert(funcTypeStack.size() != 0);
        return funcTypeStack.back().linkage;
    }
    
    RegCount& state() {
        assert(funcTypeStack.size() != 0);
        return funcTypeStack.back().state;
    }
    
    void fixup_D(IrFuncTyArg& arg);
    void fixup(IrFuncTyArg& arg);
};


// The public getter for abi.cpp
TargetABI* getX86_64TargetABI() {
    return new X86_64TargetABI;
}


bool X86_64TargetABI::returnInArg(TypeFunction* tf) {
    assert(linkage() == tf->linkage);
    Type* rt = tf->next->toBasetype();
    
    if (tf->linkage == LINKd) {
        assert(rt->ty != Tsarray && "Update calling convention for static array returns");
        
        // All non-structs can be returned in registers.
        if (rt->ty != Tstruct)
            return false;
        
        // Try to figure out whether the struct fits in return registers
        // and whether it's a good idea to put it there.
        return !x86_64_D_cc::retStructInRegs((TypeStruct*) rt);
    } else {
        if (rt == Type::tvoid || keepUnchanged(rt))
            return false;
        
        Classification cl = classify(rt);
        return cl.isMemory;
    }
}

bool X86_64TargetABI::passByVal(Type* t) {
    t = t->toBasetype();
    if (linkage() == LINKd) {
        if (t->ty != Tstruct)
            return false;
        
        // Try to be smart about which structs are passed in memory.
        return !x86_64_D_cc::passStructTypeDirectly((TypeStruct*) t);
    } else {
        // This implements the C calling convention for x86-64.
        // It might not be correct for other calling conventions.
        Classification cl = classify(t);
        if (cl.isMemory)
            return true;
        
        // Figure out how many registers we want for this arg:
        RegCount wanted = { 0, 0 };
        for (int i = 0 ; i < 2; i++) {
            if (cl.classes[i] == Integer)
                wanted.int_regs++;
            else if (cl.classes[i] == Sse)
                wanted.sse_regs++;
        }
        
        // See if they're available:
        RegCount& state = this->state();
        if (wanted.int_regs <= state.int_regs && wanted.sse_regs <= state.sse_regs) {
            state.int_regs -= wanted.int_regs;
            state.sse_regs -= wanted.sse_regs;
        } else {
            if (keepUnchanged(t)) {
                // Not enough registers available, but this is passed as if it's
                // multiple arguments. Just use the registers there are,
                // automatically spilling the rest to memory.
                if (wanted.int_regs > state.int_regs)
                    state.int_regs = 0;
                else
                    state.int_regs -= wanted.int_regs;
                
                if (wanted.sse_regs > state.sse_regs)
                    state.sse_regs = 0;
                else
                    state.sse_regs -= wanted.sse_regs;
            } else if (t->iscomplex() || t->ty == Tstruct) {
                // Spill entirely to memory, even if some of the registers are
                // available.
                
                // FIXME: Don't do this if *none* of the wanted registers are available,
                //        (i.e. only when absolutely necessary for abi-compliance)
                //        so it gets alloca'd by the callee and -scalarrepl can
                //        more easily break it up?
                // Note: this won't be necessary if the following LLVM bug gets fixed:
                //       http://llvm.org/bugs/show_bug.cgi?id=3741
                return true;
            } else {
                assert(t == Type::tfloat80 || t == Type::timaginary80 || t->size() <= 8
                    && "What other big types are there?"); // other than static arrays...
                // In any case, they shouldn't be represented as structs in LLVM:
                assert(!isaStruct(DtoType(t)));
            }
        }
        // Everything else that's passed in memory is handled by LLVM.
        return false;
    }
}

// Helper function for rewriteFunctionType.
// Structs passed or returned in registers are passed here
// to get their padding removed (if necessary).
void X86_64TargetABI::fixup_D(IrFuncTyArg& arg) {
    assert(arg.type->ty == Tstruct);
    LLType* abiTy = DtoUnpaddedStructType(arg.type);
    
    if (abiTy && abiTy != arg.ltype) {
        arg.ltype = abiTy;
        arg.rewrite = &remove_padding;
    }
}

// Helper function for rewriteFunctionType.
// Return type and parameters are passed here (unless they're already in memory)
// to get the rewrite applied (if necessary).
void X86_64TargetABI::fixup(IrFuncTyArg& arg) {
    LLType* abiTy = getAbiType(arg.type);
    
    if (abiTy && abiTy != arg.ltype) {
        assert(arg.type == Type::tcomplex32 || arg.type->ty == Tstruct);
        arg.ltype = abiTy;
        arg.rewrite = &struct_rewrite;
    }
}

void X86_64TargetABI::rewriteFunctionType(TypeFunction* tf) {
    IrFuncTy& fty = tf->fty;
    
    if (tf->linkage == LINKd) {
        if (!fty.arg_sret) {
            Type* rt = fty.ret->type->toBasetype();
            if (rt->ty == Tstruct)  {
                Logger::println("x86-64 D ABI: Transforming return type");
                fixup_D(*fty.ret);
            }
        }
        
        Logger::println("x86-64 D ABI: Transforming arguments");
        LOG_SCOPE;
        
        for (IrFuncTy::ArgIter I = fty.args.begin(), E = fty.args.end(); I != E; ++I) {
            IrFuncTyArg& arg = **I;
            
            if (Logger::enabled())
                Logger::cout() << "Arg: " << arg.type->toChars() << '\n';
            
            // Arguments that are in memory are of no interest to us.
            if (arg.byref)
                continue;
            
            Type* ty = arg.type->toBasetype();
            if (ty->ty == Tstruct)
                fixup_D(arg);
            
            if (Logger::enabled())
                Logger::cout() << "New arg type: " << *arg.ltype << '\n';
        }
        
    } else {
        // TODO: See if this is correct for more than just extern(C).
        
        if (!fty.arg_sret) {
            Logger::println("x86-64 ABI: Transforming return type");
            Type* rt = fty.ret->type->toBasetype();
            if (rt != Type::tvoid)
                fixup(*fty.ret);
        }
        
        
        Logger::println("x86-64 ABI: Transforming arguments");
        LOG_SCOPE;
        
        for (IrFuncTy::ArgIter I = fty.args.begin(), E = fty.args.end(); I != E; ++I) {
            IrFuncTyArg& arg = **I;
            
            if (Logger::enabled())
                Logger::cout() << "Arg: " << arg.type->toChars() << '\n';
            
            // Arguments that are in memory are of no interest to us.
            if (arg.byref)
                continue;
            
            Type* ty = arg.type->toBasetype();
            
            fixup(arg);
            
            if (Logger::enabled())
                Logger::cout() << "New arg type: " << *arg.ltype << '\n';
        }
    }
}
