#include "gen/llvm.h"

#include "mars.h"

#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"
#include "gen/abi.h"
#include "gen/logger.h"
#include "gen/dvalue.h"
#include "gen/abi-generic.h"
#include "ir/irfunction.h"
#include "ir/irfuncty.h"

struct X86_cfloat_rewrite : ABIRewrite
{
    // i64 -> {float,float}
    LLValue* get(Type*, DValue* dv)
    {
        LLValue* in = dv->getRVal();

        // extract real part
        LLValue* rpart = gIR->ir->CreateTrunc(in, LLType::getInt32Ty(gIR->context()));
        rpart = gIR->ir->CreateBitCast(rpart, LLType::getFloatTy(gIR->context()), ".re");

        // extract imag part
        LLValue* ipart = gIR->ir->CreateLShr(in, LLConstantInt::get(LLType::getInt64Ty(gIR->context()), 32, false));
        ipart = gIR->ir->CreateTrunc(ipart, LLType::getInt32Ty(gIR->context()));
        ipart = gIR->ir->CreateBitCast(ipart, LLType::getFloatTy(gIR->context()), ".im");

        // return {float,float} aggr pair with same bits
        return DtoAggrPair(rpart, ipart, ".final_cfloat");
    }

    // {float,float} -> i64
    LLValue* put(Type*, DValue* dv)
    {
        LLValue* v = dv->getRVal();

        // extract real
        LLValue* r = gIR->ir->CreateExtractValue(v, 0);
        // cast to i32
        r = gIR->ir->CreateBitCast(r, LLType::getInt32Ty(gIR->context()));
        // zext to i64
        r = gIR->ir->CreateZExt(r, LLType::getInt64Ty(gIR->context()));

        // extract imag
        LLValue* i = gIR->ir->CreateExtractValue(v, 1);
        // cast to i32
        i = gIR->ir->CreateBitCast(i, LLType::getInt32Ty(gIR->context()));
        // zext to i64
        i = gIR->ir->CreateZExt(i, LLType::getInt64Ty(gIR->context()));
        // shift up
        i = gIR->ir->CreateShl(i, LLConstantInt::get(LLType::getInt64Ty(gIR->context()), 32, false));

        // combine and return
        return v = gIR->ir->CreateOr(r, i);
    }

    // {float,float} -> i64
    LLType* type(Type*, LLType* t)
    {
        return LLType::getInt64Ty(gIR->context());
    }
};


struct X86TargetABI : TargetABI
{
    X87_complex_swap swapComplex;
    X86_cfloat_rewrite cfloatToInt;
    CompositeToInt compositeToInt;

    bool returnInArg(TypeFunction* tf)
    {
#if DMDV2
        if (tf->isref)
            return false;
#endif
        Type* rt = tf->next->toBasetype();
        // D only returns structs and static arrays on the stack
        if (tf->linkage == LINKd)
            return (rt->ty == Tstruct || rt->ty == Tsarray);
        // other ABI's follow C, which is cdouble and creal returned on the stack
        // as well as structs
        else
            return (rt->ty == Tstruct || rt->ty == Tcomplex64 || rt->ty == Tcomplex80);
    }

    bool passByVal(Type* t)
    {
        return t->toBasetype()->ty == Tstruct || t->toBasetype()->ty == Tsarray;
    }

    void rewriteFunctionType(TypeFunction* tf)
    {
        IrFuncTy& fty = tf->fty;
        Type* rt = fty.ret->type->toBasetype();

        // extern(D)
        if (tf->linkage == LINKd)
        {
            // RETURN VALUE

            // complex {re,im} -> {im,re}
            if (rt->iscomplex())
            {
                Logger::println("Rewriting complex return value");
                fty.ret->rewrite = &swapComplex;
            }

            // IMPLICIT PARAMETERS

            // mark this/nested params inreg
            if (fty.arg_this)
            {
                Logger::println("Putting 'this' in register");
                fty.arg_this->attrs = llvm::Attribute::InReg;
            }
            else if (fty.arg_nest)
            {
                Logger::println("Putting context ptr in register");
                fty.arg_nest->attrs = llvm::Attribute::InReg;
            }
            else if (IrFuncTyArg* sret = fty.arg_sret)
            {
                Logger::println("Putting sret ptr in register");
                // sret and inreg are incompatible, but the ABI requires the
                // sret parameter to be in EAX in this situation...
                sret->attrs = (sret->attrs | llvm::Attribute::InReg)
                                & ~llvm::Attribute::StructRet;
            }
            // otherwise try to mark the last param inreg
            else if (!fty.args.empty())
            {
                // The last parameter is passed in EAX rather than being pushed on the stack if the following conditions are met:
                //   * It fits in EAX.
                //   * It is not a 3 byte struct.
                //   * It is not a floating point type.

                IrFuncTyArg* last = fty.args.back();
                Type* lastTy = last->type->toBasetype();
                unsigned sz = lastTy->size();

                if (last->byref && !last->isByVal())
                {
                    Logger::println("Putting last (byref) parameter in register");
                    last->attrs |= llvm::Attribute::InReg;
                }
                else if (!lastTy->isfloating() && (sz == 1 || sz == 2 || sz == 4)) // right?
                {
                    // rewrite the struct into an integer to make inreg work
                    if (lastTy->ty == Tstruct || lastTy->ty == Tsarray)
                    {
                        last->rewrite = &compositeToInt;
                        last->ltype = compositeToInt.type(last->type, last->ltype);
                        last->byref = false;
                        // erase previous attributes
                        last->attrs = llvm::Attribute::None;
                    }
                    last->attrs |= llvm::Attribute::InReg;
                }
            }

            // FIXME: tf->varargs == 1 need to use C calling convention and vararg mechanism to live up to the spec:
            // "The caller is expected to clean the stack. _argptr is not passed, it is computed by the callee."

            // EXPLICIT PARAMETERS

            // reverse parameter order
            // for non variadics
            if (!fty.args.empty() && tf->varargs != 1)
            {
                fty.reverseParams = true;
            }
        }

        // extern(C) and all others
        else
        {
            // RETURN VALUE

            // cfloat -> i64
            if (tf->next->toBasetype() == Type::tcomplex32)
            {
                fty.ret->rewrite = &cfloatToInt;
                fty.ret->ltype = LLType::getInt64Ty(gIR->context());
            }

            // IMPLICIT PARAMETERS

            // EXPLICIT PARAMETERS
        }
    }
};

// The public getter for abi.cpp.
TargetABI* getX86TargetABI() {
    return new X86TargetABI;
}
