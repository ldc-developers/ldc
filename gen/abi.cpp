#include "gen/llvm.h"

#include <algorithm>

#include "mars.h"

#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"
#include "gen/abi.h"
#include "gen/logger.h"

#include "ir/irfunction.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/////////////////////              X86            ////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// simply swap of real/imag parts for proper x87 complex abi
struct X87_complex_swap : ABIRewrite
{
    LLValue* get(Type*, LLValue* v)
    {
        return DtoAggrPairSwap(v);
    }
    LLValue* put(Type*, LLValue* v)
    {
        return DtoAggrPairSwap(v);
    }
    const LLType* type(Type*, const LLType* t)
    {
        return t;
    }
};

//////////////////////////////////////////////////////////////////////////////

struct X86_cfloat_rewrite : ABIRewrite
{
    // i64 -> {float,float}
    LLValue* get(Type*, LLValue* in)
    {
        // extract real part
        LLValue* rpart = gIR->ir->CreateTrunc(in, LLType::Int32Ty);
        rpart = gIR->ir->CreateBitCast(rpart, LLType::FloatTy, ".re");

        // extract imag part
        LLValue* ipart = gIR->ir->CreateLShr(in, LLConstantInt::get(LLType::Int64Ty, 32, false));
        ipart = gIR->ir->CreateTrunc(ipart, LLType::Int32Ty);
        ipart = gIR->ir->CreateBitCast(ipart, LLType::FloatTy, ".im");

        // return {float,float} aggr pair with same bits
        return DtoAggrPair(rpart, ipart, ".final_cfloat");
    }

    // {float,float} -> i64
    LLValue* put(Type*, LLValue* v)
    {
        // extract real
        LLValue* r = gIR->ir->CreateExtractValue(v, 0);
        // cast to i32
        r = gIR->ir->CreateBitCast(r, LLType::Int32Ty);
        // zext to i64
        r = gIR->ir->CreateZExt(r, LLType::Int64Ty);

        // extract imag
        LLValue* i = gIR->ir->CreateExtractValue(v, 1);
        // cast to i32
        i = gIR->ir->CreateBitCast(i, LLType::Int32Ty);
        // zext to i64
        i = gIR->ir->CreateZExt(i, LLType::Int64Ty);
        // shift up
        i = gIR->ir->CreateShl(i, LLConstantInt::get(LLType::Int64Ty, 32, false));

        // combine and return
        return v = gIR->ir->CreateOr(r, i);
    }

    // {float,float} -> i64
    const LLType* type(Type*, const LLType* t)
    {
        return LLType::Int64Ty;
    }
};

//////////////////////////////////////////////////////////////////////////////

// FIXME: try into eliminating the alloca or if at least check
// if it gets optimized away

// convert byval struct
// when 
struct X86_struct_to_register : ABIRewrite
{
    // int -> struct
    LLValue* get(Type* dty, LLValue* v)
    {
        Logger::println("rewriting int -> struct");
        LLValue* mem = DtoAlloca(DtoType(dty), ".int_to_struct");
        DtoStore(v, DtoBitCast(mem, getPtrToType(v->getType())));
        return DtoLoad(mem);
    }
    // struct -> int
    LLValue* put(Type* dty, LLValue* v)
    {
        Logger::println("rewriting struct -> int");
        LLValue* mem = DtoAlloca(v->getType(), ".struct_to_int");
        DtoStore(v, mem);
        DtoLoad(DtoBitCast(mem, getPtrToType(type(dty, v->getType()))));
    }
    const LLType* type(Type*, const LLType* t)
    {
        size_t sz = getTypePaddedSize(t)*8;
        return LLIntegerType::get(sz);
    }
};

//////////////////////////////////////////////////////////////////////////////

struct X86TargetABI : TargetABI
{
    X87_complex_swap swapComplex;
    X86_cfloat_rewrite cfloatToInt;
    X86_struct_to_register structToReg;

    bool returnInArg(TypeFunction* tf)
    {
        Type* rt = tf->next->toBasetype();
        // D only returns structs on the stack
        if (tf->linkage == LINKd)
            return (rt->ty == Tstruct);
        // other ABI's follow C, which is cdouble and creal returned on the stack
        // as well as structs
        else
            return (rt->ty == Tstruct || rt->ty == Tcomplex64 || rt->ty == Tcomplex80);
    }

    bool passByVal(Type* t)
    {
        return t->toBasetype()->ty == Tstruct;
    }

    void rewriteFunctionType(TypeFunction* tf)
    {
        IrFuncTy* fty = tf->fty;
        Type* rt = fty->ret->type->toBasetype();

        // extern(D)
        if (tf->linkage == LINKd)
        {
            // RETURN VALUE

            // complex {re,im} -> {im,re}
            if (rt->iscomplex())
            {
                fty->ret->rewrite = &swapComplex;
            }

            // IMPLICIT PARAMETERS

            // mark this/nested params inreg
            if (fty->arg_this)
            {
                fty->arg_this->attrs = llvm::Attribute::InReg;
            }
            else if (fty->arg_nest)
            {
                fty->arg_nest->attrs = llvm::Attribute::InReg;
            }
            // otherwise try to mark the last param inreg
            else if (!fty->arg_sret && !fty->args.empty())
            {
                // The last parameter is passed in EAX rather than being pushed on the stack if the following conditions are met:
                //   * It fits in EAX.
                //   * It is not a 3 byte struct.
                //   * It is not a floating point type.

                IrFuncTyArg* last = fty->args.back();
                Type* lastTy = last->type->toBasetype();
                unsigned sz = lastTy->size();

                if (last->byref && !last->isByVal())
                {
                    last->attrs |= llvm::Attribute::InReg;
                }
                else if (!lastTy->isfloating() && (sz == 1 || sz == 2 || sz == 4)) // right?
                {
                    // rewrite the struct into an integer to make inreg work
                    if (lastTy->ty == Tstruct)
                    {
                        last->rewrite = &structToReg;
                        last->ltype = structToReg.type(last->type, last->ltype);
                        last->byref = false;
                        // erase previous attributes
                        last->attrs = 0;
                    }
                    last->attrs |= llvm::Attribute::InReg;
                }
            }

            // FIXME: tf->varargs == 1 need to use C calling convention and vararg mechanism to live up to the spec:
            // "The caller is expected to clean the stack. _argptr is not passed, it is computed by the callee."

            // EXPLICIT PARAMETERS

            // reverse parameter order
            // for non variadics
            if (!fty->args.empty() && tf->varargs != 1)
            {
                fty->reverseParams = true;
            }
        }

        // extern(C) and all others
        else
        {
            // RETURN VALUE

            // cfloat -> i64
            if (tf->next->toBasetype() == Type::tcomplex32)
            {
                fty->ret->rewrite = &cfloatToInt;
                fty->ret->ltype = LLType::Int64Ty;
            }

            // IMPLICIT PARAMETERS

            // EXPLICIT PARAMETERS
        }
    }
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
///////////////////            X86-64               //////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct X86_64_cfloat_rewrite : ABIRewrite
{
    // {double} -> {float,float}
    LLValue* get(Type*, LLValue* in)
    {
        // extract double
        LLValue* v = gIR->ir->CreateExtractValue(in, 0);
        // cast to i64
        v = gIR->ir->CreateBitCast(v, LLType::Int64Ty);

        // extract real part
        LLValue* rpart = gIR->ir->CreateTrunc(v, LLType::Int32Ty);
        rpart = gIR->ir->CreateBitCast(rpart, LLType::FloatTy, ".re");

        // extract imag part
        LLValue* ipart = gIR->ir->CreateLShr(v, LLConstantInt::get(LLType::Int64Ty, 32, false));
        ipart = gIR->ir->CreateTrunc(ipart, LLType::Int32Ty);
        ipart = gIR->ir->CreateBitCast(ipart, LLType::FloatTy, ".im");

        // return {float,float} aggr pair with same bits
        return DtoAggrPair(rpart, ipart, ".final_cfloat");
    }

    // {float,float} -> {double}
    LLValue* put(Type*, LLValue* v)
    {
        // extract real
        LLValue* r = gIR->ir->CreateExtractValue(v, 0);
        // cast to i32
        r = gIR->ir->CreateBitCast(r, LLType::Int32Ty);
        // zext to i64
        r = gIR->ir->CreateZExt(r, LLType::Int64Ty);

        // extract imag
        LLValue* i = gIR->ir->CreateExtractValue(v, 1);
        // cast to i32
        i = gIR->ir->CreateBitCast(i, LLType::Int32Ty);
        // zext to i64
        i = gIR->ir->CreateZExt(i, LLType::Int64Ty);
        // shift up
        i = gIR->ir->CreateShl(i, LLConstantInt::get(LLType::Int64Ty, 32, false));

        // combine
        v = gIR->ir->CreateOr(r, i);

        // cast to double
        v = gIR->ir->CreateBitCast(v, LLType::DoubleTy);

        // return {double}
        const LLType* t = LLStructType::get(LLType::DoubleTy, NULL);
        LLValue* undef = llvm::UndefValue::get(t);
        return gIR->ir->CreateInsertValue(undef, v, 0);
    }

    // {float,float} -> {double}
    const LLType* type(Type*, const LLType* t)
    {
        return LLStructType::get(LLType::DoubleTy, NULL);
    }
};

//////////////////////////////////////////////////////////////////////////////

struct X86_64TargetABI : TargetABI
{
    X86_64_cfloat_rewrite cfloat_rewrite;

    bool returnInArg(TypeFunction* tf)
    {
        Type* rt = tf->next->toBasetype();
        return (rt->ty == Tstruct);
    }

    bool passByVal(Type* t)
    {
        return t->toBasetype()->ty == Tstruct;
    }

    void rewriteFunctionType(TypeFunction* tf)
    {
        IrFuncTy* fty = tf->fty;
        Type* rt = fty->ret->type->toBasetype();

        // rewrite cfloat return for !extern(D)
        if (tf->linkage != LINKd && rt == Type::tcomplex32)
        {
            fty->ret->rewrite = &cfloat_rewrite;
            fty->ret->ltype = cfloat_rewrite.type(fty->ret->type, fty->ret->ltype);
        }
    }
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
///////////////////         Unknown targets         //////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Some reasonable defaults for when we don't know what ABI to use.
struct UnknownTargetABI : TargetABI
{
    bool returnInArg(TypeFunction* tf)
    {
        return (tf->next->toBasetype()->ty == Tstruct);
    }

    bool passByVal(Type* t)
    {
        return t->toBasetype()->ty == Tstruct;
    }

    void rewriteFunctionType(TypeFunction* t)
    {
        // why?
    }
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

TargetABI * TargetABI::getTarget()
{
    switch(global.params.cpu)
    {
    case ARCHx86:
        return new X86TargetABI;
    case ARCHx86_64:
        return new X86_64TargetABI;
    default:
        Logger::cout() << "WARNING: Unknown ABI, guessing...\n";
        return new UnknownTargetABI;
    }
}
