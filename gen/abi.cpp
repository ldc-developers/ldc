#include "gen/llvm.h"

#include "mars.h"

#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"
#include "gen/abi.h"
#include "gen/logger.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/////////////////////        baseclass            ////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// FIXME: Would be nice to come up a better and faster way to do this, right
// now I'm more worried about actually making this abstraction work at all ...
// It's definitely way overkill with the amount of return value rewrites we
// have right now, but I expect this to change with proper x86-64 abi support

TargetABI::TargetABI()
{
}

llvm::Value* TargetABI::getRet(TypeFunction* tf, llvm::Value* io)
{
    if (ABIRetRewrite* r = findRetRewrite(tf))
    {
        return r->get(io);
    }
    return io;
}

llvm::Value* TargetABI::putRet(TypeFunction* tf, llvm::Value* io)
{
    if (ABIRetRewrite* r = findRetRewrite(tf))
    {
        return r->put(io);
    }
    return io;
}

const llvm::Type* TargetABI::getRetType(TypeFunction* tf, const llvm::Type* t)
{
    if (ABIRetRewrite* r = findRetRewrite(tf))
    {
        return r->type(t);
    }
    return t;
}

ABIRetRewrite * TargetABI::findRetRewrite(TypeFunction * tf)
{
    size_t n = retOps.size();
    if (n)
    for (size_t i = 0; i < n; i++)
    {
        if (retOps[i]->test(tf))
            return retOps[i];
    }
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/////////////////////              X86            ////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// simply swap of real/imag parts for proper x87 complex abi
struct X87_complex_swap : ABIRetRewrite
{
    LLValue* get(LLValue* v)
    {
        return DtoAggrPairSwap(v);
    }
    LLValue* put(LLValue* v)
    {
        return DtoAggrPairSwap(v);
    }
    const LLType* type(const LLType* t)
    {
        return t;
    }
    bool test(TypeFunction* tf)
    {
        return (tf->next->toBasetype()->iscomplex());
    }
};

//////////////////////////////////////////////////////////////////////////////

struct X86TargetABI : TargetABI
{
    X86TargetABI()
    {
        retOps.push_back(new X87_complex_swap);
    }

    bool returnInArg(Type* t)
    {
        Type* rt = t->toBasetype();
        return (rt->ty == Tstruct);
    }

    bool passByRef(Type* t)
    {
        t = t->toBasetype();
        return (t->ty == Tstruct || t->ty == Tsarray);
    }
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
///////////////////            X86-64               //////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct X86_64_cfloat_rewrite : ABIRetRewrite
{
    // {double} -> {float,float}
    LLValue* get(LLValue* in)
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
    LLValue* put(LLValue* v)
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
    const LLType* type(const LLType* t)
    {
        return LLStructType::get(LLType::DoubleTy, NULL);
    }

    // test if rewrite applies to function
    bool test(TypeFunction* tf)
    {
        return (tf->next->toBasetype() == Type::tcomplex32);
    }
};

//////////////////////////////////////////////////////////////////////////////

struct X86_64TargetABI : TargetABI
{
    X86_64TargetABI()
    {
        retOps.push_back(new X86_64_cfloat_rewrite);
    }

    bool returnInArg(Type* t)
    {
        Type* rt = t->toBasetype();
        return (rt->ty == Tstruct);
    }

    bool passByRef(Type* t)
    {
        t = t->toBasetype();
        return (t->ty == Tstruct || t->ty == Tsarray);
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
    UnknownTargetABI()
    {
        // Don't push anything into retOps, assume defaults will be fine.
    }

    bool returnInArg(Type* t)
    {
        Type* rt = t->toBasetype();
        return (rt->ty == Tstruct);
    }

    bool passByRef(Type* t)
    {
        t = t->toBasetype();
        return (t->ty == Tstruct || t->ty == Tsarray);
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
