//===-- gen/abi-generic.h - Generic Target ABI helpers ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Contains helpers for handling rewrites common to more than one target ABI.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_ABI_GENERIC_H
#define LDC_GEN_ABI_GENERIC_H

#include "gen/abi.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/structs.h"
#include "gen/tollvm.h"

/// Removes padding fields for (non-union-containing!) structs
struct RemoveStructPadding : ABIRewrite {
    /// get a rewritten value back to its original form
    virtual LLValue* get(Type* dty, DValue* v) {
        LLValue* lval = DtoAlloca(dty, ".rewritetmp");
        getL(dty, v, lval);
        return lval;
    }

    /// get a rewritten value back to its original form and store result in provided lvalue
    /// this one is optional and defaults to calling the one above
    virtual void getL(Type* dty, DValue* v, llvm::Value* lval) {
        // Make sure the padding is zero, so struct comparisons work.
        // TODO: Only do this if there's padding, and/or only initialize padding.
        DtoMemSetZero(lval, DtoConstSize_t(getTypePaddedSize(DtoType(dty))));
        DtoPaddedStruct(dty->toBasetype(), v->getRVal(), lval);
    }

    /// put out rewritten value
    virtual LLValue* put(Type* dty, DValue* v) {
        return DtoUnpaddedStruct(dty->toBasetype(), v->getRVal());
    }

    /// return the transformed type for this rewrite
    virtual LLType* type(Type* dty, LLType* t) {
        return DtoUnpaddedStructType(dty->toBasetype());
    }
};

//////////////////////////////////////////////////////////////////////////////

// simply swap of real/imag parts for proper x87 complex abi
struct X87_complex_swap : ABIRewrite
{
    LLValue* get(Type*, DValue* v)
    {
        return DtoAggrPairSwap(v->getRVal());
    }
    LLValue* put(Type*, DValue* v)
    {
        return DtoAggrPairSwap(v->getRVal());
    }
    LLType* type(Type*, LLType* t)
    {
        return t;
    }
};

//////////////////////////////////////////////////////////////////////////////

// Rewrites a cfloat (2x32 bits) as 64-bit integer.
// Assumes a little-endian byte order.
struct CfloatToInt : ABIRewrite
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

//////////////////////////////////////////////////////////////////////////////

/**
 * Rewrites a composite type parameter to an integer of the same size.
 *
 * This is needed in order to be able to use LLVM's inreg attribute to put
 * struct and static array parameters into registers, because the attribute has
 * slightly different semantics. For example, LLVM would store a [4 x i8] inreg
 * in four registers (zero-extended), instead of a single 32bit one.
 *
 * The LLVM value in dv is expected to be a pointer to the parameter, as
 * generated when lowering struct/static array paramters to LLVM byval.
 */
struct CompositeToInt : ABIRewrite
{
    LLValue* get(Type* dty, DValue* dv)
    {
        Logger::println("rewriting integer -> %s", dty->toChars());
        LLValue* mem = DtoAlloca(dty, ".int_to_composite");
        LLValue* v = dv->getRVal();
        DtoStore(v, DtoBitCast(mem, getPtrToType(v->getType())));
        return DtoLoad(mem);
    }

    void getL(Type* dty, DValue* dv, llvm::Value* lval)
    {
        Logger::println("rewriting integer -> %s", dty->toChars());
        LLValue* v = dv->getRVal();
        DtoStore(v, DtoBitCast(lval, getPtrToType(v->getType())));
    }

    LLValue* put(Type* dty, DValue* dv)
    {
        Logger::println("rewriting %s -> integer", dty->toChars());
        LLType* t = LLIntegerType::get(gIR->context(), dty->size() * 8);
        return DtoLoad(DtoBitCast(dv->getRVal(), getPtrToType(t)));
    }

    LLType* type(Type* t, LLType*)
    {
        size_t sz = t->size() * 8;
        return LLIntegerType::get(gIR->context(), sz);
    }
};

#endif
