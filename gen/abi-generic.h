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
    LLValue* get(Type* dty, DValue* v) {
        LLValue* lval = DtoAlloca(dty, ".rewritetmp");
        getL(dty, v, lval);
        return lval;
    }

    /// get a rewritten value back to its original form and store result in provided lvalue
    /// this one is optional and defaults to calling the one above
    void getL(Type* dty, DValue* v, LLValue* lval) {
        // Make sure the padding is zero, so struct comparisons work.
        // TODO: Only do this if there's padding, and/or only initialize padding.
        DtoMemSetZero(lval, DtoConstSize_t(getTypePaddedSize(DtoType(dty))));
        DtoPaddedStruct(dty->toBasetype(), v->getRVal(), lval);
    }

    /// put out rewritten value
    LLValue* put(Type* dty, DValue* v) {
        return DtoUnpaddedStruct(dty->toBasetype(), v->getRVal());
    }

    /// return the transformed type for this rewrite
    LLType* type(Type* dty, LLType* t) {
        return DtoUnpaddedStructType(dty->toBasetype());
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
 */
struct CompositeToInt : ABIRewrite
{
    LLValue* get(Type* dty, DValue* dv)
    {
        Logger::println("rewriting integer -> %s", dty->toChars());

        LLValue* address = 0; // of passed Int

        if (dv->isLVal()) {                                               // dv is already in memory:
            address = dv->getLVal();                                      //   address = &<passed Int>
        } else {                                                          // dump dv to memory:
            LLValue* v = dv->getRVal();                                   //   v = <passed Int>
            address = DtoRawAlloca(v->getType(), 0, ".int_to_composite"); //   address = new Int
            DtoStore(v, address);                                         //   *address = v
        }

        LLType* pTy = getPtrToType(DtoType(dty));
        return DtoLoad(DtoBitCast(address, pTy), "get-result");           // *(Type*)address
    }

    void getL(Type* dty, DValue* dv, LLValue* lval)
    {
        Logger::println("rewriting integer -> %s", dty->toChars());
        LLValue* v = dv->getRVal();                                 // v = <passed Int>
        DtoStore(v, DtoBitCast(lval, getPtrToType(v->getType())));  // *(Int*)lval = v
    }

    LLValue* put(Type* dty, DValue* dv)
    {
        Logger::println("rewriting %s -> integer", dty->toChars());

        LLValue* address = 0; // of original parameter dv

        if (dv->getRVal()->getType()->isPointerTy()) {     // dv has been lowered to a pointer to the struct/static array:
            address = dv->getRVal();                       //   address = dv
        } else if (dv->isLVal()) {                         // dv is already in memory:
            address = dv->getLVal();                       //   address = &dv
        } else {                                           // dump dv to memory:
            address = DtoAlloca(dty, ".composite_to_int"); //   address = new Type
            DtoStore(dv->getRVal(), address);              //   *address = dv
        }

        LLType* intType = type(dty, 0);
        return DtoLoad(DtoBitCast(address, getPtrToType(intType)), "put-result"); // *(Int*)address
    }

    LLType* type(Type* t, LLType*)
    {
        size_t sz = t->size() * 8;
        return LLIntegerType::get(gIR->context(), sz);
    }
};

#endif
