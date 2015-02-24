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
 * Rewrites any parameter to an integer of the same or next bigger size via
 * bit-casting.
 */
struct IntegerRewrite : ABIRewrite
{
    LLValue* get(Type* dty, DValue* dv)
    {
        LLValue* integer = dv->getRVal();
        LLValue* integerDump = storeToMemory(integer, 0, ".IntegerRewrite_dump");

        LLType* type = DtoType(dty);
        return loadFromMemory(integerDump, type, ".IntegerRewrite_getResult");
    }

    void getL(Type* dty, DValue* dv, LLValue* lval)
    {
        LLValue* integer = dv->getRVal();
        storeToMemory(integer, lval);
    }

    LLValue* put(Type* dty, DValue* dv)
    {
        assert(dty == dv->getType());
        LLValue* address = getAddressOf(dv);
        LLType* integerType = type(dty, NULL);
        return loadFromMemory(address, integerType, ".IntegerRewrite_putResult");
    }

    LLType* type(Type* t, LLType*)
    {
        unsigned size = t->size();
        switch (size) {
          case 0:
            size = 1;
            break;
          case 3:
            size = 4;
            break;
          case 5:
          case 6:
          case 7:
            size = 8;
            break;
          default:
            break;
        }
        return LLIntegerType::get(gIR->context(), size * 8);
    }
};

//////////////////////////////////////////////////////////////////////////////

/**
 * Implements explicit ByVal semantics defined like this:
 * Instead of passing a copy of the original argument directly to the callee,
 * the caller makes a bit-copy on its stack first and then passes a pointer
 * to that copy to the callee.
 * The pointer is passed as regular parameter and hence occupies either a
 * register or a function arguments stack slot.
 *
 * This differs from LLVM's ByVal attribute for pointer parameters.
 * The ByVal attribute instructs LLVM to pass the pointed-to argument directly
 * as a copy on the function arguments stack. In this case, there's no need to
 * pass an explicit pointer; the address is implicit.
 */
struct ExplicitByvalRewrite : ABIRewrite
{
    const size_t alignment;

    ExplicitByvalRewrite(size_t alignment = 16) : alignment(alignment)
    { }

    LLValue* get(Type* dty, DValue* v)
    {
        LLValue* pointer = v->getRVal();
        return DtoLoad(pointer, ".ExplicitByvalRewrite_getResult");
    }

    void getL(Type* dty, DValue* v, LLValue* lval)
    {
        LLValue* pointer = v->getRVal();
        DtoAggrCopy(lval, pointer);
    }

    LLValue* put(Type* dty, DValue* v)
    {
        if (DtoIsPassedByRef(dty))
        {
            LLValue* originalPointer = v->getRVal();
            LLType* type = originalPointer->getType()->getPointerElementType();
            LLValue* copyForCallee = DtoRawAlloca(type, alignment, ".ExplicitByvalRewrite_putResult");
            DtoAggrCopy(copyForCallee, originalPointer);
            return copyForCallee;
        }

        LLValue* originalValue = v->getRVal();
        LLValue* copyForCallee = storeToMemory(originalValue, alignment, ".ExplicitByvalRewrite_putResult");
        return copyForCallee;
    }

    LLType* type(Type* dty, LLType* t)
    {
        return getPtrToType(DtoType(dty));
    }
};

#endif
