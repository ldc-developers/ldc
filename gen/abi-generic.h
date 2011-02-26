#ifndef LDC_GEN_ABI_GENERIC
#define LDC_GEN_ABI_GENERIC

#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"
#include "gen/structs.h"

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
    virtual const LLType* type(Type* dty, const LLType* t) {
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
    const LLType* type(Type*, const LLType* t)
    {
        return t;
    }
};

#endif
