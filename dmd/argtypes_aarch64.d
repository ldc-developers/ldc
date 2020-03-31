/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2020 by The D Language Foundation, All Rights Reserved
 * Authors:     Martin Kinkelin
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/argtypes_aarch64.d, _argtypes_aarch64.d)
 * Documentation:  https://dlang.org/phobos/dmd_argtypes_aarch64.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/argtypes_aarch64.d
 */

module dmd.argtypes_aarch64;

import dmd.mtype;

/****************************************************
 * This breaks a type down into 'simpler' types that can be passed to a function
 * in registers, and returned in registers.
 * This is the implementation for the AAPCS64 ABI, based on
 * https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst.
 * Params:
 *      t = type to break down
 * Returns:
 *      tuple of 1 type if the `t` can be passed in registers; e.g., a static array
 *      for Homogeneous Floating-point/Vector Aggregates (HFVA).
 *      A tuple of zero length means the type cannot be passed/returned in registers.
 *      null indicates a `void`.
 */
extern (C++) TypeTuple toArgTypes_aarch64(Type t)
{
    if (t == Type.terror)
        return new TypeTuple(t);

    const size = cast(size_t) t.size();
    if (size == 0)
        return null;

    Type tb = t.toBasetype();
    if (tb.ty == Tstruct || tb.ty == Tsarray || tb.ty == Tdelegate || tb.iscomplex())
    {
        Type hfvaType;
        const isHFVA = size > 4 * 16 ? false : isHFVA(tb, hfvaType);

        if ((size > 16 && !isHFVA) || !isPOD(tb))
        {
            // pass indirectly by value (pointer to hidden copy)
            return new TypeTuple();
        }

        if (isHFVA)
        {
            // pass in SIMD registers
            return new TypeTuple(hfvaType);
        }

        // pass remaining aggregates in 1 or 2 GP registers
        static Type getGPType(size_t size)
        {
            switch (size)
            {
            case 1:  return Type.tint8;
            case 2:  return Type.tint16;
            case 4:  return Type.tint32;
            case 8:  return Type.tint64;
            default: return Type.tint64.sarrayOf((size + 7) / 8);
            }
        }
        return new TypeTuple(getGPType(size));
    }

    return new TypeTuple(t);
}

private:

bool isPOD(Type t)
{
    auto baseType = t.baseElemOf();
    if (auto ts = baseType.isTypeStruct())
        return ts.sym.isPOD();
    return true;
}

bool isHFVA(Type t, ref Type rewriteType)
{
    Type fundamentalType;
    const N = getNestedHFVA(t, fundamentalType);
    if (N < 1 || N > 4)
        return false;

    rewriteType = fundamentalType.sarrayOf(N);
    return true;
}

/**
 * Recursive helper.
 * Returns size_t.max if the type isn't suited as HFVA (element) or incompatible
 * to the specified fundamental type, otherwise the number of consumed elements
 * of that fundamental type.
 * If `fundamentalType` is null, it is set on the first occasion and then left
 * untouched.
 */
size_t getNestedHFVA(Type t, ref Type fundamentalType)
{
    t = t.toBasetype();

    if (auto tarray = t.isTypeSArray())
    {
        const N = getNestedHFVA(tarray.nextOf(), fundamentalType);
        return N == size_t.max ? N : N * cast(size_t) tarray.dim.toUInteger(); // => T[0] may return 0
    }

    if (auto tstruct = t.isTypeStruct())
    {
        // check each field recursively and set fundamentalType
        bool isEmpty = true;
        foreach (field; tstruct.sym.fields)
        {
            const field_N = getNestedHFVA(field.type, fundamentalType);
            if (field_N == size_t.max)
                return field_N;
            if (field_N > 0) // might be 0 for empty static array
                isEmpty = false;
        }

        // an empty struct (no fields or only empty static arrays) is an undefined
        // byte, i.e., no HFVA
        if (isEmpty)
            return size_t.max;

        // due to possibly overlapping fields (for unions and nested anonymous
        // unions), use the overall struct size to determine N
        const structSize = t.size();
        const fundamentalSize = fundamentalType.size();
        assert(structSize % fundamentalSize == 0);
        return cast(size_t) (structSize / fundamentalSize);
    }

    Type thisFundamentalType;
    size_t N;

    if (t.isTypeVector())
    {
        thisFundamentalType = t;
        N = 1;
    }
    else if (t.isfloating()) // incl. imaginary and complex
    {
        auto ftSize = t.size();
        N = 1;

        if (t.iscomplex())
        {
            ftSize /= 2;
            N = 2;
        }

        switch (ftSize)
        {
            case  4: thisFundamentalType = Type.tfloat32; break;
            case  8: thisFundamentalType = Type.tfloat64; break;
            case 16: thisFundamentalType = Type.tfloat80; break; // IEEE quadruple
            default: assert(0, "unexpected floating-point type size");
        }
    }
    else
    {
        return size_t.max; // reject all other types
    }

    if (!fundamentalType)
        fundamentalType = thisFundamentalType; // initialize
    else if (fundamentalType != thisFundamentalType)
        return size_t.max; // incompatible fundamental types, reject

    return N;
}
