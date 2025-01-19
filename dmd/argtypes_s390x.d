/**
 * Break down a D type into basic (register) types for the IBM Z ELF ABI.
 *
 * Copyright:   Copyright (C) 2024-2025 by The D Language Foundation, All Rights Reserved
 * Authors:     Martin Kinkelin
 * License:     $(LINK2 https://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/argtypes_s390x.d, _argtypes_s390x.d)
 * Documentation:  https://dlang.org/phobos/dmd_argtypes_s390x.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/argtypes_s390x.d
 */

module dmd.argtypes_s390x;

import dmd.astenums;
import dmd.mtype;
import dmd.typesem;

/****************************************************
 * This breaks a type down into 'simpler' types that can be passed to a function
 * in registers, and returned in registers.
 * This is the implementation for the IBM Z ELF ABI,
 * based on https://github.com/IBM/s390x-abi/releases/download/v1.6/lzsabi_s390x.pdf.
 * Params:
 *      t = type to break down
 * Returns:
 *      tuple of types, each element can be passed in a register.
 *      A tuple of zero length means the type cannot be passed/returned in registers.
 *      null indicates a `void`.
 */
TypeTuple toArgTypes_s390x(Type t)
{
    if (t == Type.terror)
        return new TypeTuple(t);

    const size = cast(size_t) t.size();
    if (size == 0)
        return null;

    // TODO
    // Implement the rest of the va args passing
    //...
    Type tb = t.toBasetype();
    const isAggregate = tb.ty == Tstruct || tb.ty == Tsarray || tb.ty == Tarray || tb.ty == Tdelegate || tb.iscomplex();
    if (!isAggregate)
        return new TypeTuple(t);
    // unwrap single-float struct per ABI requirements
    if (auto tstruct = t.isTypeStruct())
    {
        if (tstruct.sym.fields.length == 1)
        {
            Type fieldType = tstruct.sym.fields[0].type.toBasetype();
            if (fieldType.isfloating())
            {
                return new TypeTuple(fieldType);
            }
        }
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
        default:
            import dmd.typesem : sarrayOf;
            return Type.tint64.sarrayOf((size + 7) / 8);
        }
    }
    return new TypeTuple(getGPType(size));
}