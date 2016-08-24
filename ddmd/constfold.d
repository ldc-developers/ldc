// Compiler implementation of the D programming language
// Copyright (c) 1999-2015 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.constfold;

import core.stdc.string;
import core.stdc.stdio;
import ddmd.arraytypes;
import ddmd.builtin;
import ddmd.complex;
import ddmd.ctfeexpr;
import ddmd.declaration;
import ddmd.dstruct;
import ddmd.errors;
import ddmd.expression;
import ddmd.func;
import ddmd.globals;
import ddmd.mtype;
import ddmd.root.ctfloat;
import ddmd.root.port;
import ddmd.root.rmem;
import ddmd.sideeffect;
import ddmd.target;
import ddmd.tokens;
import ddmd.utf;

version(IN_LLVM)
{
    import gen.dpragma;
}

private enum LOG = false;

extern (C++) Expression expType(Type type, Expression e)
{
    if (type != e.type)
    {
        e = e.copy();
        e.type = type;
    }
    return e;
}

/* ================================== isConst() ============================== */
extern (C++) int isConst(Expression e)
{
    //printf("Expression::isConst(): %s\n", e->toChars());
    switch (e.op)
    {
    case TOKint64:
    case TOKfloat64:
    case TOKcomplex80:
        return 1;
    case TOKnull:
        return 0;
    case TOKsymoff:
version(IN_LLVM)
{
        // We don't statically know anything about the address of a weak symbol
        // if there is no offset. With an offset, we can at least say that it is
        // non-zero.
        SymOffExp soe = cast(SymOffExp) e;
        if (soe.var.llvmInternal == LDCPragma.LLVMextern_weak && !soe.offset)
        {
            return 0;
        }
}
        return 2;
    default:
        return 0;
    }
    assert(0);
}

/* =============================== constFold() ============================== */
/* The constFold() functions were redundant with the optimize() ones,
 * and so have been folded in with them.
 */
/* ========================================================================== */
extern (C++) UnionExp Neg(Type type, Expression e1)
{
    UnionExp ue;
    Loc loc = e1.loc;
    if (e1.type.isreal())
    {
        emplaceExp!(RealExp)(&ue, loc, -e1.toReal(), type);
    }
    else if (e1.type.isimaginary())
    {
        emplaceExp!(RealExp)(&ue, loc, -e1.toImaginary(), type);
    }
    else if (e1.type.iscomplex())
    {
        emplaceExp!(ComplexExp)(&ue, loc, -e1.toComplex(), type);
    }
    else
    {
        emplaceExp!(IntegerExp)(&ue, loc, -e1.toInteger(), type);
    }
    return ue;
}

extern (C++) UnionExp Com(Type type, Expression e1)
{
    UnionExp ue;
    Loc loc = e1.loc;
    emplaceExp!(IntegerExp)(&ue, loc, ~e1.toInteger(), type);
    return ue;
}

extern (C++) UnionExp Not(Type type, Expression e1)
{
    UnionExp ue;
    Loc loc = e1.loc;
    emplaceExp!(IntegerExp)(&ue, loc, e1.isBool(false) ? 1 : 0, type);
    return ue;
}

extern (C++) UnionExp Bool(Type type, Expression e1)
{
    UnionExp ue;
    Loc loc = e1.loc;
    emplaceExp!(IntegerExp)(&ue, loc, e1.isBool(true) ? 1 : 0, type);
    return ue;
}

extern (C++) UnionExp Add(Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    static if (LOG)
    {
        printf("Add(e1 = %s, e2 = %s)\n", e1.toChars(), e2.toChars());
    }
    if (type.isreal())
    {
        emplaceExp!(RealExp)(&ue, loc, e1.toReal() + e2.toReal(), type);
    }
    else if (type.isimaginary())
    {
        emplaceExp!(RealExp)(&ue, loc, e1.toImaginary() + e2.toImaginary(), type);
    }
    else if (type.iscomplex())
    {
        // This rigamarole is necessary so that -0.0 doesn't get
        // converted to +0.0 by doing an extraneous add with +0.0
        auto c1 = complex_t(real_t(0));
        real_t r1 = 0;
        real_t i1 = 0;
        auto c2 = complex_t(real_t(0));
        real_t r2 = 0;
        real_t i2 = 0;
        auto v = complex_t(real_t(0));
        int x;
        if (e1.type.isreal())
        {
            r1 = e1.toReal();
            x = 0;
        }
        else if (e1.type.isimaginary())
        {
            i1 = e1.toImaginary();
            x = 3;
        }
        else
        {
            c1 = e1.toComplex();
            x = 6;
        }
        if (e2.type.isreal())
        {
            r2 = e2.toReal();
        }
        else if (e2.type.isimaginary())
        {
            i2 = e2.toImaginary();
            x += 1;
        }
        else
        {
            c2 = e2.toComplex();
            x += 2;
        }
        switch (x)
        {
        case 0 + 0:
            v = complex_t(r1 + r2);
            break;
        case 0 + 1:
            v = complex_t(r1, i2);
            break;
        case 0 + 2:
            v = complex_t(r1 + creall(c2), cimagl(c2));
            break;
        case 3 + 0:
            v = complex_t(r2, i1);
            break;
        case 3 + 1:
            v = complex_t(real_t(0), i1 + i2);
            break;
        case 3 + 2:
            v = complex_t(creall(c2), i1 + cimagl(c2));
            break;
        case 6 + 0:
            v = complex_t(creall(c1) + r2, cimagl(c2));
            break;
        case 6 + 1:
            v = complex_t(creall(c1), cimagl(c1) + i2);
            break;
        case 6 + 2:
            v = c1 + c2;
            break;
        default:
            assert(0);
        }
        emplaceExp!(ComplexExp)(&ue, loc, v, type);
    }
    else if (e1.op == TOKsymoff)
    {
        SymOffExp soe = cast(SymOffExp)e1;
        emplaceExp!(SymOffExp)(&ue, loc, soe.var, soe.offset + e2.toInteger());
        ue.exp().type = type;
    }
    else if (e2.op == TOKsymoff)
    {
        SymOffExp soe = cast(SymOffExp)e2;
        emplaceExp!(SymOffExp)(&ue, loc, soe.var, soe.offset + e1.toInteger());
        ue.exp().type = type;
    }
    else
        emplaceExp!(IntegerExp)(&ue, loc, e1.toInteger() + e2.toInteger(), type);
    return ue;
}

extern (C++) UnionExp Min(Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    if (type.isreal())
    {
        emplaceExp!(RealExp)(&ue, loc, e1.toReal() - e2.toReal(), type);
    }
    else if (type.isimaginary())
    {
        emplaceExp!(RealExp)(&ue, loc, e1.toImaginary() - e2.toImaginary(), type);
    }
    else if (type.iscomplex())
    {
        // This rigamarole is necessary so that -0.0 doesn't get
        // converted to +0.0 by doing an extraneous add with +0.0
        auto c1 = complex_t(real_t(0));
        real_t r1 = 0;
        real_t i1 = 0;
        auto c2 = complex_t(real_t(0));
        real_t r2 = 0;
        real_t i2 = 0;
        auto v = complex_t(real_t(0));
        int x;
        if (e1.type.isreal())
        {
            r1 = e1.toReal();
            x = 0;
        }
        else if (e1.type.isimaginary())
        {
            i1 = e1.toImaginary();
            x = 3;
        }
        else
        {
            c1 = e1.toComplex();
            x = 6;
        }
        if (e2.type.isreal())
        {
            r2 = e2.toReal();
        }
        else if (e2.type.isimaginary())
        {
            i2 = e2.toImaginary();
            x += 1;
        }
        else
        {
            c2 = e2.toComplex();
            x += 2;
        }
        switch (x)
        {
        case 0 + 0:
            v = complex_t(r1 - r2);
            break;
        case 0 + 1:
            v = complex_t(r1, -i2);
            break;
        case 0 + 2:
            v = complex_t(r1 - creall(c2), -cimagl(c2));
            break;
        case 3 + 0:
            v = complex_t(-r2, i1);
            break;
        case 3 + 1:
            v = complex_t(real_t(0), i1 - i2);
            break;
        case 3 + 2:
            v = complex_t(-creall(c2), i1 - cimagl(c2));
            break;
        case 6 + 0:
            v = complex_t(creall(c1) - r2, cimagl(c1));
            break;
        case 6 + 1:
            v = complex_t(creall(c1), cimagl(c1) - i2);
            break;
        case 6 + 2:
            v = c1 - c2;
            break;
        default:
            assert(0);
        }
        emplaceExp!(ComplexExp)(&ue, loc, v, type);
    }
    else if (e1.op == TOKsymoff)
    {
        SymOffExp soe = cast(SymOffExp)e1;
        emplaceExp!(SymOffExp)(&ue, loc, soe.var, soe.offset - e2.toInteger());
        ue.exp().type = type;
    }
    else
    {
        emplaceExp!(IntegerExp)(&ue, loc, e1.toInteger() - e2.toInteger(), type);
    }
    return ue;
}

extern (C++) UnionExp Mul(Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    if (type.isfloating())
    {
        auto c = complex_t(real_t(0));
        real_t r = 0;
        if (e1.type.isreal())
        {
            r = e1.toReal();
            c = e2.toComplex();
            c = complex_t(r * creall(c), r * cimagl(c));
        }
        else if (e1.type.isimaginary())
        {
            r = e1.toImaginary();
            c = e2.toComplex();
            c = complex_t(-r * cimagl(c), r * creall(c));
        }
        else if (e2.type.isreal())
        {
            r = e2.toReal();
            c = e1.toComplex();
            c = complex_t(r * creall(c), r * cimagl(c));
        }
        else if (e2.type.isimaginary())
        {
            r = e2.toImaginary();
            c = e1.toComplex();
            c = complex_t(-r * cimagl(c), r * creall(c));
        }
        else
            c = e1.toComplex() * e2.toComplex();
        if (type.isreal())
            emplaceExp!(RealExp)(&ue, loc, creall(c), type);
        else if (type.isimaginary())
            emplaceExp!(RealExp)(&ue, loc, cimagl(c), type);
        else if (type.iscomplex())
            emplaceExp!(ComplexExp)(&ue, loc, c, type);
        else
            assert(0);
    }
    else
    {
        emplaceExp!(IntegerExp)(&ue, loc, e1.toInteger() * e2.toInteger(), type);
    }
    return ue;
}

extern (C++) UnionExp Div(Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    if (type.isfloating())
    {
        auto c = complex_t(real_t(0));
        //e1->type->print();
        //e2->type->print();
        if (e2.type.isreal())
        {
            if (e1.type.isreal())
            {
                version (all)
                {
                    // Work around redundant REX.W prefix breaking Valgrind
                    // when built with affected versions of DMD.
                    // https://issues.dlang.org/show_bug.cgi?id=14952
                    // This can be removed once compiling with DMD 2.068 or
                    // older is no longer supported.
                    const r1 = e1.toReal();
                    const r2 = e2.toReal();
                    emplaceExp!(RealExp)(&ue, loc, r1 / r2, type);
                }
                else
                {
                    emplaceExp!(RealExp)(&ue, loc, e1.toReal() / e2.toReal(), type);
                }
                return ue;
            }
            const r = e2.toReal();
            c = e1.toComplex();
            c = complex_t(creall(c) / r, cimagl(c) / r);
        }
        else if (e2.type.isimaginary())
        {
            const r = e2.toImaginary();
            c = e1.toComplex();
            c = complex_t(cimagl(c) / r, -creall(c) / r);
        }
        else
        {
            c = e1.toComplex() / e2.toComplex();
        }

        if (type.isreal())
            emplaceExp!(RealExp)(&ue, loc, creall(c), type);
        else if (type.isimaginary())
            emplaceExp!(RealExp)(&ue, loc, cimagl(c), type);
        else if (type.iscomplex())
            emplaceExp!(ComplexExp)(&ue, loc, c, type);
        else
            assert(0);
    }
    else
    {
        sinteger_t n1;
        sinteger_t n2;
        sinteger_t n;
        n1 = e1.toInteger();
        n2 = e2.toInteger();
        if (n2 == 0)
        {
            e2.error("divide by 0");
            n2 = 1;
        }
        if (e1.type.isunsigned() || e2.type.isunsigned())
            n = (cast(dinteger_t)n1) / (cast(dinteger_t)n2);
        else
            n = n1 / n2;
        emplaceExp!(IntegerExp)(&ue, loc, n, type);
    }
    return ue;
}

extern (C++) UnionExp Mod(Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    if (type.isfloating())
    {
        auto c = complex_t(real_t(0));
        if (e2.type.isreal())
        {
            const r2 = e2.toReal();
            c = complex_t(e1.toReal() % r2, e1.toImaginary() % r2);
        }
        else if (e2.type.isimaginary())
        {
            const i2 = e2.toImaginary();
            c = complex_t(e1.toReal() % i2, e1.toImaginary() % i2);
        }
        else
            assert(0);
        if (type.isreal())
            emplaceExp!(RealExp)(&ue, loc, creall(c), type);
        else if (type.isimaginary())
            emplaceExp!(RealExp)(&ue, loc, cimagl(c), type);
        else if (type.iscomplex())
            emplaceExp!(ComplexExp)(&ue, loc, c, type);
        else
            assert(0);
    }
    else
    {
        sinteger_t n1;
        sinteger_t n2;
        sinteger_t n;
        n1 = e1.toInteger();
        n2 = e2.toInteger();
        if (n2 == 0)
        {
            e2.error("divide by 0");
            n2 = 1;
        }
        if (n2 == -1 && !type.isunsigned())
        {
            // Check for int.min % -1
            if (n1 == 0xFFFFFFFF80000000UL && type.toBasetype().ty != Tint64)
            {
                e2.error("integer overflow: int.min % -1");
                n2 = 1;
            }
            else if (n1 == 0x8000000000000000L) // long.min % -1
            {
                e2.error("integer overflow: long.min % -1");
                n2 = 1;
            }
        }
        if (e1.type.isunsigned() || e2.type.isunsigned())
            n = (cast(dinteger_t)n1) % (cast(dinteger_t)n2);
        else
            n = n1 % n2;
        emplaceExp!(IntegerExp)(&ue, loc, n, type);
    }
    return ue;
}

extern (C++) UnionExp Pow(Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    // Handle integer power operations.
    if (e2.type.isintegral())
    {
        dinteger_t n = e2.toInteger();
        bool neg;
        if (!e2.type.isunsigned() && cast(sinteger_t)n < 0)
        {
            if (e1.type.isintegral())
            {
                emplaceExp!(CTFEExp)(&ue, TOKcantexp);
                return ue;
            }
            // Don't worry about overflow, from now on n is unsigned.
            neg = true;
            n = -n;
        }
        else
            neg = false;
        UnionExp ur, uv;
        if (e1.type.iscomplex())
        {
            emplaceExp!(ComplexExp)(&ur, loc, e1.toComplex(), e1.type);
            emplaceExp!(ComplexExp)(&uv, loc, complex_t(real_t(1)), e1.type);
        }
        else if (e1.type.isfloating())
        {
            emplaceExp!(RealExp)(&ur, loc, e1.toReal(), e1.type);
            emplaceExp!(RealExp)(&uv, loc, real_t(1), e1.type);
        }
        else
        {
            emplaceExp!(IntegerExp)(&ur, loc, e1.toInteger(), e1.type);
            emplaceExp!(IntegerExp)(&uv, loc, 1, e1.type);
        }
        Expression r = ur.exp();
        Expression v = uv.exp();
        while (n != 0)
        {
            if (n & 1)
            {
                // v = v * r;
                uv = Mul(loc, v.type, v, r);
            }
            n >>= 1;
            // r = r * r
            ur = Mul(loc, r.type, r, r);
        }
        if (neg)
        {
            // ue = 1.0 / v
            UnionExp one;
            emplaceExp!(RealExp)(&one, loc, real_t(1), v.type);
            uv = Div(loc, v.type, one.exp(), v);
        }
        if (type.iscomplex())
            emplaceExp!(ComplexExp)(&ue, loc, v.toComplex(), type);
        else if (type.isintegral())
            emplaceExp!(IntegerExp)(&ue, loc, v.toInteger(), type);
        else
            emplaceExp!(RealExp)(&ue, loc, v.toReal(), type);
    }
    else if (e2.type.isfloating())
    {
        // x ^^ y for x < 0 and y not an integer is not defined; so set result as NaN
        if (e1.toReal() < real_t(0))
        {
            emplaceExp!(RealExp)(&ue, loc, Target.RealProperties.nan, type);
        }
        else
            emplaceExp!(CTFEExp)(&ue, TOKcantexp);
    }
    else
        emplaceExp!(CTFEExp)(&ue, TOKcantexp);
    return ue;
}

extern (C++) UnionExp Shl(Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    emplaceExp!(IntegerExp)(&ue, loc, e1.toInteger() << e2.toInteger(), type);
    return ue;
}

extern (C++) UnionExp Shr(Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    dinteger_t value = e1.toInteger();
    dinteger_t dcount = e2.toInteger();
    assert(dcount <= 0xFFFFFFFF);
    uint count = cast(uint)dcount;
    switch (e1.type.toBasetype().ty)
    {
    case Tint8:
        value = cast(d_int8)value >> count;
        break;
    case Tuns8:
    case Tchar:
        value = cast(d_uns8)value >> count;
        break;
    case Tint16:
        value = cast(d_int16)value >> count;
        break;
    case Tuns16:
    case Twchar:
        value = cast(d_uns16)value >> count;
        break;
    case Tint32:
        value = cast(d_int32)value >> count;
        break;
    case Tuns32:
    case Tdchar:
        value = cast(d_uns32)value >> count;
        break;
    case Tint64:
        value = cast(d_int64)value >> count;
        break;
    case Tuns64:
        value = cast(d_uns64)value >> count;
        break;
    case Terror:
        emplaceExp!(ErrorExp)(&ue);
        return ue;
    default:
        assert(0);
    }
    emplaceExp!(IntegerExp)(&ue, loc, value, type);
    return ue;
}

extern (C++) UnionExp Ushr(Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    dinteger_t value = e1.toInteger();
    dinteger_t dcount = e2.toInteger();
    assert(dcount <= 0xFFFFFFFF);
    uint count = cast(uint)dcount;
    switch (e1.type.toBasetype().ty)
    {
    case Tint8:
    case Tuns8:
    case Tchar:
        // Possible only with >>>=. >>> always gets promoted to int.
        value = (value & 0xFF) >> count;
        break;
    case Tint16:
    case Tuns16:
    case Twchar:
        // Possible only with >>>=. >>> always gets promoted to int.
        value = (value & 0xFFFF) >> count;
        break;
    case Tint32:
    case Tuns32:
    case Tdchar:
        value = (value & 0xFFFFFFFF) >> count;
        break;
    case Tint64:
    case Tuns64:
        value = cast(d_uns64)value >> count;
        break;
    case Terror:
        emplaceExp!(ErrorExp)(&ue);
        return ue;
    default:
        assert(0);
    }
    emplaceExp!(IntegerExp)(&ue, loc, value, type);
    return ue;
}

extern (C++) UnionExp And(Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    emplaceExp!(IntegerExp)(&ue, loc, e1.toInteger() & e2.toInteger(), type);
    return ue;
}

extern (C++) UnionExp Or(Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    emplaceExp!(IntegerExp)(&ue, loc, e1.toInteger() | e2.toInteger(), type);
    return ue;
}

extern (C++) UnionExp Xor(Loc loc, Type type, Expression e1, Expression e2)
{
    //printf("Xor(linnum = %d, e1 = %s, e2 = %s)\n", loc.linnum, e1.toChars(), e2.toChars());
    UnionExp ue;
    emplaceExp!(IntegerExp)(&ue, loc, e1.toInteger() ^ e2.toInteger(), type);
    return ue;
}

/* Also returns TOKcantexp if cannot be computed.
 */
extern (C++) UnionExp Equal(TOK op, Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    int cmp = 0;
    real_t r1 = 0;
    real_t r2 = 0;
    //printf("Equal(e1 = %s, e2 = %s)\n", e1->toChars(), e2->toChars());
    assert(op == TOKequal || op == TOKnotequal);
    if (e1.op == TOKnull)
    {
        if (e2.op == TOKnull)
            cmp = 1;
        else if (e2.op == TOKstring)
        {
            StringExp es2 = cast(StringExp)e2;
            cmp = (0 == es2.len);
        }
        else if (e2.op == TOKarrayliteral)
        {
            ArrayLiteralExp es2 = cast(ArrayLiteralExp)e2;
            cmp = !es2.elements || (0 == es2.elements.dim);
        }
        else
        {
            emplaceExp!(CTFEExp)(&ue, TOKcantexp);
            return ue;
        }
    }
    else if (e2.op == TOKnull)
    {
        if (e1.op == TOKstring)
        {
            StringExp es1 = cast(StringExp)e1;
            cmp = (0 == es1.len);
        }
        else if (e1.op == TOKarrayliteral)
        {
            ArrayLiteralExp es1 = cast(ArrayLiteralExp)e1;
            cmp = !es1.elements || (0 == es1.elements.dim);
        }
        else
        {
            emplaceExp!(CTFEExp)(&ue, TOKcantexp);
            return ue;
        }
    }
    else if (e1.op == TOKstring && e2.op == TOKstring)
    {
        StringExp es1 = cast(StringExp)e1;
        StringExp es2 = cast(StringExp)e2;
        if (es1.sz != es2.sz)
        {
            assert(global.errors);
            emplaceExp!(CTFEExp)(&ue, TOKcantexp);
            return ue;
        }
        if (es1.len == es2.len && memcmp(es1.string, es2.string, es1.sz * es1.len) == 0)
            cmp = 1;
        else
            cmp = 0;
    }
    else if (e1.op == TOKarrayliteral && e2.op == TOKarrayliteral)
    {
        ArrayLiteralExp es1 = cast(ArrayLiteralExp)e1;
        ArrayLiteralExp es2 = cast(ArrayLiteralExp)e2;
        if ((!es1.elements || !es1.elements.dim) && (!es2.elements || !es2.elements.dim))
            cmp = 1; // both arrays are empty
        else if (!es1.elements || !es2.elements)
            cmp = 0;
        else if (es1.elements.dim != es2.elements.dim)
            cmp = 0;
        else
        {
            for (size_t i = 0; i < es1.elements.dim; i++)
            {
                auto ee1 = es1.getElement(i);
                auto ee2 = es2.getElement(i);
                ue = Equal(TOKequal, loc, Type.tint32, ee1, ee2);
                if (CTFEExp.isCantExp(ue.exp()))
                    return ue;
                cmp = cast(int)ue.exp().toInteger();
                if (cmp == 0)
                    break;
            }
        }
    }
    else if (e1.op == TOKarrayliteral && e2.op == TOKstring)
    {
        // Swap operands and use common code
        Expression etmp = e1;
        e1 = e2;
        e2 = etmp;
        goto Lsa;
    }
    else if (e1.op == TOKstring && e2.op == TOKarrayliteral)
    {
    Lsa:
        StringExp es1 = cast(StringExp)e1;
        ArrayLiteralExp es2 = cast(ArrayLiteralExp)e2;
        size_t dim1 = es1.len;
        size_t dim2 = es2.elements ? es2.elements.dim : 0;
        if (dim1 != dim2)
            cmp = 0;
        else
        {
            cmp = 1; // if dim1 winds up being 0
            for (size_t i = 0; i < dim1; i++)
            {
                uinteger_t c = es1.charAt(i);
                auto ee2 = es2.getElement(i);
                if (ee2.isConst() != 1)
                {
                    emplaceExp!(CTFEExp)(&ue, TOKcantexp);
                    return ue;
                }
                cmp = (c == ee2.toInteger());
                if (cmp == 0)
                    break;
            }
        }
    }
    else if (e1.op == TOKstructliteral && e2.op == TOKstructliteral)
    {
        StructLiteralExp es1 = cast(StructLiteralExp)e1;
        StructLiteralExp es2 = cast(StructLiteralExp)e2;
        if (es1.sd != es2.sd)
            cmp = 0;
        else if ((!es1.elements || !es1.elements.dim) && (!es2.elements || !es2.elements.dim))
            cmp = 1; // both arrays are empty
        else if (!es1.elements || !es2.elements)
            cmp = 0;
        else if (es1.elements.dim != es2.elements.dim)
            cmp = 0;
        else
        {
            cmp = 1;
            for (size_t i = 0; i < es1.elements.dim; i++)
            {
                Expression ee1 = (*es1.elements)[i];
                Expression ee2 = (*es2.elements)[i];
                if (ee1 == ee2)
                    continue;
                if (!ee1 || !ee2)
                {
                    cmp = 0;
                    break;
                }
                ue = Equal(TOKequal, loc, Type.tint32, ee1, ee2);
                if (ue.exp().op == TOKcantexp)
                    return ue;
                cmp = cast(int)ue.exp().toInteger();
                if (cmp == 0)
                    break;
            }
        }
    }
    else if (e1.isConst() != 1 || e2.isConst() != 1)
    {
        emplaceExp!(CTFEExp)(&ue, TOKcantexp);
        return ue;
    }
    else if (e1.type.isreal())
    {
        r1 = e1.toReal();
        r2 = e2.toReal();
        goto L1;
    }
    else if (e1.type.isimaginary())
    {
        r1 = e1.toImaginary();
        r2 = e2.toImaginary();
    L1:
        if (CTFloat.isNaN(r1) || CTFloat.isNaN(r2)) // if unordered
        {
            cmp = 0;
        }
        else
        {
            cmp = (r1 == r2);
        }
    }
    else if (e1.type.iscomplex())
    {
        cmp = e1.toComplex() == e2.toComplex();
    }
    else if (e1.type.isintegral() || e1.type.toBasetype().ty == Tpointer)
    {
        cmp = (e1.toInteger() == e2.toInteger());
    }
    else
    {
        emplaceExp!(CTFEExp)(&ue, TOKcantexp);
        return ue;
    }
    if (op == TOKnotequal)
        cmp ^= 1;
    emplaceExp!(IntegerExp)(&ue, loc, cmp, type);
    return ue;
}

extern (C++) UnionExp Identity(TOK op, Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    int cmp;
    if (e1.op == TOKnull)
    {
        cmp = (e2.op == TOKnull);
    }
    else if (e2.op == TOKnull)
    {
        cmp = 0;
    }
    else if (e1.op == TOKsymoff && e2.op == TOKsymoff)
    {
        SymOffExp es1 = cast(SymOffExp)e1;
        SymOffExp es2 = cast(SymOffExp)e2;
        cmp = (es1.var == es2.var && es1.offset == es2.offset);
    }
    else
    {
        if (e1.type.isreal())
        {
            cmp = RealEquals(e1.toReal(), e2.toReal());
        }
        else if (e1.type.isimaginary())
        {
            cmp = RealEquals(e1.toImaginary(), e2.toImaginary());
        }
        else if (e1.type.iscomplex())
        {
            complex_t v1 = e1.toComplex();
            complex_t v2 = e2.toComplex();
            cmp = RealEquals(creall(v1), creall(v2)) && RealEquals(cimagl(v1), cimagl(v1));
        }
        else
        {
            ue = Equal((op == TOKidentity) ? TOKequal : TOKnotequal, loc, type, e1, e2);
            return ue;
        }
    }
    if (op == TOKnotidentity)
        cmp ^= 1;
    emplaceExp!(IntegerExp)(&ue, loc, cmp, type);
    return ue;
}

extern (C++) UnionExp Cmp(TOK op, Loc loc, Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    dinteger_t n;
    real_t r1 = 0;
    real_t r2 = 0;
    //printf("Cmp(e1 = %s, e2 = %s)\n", e1->toChars(), e2->toChars());
    if (e1.op == TOKstring && e2.op == TOKstring)
    {
        StringExp es1 = cast(StringExp)e1;
        StringExp es2 = cast(StringExp)e2;
        size_t sz = es1.sz;
        assert(sz == es2.sz);
        size_t len = es1.len;
        if (es2.len < len)
            len = es2.len;
        int rawCmp = memcmp(es1.string, es2.string, sz * len);
        if (rawCmp == 0)
            rawCmp = cast(int)(es1.len - es2.len);
        n = specificCmp(op, rawCmp);
    }
    else if (e1.isConst() != 1 || e2.isConst() != 1)
    {
        emplaceExp!(CTFEExp)(&ue, TOKcantexp);
        return ue;
    }
    else if (e1.type.isreal())
    {
        r1 = e1.toReal();
        r2 = e2.toReal();
        goto L1;
    }
    else if (e1.type.isimaginary())
    {
        r1 = e1.toImaginary();
        r2 = e2.toImaginary();
    L1:
        n = realCmp(op, r1, r2);
    }
    else if (e1.type.iscomplex())
    {
        assert(0);
    }
    else
    {
        sinteger_t n1;
        sinteger_t n2;
        n1 = e1.toInteger();
        n2 = e2.toInteger();
        if (e1.type.isunsigned() || e2.type.isunsigned())
            n = intUnsignedCmp(op, n1, n2);
        else
            n = intSignedCmp(op, n1, n2);
    }
    emplaceExp!(IntegerExp)(&ue, loc, n, type);
    return ue;
}

/* Also returns TOKcantexp if cannot be computed.
 *  to: type to cast to
 *  type: type to paint the result
 */
extern (C++) UnionExp Cast(Loc loc, Type type, Type to, Expression e1)
{
    UnionExp ue;
    Type tb = to.toBasetype();
    Type typeb = type.toBasetype();
    //printf("Cast(type = %s, to = %s, e1 = %s)\n", type->toChars(), to->toChars(), e1->toChars());
    //printf("\te1->type = %s\n", e1->type->toChars());
    if (e1.type.equals(type) && type.equals(to))
    {
        emplaceExp!(UnionExp)(&ue, e1);
        return ue;
    }
    if (e1.op == TOKvector && (cast(TypeVector)e1.type).basetype.equals(type) && type.equals(to))
    {
        Expression ex = (cast(VectorExp)e1).e1;
        emplaceExp!(UnionExp)(&ue, ex);
        return ue;
    }
    if (e1.type.implicitConvTo(to) >= MATCHconst || to.implicitConvTo(e1.type) >= MATCHconst)
    {
        goto L1;
    }
    // Allow covariant converions of delegates
    // (Perhaps implicit conversion from pure to impure should be a MATCHconst,
    // then we wouldn't need this extra check.)
    if (e1.type.toBasetype().ty == Tdelegate && e1.type.implicitConvTo(to) == MATCHconvert)
    {
        goto L1;
    }
    /* Allow casting from one string type to another
     */
    if (e1.op == TOKstring)
    {
        if (tb.ty == Tarray && typeb.ty == Tarray && tb.nextOf().size() == typeb.nextOf().size())
        {
            goto L1;
        }
    }
    if (e1.op == TOKarrayliteral && typeb == tb)
    {
    L1:
        Expression ex = expType(to, e1);
        emplaceExp!(UnionExp)(&ue, ex);
        return ue;
    }
    if (e1.isConst() != 1)
    {
        emplaceExp!(CTFEExp)(&ue, TOKcantexp);
    }
    else if (tb.ty == Tbool)
    {
        emplaceExp!(IntegerExp)(&ue, loc, e1.toInteger() != 0, type);
    }
    else if (type.isintegral())
    {
        if (e1.type.isfloating())
        {
            dinteger_t result;
            real_t r = e1.toReal();
            switch (typeb.ty)
            {
            case Tint8:
                result = cast(d_int8)cast(sinteger_t)r;
                break;
            case Tchar:
            case Tuns8:
                result = cast(d_uns8)cast(dinteger_t)r;
                break;
            case Tint16:
                result = cast(d_int16)cast(sinteger_t)r;
                break;
            case Twchar:
            case Tuns16:
                result = cast(d_uns16)cast(dinteger_t)r;
                break;
            case Tint32:
                result = cast(d_int32)r;
                break;
            case Tdchar:
            case Tuns32:
                result = cast(d_uns32)r;
                break;
            case Tint64:
                result = cast(d_int64)r;
                break;
            case Tuns64:
                result = cast(d_uns64)r;
                break;
            default:
                assert(0);
            }
            emplaceExp!(IntegerExp)(&ue, loc, result, type);
        }
        else if (type.isunsigned())
            emplaceExp!(IntegerExp)(&ue, loc, e1.toUInteger(), type);
        else
            emplaceExp!(IntegerExp)(&ue, loc, e1.toInteger(), type);
    }
    else if (tb.isreal())
    {
        real_t value = e1.toReal();
        emplaceExp!(RealExp)(&ue, loc, value, type);
    }
    else if (tb.isimaginary())
    {
        real_t value = e1.toImaginary();
        emplaceExp!(RealExp)(&ue, loc, value, type);
    }
    else if (tb.iscomplex())
    {
        complex_t value = e1.toComplex();
        emplaceExp!(ComplexExp)(&ue, loc, value, type);
    }
    else if (tb.isscalar())
    {
        emplaceExp!(IntegerExp)(&ue, loc, e1.toInteger(), type);
    }
    else if (tb.ty == Tvoid)
    {
        emplaceExp!(CTFEExp)(&ue, TOKcantexp);
    }
    else if (tb.ty == Tstruct && e1.op == TOKint64)
    {
        // Struct = 0;
        StructDeclaration sd = tb.toDsymbol(null).isStructDeclaration();
        assert(sd);
        auto elements = new Expressions();
        for (size_t i = 0; i < sd.fields.dim; i++)
        {
            VarDeclaration v = sd.fields[i];
            UnionExp zero;
            emplaceExp!(IntegerExp)(&zero, 0);
            ue = Cast(loc, v.type, v.type, zero.exp());
            if (ue.exp().op == TOKcantexp)
                return ue;
            elements.push(ue.exp().copy());
        }
        emplaceExp!(StructLiteralExp)(&ue, loc, sd, elements);
        ue.exp().type = type;
    }
    else
    {
        if (type != Type.terror)
        {
            // have to change to Internal Compiler Error
            // all invalid casts should be handled already in Expression::castTo().
            error(loc, "cannot cast %s to %s", e1.type.toChars(), type.toChars());
        }
        emplaceExp!(ErrorExp)(&ue);
    }
    return ue;
}

extern (C++) UnionExp ArrayLength(Type type, Expression e1)
{
    UnionExp ue;
    Loc loc = e1.loc;
    if (e1.op == TOKstring)
    {
        StringExp es1 = cast(StringExp)e1;
        emplaceExp!(IntegerExp)(&ue, loc, es1.len, type);
    }
    else if (e1.op == TOKarrayliteral)
    {
        ArrayLiteralExp ale = cast(ArrayLiteralExp)e1;
        size_t dim = ale.elements ? ale.elements.dim : 0;
        emplaceExp!(IntegerExp)(&ue, loc, dim, type);
    }
    else if (e1.op == TOKassocarrayliteral)
    {
        AssocArrayLiteralExp ale = cast(AssocArrayLiteralExp)e1;
        size_t dim = ale.keys.dim;
        emplaceExp!(IntegerExp)(&ue, loc, dim, type);
    }
    else if (e1.type.toBasetype().ty == Tsarray)
    {
        Expression e = (cast(TypeSArray)e1.type.toBasetype()).dim;
        emplaceExp!(UnionExp)(&ue, e);
    }
    else
        emplaceExp!(CTFEExp)(&ue, TOKcantexp);
    return ue;
}

/* Also return TOKcantexp if this fails
 */
extern (C++) UnionExp Index(Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    Loc loc = e1.loc;
    //printf("Index(e1 = %s, e2 = %s)\n", e1->toChars(), e2->toChars());
    assert(e1.type);
    if (e1.op == TOKstring && e2.op == TOKint64)
    {
        StringExp es1 = cast(StringExp)e1;
        uinteger_t i = e2.toInteger();
        if (i >= es1.len)
        {
            e1.error("string index %llu is out of bounds [0 .. %llu]", i, cast(ulong)es1.len);
            emplaceExp!(ErrorExp)(&ue);
        }
        else
        {
            emplaceExp!(IntegerExp)(&ue, loc, es1.charAt(i), type);
        }
    }
    else if (e1.type.toBasetype().ty == Tsarray && e2.op == TOKint64)
    {
        TypeSArray tsa = cast(TypeSArray)e1.type.toBasetype();
        uinteger_t length = tsa.dim.toInteger();
        uinteger_t i = e2.toInteger();
        if (i >= length)
        {
            e1.error("array index %llu is out of bounds %s[0 .. %llu]", i, e1.toChars(), length);
            emplaceExp!(ErrorExp)(&ue);
        }
        else if (e1.op == TOKarrayliteral)
        {
            ArrayLiteralExp ale = cast(ArrayLiteralExp)e1;
            auto e = ale.getElement(cast(size_t)i);
            e.type = type;
            e.loc = loc;
            if (hasSideEffect(e))
                emplaceExp!(CTFEExp)(&ue, TOKcantexp);
            else
                emplaceExp!(UnionExp)(&ue, e);
        }
        else
            emplaceExp!(CTFEExp)(&ue, TOKcantexp);
    }
    else if (e1.type.toBasetype().ty == Tarray && e2.op == TOKint64)
    {
        uinteger_t i = e2.toInteger();
        if (e1.op == TOKarrayliteral)
        {
            ArrayLiteralExp ale = cast(ArrayLiteralExp)e1;
            if (i >= ale.elements.dim)
            {
                e1.error("array index %llu is out of bounds %s[0 .. %u]", i, e1.toChars(), ale.elements.dim);
                emplaceExp!(ErrorExp)(&ue);
            }
            else
            {
                auto e = ale.getElement(cast(size_t)i);
                e.type = type;
                e.loc = loc;
                if (hasSideEffect(e))
                    emplaceExp!(CTFEExp)(&ue, TOKcantexp);
                else
                    emplaceExp!(UnionExp)(&ue, e);
            }
        }
        else
            emplaceExp!(CTFEExp)(&ue, TOKcantexp);
    }
    else if (e1.op == TOKassocarrayliteral)
    {
        AssocArrayLiteralExp ae = cast(AssocArrayLiteralExp)e1;
        /* Search the keys backwards, in case there are duplicate keys
         */
        for (size_t i = ae.keys.dim; i;)
        {
            i--;
            Expression ekey = (*ae.keys)[i];
            ue = Equal(TOKequal, loc, Type.tbool, ekey, e2);
            if (CTFEExp.isCantExp(ue.exp()))
                return ue;
            if (ue.exp().isBool(true))
            {
                Expression e = (*ae.values)[i];
                e.type = type;
                e.loc = loc;
                if (hasSideEffect(e))
                    emplaceExp!(CTFEExp)(&ue, TOKcantexp);
                else
                    emplaceExp!(UnionExp)(&ue, e);
                return ue;
            }
        }
        emplaceExp!(CTFEExp)(&ue, TOKcantexp);
    }
    else
        emplaceExp!(CTFEExp)(&ue, TOKcantexp);
    return ue;
}

/* Also return TOKcantexp if this fails
 */
extern (C++) UnionExp Slice(Type type, Expression e1, Expression lwr, Expression upr)
{
    UnionExp ue;
    Loc loc = e1.loc;
    static if (LOG)
    {
        printf("Slice()\n");
        if (lwr)
        {
            printf("\te1 = %s\n", e1.toChars());
            printf("\tlwr = %s\n", lwr.toChars());
            printf("\tupr = %s\n", upr.toChars());
        }
    }
    if (e1.op == TOKstring && lwr.op == TOKint64 && upr.op == TOKint64)
    {
        StringExp es1 = cast(StringExp)e1;
        uinteger_t ilwr = lwr.toInteger();
        uinteger_t iupr = upr.toInteger();
        if (iupr > es1.len || ilwr > iupr)
        {
            e1.error("string slice [%llu .. %llu] is out of bounds", ilwr, iupr);
            emplaceExp!(ErrorExp)(&ue);
        }
        else
        {
            size_t len = cast(size_t)(iupr - ilwr);
            ubyte sz = es1.sz;
            void* s = mem.xmalloc(len * sz);
            memcpy(cast(char*)s, es1.string + ilwr * sz, len * sz);
            emplaceExp!(StringExp)(&ue, loc, s, len, es1.postfix);
            StringExp es = cast(StringExp)ue.exp();
            es.sz = sz;
            es.committed = es1.committed;
            es.type = type;
        }
    }
    else if (e1.op == TOKarrayliteral && lwr.op == TOKint64 && upr.op == TOKint64 && !hasSideEffect(e1))
    {
        ArrayLiteralExp es1 = cast(ArrayLiteralExp)e1;
        uinteger_t ilwr = lwr.toInteger();
        uinteger_t iupr = upr.toInteger();
        if (iupr > es1.elements.dim || ilwr > iupr)
        {
            e1.error("array slice [%llu .. %llu] is out of bounds", ilwr, iupr);
            emplaceExp!(ErrorExp)(&ue);
        }
        else
        {
            auto elements = new Expressions();
            elements.setDim(cast(size_t)(iupr - ilwr));
            memcpy(elements.tdata(), es1.elements.tdata() + ilwr, cast(size_t)(iupr - ilwr) * ((*es1.elements)[0]).sizeof);
            emplaceExp!(ArrayLiteralExp)(&ue, e1.loc, elements);
            ue.exp().type = type;
        }
    }
    else
        emplaceExp!(CTFEExp)(&ue, TOKcantexp);
    assert(ue.exp().type);
    return ue;
}

/* Set a slice of char/integer array literal 'existingAE' from a string 'newval'.
 * existingAE[firstIndex..firstIndex+newval.length] = newval.
 */
extern (C++) void sliceAssignArrayLiteralFromString(ArrayLiteralExp existingAE, const StringExp newval, size_t firstIndex)
{
    const len = newval.len;
    Type elemType = existingAE.type.nextOf();
    foreach (j; 0 .. len)
    {
        const val = newval.getCodeUnit(j);
        (*existingAE.elements)[j + firstIndex] = new IntegerExp(newval.loc, val, elemType);
    }
}

/* Set a slice of string 'existingSE' from a char array literal 'newae'.
 *   existingSE[firstIndex..firstIndex+newae.length] = newae.
 */
extern (C++) void sliceAssignStringFromArrayLiteral(StringExp existingSE, ArrayLiteralExp newae, size_t firstIndex)
{
    assert(existingSE.ownedByCtfe != OWNEDcode);
    foreach (j; 0 .. newae.elements.dim)
    {
        existingSE.setCodeUnit(firstIndex + j, cast(dchar)newae.getElement(j).toInteger());
    }
}

/* Set a slice of string 'existingSE' from a string 'newstr'.
 *   existingSE[firstIndex..firstIndex+newstr.length] = newstr.
 */
extern (C++) void sliceAssignStringFromString(StringExp existingSE, const StringExp newstr, size_t firstIndex)
{
    assert(existingSE.ownedByCtfe != OWNEDcode);
    size_t sz = existingSE.sz;
    assert(sz == newstr.sz);
    memcpy(existingSE.string + firstIndex * sz, newstr.string, sz * newstr.len);
}

/* Compare a string slice with another string slice.
 * Conceptually equivalent to memcmp( se1[lo1..lo1+len],  se2[lo2..lo2+len])
 */
extern (C++) int sliceCmpStringWithString(const StringExp se1, const StringExp se2, size_t lo1, size_t lo2, size_t len)
{
    size_t sz = se1.sz;
    assert(sz == se2.sz);
    return memcmp(se1.string + sz * lo1, se2.string + sz * lo2, sz * len);
}

/* Compare a string slice with an array literal slice
 * Conceptually equivalent to memcmp( se1[lo1..lo1+len],  ae2[lo2..lo2+len])
 */
extern (C++) int sliceCmpStringWithArray(const StringExp se1, ArrayLiteralExp ae2, size_t lo1, size_t lo2, size_t len)
{
    foreach (j; 0 .. len)
    {
        const val2 = cast(dchar)ae2.getElement(j + lo2).toInteger();
        const val1 = se1.getCodeUnit(j + lo1);
        const int c = val1 - val2;
        if (c)
            return c;
    }
    return 0;
}

/* Also return TOKcantexp if this fails
 */
extern (C++) UnionExp Cat(Type type, Expression e1, Expression e2)
{
    UnionExp ue;
    Expression e = CTFEExp.cantexp;
    Loc loc = e1.loc;
    Type t;
    Type t1 = e1.type.toBasetype();
    Type t2 = e2.type.toBasetype();
    //printf("Cat(e1 = %s, e2 = %s)\n", e1->toChars(), e2->toChars());
    //printf("\tt1 = %s, t2 = %s, type = %s\n", t1->toChars(), t2->toChars(), type->toChars());
    if (e1.op == TOKnull && (e2.op == TOKint64 || e2.op == TOKstructliteral))
    {
        e = e2;
        t = t1;
        goto L2;
    }
    else if ((e1.op == TOKint64 || e1.op == TOKstructliteral) && e2.op == TOKnull)
    {
        e = e1;
        t = t2;
    L2:
        Type tn = e.type.toBasetype();
        if (tn.ty == Tchar || tn.ty == Twchar || tn.ty == Tdchar)
        {
            // Create a StringExp
            if (t.nextOf())
                t = t.nextOf().toBasetype();
            ubyte sz = cast(ubyte)t.size();
            dinteger_t v = e.toInteger();
            size_t len = (t.ty == tn.ty) ? 1 : utf_codeLength(sz, cast(dchar)v);
            void* s = mem.xmalloc(len * sz);
            if (t.ty == tn.ty)
                Port.valcpy(s, v, sz);
            else
                utf_encode(sz, s, cast(dchar)v);
            emplaceExp!(StringExp)(&ue, loc, s, len);
            StringExp es = cast(StringExp)ue.exp();
            es.sz = sz;
            es.committed = 1;
        }
        else
        {
            // Create an ArrayLiteralExp
            auto elements = new Expressions();
            elements.push(e);
            emplaceExp!(ArrayLiteralExp)(&ue, e.loc, elements);
        }
        ue.exp().type = type;
        assert(ue.exp().type);
        return ue;
    }
    else if (e1.op == TOKnull && e2.op == TOKnull)
    {
        if (type == e1.type)
        {
            // Handle null ~= null
            if (t1.ty == Tarray && t2 == t1.nextOf())
            {
                emplaceExp!(ArrayLiteralExp)(&ue, e1.loc, e2);
                ue.exp().type = type;
                assert(ue.exp().type);
                return ue;
            }
            else
            {
                emplaceExp!(UnionExp)(&ue, e1);
                assert(ue.exp().type);
                return ue;
            }
        }
        if (type == e2.type)
        {
            emplaceExp!(UnionExp)(&ue, e2);
            assert(ue.exp().type);
            return ue;
        }
        emplaceExp!(NullExp)(&ue, e1.loc, type);
        assert(ue.exp().type);
        return ue;
    }
    else if (e1.op == TOKstring && e2.op == TOKstring)
    {
        // Concatenate the strings
        StringExp es1 = cast(StringExp)e1;
        StringExp es2 = cast(StringExp)e2;
        size_t len = es1.len + es2.len;
        ubyte sz = es1.sz;
        if (sz != es2.sz)
        {
            /* Can happen with:
             *   auto s = "foo"d ~ "bar"c;
             */
            assert(global.errors);
            emplaceExp!(CTFEExp)(&ue, TOKcantexp);
            assert(ue.exp().type);
            return ue;
        }
        void* s = mem.xmalloc(len * sz);
        memcpy(cast(char*)s, es1.string, es1.len * sz);
        memcpy(cast(char*)s + es1.len * sz, es2.string, es2.len * sz);
        emplaceExp!(StringExp)(&ue, loc, s, len);
        StringExp es = cast(StringExp)ue.exp();
        es.sz = sz;
        es.committed = es1.committed | es2.committed;
        es.type = type;
        assert(ue.exp().type);
        return ue;
    }
    else if (e2.op == TOKstring && e1.op == TOKarrayliteral && t1.nextOf().isintegral())
    {
        // [chars] ~ string --> [chars]
        StringExp es = cast(StringExp)e2;
        ArrayLiteralExp ea = cast(ArrayLiteralExp)e1;
        size_t len = es.len + ea.elements.dim;
        auto elems = new Expressions();
        elems.setDim(len);
        for (size_t i = 0; i < ea.elements.dim; ++i)
        {
            (*elems)[i] = ea.getElement(i);
        }
        emplaceExp!(ArrayLiteralExp)(&ue, e1.loc, elems);
        ArrayLiteralExp dest = cast(ArrayLiteralExp)ue.exp();
        dest.type = type;
        sliceAssignArrayLiteralFromString(dest, es, ea.elements.dim);
        assert(ue.exp().type);
        return ue;
    }
    else if (e1.op == TOKstring && e2.op == TOKarrayliteral && t2.nextOf().isintegral())
    {
        // string ~ [chars] --> [chars]
        StringExp es = cast(StringExp)e1;
        ArrayLiteralExp ea = cast(ArrayLiteralExp)e2;
        size_t len = es.len + ea.elements.dim;
        auto elems = new Expressions();
        elems.setDim(len);
        for (size_t i = 0; i < ea.elements.dim; ++i)
        {
            (*elems)[es.len + i] = ea.getElement(i);
        }
        emplaceExp!(ArrayLiteralExp)(&ue, e1.loc, elems);
        ArrayLiteralExp dest = cast(ArrayLiteralExp)ue.exp();
        dest.type = type;
        sliceAssignArrayLiteralFromString(dest, es, 0);
        assert(ue.exp().type);
        return ue;
    }
    else if (e1.op == TOKstring && e2.op == TOKint64)
    {
        // string ~ char --> string
        StringExp es1 = cast(StringExp)e1;
        StringExp es;
        ubyte sz = es1.sz;
        dinteger_t v = e2.toInteger();
        // Is it a concatentation of homogenous types?
        // (char[] ~ char, wchar[]~wchar, or dchar[]~dchar)
        bool homoConcat = (sz == t2.size());
        size_t len = es1.len;
        len += homoConcat ? 1 : utf_codeLength(sz, cast(dchar)v);
        void* s = mem.xmalloc(len * sz);
        memcpy(s, es1.string, es1.len * sz);
        if (homoConcat)
            Port.valcpy(cast(char*)s + (sz * es1.len), v, sz);
        else
            utf_encode(sz, cast(char*)s + (sz * es1.len), cast(dchar)v);
        emplaceExp!(StringExp)(&ue, loc, s, len);
        es = cast(StringExp)ue.exp();
        es.sz = sz;
        es.committed = es1.committed;
        es.type = type;
        assert(ue.exp().type);
        return ue;
    }
    else if (e1.op == TOKint64 && e2.op == TOKstring)
    {
        // Concatenate the strings
        StringExp es2 = cast(StringExp)e2;
        size_t len = 1 + es2.len;
        ubyte sz = es2.sz;
        dinteger_t v = e1.toInteger();
        void* s = mem.xmalloc(len * sz);
        memcpy(cast(char*)s, &v, sz);
        memcpy(cast(char*)s + sz, es2.string, es2.len * sz);
        emplaceExp!(StringExp)(&ue, loc, s, len);
        StringExp es = cast(StringExp)ue.exp();
        es.sz = sz;
        es.committed = es2.committed;
        es.type = type;
        assert(ue.exp().type);
        return ue;
    }
    else if (e1.op == TOKarrayliteral && e2.op == TOKarrayliteral && t1.nextOf().equals(t2.nextOf()))
    {
        // Concatenate the arrays
        auto elems = ArrayLiteralExp.copyElements(e1, e2);

        emplaceExp!(ArrayLiteralExp)(&ue, e1.loc, elems);

        e = ue.exp();
        if (type.toBasetype().ty == Tsarray)
        {
            e.type = t1.nextOf().sarrayOf(elems.dim);
        }
        else
            e.type = type;
        assert(ue.exp().type);
        return ue;
    }
    else if (e1.op == TOKarrayliteral && e2.op == TOKnull && t1.nextOf().equals(t2.nextOf()))
    {
        e = e1;
        goto L3;
    }
    else if (e1.op == TOKnull && e2.op == TOKarrayliteral && t1.nextOf().equals(t2.nextOf()))
    {
        e = e2;
    L3:
        // Concatenate the array with null
        auto elems = ArrayLiteralExp.copyElements(e);

        emplaceExp!(ArrayLiteralExp)(&ue, e.loc, elems);

        e = ue.exp();
        if (type.toBasetype().ty == Tsarray)
        {
            e.type = t1.nextOf().sarrayOf(elems.dim);
        }
        else
            e.type = type;
        assert(ue.exp().type);
        return ue;
    }
    else if ((e1.op == TOKarrayliteral || e1.op == TOKnull) && e1.type.toBasetype().nextOf() && e1.type.toBasetype().nextOf().equals(e2.type))
    {
        auto elems = (e1.op == TOKarrayliteral)
                ? ArrayLiteralExp.copyElements(e1) : new Expressions();
        elems.push(e2);

        emplaceExp!(ArrayLiteralExp)(&ue, e1.loc, elems);

        e = ue.exp();
        if (type.toBasetype().ty == Tsarray)
        {
            e.type = e2.type.sarrayOf(elems.dim);
        }
        else
            e.type = type;
        assert(ue.exp().type);
        return ue;
    }
    else if (e2.op == TOKarrayliteral && e2.type.toBasetype().nextOf().equals(e1.type))
    {
        auto elems = ArrayLiteralExp.copyElements(e1, e2);

        emplaceExp!(ArrayLiteralExp)(&ue, e2.loc, elems);

        e = ue.exp();
        if (type.toBasetype().ty == Tsarray)
        {
            e.type = e1.type.sarrayOf(elems.dim);
        }
        else
            e.type = type;
        assert(ue.exp().type);
        return ue;
    }
    else if (e1.op == TOKnull && e2.op == TOKstring)
    {
        t = e1.type;
        e = e2;
        goto L1;
    }
    else if (e1.op == TOKstring && e2.op == TOKnull)
    {
        e = e1;
        t = e2.type;
    L1:
        Type tb = t.toBasetype();
        if (tb.ty == Tarray && tb.nextOf().equivalent(e.type))
        {
            auto expressions = new Expressions();
            expressions.push(e);
            emplaceExp!(ArrayLiteralExp)(&ue, loc, expressions);
            e = ue.exp();
            e.type = t;
        }
        else
        {
            emplaceExp!(UnionExp)(&ue, e);
            e = ue.exp();
        }
        if (!e.type.equals(type))
        {
            StringExp se = cast(StringExp)e.copy();
            e = se.castTo(null, type);
            emplaceExp!(UnionExp)(&ue, e);
            e = ue.exp();
        }
    }
    else
        emplaceExp!(CTFEExp)(&ue, TOKcantexp);
    assert(ue.exp().type);
    return ue;
}

extern (C++) UnionExp Ptr(Type type, Expression e1)
{
    //printf("Ptr(e1 = %s)\n", e1->toChars());
    UnionExp ue;
    if (e1.op == TOKadd)
    {
        AddExp ae = cast(AddExp)e1;
        if (ae.e1.op == TOKaddress && ae.e2.op == TOKint64)
        {
            AddrExp ade = cast(AddrExp)ae.e1;
            if (ade.e1.op == TOKstructliteral)
            {
                StructLiteralExp se = cast(StructLiteralExp)ade.e1;
                uint offset = cast(uint)ae.e2.toInteger();
                Expression e = se.getField(type, offset);
                if (e)
                {
                    emplaceExp!(UnionExp)(&ue, e);
                    return ue;
                }
            }
        }
    }
    emplaceExp!(CTFEExp)(&ue, TOKcantexp);
    return ue;
}
