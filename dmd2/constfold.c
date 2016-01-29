
/* Compiler implementation of the D programming language
 * Copyright (c) 1999-2014 by Digital Mars
 * All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/D-Programming-Language/dmd/blob/master/src/constfold.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>                     // mem{cpy|set|cmp}()
#include <math.h>
#include <new>

#include "rmem.h"
#include "root.h"
#include "port.h"

#include "mtype.h"
#include "expression.h"
#include "aggregate.h"
#include "declaration.h"
#include "utf.h"
#include "ctfe.h"

#if IN_LLVM
#include "gen/pragma.h"
#endif

#define LOG 0

int RealEquals(real_t x1, real_t x2);

Expression *expType(Type *type, Expression *e)
{
    if (type != e->type)
    {
        e = e->copy();
        e->type = type;
    }
    return e;
}

/* ================================== isConst() ============================== */

int isConst(Expression *e)
{
    //printf("Expression::isConst(): %s\n", e->toChars());
    switch (e->op)
    {
        case TOKint64:
        case TOKfloat64:
        case TOKcomplex80:
            return 1;
        case TOKnull:
            return 0;
        case TOKsymoff:
#if IN_LLVM
            // We don't statically know anything about the address of a weak symbol
            // if there is no offset. With an offset, we can at least say that it is
            // non-zero.
            {
                SymOffExp *soe = (SymOffExp*)e;
                if (soe->var->llvmInternal == LLVMextern_weak && !soe->offset)
                    return 0;
            }
#endif
            return 2;
        default:
            return 0;
    }
    assert(0);
    return 0;
}

/* =============================== constFold() ============================== */

/* The constFold() functions were redundant with the optimize() ones,
 * and so have been folded in with them.
 */

/* ========================================================================== */

UnionExp Neg(Type *type, Expression *e1)
{
    UnionExp ue;
    Loc loc = e1->loc;

    if (e1->type->isreal())
    {
        new(&ue) RealExp(loc, -e1->toReal(), type);
    }
    else if (e1->type->isimaginary())
    {
        new(&ue) RealExp(loc, -e1->toImaginary(), type);
    }
    else if (e1->type->iscomplex())
    {
        new(&ue) ComplexExp(loc, -e1->toComplex(), type);
    }
    else
    {
        new(&ue) IntegerExp(loc, -e1->toInteger(), type);
    }
    return ue;
}

UnionExp Com(Type *type, Expression *e1)
{
    UnionExp ue;
    Loc loc = e1->loc;

    new(&ue) IntegerExp(loc, ~e1->toInteger(), type);
    return ue;
}

UnionExp Not(Type *type, Expression *e1)
{
    UnionExp ue;
    Loc loc = e1->loc;

    new(&ue) IntegerExp(loc, e1->isBool(false) ? 1 : 0, type);
    return ue;
}

UnionExp Bool(Type *type, Expression *e1)
{
    UnionExp ue;
    Loc loc = e1->loc;

    new(&ue) IntegerExp(loc, e1->isBool(true) ? 1 : 0, type);
    return ue;
}

UnionExp Add(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Loc loc = e1->loc;

#if LOG
    printf("Add(e1 = %s, e2 = %s)\n", e1->toChars(), e2->toChars());
#endif
    if (type->isreal())
    {
        new(&ue) RealExp(loc, e1->toReal() + e2->toReal(), type);
    }
    else if (type->isimaginary())
    {
        new(&ue) RealExp(loc, e1->toImaginary() + e2->toImaginary(), type);
    }
    else if (type->iscomplex())
    {
        // This rigamarole is necessary so that -0.0 doesn't get
        // converted to +0.0 by doing an extraneous add with +0.0
        complex_t c1;
        real_t r1 = ldouble (0.0);
        real_t i1 = ldouble (0.0);

        complex_t c2;
        real_t r2 = ldouble (0.0);
        real_t i2 = ldouble (0.0);

        complex_t v;
        int x;

        if (e1->type->isreal())
        {
            r1 = e1->toReal();
            x = 0;
        }
        else if (e1->type->isimaginary())
        {
            i1 = e1->toImaginary();
            x = 3;
        }
        else
        {
            c1 = e1->toComplex();
            x = 6;
        }

        if (e2->type->isreal())
        {
            r2 = e2->toReal();
        }
        else if (e2->type->isimaginary())
        {
            i2 = e2->toImaginary();
            x += 1;
        }
        else
        {
            c2 = e2->toComplex();
            x += 2;
        }

        switch (x)
        {
            case 0+0:   v = complex_t(r1 + r2, 0);      break;
            case 0+1:   v = complex_t(r1, i2);          break;
            case 0+2:   v = complex_t(r1 + creall(c2), cimagl(c2));     break;
            case 3+0:   v = complex_t(r2, i1);          break;
            case 3+1:   v = complex_t(0, i1 + i2);      break;
            case 3+2:   v = complex_t(creall(c2), i1 + cimagl(c2));     break;
            case 6+0:   v = complex_t(creall(c1) + r2, cimagl(c2));     break;
            case 6+1:   v = complex_t(creall(c1), cimagl(c1) + i2);     break;
            case 6+2:   v = c1 + c2;                    break;
            default: assert(0);
        }
        new(&ue) ComplexExp(loc, v, type);
    }
    else if (e1->op == TOKsymoff)
    {
        SymOffExp *soe = (SymOffExp *)e1;
        new(&ue) SymOffExp(loc, soe->var, soe->offset + e2->toInteger());
        ue.exp()->type = type;
    }
    else if (e2->op == TOKsymoff)
    {
        SymOffExp *soe = (SymOffExp *)e2;
        new(&ue) SymOffExp(loc, soe->var, soe->offset + e1->toInteger());
        ue.exp()->type = type;
    }
    else
        new(&ue) IntegerExp(loc, e1->toInteger() + e2->toInteger(), type);
    return ue;
}


UnionExp Min(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Loc loc = e1->loc;

    if (type->isreal())
    {
        new(&ue) RealExp(loc, e1->toReal() - e2->toReal(), type);
    }
    else if (type->isimaginary())
    {
        new(&ue) RealExp(loc, e1->toImaginary() - e2->toImaginary(), type);
    }
    else if (type->iscomplex())
    {
        // This rigamarole is necessary so that -0.0 doesn't get
        // converted to +0.0 by doing an extraneous add with +0.0
        complex_t c1;
        real_t r1 = ldouble (0.0);
        real_t i1 = ldouble (0.0);

        complex_t c2;
        real_t r2 = ldouble (0.0);
        real_t i2 = ldouble (0.0);

        complex_t v;
        int x;

        if (e1->type->isreal())
        {
            r1 = e1->toReal();
            x = 0;
        }
        else if (e1->type->isimaginary())
        {
            i1 = e1->toImaginary();
            x = 3;
        }
        else
        {
            c1 = e1->toComplex();
            x = 6;
        }

        if (e2->type->isreal())
        {
            r2 = e2->toReal();
        }
        else if (e2->type->isimaginary())
        {
            i2 = e2->toImaginary();
            x += 1;
        }
        else
        {
            c2 = e2->toComplex();
            x += 2;
        }

        switch (x)
        {
            case 0+0:   v = complex_t(r1 - r2, 0);      break;
            case 0+1:   v = complex_t(r1, -i2);         break;
            case 0+2:   v = complex_t(r1 - creall(c2), -cimagl(c2));    break;
            case 3+0:   v = complex_t(-r2, i1);         break;
            case 3+1:   v = complex_t(0, i1 - i2);      break;
            case 3+2:   v = complex_t(-creall(c2), i1 - cimagl(c2));    break;
            case 6+0:   v = complex_t(creall(c1) - r2, cimagl(c1));     break;
            case 6+1:   v = complex_t(creall(c1), cimagl(c1) - i2);     break;
            case 6+2:   v = c1 - c2;                    break;
            default: assert(0);
        }
        new(&ue) ComplexExp(loc, v, type);
    }
    else if (e1->op == TOKsymoff)
    {
        SymOffExp *soe = (SymOffExp *)e1;
        new(&ue) SymOffExp(loc, soe->var, soe->offset - e2->toInteger());
        ue.exp()->type = type;
    }
    else
    {
        new(&ue) IntegerExp(loc, e1->toInteger() - e2->toInteger(), type);
    }
    return ue;
}

UnionExp Mul(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Loc loc = e1->loc;

    if (type->isfloating())
    {
        complex_t c;
        d_float80 r;

        if (e1->type->isreal())
        {
            r = e1->toReal();
            c = e2->toComplex();
            c = complex_t(r * creall(c), r * cimagl(c));
        }
        else if (e1->type->isimaginary())
        {
            r = e1->toImaginary();
            c = e2->toComplex();
            c = complex_t(-r * cimagl(c), r * creall(c));
        }
        else if (e2->type->isreal())
        {
            r = e2->toReal();
            c = e1->toComplex();
            c = complex_t(r * creall(c), r * cimagl(c));
        }
        else if (e2->type->isimaginary())
        {
            r = e2->toImaginary();
            c = e1->toComplex();
            c = complex_t(-r * cimagl(c), r * creall(c));
        }
        else
            c = e1->toComplex() * e2->toComplex();

        if (type->isreal())
            new(&ue) RealExp(loc, creall(c), type);
        else if (type->isimaginary())
            new(&ue) RealExp(loc, cimagl(c), type);
        else if (type->iscomplex())
            new(&ue) ComplexExp(loc, c, type);
        else
            assert(0);
    }
    else
    {
        new(&ue) IntegerExp(loc, e1->toInteger() * e2->toInteger(), type);
    }
    return ue;
}

UnionExp Div(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Loc loc = e1->loc;

    if (type->isfloating())
    {
        complex_t c;
        d_float80 r;

        //e1->type->print();
        //e2->type->print();
        if (e2->type->isreal())
        {
            if (e1->type->isreal())
            {
                new(&ue) RealExp(loc, e1->toReal() / e2->toReal(), type);
                return ue;
            }
            r = e2->toReal();
            c = e1->toComplex();
            c = complex_t(creall(c) / r, cimagl(c) / r);
        }
        else if (e2->type->isimaginary())
        {
            r = e2->toImaginary();
            c = e1->toComplex();
            c = complex_t(cimagl(c) / r, -creall(c) / r);
        }
        else
        {
            c = e1->toComplex() / e2->toComplex();
        }

        if (type->isreal())
            new(&ue) RealExp(loc, creall(c), type);
        else if (type->isimaginary())
            new(&ue) RealExp(loc, cimagl(c), type);
        else if (type->iscomplex())
            new(&ue) ComplexExp(loc, c, type);
        else
            assert(0);
    }
    else
    {
        sinteger_t n1;
        sinteger_t n2;
        sinteger_t n;

        n1 = e1->toInteger();
        n2 = e2->toInteger();
        if (n2 == 0)
        {
            e2->error("divide by 0");
            n2 = 1;
        }
        if (e1->type->isunsigned() || e2->type->isunsigned())
            n = ((dinteger_t) n1) / ((dinteger_t) n2);
        else
            n = n1 / n2;
        new(&ue) IntegerExp(loc, n, type);
    }
    return ue;
}

UnionExp Mod(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Loc loc = e1->loc;

    if (type->isfloating())
    {
        complex_t c;

        if (e2->type->isreal())
        {
            real_t r2 = e2->toReal();

            c = complex_t(Port::fmodl(e1->toReal(), r2), Port::fmodl(e1->toImaginary(), r2));
        }
        else if (e2->type->isimaginary())
        {
            real_t i2 = e2->toImaginary();

            c = complex_t(Port::fmodl(e1->toReal(), i2), Port::fmodl(e1->toImaginary(), i2));
        }
        else
            assert(0);

        if (type->isreal())
            new(&ue) RealExp(loc, creall(c), type);
        else if (type->isimaginary())
            new(&ue) RealExp(loc, cimagl(c), type);
        else if (type->iscomplex())
            new(&ue) ComplexExp(loc, c, type);
        else
            assert(0);
    }
    else
    {
        sinteger_t n1;
        sinteger_t n2;
        sinteger_t n;

        n1 = e1->toInteger();
        n2 = e2->toInteger();
        if (n2 == 0)
        {
            e2->error("divide by 0");
            n2 = 1;
        }
        if (n2 == -1 && !type->isunsigned())
        {
            // Check for int.min % -1
            if (n1 == 0xFFFFFFFF80000000ULL && type->toBasetype()->ty != Tint64)
            {
                e2->error("integer overflow: int.min % -1");
                n2 = 1;
            }
            else if (n1 == 0x8000000000000000LL) // long.min % -1
            {
                e2->error("integer overflow: long.min % -1");
                n2 = 1;
            }
        }
        if (e1->type->isunsigned() || e2->type->isunsigned())
            n = ((dinteger_t) n1) % ((dinteger_t) n2);
        else
            n = n1 % n2;
        new(&ue) IntegerExp(loc, n, type);
    }
    return ue;
}

UnionExp Pow(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Loc loc = e1->loc;

    // Handle integer power operations.
    if (e2->type->isintegral())
    {
        dinteger_t n = e2->toInteger();
        bool neg;

        if (!e2->type->isunsigned() && (sinteger_t)n < 0)
        {
            if (e1->type->isintegral())
            {
                new(&ue) CTFEExp(TOKcantexp);
                return ue;
            }

            // Don't worry about overflow, from now on n is unsigned.
            neg = true;
            n = -n;
        }
        else
            neg = false;

        UnionExp ur, uv;
        if (e1->type->iscomplex())
        {
            new(&ur) ComplexExp(loc, e1->toComplex(), e1->type);
            new(&uv) ComplexExp(loc, complex_t(1.0, 0.0), e1->type);
        }
        else if (e1->type->isfloating())
        {
            new(&ur) RealExp(loc, e1->toReal(), e1->type);
            new(&uv) RealExp(loc, ldouble(1.0), e1->type);
        }
        else
        {
            new(&ur) IntegerExp(loc, e1->toInteger(), e1->type);
            new(&uv) IntegerExp(loc, 1, e1->type);
        }

        Expression* r = ur.exp();
        Expression* v = uv.exp();
        while (n != 0)
        {
            if (n & 1)
            {
                // v = v * r;
                uv = Mul(v->type, v, r);
            }
            n >>= 1;
            // r = r * r
            ur = Mul(r->type, r, r);
        }

        if (neg)
        {
            // ue = 1.0 / v
            UnionExp one;
            new(&one) RealExp(loc, ldouble(1.0), v->type);
            uv = Div(v->type, one.exp(), v);
        }

        if (type->iscomplex())
            new(&ue) ComplexExp(loc, v->toComplex(), type);
        else if (type->isintegral())
            new(&ue) IntegerExp(loc, v->toInteger(), type);
        else
            new(&ue) RealExp(loc, v->toReal(), type);
    }
    else if (e2->type->isfloating())
    {
        // x ^^ y for x < 0 and y not an integer is not defined; so set result as NaN
        if (e1->toReal() < 0.0)
        {
            new(&ue) RealExp(loc, Port::ldbl_nan, type);
        }
        else
            new(&ue) CTFEExp(TOKcantexp);
    }
    else
        new(&ue) CTFEExp(TOKcantexp);

    return ue;
}

UnionExp Shl(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Loc loc = e1->loc;

    new(&ue) IntegerExp(loc, e1->toInteger() << e2->toInteger(), type);
    return ue;
}

UnionExp Shr(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Loc loc = e1->loc;

    dinteger_t value = e1->toInteger();
    dinteger_t dcount = e2->toInteger();
    assert(dcount <= 0xFFFFFFFF);
    unsigned count = (unsigned)dcount;
    switch (e1->type->toBasetype()->ty)
    {
        case Tint8:
                value = (d_int8)(value) >> count;
                break;

        case Tuns8:
        case Tchar:
                value = (d_uns8)(value) >> count;
                break;

        case Tint16:
                value = (d_int16)(value) >> count;
                break;

        case Tuns16:
        case Twchar:
                value = (d_uns16)(value) >> count;
                break;

        case Tint32:
                value = (d_int32)(value) >> count;
                break;

        case Tuns32:
        case Tdchar:
                value = (d_uns32)(value) >> count;
                break;

        case Tint64:
                value = (d_int64)(value) >> count;
                break;

        case Tuns64:
                value = (d_uns64)(value) >> count;
                break;

        case Terror:
                new(&ue) ErrorExp();
                return ue;

        default:
                assert(0);
    }
    new(&ue) IntegerExp(loc, value, type);
    return ue;
}

UnionExp Ushr(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Loc loc = e1->loc;

    dinteger_t value = e1->toInteger();
    dinteger_t dcount = e2->toInteger();
    assert(dcount <= 0xFFFFFFFF);
    unsigned count = (unsigned)dcount;
    switch (e1->type->toBasetype()->ty)
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
                value = (d_uns64)(value) >> count;
                break;

        case Terror:
                new(&ue) ErrorExp();
                return ue;

        default:
                assert(0);
    }
    new(&ue) IntegerExp(loc, value, type);
    return ue;
}

UnionExp And(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    new(&ue) IntegerExp(e1->loc, e1->toInteger() & e2->toInteger(), type);
    return ue;
}

UnionExp Or(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    new(&ue) IntegerExp(e1->loc, e1->toInteger() | e2->toInteger(), type);
    return ue;
}

UnionExp Xor(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    new(&ue) IntegerExp(e1->loc, e1->toInteger() ^ e2->toInteger(), type);
    return ue;
}

/* Also returns TOKcantexp if cannot be computed.
 */
UnionExp Equal(TOK op, Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Loc loc = e1->loc;
    int cmp = 0;
    real_t r1;
    real_t r2;

    //printf("Equal(e1 = %s, e2 = %s)\n", e1->toChars(), e2->toChars());

    assert(op == TOKequal || op == TOKnotequal);

    if (e1->op == TOKnull)
    {
        if (e2->op == TOKnull)
            cmp = 1;
        else if (e2->op == TOKstring)
        {
            StringExp *es2 = (StringExp *)e2;
            cmp = (0 == es2->len);
        }
        else if (e2->op == TOKarrayliteral)
        {
            ArrayLiteralExp *es2 = (ArrayLiteralExp *)e2;
            cmp = !es2->elements || (0 == es2->elements->dim);
        }
        else
        {
            new(&ue) CTFEExp(TOKcantexp);
            return ue;
        }
    }
    else if (e2->op == TOKnull)
    {
        if (e1->op == TOKstring)
        {
            StringExp *es1 = (StringExp *)e1;
            cmp = (0 == es1->len);
        }
        else if (e1->op == TOKarrayliteral)
        {
            ArrayLiteralExp *es1 = (ArrayLiteralExp *)e1;
            cmp = !es1->elements || (0 == es1->elements->dim);
        }
        else
        {
            new(&ue) CTFEExp(TOKcantexp);
            return ue;
        }
    }
    else if (e1->op == TOKstring && e2->op == TOKstring)
    {
        StringExp *es1 = (StringExp *)e1;
        StringExp *es2 = (StringExp *)e2;

        if (es1->sz != es2->sz)
        {
            assert(global.errors);
            new(&ue) CTFEExp(TOKcantexp);
            return ue;
        }
        if (es1->len == es2->len &&
            memcmp(es1->string, es2->string, es1->sz * es1->len) == 0)
            cmp = 1;
        else
            cmp = 0;
    }
    else if (e1->op == TOKarrayliteral && e2->op == TOKarrayliteral)
    {
        ArrayLiteralExp *es1 = (ArrayLiteralExp *)e1;
        ArrayLiteralExp *es2 = (ArrayLiteralExp *)e2;

        if ((!es1->elements || !es1->elements->dim) &&
            (!es2->elements || !es2->elements->dim))
            cmp = 1;            // both arrays are empty
        else if (!es1->elements || !es2->elements)
            cmp = 0;
        else if (es1->elements->dim != es2->elements->dim)
            cmp = 0;
        else
        {
            for (size_t i = 0; i < es1->elements->dim; i++)
            {
                Expression *ee1 = (*es1->elements)[i];
                Expression *ee2 = (*es2->elements)[i];

                ue = Equal(TOKequal, Type::tint32, ee1, ee2);
                if (CTFEExp::isCantExp(ue.exp()))
                    return ue;
                cmp = (int)ue.exp()->toInteger();
                if (cmp == 0)
                    break;
            }
        }
    }
    else if (e1->op == TOKarrayliteral && e2->op == TOKstring)
    {
        // Swap operands and use common code
        Expression *etmp = e1;
        e1 = e2;
        e2 = etmp;
        goto Lsa;
    }
    else if (e1->op == TOKstring && e2->op == TOKarrayliteral)
    {
     Lsa:
        StringExp *es1 = (StringExp *)e1;
        ArrayLiteralExp *es2 = (ArrayLiteralExp *)e2;
        size_t dim1 = es1->len;
        size_t dim2 = es2->elements ? es2->elements->dim : 0;
        if (dim1 != dim2)
            cmp = 0;
        else
        {
            cmp = 1;            // if dim1 winds up being 0
            for (size_t i = 0; i < dim1; i++)
            {
                uinteger_t c = es1->charAt(i);
                Expression *ee2 = (*es2->elements)[i];
                if (ee2->isConst() != 1)
                {
                    new(&ue) CTFEExp(TOKcantexp);
                    return ue;
                }
                cmp = (c == ee2->toInteger());
                if (cmp == 0)
                    break;
            }
        }
    }
    else if (e1->op == TOKstructliteral && e2->op == TOKstructliteral)
    {
        StructLiteralExp *es1 = (StructLiteralExp *)e1;
        StructLiteralExp *es2 = (StructLiteralExp *)e2;

        if (es1->sd != es2->sd)
            cmp = 0;
        else if ((!es1->elements || !es1->elements->dim) &&
            (!es2->elements || !es2->elements->dim))
            cmp = 1;            // both arrays are empty
        else if (!es1->elements || !es2->elements)
            cmp = 0;
        else if (es1->elements->dim != es2->elements->dim)
            cmp = 0;
        else
        {
            cmp = 1;
            for (size_t i = 0; i < es1->elements->dim; i++)
            {
                Expression *ee1 = (*es1->elements)[i];
                Expression *ee2 = (*es2->elements)[i];

                if (ee1 == ee2)
                    continue;
                if (!ee1 || !ee2)
                {
                    cmp = 0;
                    break;
                }
                ue = Equal(TOKequal, Type::tint32, ee1, ee2);
                if (ue.exp()->op == TOKcantexp)
                    return ue;
                cmp = (int)ue.exp()->toInteger();
                if (cmp == 0)
                    break;
            }
        }
        if (cmp && es1->type->needsNested())
        {
            if ((es1->sinit != NULL) != (es2->sinit != NULL))
                cmp = 0;
        }
    }
    else if (e1->isConst() != 1 || e2->isConst() != 1)
    {
        new(&ue) CTFEExp(TOKcantexp);
        return ue;
    }
    else if (e1->type->isreal())
    {
        r1 = e1->toReal();
        r2 = e2->toReal();
        goto L1;
    }
    else if (e1->type->isimaginary())
    {
        r1 = e1->toImaginary();
        r2 = e2->toImaginary();
     L1:
        if (Port::isNan(r1) || Port::isNan(r2)) // if unordered
        {
            cmp = 0;
        }
        else
        {
            cmp = (r1 == r2);
        }
    }
    else if (e1->type->iscomplex())
    {
        cmp = e1->toComplex() == e2->toComplex();
    }
    else if (e1->type->isintegral() || e1->type->toBasetype()->ty == Tpointer)
    {
        cmp = (e1->toInteger() == e2->toInteger());
    }
    else
    {
        new(&ue) CTFEExp(TOKcantexp);
        return ue;
    }

    if (op == TOKnotequal)
        cmp ^= 1;
    new(&ue) IntegerExp(loc, cmp, type);
    return ue;
}

UnionExp Identity(TOK op, Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Loc loc = e1->loc;
    int cmp;

    if (e1->op == TOKnull)
    {
        cmp = (e2->op == TOKnull);
    }
    else if (e2->op == TOKnull)
    {
        cmp = 0;
    }
    else if (e1->op == TOKsymoff && e2->op == TOKsymoff)
    {
        SymOffExp *es1 = (SymOffExp *)e1;
        SymOffExp *es2 = (SymOffExp *)e2;

        cmp = (es1->var == es2->var && es1->offset == es2->offset);
    }
    else
    {
       if (e1->type->isreal())
       {
           cmp = RealEquals(e1->toReal(), e2->toReal());
       }
       else if (e1->type->isimaginary())
       {
           cmp = RealEquals(e1->toImaginary(), e2->toImaginary());
       }
       else if (e1->type->iscomplex())
       {
           complex_t v1 = e1->toComplex();
           complex_t v2 = e2->toComplex();
           cmp = RealEquals(creall(v1), creall(v2)) &&
                 RealEquals(cimagl(v1), cimagl(v1));
        }
        else
        {
           ue = Equal((op == TOKidentity) ? TOKequal : TOKnotequal,
                   type, e1, e2);
           return ue;
        }
    }
    if (op == TOKnotidentity)
        cmp ^= 1;
    new(&ue) IntegerExp(loc, cmp, type);
    return ue;
}


UnionExp Cmp(TOK op, Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Loc loc = e1->loc;
    dinteger_t n;
    real_t r1;
    real_t r2;

    //printf("Cmp(e1 = %s, e2 = %s)\n", e1->toChars(), e2->toChars());

    if (e1->op == TOKstring && e2->op == TOKstring)
    {
        StringExp *es1 = (StringExp *)e1;
        StringExp *es2 = (StringExp *)e2;
        size_t sz = es1->sz;
        assert(sz == es2->sz);

        size_t len = es1->len;
        if (es2->len < len)
            len = es2->len;

        int cmp = memcmp(es1->string, es2->string, sz * len);
        if (cmp == 0)
            cmp = (int)(es1->len - es2->len);

        switch (op)
        {
            case TOKlt: n = cmp <  0;   break;
            case TOKle: n = cmp <= 0;   break;
            case TOKgt: n = cmp >  0;   break;
            case TOKge: n = cmp >= 0;   break;

            case TOKleg:   n = 1;               break;
            case TOKlg:    n = cmp != 0;        break;
            case TOKunord: n = 0;               break;
            case TOKue:    n = cmp == 0;        break;
            case TOKug:    n = cmp >  0;        break;
            case TOKuge:   n = cmp >= 0;        break;
            case TOKul:    n = cmp <  0;        break;
            case TOKule:   n = cmp <= 0;        break;

            default:
                assert(0);
        }
    }
    else if (e1->isConst() != 1 || e2->isConst() != 1)
    {
        new(&ue) CTFEExp(TOKcantexp);
        return ue;
    }
    else if (e1->type->isreal())
    {
        r1 = e1->toReal();
        r2 = e2->toReal();
        goto L1;
    }
    else if (e1->type->isimaginary())
    {
        r1 = e1->toImaginary();
        r2 = e2->toImaginary();
     L1:
        // Don't rely on compiler, handle NAN arguments separately
        // (DMC does do it correctly)
        if (Port::isNan(r1) || Port::isNan(r2)) // if unordered
        {
            switch (op)
            {
                case TOKlt:     n = 0;  break;
                case TOKle:     n = 0;  break;
                case TOKgt:     n = 0;  break;
                case TOKge:     n = 0;  break;

                case TOKleg:    n = 0;  break;
                case TOKlg:     n = 0;  break;
                case TOKunord:  n = 1;  break;
                case TOKue:     n = 1;  break;
                case TOKug:     n = 1;  break;
                case TOKuge:    n = 1;  break;
                case TOKul:     n = 1;  break;
                case TOKule:    n = 1;  break;

                default:
                    assert(0);
            }
        }
        else
        {
            switch (op)
            {
                case TOKlt:     n = r1 <  r2;   break;
                case TOKle:     n = r1 <= r2;   break;
                case TOKgt:     n = r1 >  r2;   break;
                case TOKge:     n = r1 >= r2;   break;

                case TOKleg:    n = 1;          break;
                case TOKlg:     n = r1 != r2;   break;
                case TOKunord:  n = 0;          break;
                case TOKue:     n = r1 == r2;   break;
                case TOKug:     n = r1 >  r2;   break;
                case TOKuge:    n = r1 >= r2;   break;
                case TOKul:     n = r1 <  r2;   break;
                case TOKule:    n = r1 <= r2;   break;

                default:
                    assert(0);
            }
        }
    }
    else if (e1->type->iscomplex())
    {
        assert(0);
    }
    else
    {
        sinteger_t n1;
        sinteger_t n2;

        n1 = e1->toInteger();
        n2 = e2->toInteger();
        if (e1->type->isunsigned() || e2->type->isunsigned())
        {
            switch (op)
            {
                case TOKlt:     n = ((dinteger_t) n1) <  ((dinteger_t) n2);   break;
                case TOKle:     n = ((dinteger_t) n1) <= ((dinteger_t) n2);   break;
                case TOKgt:     n = ((dinteger_t) n1) >  ((dinteger_t) n2);   break;
                case TOKge:     n = ((dinteger_t) n1) >= ((dinteger_t) n2);   break;

                case TOKleg:    n = 1;                                        break;
                case TOKlg:     n = ((dinteger_t) n1) != ((dinteger_t) n2);   break;
                case TOKunord:  n = 0;                                        break;
                case TOKue:     n = ((dinteger_t) n1) == ((dinteger_t) n2);   break;
                case TOKug:     n = ((dinteger_t) n1) >  ((dinteger_t) n2);   break;
                case TOKuge:    n = ((dinteger_t) n1) >= ((dinteger_t) n2);   break;
                case TOKul:     n = ((dinteger_t) n1) <  ((dinteger_t) n2);   break;
                case TOKule:    n = ((dinteger_t) n1) <= ((dinteger_t) n2);   break;

                default:
                    assert(0);
            }
        }
        else
        {
            switch (op)
            {
                case TOKlt:     n = n1 <  n2;   break;
                case TOKle:     n = n1 <= n2;   break;
                case TOKgt:     n = n1 >  n2;   break;
                case TOKge:     n = n1 >= n2;   break;

                case TOKleg:    n = 1;          break;
                case TOKlg:     n = n1 != n2;   break;
                case TOKunord:  n = 0;          break;
                case TOKue:     n = n1 == n2;   break;
                case TOKug:     n = n1 >  n2;   break;
                case TOKuge:    n = n1 >= n2;   break;
                case TOKul:     n = n1 <  n2;   break;
                case TOKule:    n = n1 <= n2;   break;

                default:
                    assert(0);
            }
        }
    }
    new(&ue) IntegerExp(loc, n, type);
    return ue;
}

/* Also returns TOKcantexp if cannot be computed.
 *  to: type to cast to
 *  type: type to paint the result
 */

UnionExp Cast(Type *type, Type *to, Expression *e1)
{
    UnionExp ue;
    Loc loc = e1->loc;

    Type *tb = to->toBasetype();
    Type *typeb = type->toBasetype();

    //printf("Cast(type = %s, to = %s, e1 = %s)\n", type->toChars(), to->toChars(), e1->toChars());
    //printf("\te1->type = %s\n", e1->type->toChars());
    if (e1->type->equals(type) && type->equals(to))
    {
        new(&ue) UnionExp(e1);
        return ue;
    }

    if (e1->op == TOKvector && ((TypeVector *)e1->type)->basetype->equals(type) && type->equals(to))
    {
        Expression *ex = ((VectorExp *)e1)->e1;
        new(&ue) UnionExp(ex);
        return ue;
    }

    if (e1->type->implicitConvTo(to) >= MATCHconst ||
        to->implicitConvTo(e1->type) >= MATCHconst)
    {
        goto L1;
    }

    // Allow covariant converions of delegates
    // (Perhaps implicit conversion from pure to impure should be a MATCHconst,
    // then we wouldn't need this extra check.)
    if (e1->type->toBasetype()->ty == Tdelegate &&
        e1->type->implicitConvTo(to) == MATCHconvert)
    {
        goto L1;
    }

    /* Allow casting from one string type to another
     */
    if (e1->op == TOKstring)
    {
        if (tb->ty == Tarray && typeb->ty == Tarray &&
            tb->nextOf()->size() == typeb->nextOf()->size())
        {
            goto L1;
        }
    }

    if (e1->op == TOKarrayliteral && typeb == tb)
    {
L1:
        Expression *ex = expType(to, e1);
        new(&ue) UnionExp(ex);
        return ue;
    }

    if (e1->isConst() != 1)
    {
        new(&ue) CTFEExp(TOKcantexp);
    }
    else if (tb->ty == Tbool)
    {
        new(&ue) IntegerExp(loc, e1->toInteger() != 0, type);
    }
    else if (type->isintegral())
    {
        if (e1->type->isfloating())
        {
            dinteger_t result;
            real_t r = e1->toReal();

            switch (typeb->ty)
            {
                case Tint8:     result = (d_int8)r;     break;
                case Tchar:
                case Tuns8:     result = (d_uns8)r;     break;
                case Tint16:    result = (d_int16)r;    break;
                case Twchar:
                case Tuns16:    result = (d_uns16)r;    break;
                case Tint32:    result = (d_int32)r;    break;
                case Tdchar:
                case Tuns32:    result = (d_uns32)r;    break;
                case Tint64:    result = (d_int64)r;    break;
                case Tuns64:    result = (d_uns64)r;    break;
                default:
                    assert(0);
            }

            new(&ue) IntegerExp(loc, result, type);
        }
        else if (type->isunsigned())
            new(&ue) IntegerExp(loc, e1->toUInteger(), type);
        else
            new(&ue) IntegerExp(loc, e1->toInteger(), type);
    }
    else if (tb->isreal())
    {
        real_t value = e1->toReal();

        new(&ue) RealExp(loc, value, type);
    }
    else if (tb->isimaginary())
    {
        real_t value = e1->toImaginary();

        new(&ue) RealExp(loc, value, type);
    }
    else if (tb->iscomplex())
    {
        complex_t value = e1->toComplex();

        new(&ue) ComplexExp(loc, value, type);
    }
    else if (tb->isscalar())
    {
        new(&ue) IntegerExp(loc, e1->toInteger(), type);
    }
    else if (tb->ty == Tvoid)
    {
        new(&ue) CTFEExp(TOKcantexp);
    }
    else if (tb->ty == Tstruct && e1->op == TOKint64)
    {
        // Struct = 0;
        StructDeclaration *sd = tb->toDsymbol(NULL)->isStructDeclaration();
        assert(sd);
        Expressions *elements = new Expressions;
        for (size_t i = 0; i < sd->fields.dim; i++)
        {
            VarDeclaration *v = sd->fields[i];
            UnionExp zero;
            new(&zero) IntegerExp(0);
            ue = Cast(v->type, v->type, zero.exp());
            if (ue.exp()->op == TOKcantexp)
                return ue;
            elements->push(ue.exp()->copy());
        }
        new(&ue) StructLiteralExp(loc, sd, elements);
        ue.exp()->type = type;
    }
    else
    {
        if (type != Type::terror)
        {
            // have to change to Internal Compiler Error
            // all invalid casts should be handled already in Expression::castTo().
            error(loc, "cannot cast %s to %s", e1->type->toChars(), type->toChars());
        }
        new(&ue) ErrorExp();
    }
    return ue;
}


UnionExp ArrayLength(Type *type, Expression *e1)
{
    UnionExp ue;
    Loc loc = e1->loc;

    if (e1->op == TOKstring)
    {
        StringExp *es1 = (StringExp *)e1;

        new(&ue) IntegerExp(loc, es1->len, type);
    }
    else if (e1->op == TOKarrayliteral)
    {
        ArrayLiteralExp *ale = (ArrayLiteralExp *)e1;
        size_t dim = ale->elements ? ale->elements->dim : 0;

        new(&ue) IntegerExp(loc, dim, type);
    }
    else if (e1->op == TOKassocarrayliteral)
    {
        AssocArrayLiteralExp *ale = (AssocArrayLiteralExp *)e1;
        size_t dim = ale->keys->dim;

        new(&ue) IntegerExp(loc, dim, type);
    }
    else if (e1->type->toBasetype()->ty == Tsarray)
    {
        Expression *e = ((TypeSArray *)e1->type->toBasetype())->dim;
        new(&ue) UnionExp(e);
    }
    else
        new(&ue) CTFEExp(TOKcantexp);
    return ue;
}

/* Also return TOKcantexp if this fails
 */
UnionExp Index(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Loc loc = e1->loc;

    //printf("Index(e1 = %s, e2 = %s)\n", e1->toChars(), e2->toChars());
    assert(e1->type);
    if (e1->op == TOKstring && e2->op == TOKint64)
    {
        StringExp *es1 = (StringExp *)e1;
        uinteger_t i = e2->toInteger();

        if (i >= es1->len)
        {
            e1->error("string index %llu is out of bounds [0 .. %llu]", i, (ulonglong)es1->len);
            new(&ue) ErrorExp();
        }
        else
        {
            new(&ue) IntegerExp(loc, es1->charAt(i), type);
        }
    }
    else if (e1->type->toBasetype()->ty == Tsarray && e2->op == TOKint64)
    {
        TypeSArray *tsa = (TypeSArray *)e1->type->toBasetype();
        uinteger_t length = tsa->dim->toInteger();
        uinteger_t i = e2->toInteger();

        if (i >= length)
        {
            e1->error("array index %llu is out of bounds %s[0 .. %llu]", i, e1->toChars(), length);
            new(&ue) ErrorExp();
        }
        else if (e1->op == TOKarrayliteral)
        {
            ArrayLiteralExp *ale = (ArrayLiteralExp *)e1;
            Expression *e = (*ale->elements)[(size_t)i];
            e->type = type;
            e->loc = loc;
            if (hasSideEffect(e))
                new(&ue) CTFEExp(TOKcantexp);
            else
                new(&ue) UnionExp(e);
        }
        else
            new(&ue) CTFEExp(TOKcantexp);
    }
    else if (e1->type->toBasetype()->ty == Tarray && e2->op == TOKint64)
    {
        uinteger_t i = e2->toInteger();

        if (e1->op == TOKarrayliteral)
        {
            ArrayLiteralExp *ale = (ArrayLiteralExp *)e1;
            if (i >= ale->elements->dim)
            {
                e1->error("array index %llu is out of bounds %s[0 .. %u]", i, e1->toChars(), ale->elements->dim);
                new(&ue) ErrorExp();
            }
            else
            {
                Expression *e = (*ale->elements)[(size_t)i];
                e->type = type;
                e->loc = loc;
                if (hasSideEffect(e))
                    new(&ue) CTFEExp(TOKcantexp);
                else
                    new(&ue) UnionExp(e);
            }
        }
        else
            new(&ue) CTFEExp(TOKcantexp);
    }
    else if (e1->op == TOKassocarrayliteral)
    {
        AssocArrayLiteralExp *ae = (AssocArrayLiteralExp *)e1;
        /* Search the keys backwards, in case there are duplicate keys
         */
        for (size_t i = ae->keys->dim; i;)
        {
            i--;
            Expression *ekey = (*ae->keys)[i];
            ue = Equal(TOKequal, Type::tbool, ekey, e2);
            if (CTFEExp::isCantExp(ue.exp()))
                return ue;
            if (ue.exp()->isBool(true))
            {
                Expression *e = (*ae->values)[i];
                e->type = type;
                e->loc = loc;
                if (hasSideEffect(e))
                    new(&ue) CTFEExp(TOKcantexp);
                else
                    new(&ue) UnionExp(e);
                return ue;
            }
        }
        new(&ue) CTFEExp(TOKcantexp);
    }
    else
        new(&ue) CTFEExp(TOKcantexp);
    return ue;
}

/* Also return TOKcantexp if this fails
 */
UnionExp Slice(Type *type, Expression *e1, Expression *lwr, Expression *upr)
{
    UnionExp ue;
    Loc loc = e1->loc;

#if LOG
    printf("Slice()\n");
    if (lwr)
    {
        printf("\te1 = %s\n", e1->toChars());
        printf("\tlwr = %s\n", lwr->toChars());
        printf("\tupr = %s\n", upr->toChars());
    }
#endif
    if (e1->op == TOKstring && lwr->op == TOKint64 && upr->op == TOKint64)
    {
        StringExp *es1 = (StringExp *)e1;
        uinteger_t ilwr = lwr->toInteger();
        uinteger_t iupr = upr->toInteger();

        if (iupr > es1->len || ilwr > iupr)
        {
            e1->error("string slice [%llu .. %llu] is out of bounds", ilwr, iupr);
            new(&ue) ErrorExp();
        }
        else
        {
            size_t len = (size_t)(iupr - ilwr);
            unsigned char sz = es1->sz;

            void *s = mem.xmalloc((len + 1) * sz);
            memcpy((char *)s, (char *)es1->string + ilwr * sz, len * sz);
            memset((char *)s + len * sz, 0, sz);

            new(&ue) StringExp(loc, s, len, es1->postfix);
            StringExp *es = (StringExp *)ue.exp();
            es->sz = sz;
            es->committed = es1->committed;
            es->type = type;
        }
    }
    else if (e1->op == TOKarrayliteral &&
            lwr->op == TOKint64 && upr->op == TOKint64 &&
            !hasSideEffect(e1))
    {
        ArrayLiteralExp *es1 = (ArrayLiteralExp *)e1;
        uinteger_t ilwr = lwr->toInteger();
        uinteger_t iupr = upr->toInteger();

        if (iupr > es1->elements->dim || ilwr > iupr)
        {
            e1->error("array slice [%llu .. %llu] is out of bounds", ilwr, iupr);
            new(&ue) ErrorExp();
        }
        else
        {
            Expressions *elements = new Expressions();
            elements->setDim((size_t)(iupr - ilwr));
            memcpy(elements->tdata(),
                   es1->elements->tdata() + ilwr,
                   (size_t)(iupr - ilwr) * sizeof((*es1->elements)[0]));
            new(&ue) ArrayLiteralExp(e1->loc, elements);
            ue.exp()->type = type;
        }
    }
    else
        new(&ue) CTFEExp(TOKcantexp);
    assert(ue.exp()->type);
    return ue;
}

/* Set a slice of char/integer array literal 'existingAE' from a string 'newval'.
 * existingAE[firstIndex..firstIndex+newval.length] = newval.
 */
void sliceAssignArrayLiteralFromString(ArrayLiteralExp *existingAE, StringExp *newval, size_t firstIndex)
{
    size_t newlen =  newval->len;
    size_t sz = newval->sz;
    void *s = newval->string;
    Type *elemType = existingAE->type->nextOf();
    for (size_t j = 0; j < newlen; j++)
    {
        dinteger_t val;
        switch (sz)
        {
            case 1:     val = (( utf8_t *)s)[j];    break;
            case 2:     val = ((utf16_t *)s)[j];    break;
            case 4:     val = ((utf32_t *)s)[j];    break;
            default:    assert(0);                  break;
        }
        (*existingAE->elements)[j + firstIndex]
            = new IntegerExp(newval->loc, val, elemType);
    }
}

/* Set a slice of string 'existingSE' from a char array literal 'newae'.
 *   existingSE[firstIndex..firstIndex+newae.length] = newae.
 */
void sliceAssignStringFromArrayLiteral(StringExp *existingSE, ArrayLiteralExp *newae, size_t firstIndex)
{
    void *s = existingSE->string;
    for (size_t j = 0; j < newae->elements->dim; j++)
    {
        unsigned val = (unsigned)((*newae->elements)[j]->toInteger());
        switch (existingSE->sz)
        {
            case 1:     (( utf8_t *)s)[j + firstIndex] = ( utf8_t)val;  break;
            case 2:     ((utf16_t *)s)[j + firstIndex] = (utf16_t)val;  break;
            case 4:     ((utf32_t *)s)[j + firstIndex] = (utf32_t)val;  break;
            default:    assert(0);                                      break;
        }
    }
}

/* Set a slice of string 'existingSE' from a string 'newstr'.
 *   existingSE[firstIndex..firstIndex+newstr.length] = newstr.
 */
void sliceAssignStringFromString(StringExp *existingSE, StringExp *newstr, size_t firstIndex)
{
    void *s = existingSE->string;
    size_t sz = existingSE->sz;
    assert(sz == newstr->sz);
    memcpy((char *)s + firstIndex * sz, newstr->string, sz * newstr->len);
}

/* Compare a string slice with another string slice.
 * Conceptually equivalent to memcmp( se1[lo1..lo1+len],  se2[lo2..lo2+len])
 */
int sliceCmpStringWithString(StringExp *se1, StringExp *se2, size_t lo1, size_t lo2, size_t len)
{
    void *s1 = se1->string;
    void *s2 = se2->string;
    size_t sz = se1->sz;
    assert(sz == se2->sz);
    return memcmp((char *)s1 + sz * lo1, (char *)s2 + sz * lo2, sz * len);
}

/* Compare a string slice with an array literal slice
 * Conceptually equivalent to memcmp( se1[lo1..lo1+len],  ae2[lo2..lo2+len])
 */
int sliceCmpStringWithArray(StringExp *se1, ArrayLiteralExp *ae2, size_t lo1, size_t lo2, size_t len)
{
    void *s = se1->string;
    size_t sz = se1->sz;

    for (size_t j = 0; j < len; j++)
    {
        unsigned val2 = (unsigned)((*ae2->elements)[j + lo2]->toInteger());
        unsigned val1;
        switch (sz)
        {
            case 1:     val1 = (( utf8_t *)s)[j + lo1]; break;
            case 2:     val1 = ((utf16_t *)s)[j + lo1]; break;
            case 4:     val1 = ((utf32_t *)s)[j + lo1]; break;
            default:    assert(0);                      break;
        }
        int c = val1 - val2;
        if (c)
            return c;
    }
    return 0;
}

/* Also return TOKcantexp if this fails
 */
UnionExp Cat(Type *type, Expression *e1, Expression *e2)
{
    UnionExp ue;
    Expression *e = CTFEExp::cantexp;
    Loc loc = e1->loc;
    Type *t;
    Type *t1 = e1->type->toBasetype();
    Type *t2 = e2->type->toBasetype();

    //printf("Cat(e1 = %s, e2 = %s)\n", e1->toChars(), e2->toChars());
    //printf("\tt1 = %s, t2 = %s, type = %s\n", t1->toChars(), t2->toChars(), type->toChars());

    if (e1->op == TOKnull && (e2->op == TOKint64 || e2->op == TOKstructliteral))
    {
        e = e2;
        t = t1;
        goto L2;
    }
    else if ((e1->op == TOKint64 || e1->op == TOKstructliteral) && e2->op == TOKnull)
    {
        e = e1;
        t = t2;
     L2:
        Type *tn = e->type->toBasetype();
        if (tn->ty == Tchar || tn->ty == Twchar || tn->ty == Tdchar)
        {
            // Create a StringExp
            if (t->nextOf())
                t = t->nextOf()->toBasetype();
            unsigned char sz = (unsigned char)t->size();

            dinteger_t v = e->toInteger();

            size_t len = (t->ty == tn->ty) ? 1 : utf_codeLength(sz, (dchar_t)v);
            void *s = mem.xmalloc((len + 1) * sz);
            if (t->ty == tn->ty)
#if IN_LLVM
#if __LITTLE_ENDIAN__
                memcpy((unsigned char *)s, &v, sz);
#else
                memcpy((unsigned char *)s,
                       (unsigned char *)&v + (sizeof(dinteger_t) - sz), sz);
#endif
#else
                memcpy(s, &v, sz);
#endif
            else
                utf_encode(sz, s, (dchar_t)v);

            // Add terminating 0
            memset((char *)s + len * sz, 0, sz);

            new(&ue) StringExp(loc, s, len);
            StringExp *es = (StringExp *)ue.exp();
            es->sz = sz;
            es->committed = 1;
        }
        else
        {
            // Create an ArrayLiteralExp
            Expressions *elements = new Expressions();
            elements->push(e);
            new(&ue) ArrayLiteralExp(e->loc, elements);
        }
        ue.exp()->type = type;
        assert(ue.exp()->type);
        return ue;
    }
    else if (e1->op == TOKnull && e2->op == TOKnull)
    {
        if (type == e1->type)
        {
            // Handle null ~= null
            if (t1->ty == Tarray && t2 == t1->nextOf())
            {
                new(&ue) ArrayLiteralExp(e1->loc, e2);
                ue.exp()->type = type;
                assert(ue.exp()->type);
                return ue;
            }
            else
            {
                new(&ue) UnionExp(e1);
                assert(ue.exp()->type);
                return ue;
            }
        }
        if (type == e2->type)
        {
            new(&ue) UnionExp(e2);
            assert(ue.exp()->type);
            return ue;
        }
        new(&ue) NullExp(e1->loc, type);
        assert(ue.exp()->type);
        return ue;
    }
    else if (e1->op == TOKstring && e2->op == TOKstring)
    {
        // Concatenate the strings
        StringExp *es1 = (StringExp *)e1;
        StringExp *es2 = (StringExp *)e2;
        size_t len = es1->len + es2->len;
        unsigned char sz = es1->sz;

        if (sz != es2->sz)
        {
            /* Can happen with:
             *   auto s = "foo"d ~ "bar"c;
             */
            assert(global.errors);
            new(&ue) CTFEExp(TOKcantexp);
            assert(ue.exp()->type);
            return ue;
        }
        void *s = mem.xmalloc((len + 1) * sz);
        memcpy((char *)s, es1->string, es1->len * sz);
        memcpy((char *)s + es1->len * sz, es2->string, es2->len * sz);

        // Add terminating 0
        memset((char *)s + len * sz, 0, sz);

        new(&ue) StringExp(loc, s, len);
        StringExp *es = (StringExp *)ue.exp();
        es->sz = sz;
        es->committed = es1->committed | es2->committed;
        es->type = type;
        assert(ue.exp()->type);
        return ue;
    }
    else if (e2->op == TOKstring && e1->op == TOKarrayliteral &&
        t1->nextOf()->isintegral())
    {
        // [chars] ~ string --> [chars]
        StringExp *es = (StringExp *)e2;
        ArrayLiteralExp *ea = (ArrayLiteralExp *)e1;
        size_t len = es->len + ea->elements->dim;
        Expressions * elems = new Expressions;
        elems->setDim(len);
        for (size_t i= 0; i < ea->elements->dim; ++i)
        {
            (*elems)[i] = (*ea->elements)[i];
        }
        new(&ue) ArrayLiteralExp(e1->loc, elems);
        ArrayLiteralExp *dest = (ArrayLiteralExp *)ue.exp();
        dest->type = type;
        sliceAssignArrayLiteralFromString(dest, es, ea->elements->dim);
        assert(ue.exp()->type);
        return ue;
    }
    else if (e1->op == TOKstring && e2->op == TOKarrayliteral &&
        t2->nextOf()->isintegral())
    {
        // string ~ [chars] --> [chars]
        StringExp *es = (StringExp *)e1;
        ArrayLiteralExp *ea = (ArrayLiteralExp *)e2;
        size_t len = es->len + ea->elements->dim;
        Expressions * elems = new Expressions;
        elems->setDim(len);
        for (size_t i= 0; i < ea->elements->dim; ++i)
        {
            (*elems)[es->len + i] = (*ea->elements)[i];
        }
        new(&ue) ArrayLiteralExp(e1->loc, elems);
        ArrayLiteralExp *dest = (ArrayLiteralExp *)ue.exp();
        dest->type = type;
        sliceAssignArrayLiteralFromString(dest, es, 0);
        assert(ue.exp()->type);
        return ue;
    }
    else if (e1->op == TOKstring && e2->op == TOKint64)
    {
        // string ~ char --> string
        StringExp *es1 = (StringExp *)e1;
        StringExp *es;
        unsigned char sz = es1->sz;
        dinteger_t v = e2->toInteger();

        // Is it a concatentation of homogenous types?
        // (char[] ~ char, wchar[]~wchar, or dchar[]~dchar)
        bool homoConcat = (sz == t2->size());
        size_t len = es1->len;
        len += homoConcat ? 1 : utf_codeLength(sz, (dchar_t)v);

        void *s = mem.xmalloc((len + 1) * sz);
        memcpy(s, es1->string, es1->len * sz);
        if (homoConcat)
#if IN_LLVM
#if __LITTLE_ENDIAN__
             memcpy((unsigned char *)s + (sz * es1->len), &v, sz);
#else
             memcpy((unsigned char *)s + (sz * es1->len),
                    (unsigned char *)&v + (sizeof(dinteger_t) - sz), sz);
#endif
#else
             memcpy((char *)s + (sz * es1->len), &v, sz);
#endif
        else
             utf_encode(sz, (char *)s + (sz * es1->len), (dchar_t)v);

        // Add terminating 0
        memset((char *)s + len * sz, 0, sz);

        new(&ue) StringExp(loc, s, len);
        es = (StringExp *)ue.exp();
        es->sz = sz;
        es->committed = es1->committed;
        es->type = type;
        assert(ue.exp()->type);
        return ue;
    }
    else if (e1->op == TOKint64 && e2->op == TOKstring)
    {
        // Concatenate the strings
        StringExp *es2 = (StringExp *)e2;
        size_t len = 1 + es2->len;
        unsigned char sz = es2->sz;
        dinteger_t v = e1->toInteger();

        void *s = mem.xmalloc((len + 1) * sz);
#if IN_LLVM
#if __LITTLE_ENDIAN__
        memcpy((unsigned char *)s, &v, sz);
#else
        memcpy((unsigned char *)s,
               (unsigned char *)&v + (sizeof(dinteger_t) - sz), sz);
#endif
#else
        memcpy((char *)s, &v, sz);
#endif
        memcpy((char *)s + sz, es2->string, es2->len * sz);

        // Add terminating 0
        memset((char *)s + len * sz, 0, sz);

        new(&ue) StringExp(loc, s, len);
        StringExp *es = (StringExp *)ue.exp();
        es->sz = sz;
        es->committed = es2->committed;
        es->type = type;
        assert(ue.exp()->type);
        return ue;
    }
    else if (e1->op == TOKarrayliteral && e2->op == TOKarrayliteral &&
        t1->nextOf()->equals(t2->nextOf()))
    {
        // Concatenate the arrays
        ArrayLiteralExp *es1 = (ArrayLiteralExp *)e1;
        ArrayLiteralExp *es2 = (ArrayLiteralExp *)e2;

        new(&ue) ArrayLiteralExp(es1->loc, (Expressions *)es1->elements->copy());
        es1 = (ArrayLiteralExp *)ue.exp();
        es1->elements->insert(es1->elements->dim, es2->elements);
        e = es1;

        if (type->toBasetype()->ty == Tsarray)
        {
            e->type = t1->nextOf()->sarrayOf(es1->elements->dim);
        }
        else
            e->type = type;
        assert(ue.exp()->type);
        return ue;
    }
    else if (e1->op == TOKarrayliteral && e2->op == TOKnull &&
        t1->nextOf()->equals(t2->nextOf()))
    {
        e = e1;
        goto L3;
    }
    else if (e1->op == TOKnull && e2->op == TOKarrayliteral &&
        t1->nextOf()->equals(t2->nextOf()))
    {
        e = e2;
     L3:
        // Concatenate the array with null
        ArrayLiteralExp *es = (ArrayLiteralExp *)e;

        new(&ue) ArrayLiteralExp(es->loc, (Expressions *)es->elements->copy());
        es = (ArrayLiteralExp *)ue.exp();
        e = es;

        if (type->toBasetype()->ty == Tsarray)
        {
            e->type = t1->nextOf()->sarrayOf(es->elements->dim);
        }
        else
            e->type = type;
        assert(ue.exp()->type);
        return ue;
    }
    else if ((e1->op == TOKarrayliteral || e1->op == TOKnull) &&
        e1->type->toBasetype()->nextOf() &&
        e1->type->toBasetype()->nextOf()->equals(e2->type))
    {
        ArrayLiteralExp *es1;
        if (e1->op == TOKarrayliteral)
        {
            es1 = (ArrayLiteralExp *)e1;
            new(&ue) ArrayLiteralExp(es1->loc, (Expressions *)es1->elements->copy());
            es1 = (ArrayLiteralExp *)ue.exp();
            es1->elements->push(e2);
        }
        else
        {
            new(&ue) ArrayLiteralExp(e1->loc, e2);
            es1 = (ArrayLiteralExp *)ue.exp();
        }
        e = es1;

        if (type->toBasetype()->ty == Tsarray)
        {
            e->type = e2->type->sarrayOf(es1->elements->dim);
        }
        else
            e->type = type;
        assert(ue.exp()->type);
        return ue;
    }
    else if (e2->op == TOKarrayliteral &&
        e2->type->toBasetype()->nextOf()->equals(e1->type))
    {
        ArrayLiteralExp *es2 = (ArrayLiteralExp *)e2;

        new(&ue) ArrayLiteralExp(es2->loc, (Expressions *)es2->elements->copy());
        es2 = (ArrayLiteralExp *)ue.exp();
        es2->elements->shift(e1);
        e = es2;

        if (type->toBasetype()->ty == Tsarray)
        {
            e->type = e1->type->sarrayOf(es2->elements->dim);
        }
        else
            e->type = type;
        assert(ue.exp()->type);
        return ue;
    }
    else if (e1->op == TOKnull && e2->op == TOKstring)
    {
        t = e1->type;
        e = e2;
        goto L1;
    }
    else if (e1->op == TOKstring && e2->op == TOKnull)
    {
        e = e1;
        t = e2->type;
      L1:
        Type *tb = t->toBasetype();
        if (tb->ty == Tarray && tb->nextOf()->equivalent(e->type))
        {
            Expressions *expressions = new Expressions();
            expressions->push(e);
            new(&ue) ArrayLiteralExp(loc, expressions);
            e = ue.exp();
            e->type = t;
        }
        else
        {
            new(&ue) UnionExp(e);
            e = ue.exp();
        }
        if (!e->type->equals(type))
        {
            StringExp *se = (StringExp *)e->copy();
            e = se->castTo(NULL, type);
            new(&ue) UnionExp(e);
            e = ue.exp();
        }
    }
    else
        new(&ue) CTFEExp(TOKcantexp);
    assert(ue.exp()->type);
    return ue;
}

UnionExp Ptr(Type *type, Expression *e1)
{
    //printf("Ptr(e1 = %s)\n", e1->toChars());
    UnionExp ue;
    if (e1->op == TOKadd)
    {
        AddExp *ae = (AddExp *)e1;
        if (ae->e1->op == TOKaddress && ae->e2->op == TOKint64)
        {
            AddrExp *ade = (AddrExp *)ae->e1;
            if (ade->e1->op == TOKstructliteral)
            {
                StructLiteralExp *se = (StructLiteralExp *)ade->e1;
                unsigned offset = (unsigned)ae->e2->toInteger();
                Expression *e = se->getField(type, offset);
                if (e)
                {
                    new(&ue) UnionExp(e);
                    return ue;
                }
            }
        }
    }
    new(&ue) CTFEExp(TOKcantexp);
    return ue;
}
