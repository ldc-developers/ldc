// Compiler implementation of the D programming language
// Copyright (c) 1999-2015 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.dcast;

import core.stdc.stdio;
import core.stdc.string;
import ddmd.aggregate;
import ddmd.aliasthis;
import ddmd.arrayop;
import ddmd.arraytypes;
import ddmd.dclass;
import ddmd.declaration;
import ddmd.dscope;
import ddmd.dstruct;
import ddmd.dsymbol;
import ddmd.dtemplate;
import ddmd.errors;
import ddmd.expression;
import ddmd.func;
import ddmd.globals;
import ddmd.id;
import ddmd.impcnvtab;
import ddmd.init;
import ddmd.intrange;
import ddmd.mtype;
import ddmd.opover;
import ddmd.root.ctfloat;
import ddmd.root.outbuffer;
import ddmd.root.rmem;
import ddmd.root.rootobject;
import ddmd.tokens;
import ddmd.utf;
import ddmd.visitor;

enum LOG = false;

/**************************************
 * Do an implicit cast.
 * Issue error if it can't be done.
 */
extern (C++) Expression implicitCastTo(Expression e, Scope* sc, Type t)
{
    extern (C++) final class ImplicitCastTo : Visitor
    {
        alias visit = super.visit;
    public:
        Type t;
        Scope* sc;
        Expression result;

        extern (D) this(Scope* sc, Type t)
        {
            this.sc = sc;
            this.t = t;
        }

        override void visit(Expression e)
        {
            //printf("Expression::implicitCastTo(%s of type %s) => %s\n", e->toChars(), e->type->toChars(), t->toChars());
            MATCH match = e.implicitConvTo(t);
            if (match)
            {
                if (match == MATCHconst && (e.type.constConv(t) || !e.isLvalue() && e.type.equivalent(t)))
                {
                    /* Do not emit CastExp for const conversions and
                     * unique conversions on rvalue.
                     */
                    result = e.copy();
                    result.type = t;
                    return;
                }
                result = e.castTo(sc, t);
                return;
            }
            result = e.optimize(WANTvalue);
            if (result != e)
            {
                result.accept(this);
                return;
            }
            if (t.ty != Terror && e.type.ty != Terror)
            {
                if (!t.deco)
                {
                    e.error("forward reference to type %s", t.toChars());
                }
                else
                {
                    //printf("type %p ty %d deco %p\n", type, type.ty, type.deco);
                    //type = type.semantic(loc, sc);
                    //printf("type %s t %s\n", type.deco, t.deco);
                    e.error("cannot implicitly convert expression (%s) of type %s to %s",
                        e.toChars(), e.type.toChars(), t.toChars());
                }
            }
            result = new ErrorExp();
        }

        override void visit(StringExp e)
        {
            //printf("StringExp::implicitCastTo(%s of type %s) => %s\n", e->toChars(), e->type->toChars(), t->toChars());
            visit(cast(Expression)e);
            if (result.op == TOKstring)
            {
                // Retain polysemous nature if it started out that way
                (cast(StringExp)result).committed = e.committed;
            }
        }

        override void visit(ErrorExp e)
        {
            result = e;
        }

        override void visit(FuncExp e)
        {
            //printf("FuncExp::implicitCastTo type = %p %s, t = %s\n", e->type, e->type ? e->type->toChars() : NULL, t->toChars());
            FuncExp fe;
            if (e.matchType(t, sc, &fe) > MATCHnomatch)
            {
                result = fe;
                return;
            }
            visit(cast(Expression)e);
        }

        override void visit(ArrayLiteralExp e)
        {
            visit(cast(Expression)e);
            Type tb = result.type.toBasetype();
            if (tb.ty == Tarray)
                semanticTypeInfo(sc, (cast(TypeDArray)tb).next);
        }

        override void visit(SliceExp e)
        {
            visit(cast(Expression)e);
            if (result.op != TOKslice)
                return;
            e = cast(SliceExp)result;
            if (e.e1.op == TOKarrayliteral)
            {
                ArrayLiteralExp ale = cast(ArrayLiteralExp)e.e1;
                Type tb = t.toBasetype();
                Type tx;
                if (tb.ty == Tsarray)
                    tx = tb.nextOf().sarrayOf(ale.elements ? ale.elements.dim : 0);
                else
                    tx = tb.nextOf().arrayOf();
                e.e1 = ale.implicitCastTo(sc, tx);
            }
        }
    }

    scope ImplicitCastTo v = new ImplicitCastTo(sc, t);
    e.accept(v);
    return v.result;
}

/*******************************************
 * Return MATCH level of implicitly converting e to type t.
 * Don't do the actual cast; don't change e.
 */
extern (C++) MATCH implicitConvTo(Expression e, Type t)
{
    extern (C++) final class ImplicitConvTo : Visitor
    {
        alias visit = super.visit;
    public:
        Type t;
        MATCH result;

        extern (D) this(Type t)
        {
            this.t = t;
            result = MATCHnomatch;
        }

        override void visit(Expression e)
        {
            version (none)
            {
                printf("Expression::implicitConvTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            //static int nest; if (++nest == 10) assert(0);
            if (t == Type.terror)
                return;
            if (!e.type)
            {
                e.error("%s is not an expression", e.toChars());
                e.type = Type.terror;
            }
            Expression ex = e.optimize(WANTvalue);
            if (ex.type.equals(t))
            {
                result = MATCHexact;
                return;
            }
            if (ex != e)
            {
                //printf("\toptimized to %s of type %s\n", e->toChars(), e->type->toChars());
                result = ex.implicitConvTo(t);
                return;
            }
            MATCH match = e.type.implicitConvTo(t);
            if (match != MATCHnomatch)
            {
                result = match;
                return;
            }
            /* See if we can do integral narrowing conversions
             */
            if (e.type.isintegral() && t.isintegral() && e.type.isTypeBasic() && t.isTypeBasic())
            {
                IntRange src = getIntRange(e);
                IntRange target = IntRange.fromType(t);
                if (target.contains(src))
                {
                    result = MATCHconvert;
                    return;
                }
            }
        }

        /******
         * Given expression e of type t, see if we can implicitly convert e
         * to type tprime, where tprime is type t with mod bits added.
         * Returns:
         *      match level
         */
        static MATCH implicitMod(Expression e, Type t, MOD mod)
        {
            Type tprime;
            if (t.ty == Tpointer)
                tprime = t.nextOf().castMod(mod).pointerTo();
            else if (t.ty == Tarray)
                tprime = t.nextOf().castMod(mod).arrayOf();
            else if (t.ty == Tsarray)
                tprime = t.nextOf().castMod(mod).sarrayOf(t.size() / t.nextOf().size());
            else
                tprime = t.castMod(mod);
            return e.implicitConvTo(tprime);
        }

        static MATCH implicitConvToAddMin(BinExp e, Type t)
        {
            /* Is this (ptr +- offset)? If so, then ask ptr
             * if the conversion can be done.
             * This is to support doing things like implicitly converting a mutable unique
             * pointer to an immutable pointer.
             */

            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();

            if (typeb.ty != Tpointer || tb.ty != Tpointer)
                return MATCHnomatch;
            Type t1b = e.e1.type.toBasetype();
            Type t2b = e.e2.type.toBasetype();
            if (t1b.ty == Tpointer && t2b.isintegral() && t1b.equivalent(tb))
            {
                // ptr + offset
                // ptr - offset
                MATCH m = e.e1.implicitConvTo(t);
                return (m > MATCHconst) ? MATCHconst : m;
            }
            if (t2b.ty == Tpointer && t1b.isintegral() && t2b.equivalent(tb))
            {
                // offset + ptr
                MATCH m = e.e2.implicitConvTo(t);
                return (m > MATCHconst) ? MATCHconst : m;
            }
            return MATCHnomatch;
        }

        override void visit(AddExp e)
        {
            version (none)
            {
                printf("AddExp::implicitConvTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            visit(cast(Expression)e);
            if (result == MATCHnomatch)
                result = implicitConvToAddMin(e, t);
        }

        override void visit(MinExp e)
        {
            version (none)
            {
                printf("MinExp::implicitConvTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            visit(cast(Expression)e);
            if (result == MATCHnomatch)
                result = implicitConvToAddMin(e, t);
        }

        override void visit(IntegerExp e)
        {
            version (none)
            {
                printf("IntegerExp::implicitConvTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            MATCH m = e.type.implicitConvTo(t);
            if (m >= MATCHconst)
            {
                result = m;
                return;
            }
            TY ty = e.type.toBasetype().ty;
            TY toty = t.toBasetype().ty;
            TY oldty = ty;
            if (m == MATCHnomatch && t.ty == Tenum)
                return;
            if (t.ty == Tvector)
            {
                TypeVector tv = cast(TypeVector)t;
                TypeBasic tb = tv.elementType();
                if (tb.ty == Tvoid)
                    return;
                toty = tb.ty;
            }
            switch (ty)
            {
            case Tbool:
            case Tint8:
            case Tchar:
            case Tuns8:
            case Tint16:
            case Tuns16:
            case Twchar:
                ty = Tint32;
                break;
            case Tdchar:
                ty = Tuns32;
                break;
            default:
                break;
            }
            // Only allow conversion if no change in value
            dinteger_t value = e.toInteger();
            switch (toty)
            {
            case Tbool:
                if ((value & 1) != value)
                    return;
                break;
            case Tint8:
                if (ty == Tuns64 && value & ~0x7FU)
                    return;
                else if (cast(byte)value != value)
                    return;
                break;
            case Tchar:
                if ((oldty == Twchar || oldty == Tdchar) && value > 0x7F)
                    return;
                goto case Tuns8;
            case Tuns8:
                //printf("value = %llu %llu\n", (dinteger_t)(unsigned char)value, value);
                if (cast(ubyte)value != value)
                    return;
                break;
            case Tint16:
                if (ty == Tuns64 && value & ~0x7FFFU)
                    return;
                else if (cast(short)value != value)
                    return;
                break;
            case Twchar:
                if (oldty == Tdchar && value > 0xD7FF && value < 0xE000)
                    return;
                goto case Tuns16;
            case Tuns16:
                if (cast(ushort)value != value)
                    return;
                break;
            case Tint32:
                if (ty == Tuns32)
                {
                }
                else if (ty == Tuns64 && value & ~0x7FFFFFFFU)
                    return;
                else if (cast(int)value != value)
                    return;
                break;
            case Tuns32:
                if (ty == Tint32)
                {
                }
                else if (cast(uint)value != value)
                    return;
                break;
            case Tdchar:
                if (value > 0x10FFFFU)
                    return;
                break;
            case Tfloat32:
                {
                    float f;
                    if (e.type.isunsigned())
                    {
                        f = cast(float)value;
                        if (f != value)
                            return;
                    }
                    else
                    {
                        f = cast(float)cast(sinteger_t)value;
                        if (f != cast(sinteger_t)value)
                            return;
                    }
                    break;
                }
            case Tfloat64:
                {
                    double f;
                    if (e.type.isunsigned())
                    {
                        f = cast(double)value;
                        if (f != value)
                            return;
                    }
                    else
                    {
                        f = cast(double)cast(sinteger_t)value;
                        if (f != cast(sinteger_t)value)
                            return;
                    }
                    break;
                }
            case Tfloat80:
                {
                    if (e.type.isunsigned())
                    {
                        const f = real_t(value);
                        if (cast(dinteger_t)f != value) // isn't this a noop, because the compiler prefers ld
                            return;
                    }
                    else
                    {
                        const f = real_t(cast(sinteger_t)value);
                        if (cast(sinteger_t)f != cast(sinteger_t)value)
                            return;
                    }
                    break;
                }
            case Tpointer:
                //printf("type = %s\n", type->toBasetype()->toChars());
                //printf("t = %s\n", t->toBasetype()->toChars());
                if (ty == Tpointer && e.type.toBasetype().nextOf().ty == t.toBasetype().nextOf().ty)
                {
                    /* Allow things like:
                     *      const char* P = cast(char *)3;
                     *      char* q = P;
                     */
                    break;
                }
                goto default;
            default:
                visit(cast(Expression)e);
                return;
            }
            //printf("MATCHconvert\n");
            result = MATCHconvert;
        }

        override void visit(ErrorExp e)
        {
            // no match
        }

        override void visit(NullExp e)
        {
            version (none)
            {
                printf("NullExp::implicitConvTo(this=%s, type=%s, t=%s, committed = %d)\n", e.toChars(), e.type.toChars(), t.toChars(), e.committed);
            }
            if (e.type.equals(t))
            {
                result = MATCHexact;
                return;
            }
            /* Allow implicit conversions from immutable to mutable|const,
             * and mutable to immutable. It works because, after all, a null
             * doesn't actually point to anything.
             */
            if (t.equivalent(e.type))
            {
                result = MATCHconst;
                return;
            }
            visit(cast(Expression)e);
        }

        override void visit(StructLiteralExp e)
        {
            version (none)
            {
                printf("StructLiteralExp::implicitConvTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            visit(cast(Expression)e);
            if (result != MATCHnomatch)
                return;
            if (e.type.ty == t.ty && e.type.ty == Tstruct && (cast(TypeStruct)e.type).sym == (cast(TypeStruct)t).sym)
            {
                result = MATCHconst;
                for (size_t i = 0; i < e.elements.dim; i++)
                {
                    Expression el = (*e.elements)[i];
                    if (!el)
                        continue;
                    Type te = el.type;
                    te = e.sd.fields[i].type.addMod(t.mod);
                    MATCH m2 = el.implicitConvTo(te);
                    //printf("\t%s => %s, match = %d\n", el->toChars(), te->toChars(), m2);
                    if (m2 < result)
                        result = m2;
                }
            }
        }

        override void visit(StringExp e)
        {
            version (none)
            {
                printf("StringExp::implicitConvTo(this=%s, committed=%d, type=%s, t=%s)\n", e.toChars(), e.committed, e.type.toChars(), t.toChars());
            }
            if (!e.committed && t.ty == Tpointer && t.nextOf().ty == Tvoid)
                return;
            if (e.type.ty == Tsarray || e.type.ty == Tarray || e.type.ty == Tpointer)
            {
                TY tyn = e.type.nextOf().ty;
                if (tyn == Tchar || tyn == Twchar || tyn == Tdchar)
                {
                    switch (t.ty)
                    {
                    case Tsarray:
                        if (e.type.ty == Tsarray)
                        {
                            TY tynto = t.nextOf().ty;
                            if (tynto == tyn)
                            {
                                if ((cast(TypeSArray)e.type).dim.toInteger() == (cast(TypeSArray)t).dim.toInteger())
                                {
                                    result = MATCHexact;
                                }
                                return;
                            }
                            if (tynto == Tchar || tynto == Twchar || tynto == Tdchar)
                            {
                                if (e.committed && tynto != tyn)
                                    return;
                                size_t fromlen = e.numberOfCodeUnits(tynto);
                                size_t tolen = cast(size_t)(cast(TypeSArray)t).dim.toInteger();
                                if (tolen < fromlen)
                                    return;
                                if (tolen != fromlen)
                                {
                                    // implicit length extending
                                    result = MATCHconvert;
                                    return;
                                }
                            }
                            if (!e.committed && (tynto == Tchar || tynto == Twchar || tynto == Tdchar))
                            {
                                result = MATCHexact;
                                return;
                            }
                        }
                        else if (e.type.ty == Tarray)
                        {
                            TY tynto = t.nextOf().ty;
                            if (tynto == Tchar || tynto == Twchar || tynto == Tdchar)
                            {
                                if (e.committed && tynto != tyn)
                                    return;
                                size_t fromlen = e.numberOfCodeUnits(tynto);
                                size_t tolen = cast(size_t)(cast(TypeSArray)t).dim.toInteger();
                                if (tolen < fromlen)
                                    return;
                                if (tolen != fromlen)
                                {
                                    // implicit length extending
                                    result = MATCHconvert;
                                    return;
                                }
                            }
                            if (tynto == tyn)
                            {
                                result = MATCHexact;
                                return;
                            }
                            if (!e.committed && (tynto == Tchar || tynto == Twchar || tynto == Tdchar))
                            {
                                result = MATCHexact;
                                return;
                            }
                        }
                        goto case; /+ fall through +/
                    case Tarray:
                    case Tpointer:
                        Type tn = t.nextOf();
                        MATCH m = MATCHexact;
                        if (e.type.nextOf().mod != tn.mod)
                        {
                            if (!tn.isConst())
                                return;
                            m = MATCHconst;
                        }
                        if (!e.committed)
                        {
                            switch (tn.ty)
                            {
                            case Tchar:
                                if (e.postfix == 'w' || e.postfix == 'd')
                                    m = MATCHconvert;
                                result = m;
                                return;
                            case Twchar:
                                if (e.postfix != 'w')
                                    m = MATCHconvert;
                                result = m;
                                return;
                            case Tdchar:
                                if (e.postfix != 'd')
                                    m = MATCHconvert;
                                result = m;
                                return;
                            default:
                                break;
                            }
                        }
                        break;
                    default:
                        break;
                    }
                }
            }
            visit(cast(Expression)e);
        }

        override void visit(ArrayLiteralExp e)
        {
            version (none)
            {
                printf("ArrayLiteralExp::implicitConvTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();

            if ((tb.ty == Tarray || tb.ty == Tsarray) &&
                (typeb.ty == Tarray || typeb.ty == Tsarray))
            {
                result = MATCHexact;
                Type typen = typeb.nextOf().toBasetype();
                if (tb.ty == Tsarray)
                {
                    TypeSArray tsa = cast(TypeSArray)tb;
                    if (e.elements.dim != tsa.dim.toInteger())
                        result = MATCHnomatch;
                }
                Type telement = tb.nextOf();
                if (!e.elements.dim)
                {
                    if (typen.ty != Tvoid)
                        result = typen.implicitConvTo(telement);
                }
                else
                {
                    if (e.basis)
                    {
                        MATCH m = e.basis.implicitConvTo(telement);
                        if (m < result)
                            result = m;
                    }
                    for (size_t i = 0; i < e.elements.dim; i++)
                    {
                        Expression el = (*e.elements)[i];
                        if (result == MATCHnomatch)
                            break;
                        if (!el)
                            continue;
                        MATCH m = el.implicitConvTo(telement);
                        if (m < result)
                            result = m; // remember worst match
                    }
                }
                if (!result)
                    result = e.type.implicitConvTo(t);
                return;
            }
            else if (tb.ty == Tvector && (typeb.ty == Tarray || typeb.ty == Tsarray))
            {
                result = MATCHexact;
                // Convert array literal to vector type
                TypeVector tv = cast(TypeVector)tb;
                TypeSArray tbase = cast(TypeSArray)tv.basetype;
                assert(tbase.ty == Tsarray);
                if (e.elements.dim != tbase.dim.toInteger())
                {
                    result = MATCHnomatch;
                    return;
                }
                Type telement = tv.elementType();
                for (size_t i = 0; i < e.elements.dim; i++)
                {
                    Expression el = (*e.elements)[i];
                    MATCH m = el.implicitConvTo(telement);
                    if (m < result)
                        result = m; // remember worst match
                    if (result == MATCHnomatch)
                        break;
                    // no need to check for worse
                }
                return;
            }
            visit(cast(Expression)e);
        }

        override void visit(AssocArrayLiteralExp e)
        {
            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();

            if (tb.ty == Taarray && typeb.ty == Taarray)
            {
                result = MATCHexact;
                for (size_t i = 0; i < e.keys.dim; i++)
                {
                    Expression el = (*e.keys)[i];
                    MATCH m = el.implicitConvTo((cast(TypeAArray)tb).index);
                    if (m < result)
                        result = m; // remember worst match
                    if (result == MATCHnomatch)
                        break;
                    // no need to check for worse
                    el = (*e.values)[i];
                    m = el.implicitConvTo(tb.nextOf());
                    if (m < result)
                        result = m; // remember worst match
                    if (result == MATCHnomatch)
                        break;
                    // no need to check for worse
                }
                return;
            }
            else
                visit(cast(Expression)e);
        }

        override void visit(CallExp e)
        {
            enum LOG = 0;
            static if (LOG)
            {
                printf("CallExp::implicitConvTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            visit(cast(Expression)e);
            if (result != MATCHnomatch)
                return;
            /* Allow the result of strongly pure functions to
             * convert to immutable
             */
            if (e.f && e.f.isolateReturn())
            {
                result = e.type.immutableOf().implicitConvTo(t);
                if (result > MATCHconst) // Match level is MATCHconst at best.
                    result = MATCHconst;
                return;
            }
            /* Conversion is 'const' conversion if:
             * 1. function is pure (weakly pure is ok)
             * 2. implicit conversion only fails because of mod bits
             * 3. each function parameter can be implicitly converted to the mod bits
             */
            Type tx = e.f ? e.f.type : e.e1.type;
            tx = tx.toBasetype();
            if (tx.ty != Tfunction)
                return;
            TypeFunction tf = cast(TypeFunction)tx;
            if (tf.purity == PUREimpure)
                return;
            if (e.f && e.f.isNested())
                return;
            /* See if fail only because of mod bits.
             *
             * Bugzilla 14155: All pure functions can access global immutable data.
             * So the returned pointer may refer an immutable global data,
             * and then the returned pointer that points non-mutable object
             * cannot be unique pointer.
             *
             * Example:
             *  immutable g;
             *  static this() { g = 1; }
             *  const(int*) foo() pure { return &g; }
             *  void test() {
             *    immutable(int*) ip = foo(); // OK
             *    int* mp = foo();            // should be disallowed
             *  }
             */
            if (e.type.immutableOf().implicitConvTo(t) < MATCHconst && e.type.addMod(MODshared).implicitConvTo(t) < MATCHconst && e.type.implicitConvTo(t.addMod(MODshared)) < MATCHconst)
            {
                return;
            }
            // Allow a conversion to immutable type, or
            // conversions of mutable types between thread-local and shared.
            /* Get mod bits of what we're converting to
             */
            Type tb = t.toBasetype();
            MOD mod = tb.mod;
            if (tf.isref)
            {
            }
            else
            {
                Type ti = getIndirection(t);
                if (ti)
                    mod = ti.mod;
            }
            static if (LOG)
            {
                printf("mod = x%x\n", mod);
            }
            if (mod & MODwild)
                return; // not sure what to do with this
            /* Apply mod bits to each function parameter,
             * and see if we can convert the function argument to the modded type
             */
            size_t nparams = Parameter.dim(tf.parameters);
            size_t j = (tf.linkage == LINKd && tf.varargs == 1); // if TypeInfoArray was prepended
            if (e.e1.op == TOKdotvar)
            {
                /* Treat 'this' as just another function argument
                 */
                DotVarExp dve = cast(DotVarExp)e.e1;
                Type targ = dve.e1.type;
                if (targ.constConv(targ.castMod(mod)) == MATCHnomatch)
                    return;
            }
            for (size_t i = j; i < e.arguments.dim; ++i)
            {
                Expression earg = (*e.arguments)[i];
                Type targ = earg.type.toBasetype();
                static if (LOG)
                {
                    printf("[%d] earg: %s, targ: %s\n", cast(int)i, earg.toChars(), targ.toChars());
                }
                if (i - j < nparams)
                {
                    Parameter fparam = Parameter.getNth(tf.parameters, i - j);
                    if (fparam.storageClass & STClazy)
                        return; // not sure what to do with this
                    Type tparam = fparam.type;
                    if (!tparam)
                        continue;
                    if (fparam.storageClass & (STCout | STCref))
                    {
                        if (targ.constConv(tparam.castMod(mod)) == MATCHnomatch)
                            return;
                        continue;
                    }
                }
                static if (LOG)
                {
                    printf("[%d] earg: %s, targm: %s\n", cast(int)i, earg.toChars(), targ.addMod(mod).toChars());
                }
                if (implicitMod(earg, targ, mod) == MATCHnomatch)
                    return;
            }
            /* Success
             */
            result = MATCHconst;
        }

        override void visit(AddrExp e)
        {
            version (none)
            {
                printf("AddrExp::implicitConvTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            result = e.type.implicitConvTo(t);
            //printf("\tresult = %d\n", result);
            if (result != MATCHnomatch)
                return;

            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();

            // Look for pointers to functions where the functions are overloaded.
            if (e.e1.op == TOKoverloadset &&
                (tb.ty == Tpointer || tb.ty == Tdelegate) && tb.nextOf().ty == Tfunction)
            {
                OverExp eo = cast(OverExp)e.e1;
                FuncDeclaration f = null;
                for (size_t i = 0; i < eo.vars.a.dim; i++)
                {
                    Dsymbol s = eo.vars.a[i];
                    FuncDeclaration f2 = s.isFuncDeclaration();
                    assert(f2);
                    if (f2.overloadExactMatch(tb.nextOf()))
                    {
                        if (f)
                        {
                            /* Error if match in more than one overload set,
                             * even if one is a 'better' match than the other.
                             */
                            ScopeDsymbol.multiplyDefined(e.loc, f, f2);
                        }
                        else
                            f = f2;
                        result = MATCHexact;
                    }
                }
            }

            if (e.e1.op == TOKvar &&
                typeb.ty == Tpointer && typeb.nextOf().ty == Tfunction &&
                tb.ty == Tpointer && tb.nextOf().ty == Tfunction)
            {
                /* I don't think this can ever happen -
                 * it should have been
                 * converted to a SymOffExp.
                 */
                assert(0);
            }
            //printf("\tresult = %d\n", result);
        }

        override void visit(SymOffExp e)
        {
            version (none)
            {
                printf("SymOffExp::implicitConvTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            result = e.type.implicitConvTo(t);
            //printf("\tresult = %d\n", result);
            if (result != MATCHnomatch)
                return;

            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();

            // Look for pointers to functions where the functions are overloaded.
            if (typeb.ty == Tpointer && typeb.nextOf().ty == Tfunction &&
                (tb.ty == Tpointer || tb.ty == Tdelegate) && tb.nextOf().ty == Tfunction)
            {
                if (FuncDeclaration f = e.var.isFuncDeclaration())
                {
                    f = f.overloadExactMatch(tb.nextOf());
                    if (f)
                    {
                        if ((tb.ty == Tdelegate && (f.needThis() || f.isNested())) ||
                            (tb.ty == Tpointer && !(f.needThis() || f.isNested())))
                        {
                            result = MATCHexact;
                        }
                    }
                }
            }
            //printf("\tresult = %d\n", result);
        }

        override void visit(DelegateExp e)
        {
            version (none)
            {
                printf("DelegateExp::implicitConvTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            result = e.type.implicitConvTo(t);
            if (result != MATCHnomatch)
                return;

            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();

            // Look for pointers to functions where the functions are overloaded.
            if (typeb.ty == Tdelegate && tb.ty == Tdelegate)
            {
                if (e.func && e.func.overloadExactMatch(tb.nextOf()))
                    result = MATCHexact;
            }
        }

        override void visit(FuncExp e)
        {
            //printf("FuncExp::implicitConvTo type = %p %s, t = %s\n", e->type, e->type ? e->type->toChars() : NULL, t->toChars());
            MATCH m = e.matchType(t, null, null, 1);
            if (m > MATCHnomatch)
            {
                result = m;
                return;
            }
            visit(cast(Expression)e);
        }

        override void visit(OrExp e)
        {
            visit(cast(Expression)e);
            if (result != MATCHnomatch)
                return;
            MATCH m1 = e.e1.implicitConvTo(t);
            MATCH m2 = e.e2.implicitConvTo(t);
            // Pick the worst match
            result = (m1 < m2) ? m1 : m2;
        }

        override void visit(XorExp e)
        {
            visit(cast(Expression)e);
            if (result != MATCHnomatch)
                return;
            MATCH m1 = e.e1.implicitConvTo(t);
            MATCH m2 = e.e2.implicitConvTo(t);
            // Pick the worst match
            result = (m1 < m2) ? m1 : m2;
        }

        override void visit(CondExp e)
        {
            MATCH m1 = e.e1.implicitConvTo(t);
            MATCH m2 = e.e2.implicitConvTo(t);
            //printf("CondExp: m1 %d m2 %d\n", m1, m2);
            // Pick the worst match
            result = (m1 < m2) ? m1 : m2;
        }

        override void visit(CommaExp e)
        {
            e.e2.accept(this);
        }

        override void visit(CastExp e)
        {
            version (none)
            {
                printf("CastExp::implicitConvTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            result = e.type.implicitConvTo(t);
            if (result != MATCHnomatch)
                return;
            if (t.isintegral() && e.e1.type.isintegral() && e.e1.implicitConvTo(t) != MATCHnomatch)
                result = MATCHconvert;
            else
                visit(cast(Expression)e);
        }

        override void visit(NewExp e)
        {
            version (none)
            {
                printf("NewExp::implicitConvTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            visit(cast(Expression)e);
            if (result != MATCHnomatch)
                return;
            /* Calling new() is like calling a pure function. We can implicitly convert the
             * return from new() to t using the same algorithm as in CallExp, with the function
             * 'arguments' being:
             *    thisexp
             *    newargs
             *    arguments
             *    .init
             * 'member' and 'allocator' need to be pure.
             */
            /* See if fail only because of mod bits
             */
            if (e.type.immutableOf().implicitConvTo(t.immutableOf()) == MATCHnomatch)
                return;
            /* Get mod bits of what we're converting to
             */
            Type tb = t.toBasetype();
            MOD mod = tb.mod;
            if (Type ti = getIndirection(t))
                mod = ti.mod;
            static if (LOG)
            {
                printf("mod = x%x\n", mod);
            }
            if (mod & MODwild)
                return; // not sure what to do with this
            /* Apply mod bits to each argument,
             * and see if we can convert the argument to the modded type
             */
            if (e.thisexp)
            {
                /* Treat 'this' as just another function argument
                 */
                Type targ = e.thisexp.type;
                if (targ.constConv(targ.castMod(mod)) == MATCHnomatch)
                    return;
            }
            /* Check call to 'allocator', then 'member'
             */
            FuncDeclaration fd = e.allocator;
            for (int count = 0; count < 2; ++count, (fd = e.member))
            {
                if (!fd)
                    continue;
                if (fd.errors || fd.type.ty != Tfunction)
                    return; // error
                TypeFunction tf = cast(TypeFunction)fd.type;
                if (tf.purity == PUREimpure)
                    return; // impure
                if (fd == e.member)
                {
                    if (e.type.immutableOf().implicitConvTo(t) < MATCHconst && e.type.addMod(MODshared).implicitConvTo(t) < MATCHconst && e.type.implicitConvTo(t.addMod(MODshared)) < MATCHconst)
                    {
                        return;
                    }
                    // Allow a conversion to immutable type, or
                    // conversions of mutable types between thread-local and shared.
                }
                Expressions* args = (fd == e.allocator) ? e.newargs : e.arguments;
                size_t nparams = Parameter.dim(tf.parameters);
                size_t j = (tf.linkage == LINKd && tf.varargs == 1); // if TypeInfoArray was prepended
                for (size_t i = j; i < e.arguments.dim; ++i)
                {
                    Expression earg = (*args)[i];
                    Type targ = earg.type.toBasetype();
                    static if (LOG)
                    {
                        printf("[%d] earg: %s, targ: %s\n", cast(int)i, earg.toChars(), targ.toChars());
                    }
                    if (i - j < nparams)
                    {
                        Parameter fparam = Parameter.getNth(tf.parameters, i - j);
                        if (fparam.storageClass & STClazy)
                            return; // not sure what to do with this
                        Type tparam = fparam.type;
                        if (!tparam)
                            continue;
                        if (fparam.storageClass & (STCout | STCref))
                        {
                            if (targ.constConv(tparam.castMod(mod)) == MATCHnomatch)
                                return;
                            continue;
                        }
                    }
                    static if (LOG)
                    {
                        printf("[%d] earg: %s, targm: %s\n", cast(int)i, earg.toChars(), targ.addMod(mod).toChars());
                    }
                    if (implicitMod(earg, targ, mod) == MATCHnomatch)
                        return;
                }
            }
            /* If no 'member', then construction is by simple assignment,
             * and just straight check 'arguments'
             */
            if (!e.member && e.arguments)
            {
                for (size_t i = 0; i < e.arguments.dim; ++i)
                {
                    Expression earg = (*e.arguments)[i];
                    if (!earg) // Bugzilla 14853: if it's on overlapped field
                        continue;
                    Type targ = earg.type.toBasetype();
                    static if (LOG)
                    {
                        printf("[%d] earg: %s, targ: %s\n", cast(int)i, earg.toChars(), targ.toChars());
                        printf("[%d] earg: %s, targm: %s\n", cast(int)i, earg.toChars(), targ.addMod(mod).toChars());
                    }
                    if (implicitMod(earg, targ, mod) == MATCHnomatch)
                        return;
                }
            }
            /* Consider the .init expression as an argument
             */
            Type ntb = e.newtype.toBasetype();
            if (ntb.ty == Tarray)
                ntb = ntb.nextOf().toBasetype();
            if (ntb.ty == Tstruct)
            {
                // Don't allow nested structs - uplevel reference may not be convertible
                StructDeclaration sd = (cast(TypeStruct)ntb).sym;
                sd.size(e.loc); // resolve any forward references
                if (sd.isNested())
                    return;
            }
            if (ntb.isZeroInit(e.loc))
            {
                /* Zeros are implicitly convertible, except for special cases.
                 */
                if (ntb.ty == Tclass)
                {
                    /* With new() must look at the class instance initializer.
                     */
                    ClassDeclaration cd = (cast(TypeClass)ntb).sym;
                    cd.size(e.loc); // resolve any forward references
                    if (cd.isNested())
                        return; // uplevel reference may not be convertible
                    assert(!cd.isInterfaceDeclaration());
                    struct ClassCheck
                    {
                        extern (C++) static bool convertible(Loc loc, ClassDeclaration cd, MOD mod)
                        {
                            for (size_t i = 0; i < cd.fields.dim; i++)
                            {
                                VarDeclaration v = cd.fields[i];
                                Initializer _init = v._init;
                                if (_init)
                                {
                                    if (_init.isVoidInitializer())
                                    {
                                    }
                                    else if (ExpInitializer ei = _init.isExpInitializer())
                                    {
                                        Type tb = v.type.toBasetype();
                                        if (implicitMod(ei.exp, tb, mod) == MATCHnomatch)
                                            return false;
                                    }
                                    else
                                    {
                                        /* Enhancement: handle StructInitializer and ArrayInitializer
                                         */
                                        return false;
                                    }
                                }
                                else if (!v.type.isZeroInit(loc))
                                    return false;
                            }
                            return cd.baseClass ? convertible(loc, cd.baseClass, mod) : true;
                        }
                    }

                    if (!ClassCheck.convertible(e.loc, cd, mod))
                        return;
                }
            }
            else
            {
                Expression earg = e.newtype.defaultInitLiteral(e.loc);
                Type targ = e.newtype.toBasetype();
                if (implicitMod(earg, targ, mod) == MATCHnomatch)
                    return;
            }
            /* Success
             */
            result = MATCHconst;
        }

        override void visit(SliceExp e)
        {
            //printf("SliceExp::implicitConvTo e = %s, type = %s\n", e->toChars(), e->type->toChars());
            visit(cast(Expression)e);
            if (result != MATCHnomatch)
                return;
            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();
            if (tb.ty == Tsarray && typeb.ty == Tarray)
            {
                typeb = toStaticArrayType(e);
                if (typeb)
                    result = typeb.implicitConvTo(t);
                return;
            }
            /* If the only reason it won't convert is because of the mod bits,
             * then test for conversion by seeing if e1 can be converted with those
             * same mod bits.
             */
            Type t1b = e.e1.type.toBasetype();
            if (tb.ty == Tarray && typeb.equivalent(tb))
            {
                Type tbn = tb.nextOf();
                Type tx = null;
                /* If e->e1 is dynamic array or pointer, the uniqueness of e->e1
                 * is equivalent with the uniqueness of the referred data. And in here
                 * we can have arbitrary typed reference for that.
                 */
                if (t1b.ty == Tarray)
                    tx = tbn.arrayOf();
                if (t1b.ty == Tpointer)
                    tx = tbn.pointerTo();
                /* If e->e1 is static array, at least it should be an rvalue.
                 * If not, e->e1 is a reference, and its uniqueness does not link
                 * to the uniqueness of the referred data.
                 */
                if (t1b.ty == Tsarray && !e.e1.isLvalue())
                    tx = tbn.sarrayOf(t1b.size() / tbn.size());
                if (tx)
                {
                    result = e.e1.implicitConvTo(tx);
                    if (result > MATCHconst) // Match level is MATCHconst at best.
                        result = MATCHconst;
                }
            }
            // Enhancement 10724
            if (tb.ty == Tpointer && e.e1.op == TOKstring)
                e.e1.accept(this);
        }
    }

    scope ImplicitConvTo v = new ImplicitConvTo(t);
    e.accept(v);
    return v.result;
}

extern (C++) Type toStaticArrayType(SliceExp e)
{
    if (e.lwr && e.upr)
    {
        // For the following code to work, e should be optimized beforehand.
        // (eg. $ in lwr and upr should be already resolved, if possible)
        Expression lwr = e.lwr.optimize(WANTvalue);
        Expression upr = e.upr.optimize(WANTvalue);
        if (lwr.isConst() && upr.isConst())
        {
            size_t len = cast(size_t)(upr.toUInteger() - lwr.toUInteger());
            return e.type.toBasetype().nextOf().sarrayOf(len);
        }
    }
    else
    {
        Type t1b = e.e1.type.toBasetype();
        if (t1b.ty == Tsarray)
            return t1b;
    }
    return null;
}

/**************************************
 * Do an explicit cast.
 * Assume that the 'this' expression does not have any indirections.
 */
extern (C++) Expression castTo(Expression e, Scope* sc, Type t)
{
    extern (C++) final class CastTo : Visitor
    {
        alias visit = super.visit;
    public:
        Type t;
        Scope* sc;
        Expression result;

        extern (D) this(Scope* sc, Type t)
        {
            this.sc = sc;
            this.t = t;
        }

        override void visit(Expression e)
        {
            //printf("Expression::castTo(this=%s, t=%s)\n", e->toChars(), t->toChars());
            version (none)
            {
                printf("Expression::castTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            if (e.type.equals(t))
            {
                result = e;
                return;
            }
            if (e.op == TOKvar)
            {
                VarDeclaration v = (cast(VarExp)e).var.isVarDeclaration();
                if (v && v.storage_class & STCmanifest)
                {
                    result = e.ctfeInterpret();
                    result = result.castTo(sc, t);
                    return;
                }
            }
            Type tob = t.toBasetype();
            Type t1b = e.type.toBasetype();
            if (tob.equals(t1b))
            {
                result = e.copy(); // because of COW for assignment to e->type
                result.type = t;
                return;
            }
            /* Make semantic error against invalid cast between concrete types.
             * Assume that 'e' is never be any placeholder expressions.
             * The result of these checks should be consistent with CastExp::toElem().
             */
            // Fat Value types
            const(bool) tob_isFV = (tob.ty == Tstruct || tob.ty == Tsarray);
            const(bool) t1b_isFV = (t1b.ty == Tstruct || t1b.ty == Tsarray);
            // Fat Reference types
            const(bool) tob_isFR = (tob.ty == Tarray || tob.ty == Tdelegate);
            const(bool) t1b_isFR = (t1b.ty == Tarray || t1b.ty == Tdelegate);
            // Reference types
            const(bool) tob_isR = (tob_isFR || tob.ty == Tpointer || tob.ty == Taarray || tob.ty == Tclass);
            const(bool) t1b_isR = (t1b_isFR || t1b.ty == Tpointer || t1b.ty == Taarray || t1b.ty == Tclass);
            // Arithmetic types (== valueable basic types)
            const(bool) tob_isA = (tob.isintegral() || tob.isfloating());
            const(bool) t1b_isA = (t1b.isintegral() || t1b.isfloating());
            if (AggregateDeclaration t1ad = isAggregate(t1b))
            {
                AggregateDeclaration toad = isAggregate(tob);
                if (t1ad != toad && t1ad.aliasthis)
                {
                    if (t1b.ty == Tclass && tob.ty == Tclass)
                    {
                        ClassDeclaration t1cd = t1b.isClassHandle();
                        ClassDeclaration tocd = tob.isClassHandle();
                        int offset;
                        if (tocd.isBaseOf(t1cd, &offset))
                            goto Lok;
                    }
                    /* Forward the cast to our alias this member, rewrite to:
                     *   cast(to)e1.aliasthis
                     */
                    result = resolveAliasThis(sc, e);
                    result = result.castTo(sc, t);
                    return;
                }
            }
            else if (tob.ty == Tvector && t1b.ty != Tvector)
            {
                //printf("test1 e = %s, e->type = %s, tob = %s\n", e->toChars(), e->type->toChars(), tob->toChars());
                TypeVector tv = cast(TypeVector)tob;
                result = new CastExp(e.loc, e, tv.elementType());
                result = new VectorExp(e.loc, result, tob);
                result = result.semantic(sc);
                return;
            }
            else if (tob.ty != Tvector && t1b.ty == Tvector)
            {
                // T[n] <-- __vector(U[m])
                if (tob.ty == Tsarray)
                {
                    if (t1b.size(e.loc) == tob.size(e.loc))
                        goto Lok;
                }
                goto Lfail;
            }
            else if (t1b.implicitConvTo(tob) == MATCHconst && t.equals(e.type.constOf()))
            {
                result = e.copy();
                result.type = t;
                return;
            }
            // arithmetic values vs. other arithmetic values
            // arithmetic values vs. T*
            if (tob_isA && (t1b_isA || t1b.ty == Tpointer) || t1b_isA && (tob_isA || tob.ty == Tpointer))
            {
                goto Lok;
            }
            // arithmetic values vs. references or fat values
            if (tob_isA && (t1b_isR || t1b_isFV) || t1b_isA && (tob_isR || tob_isFV))
            {
                goto Lfail;
            }
            // Bugzlla 3133: A cast between fat values is possible only when the sizes match.
            if (tob_isFV && t1b_isFV)
            {
                if (t1b.size(e.loc) == tob.size(e.loc))
                    goto Lok;
                e.error("cannot cast expression %s of type %s to %s because of different sizes", e.toChars(), e.type.toChars(), t.toChars());
                result = new ErrorExp();
                return;
            }
            // Fat values vs. null or references
            if (tob_isFV && (t1b.ty == Tnull || t1b_isR) || t1b_isFV && (tob.ty == Tnull || tob_isR))
            {
                if (tob.ty == Tpointer && t1b.ty == Tsarray)
                {
                    // T[n] sa;
                    // cast(U*)sa; // ==> cast(U*)sa.ptr;
                    result = new AddrExp(e.loc, e);
                    result.type = t;
                    return;
                }
                if (tob.ty == Tarray && t1b.ty == Tsarray)
                {
                    // T[n] sa;
                    // cast(U[])sa; // ==> cast(U[])sa[];
                    d_uns64 fsize = t1b.nextOf().size();
                    d_uns64 tsize = tob.nextOf().size();
                    if (((cast(TypeSArray)t1b).dim.toInteger() * fsize) % tsize != 0)
                    {
                        // copied from sarray_toDarray() in e2ir.c
                        e.error("cannot cast expression %s of type %s to %s since sizes don't line up", e.toChars(), e.type.toChars(), t.toChars());
                        result = new ErrorExp();
                        return;
                    }
                    goto Lok;
                }
                goto Lfail;
            }
            /* For references, any reinterpret casts are allowed to same 'ty' type.
             *      T* to U*
             *      R1 function(P1) to R2 function(P2)
             *      R1 delegate(P1) to R2 delegate(P2)
             *      T[] to U[]
             *      V1[K1] to V2[K2]
             *      class/interface A to B  (will be a dynamic cast if possible)
             */
            if (tob.ty == t1b.ty && tob_isR && t1b_isR)
                goto Lok;
            // typeof(null) <-- non-null references or values
            if (tob.ty == Tnull && t1b.ty != Tnull)
                goto Lfail;
            // Bugzilla 14629
            // typeof(null) --> non-null references or arithmetic values
            if (t1b.ty == Tnull && tob.ty != Tnull)
                goto Lok;
            // Check size mismatch of references.
            // Tarray and Tdelegate are (void*).sizeof*2, but others have (void*).sizeof.
            if (tob_isFR && t1b_isR || t1b_isFR && tob_isR)
            {
                if (tob.ty == Tpointer && t1b.ty == Tarray)
                {
                    // T[] da;
                    // cast(U*)da; // ==> cast(U*)da.ptr;
                    goto Lok;
                }
                if (tob.ty == Tpointer && t1b.ty == Tdelegate)
                {
                    // void delegate() dg;
                    // cast(U*)dg; // ==> cast(U*)dg.ptr;
                    // Note that it happens even when U is a Tfunction!
                    e.deprecation("casting from %s to %s is deprecated", e.type.toChars(), t.toChars());
                    goto Lok;
                }
                goto Lfail;
            }
            if (t1b.ty == Tvoid && tob.ty != Tvoid)
            {
            Lfail:
                e.error("cannot cast expression %s of type %s to %s", e.toChars(), e.type.toChars(), t.toChars());
                result = new ErrorExp();
                return;
            }
        Lok:
            result = new CastExp(e.loc, e, tob);
            result.type = t; // Don't call semantic()
            //printf("Returning: %s\n", result->toChars());
        }

        override void visit(ErrorExp e)
        {
            result = e;
        }

        override void visit(RealExp e)
        {
            if (!e.type.equals(t))
            {
                if ((e.type.isreal() && t.isreal()) || (e.type.isimaginary() && t.isimaginary()))
                {
                    result = e.copy();
                    result.type = t;
                }
                else
                    visit(cast(Expression)e);
                return;
            }
            result = e;
        }

        override void visit(ComplexExp e)
        {
            if (!e.type.equals(t))
            {
                if (e.type.iscomplex() && t.iscomplex())
                {
                    result = e.copy();
                    result.type = t;
                }
                else
                    visit(cast(Expression)e);
                return;
            }
            result = e;
        }

        override void visit(NullExp e)
        {
            //printf("NullExp::castTo(t = %s) %s\n", t->toChars(), toChars());
            visit(cast(Expression)e);
            if (result.op == TOKnull)
            {
                NullExp ex = cast(NullExp)result;
                ex.committed = 1;
                return;
            }
        }

        override void visit(StructLiteralExp e)
        {
            visit(cast(Expression)e);
            if (result.op == TOKstructliteral)
                (cast(StructLiteralExp)result).stype = t; // commit type
        }

        override void visit(StringExp e)
        {
            /* This follows copy-on-write; any changes to 'this'
             * will result in a copy.
             * The this->string member is considered immutable.
             */
            int copied = 0;
            //printf("StringExp::castTo(t = %s), '%s' committed = %d\n", t->toChars(), e->toChars(), e->committed);
            if (!e.committed && t.ty == Tpointer && t.nextOf().ty == Tvoid)
            {
                e.error("cannot convert string literal to void*");
                result = new ErrorExp();
                return;
            }
            StringExp se = e;
            if (!e.committed)
            {
                se = cast(StringExp)e.copy();
                se.committed = 1;
                copied = 1;
            }
            if (e.type.equals(t))
            {
                result = se;
                return;
            }

            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();

            //printf("\ttype = %s\n", e->type->toChars());
            if (tb.ty == Tdelegate && typeb.ty != Tdelegate)
            {
                visit(cast(Expression)e);
                return;
            }

            if (typeb.equals(tb))
            {
                if (!copied)
                {
                    se = cast(StringExp)e.copy();
                    copied = 1;
                }
                se.type = t;
                result = se;
                return;
            }
            /* Handle reinterpret casts:
             *  cast(wchar[3])"abcd"c --> [\u6261, \u6463, \u0000]
             *  cast(wchar[2])"abcd"c --> [\u6261, \u6463]
             *  cast(wchar[1])"abcd"c --> [\u6261]
             */
            if (e.committed && tb.ty == Tsarray && typeb.ty == Tarray)
            {
                se = cast(StringExp)e.copy();
                d_uns64 szx = tb.nextOf().size();
                assert(szx <= 255);
                se.sz = cast(ubyte)szx;
                se.len = cast(size_t)(cast(TypeSArray)tb).dim.toInteger();
                se.committed = 1;
                se.type = t;
                /* Assure space for terminating 0
                 */
                if ((se.len + 1) * se.sz > (e.len + 1) * e.sz)
                {
                    void* s = mem.xmalloc((se.len + 1) * se.sz);
                    memcpy(s, se.string, se.len * se.sz);
                    memset(s + se.len * se.sz, 0, se.sz);
                    se.string = cast(char*)s;
                }
                result = se;
                return;
            }
            if (tb.ty != Tsarray && tb.ty != Tarray && tb.ty != Tpointer)
            {
                if (!copied)
                {
                    se = cast(StringExp)e.copy();
                    copied = 1;
                }
                goto Lcast;
            }
            if (typeb.ty != Tsarray && typeb.ty != Tarray && typeb.ty != Tpointer)
            {
                if (!copied)
                {
                    se = cast(StringExp)e.copy();
                    copied = 1;
                }
                goto Lcast;
            }
            if (typeb.nextOf().size() == tb.nextOf().size())
            {
                if (!copied)
                {
                    se = cast(StringExp)e.copy();
                    copied = 1;
                }
                if (tb.ty == Tsarray)
                    goto L2;
                // handle possible change in static array dimension
                se.type = t;
                result = se;
                return;
            }
            if (e.committed)
                goto Lcast;
            auto X(T, U)(T tf, U tt)
            {
                return (cast(int)tf * 256 + cast(int)tt);
            }

            {
                OutBuffer buffer;
                size_t newlen = 0;
                int tfty = typeb.nextOf().toBasetype().ty;
                int ttty = tb.nextOf().toBasetype().ty;
                switch (X(tfty, ttty))
                {
                case X(Tchar, Tchar):
                case X(Twchar, Twchar):
                case X(Tdchar, Tdchar):
                    break;
                case X(Tchar, Twchar):
                    for (size_t u = 0; u < e.len;)
                    {
                        dchar c;
                        const p = utf_decodeChar(se.string, e.len, u, c);
                        if (p)
                            e.error("%s", p);
                        else
                            buffer.writeUTF16(c);
                    }
                    newlen = buffer.offset / 2;
                    buffer.writeUTF16(0);
                    goto L1;
                case X(Tchar, Tdchar):
                    for (size_t u = 0; u < e.len;)
                    {
                        dchar c;
                        const p = utf_decodeChar(se.string, e.len, u, c);
                        if (p)
                            e.error("%s", p);
                        buffer.write4(c);
                        newlen++;
                    }
                    buffer.write4(0);
                    goto L1;
                case X(Twchar, Tchar):
                    for (size_t u = 0; u < e.len;)
                    {
                        dchar c;
                        const p = utf_decodeWchar(se.wstring, e.len, u, c);
                        if (p)
                            e.error("%s", p);
                        else
                            buffer.writeUTF8(c);
                    }
                    newlen = buffer.offset;
                    buffer.writeUTF8(0);
                    goto L1;
                case X(Twchar, Tdchar):
                    for (size_t u = 0; u < e.len;)
                    {
                        dchar c;
                        const p = utf_decodeWchar(se.wstring, e.len, u, c);
                        if (p)
                            e.error("%s", p);
                        buffer.write4(c);
                        newlen++;
                    }
                    buffer.write4(0);
                    goto L1;
                case X(Tdchar, Tchar):
                    for (size_t u = 0; u < e.len; u++)
                    {
                        uint c = se.dstring[u];
                        if (!utf_isValidDchar(c))
                            e.error("invalid UCS-32 char \\U%08x", c);
                        else
                            buffer.writeUTF8(c);
                        newlen++;
                    }
                    newlen = buffer.offset;
                    buffer.writeUTF8(0);
                    goto L1;
                case X(Tdchar, Twchar):
                    for (size_t u = 0; u < e.len; u++)
                    {
                        uint c = se.dstring[u];
                        if (!utf_isValidDchar(c))
                            e.error("invalid UCS-32 char \\U%08x", c);
                        else
                            buffer.writeUTF16(c);
                        newlen++;
                    }
                    newlen = buffer.offset / 2;
                    buffer.writeUTF16(0);
                    goto L1;
                L1:
                    if (!copied)
                    {
                        se = cast(StringExp)e.copy();
                        copied = 1;
                    }
                    se.string = buffer.extractData();
                    se.len = newlen;
                    {
                        d_uns64 szx = tb.nextOf().size();
                        assert(szx <= 255);
                        se.sz = cast(ubyte)szx;
                    }
                    break;
                default:
                    assert(typeb.nextOf().size() != tb.nextOf().size());
                    goto Lcast;
                }
            }
        L2:
            assert(copied);
            // See if need to truncate or extend the literal
            if (tb.ty == Tsarray)
            {
                size_t dim2 = cast(size_t)(cast(TypeSArray)tb).dim.toInteger();
                //printf("dim from = %d, to = %d\n", (int)se->len, (int)dim2);
                // Changing dimensions
                if (dim2 != se.len)
                {
                    // Copy when changing the string literal
                    size_t newsz = se.sz;
                    size_t d = (dim2 < se.len) ? dim2 : se.len;
                    void* s = mem.xmalloc((dim2 + 1) * newsz);
                    memcpy(s, se.string, d * newsz);
                    // Extend with 0, add terminating 0
                    memset(s + d * newsz, 0, (dim2 + 1 - d) * newsz);
                    se.string = cast(char*)s;
                    se.len = dim2;
                }
            }
            se.type = t;
            result = se;
            return;
        Lcast:
            result = new CastExp(e.loc, se, t);
            result.type = t; // so semantic() won't be run on e
        }

        override void visit(AddrExp e)
        {
            version (none)
            {
                printf("AddrExp::castTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            result = e;

            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();

            if (tb.equals(typeb))
            {
                result = e.copy();
                result.type = t;
                return;
            }

            // Look for pointers to functions where the functions are overloaded.
            if (e.e1.op == TOKoverloadset &&
                (tb.ty == Tpointer || tb.ty == Tdelegate) && tb.nextOf().ty == Tfunction)
            {
                OverExp eo = cast(OverExp)e.e1;
                FuncDeclaration f = null;
                for (size_t i = 0; i < eo.vars.a.dim; i++)
                {
                    auto s = eo.vars.a[i];
                    auto f2 = s.isFuncDeclaration();
                    assert(f2);
                    if (f2.overloadExactMatch(tb.nextOf()))
                    {
                        if (f)
                        {
                            /* Error if match in more than one overload set,
                             * even if one is a 'better' match than the other.
                             */
                            ScopeDsymbol.multiplyDefined(e.loc, f, f2);
                        }
                        else
                            f = f2;
                    }
                }
                if (f)
                {
                    f.tookAddressOf++;
                    auto se = new SymOffExp(e.loc, f, 0, false);
                    se.semantic(sc);
                    // Let SymOffExp::castTo() do the heavy lifting
                    visit(se);
                    return;
                }
            }

            if (e.e1.op == TOKvar &&
                typeb.ty == Tpointer && typeb.nextOf().ty == Tfunction &&
                tb.ty == Tpointer && tb.nextOf().ty == Tfunction)
            {
                auto ve = cast(VarExp)e.e1;
                auto f = ve.var.isFuncDeclaration();
                if (f)
                {
                    assert(f.isImportedSymbol());
                    f = f.overloadExactMatch(tb.nextOf());
                    if (f)
                    {
                        result = new VarExp(e.loc, f, false);
                        result.type = f.type;
                        result = new AddrExp(e.loc, result);
                        result.type = t;
                        return;
                    }
                }
            }

            if (auto f = isFuncAddress(e))
            {
                if (f.checkForwardRef(e.loc))
                {
                    result = new ErrorExp();
                    return;
                }
            }

            visit(cast(Expression)e);
        }

        override void visit(TupleExp e)
        {
            if (e.type.equals(t))
            {
                result = e;
                return;
            }
            TupleExp te = cast(TupleExp)e.copy();
            te.e0 = e.e0 ? e.e0.copy() : null;
            te.exps = e.exps.copy();
            for (size_t i = 0; i < te.exps.dim; i++)
            {
                Expression ex = (*te.exps)[i];
                ex = ex.castTo(sc, t);
                (*te.exps)[i] = ex;
            }
            result = te;
            /* Questionable behavior: In here, result->type is not set to t.
             * Therefoe:
             *  TypeTuple!(int, int) values;
             *  auto values2 = cast(long)values;
             *  // typeof(values2) == TypeTuple!(int, int) !!
             *
             * Only when the casted tuple is immediately expanded, it would work.
             *  auto arr = [cast(long)values];
             *  // typeof(arr) == long[]
             */
        }

        override void visit(ArrayLiteralExp e)
        {
            version (none)
            {
                printf("ArrayLiteralExp::castTo(this=%s, type=%s, => %s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            if (e.type == t)
            {
                result = e;
                return;
            }
            ArrayLiteralExp ae = e;

            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();

            if ((tb.ty == Tarray || tb.ty == Tsarray) &&
                (typeb.ty == Tarray || typeb.ty == Tsarray))
            {
                if (tb.nextOf().toBasetype().ty == Tvoid && typeb.nextOf().toBasetype().ty != Tvoid)
                {
                    // Don't do anything to cast non-void[] to void[]
                }
                else if (typeb.ty == Tsarray && typeb.nextOf().toBasetype().ty == Tvoid)
                {
                    // Don't do anything for casting void[n] to others
                }
                else
                {
                    if (tb.ty == Tsarray)
                    {
                        TypeSArray tsa = cast(TypeSArray)tb;
                        if (e.elements.dim != tsa.dim.toInteger())
                            goto L1;
                    }
                    ae = cast(ArrayLiteralExp)e.copy();
                    if (e.basis)
                        ae.basis = e.basis.castTo(sc, tb.nextOf());
                    ae.elements = e.elements.copy();
                    for (size_t i = 0; i < e.elements.dim; i++)
                    {
                        Expression ex = (*e.elements)[i];
                        if (!ex)
                            continue;
                        ex = ex.castTo(sc, tb.nextOf());
                        (*ae.elements)[i] = ex;
                    }
                    ae.type = t;
                    result = ae;
                    return;
                }
            }
            else if (tb.ty == Tpointer && typeb.ty == Tsarray)
            {
                Type tp = typeb.nextOf().pointerTo();
                if (!tp.equals(ae.type))
                {
                    ae = cast(ArrayLiteralExp)e.copy();
                    ae.type = tp;
                }
            }
            else if (tb.ty == Tvector && (typeb.ty == Tarray || typeb.ty == Tsarray))
            {
                // Convert array literal to vector type
                TypeVector tv = cast(TypeVector)tb;
                TypeSArray tbase = cast(TypeSArray)tv.basetype;
                assert(tbase.ty == Tsarray);
                if (e.elements.dim != tbase.dim.toInteger())
                    goto L1;
                ae = cast(ArrayLiteralExp)e.copy();
                ae.type = tbase; // Bugzilla 12642
                ae.elements = e.elements.copy();
                Type telement = tv.elementType();
                for (size_t i = 0; i < e.elements.dim; i++)
                {
                    Expression ex = (*e.elements)[i];
                    ex = ex.castTo(sc, telement);
                    (*ae.elements)[i] = ex;
                }
                Expression ev = new VectorExp(e.loc, ae, tb);
                ev = ev.semantic(sc);
                result = ev;
                return;
            }
        L1:
            visit(cast(Expression)ae);
        }

        override void visit(AssocArrayLiteralExp e)
        {
            if (e.type == t)
            {
                result = e;
                return;
            }

            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();

            if (tb.ty == Taarray && typeb.ty == Taarray &&
                tb.nextOf().toBasetype().ty != Tvoid)
            {
                AssocArrayLiteralExp ae = cast(AssocArrayLiteralExp)e.copy();
                ae.keys = e.keys.copy();
                ae.values = e.values.copy();
                assert(e.keys.dim == e.values.dim);
                for (size_t i = 0; i < e.keys.dim; i++)
                {
                    Expression ex = (*e.values)[i];
                    ex = ex.castTo(sc, tb.nextOf());
                    (*ae.values)[i] = ex;
                    ex = (*e.keys)[i];
                    ex = ex.castTo(sc, (cast(TypeAArray)tb).index);
                    (*ae.keys)[i] = ex;
                }
                ae.type = t;
                result = ae;
                return;
            }
            visit(cast(Expression)e);
        }

        override void visit(SymOffExp e)
        {
            version (none)
            {
                printf("SymOffExp::castTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            if (e.type == t && !e.hasOverloads)
            {
                result = e;
                return;
            }

            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();

            if (tb.equals(typeb))
            {
                result = e.copy();
                result.type = t;
                (cast(SymOffExp)result).hasOverloads = false;
                return;
            }

            // Look for pointers to functions where the functions are overloaded.
            if (e.hasOverloads &&
                typeb.ty == Tpointer && typeb.nextOf().ty == Tfunction &&
                (tb.ty == Tpointer || tb.ty == Tdelegate) && tb.nextOf().ty == Tfunction)
            {
                FuncDeclaration f = e.var.isFuncDeclaration();
                f = f ? f.overloadExactMatch(tb.nextOf()) : null;
                if (f)
                {
                    if (tb.ty == Tdelegate)
                    {
                        if (f.needThis() && hasThis(sc))
                        {
                            result = new DelegateExp(e.loc, new ThisExp(e.loc), f, false);
                            result = result.semantic(sc);
                        }
                        else if (f.isNested())
                        {
                            result = new DelegateExp(e.loc, new IntegerExp(0), f, false);
                            result = result.semantic(sc);
                        }
                        else if (f.needThis())
                        {
                            e.error("no 'this' to create delegate for %s", f.toChars());
                            result = new ErrorExp();
                            return;
                        }
                        else
                        {
                            e.error("cannot cast from function pointer to delegate");
                            result = new ErrorExp();
                            return;
                        }
                    }
                    else
                    {
                        result = new SymOffExp(e.loc, f, 0, false);
                        result.type = t;
                    }
                    f.tookAddressOf++;
                    return;
                }
            }

            if (auto f = isFuncAddress(e))
            {
                if (f.checkForwardRef(e.loc))
                {
                    result = new ErrorExp();
                    return;
                }
            }

            visit(cast(Expression)e);
        }

        override void visit(DelegateExp e)
        {
            version (none)
            {
                printf("DelegateExp::castTo(this=%s, type=%s, t=%s)\n", e.toChars(), e.type.toChars(), t.toChars());
            }
            static __gshared const(char)* msg = "cannot form delegate due to covariant return type";
            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();

            if (tb.equals(typeb) && !e.hasOverloads)
            {
                int offset;
                e.func.tookAddressOf++;
                if (e.func.tintro && e.func.tintro.nextOf().isBaseOf(e.func.type.nextOf(), &offset) && offset)
                    e.error("%s", msg);
                result = e.copy();
                result.type = t;
                return;
            }

            // Look for delegates to functions where the functions are overloaded.
            if (typeb.ty == Tdelegate && tb.ty == Tdelegate)
            {
                if (e.func)
                {
                    auto f = e.func.overloadExactMatch(tb.nextOf());
                    if (f)
                    {
                        int offset;
                        if (f.tintro && f.tintro.nextOf().isBaseOf(f.type.nextOf(), &offset) && offset)
                            e.error("%s", msg);
                        if (f != e.func)    // if address not already marked as taken
                            f.tookAddressOf++;
                        result = new DelegateExp(e.loc, e.e1, f, false);
                        result.type = t;
                        return;
                    }
                    if (e.func.tintro)
                        e.error("%s", msg);
                }
            }

            if (auto f = isFuncAddress(e))
            {
                if (f.checkForwardRef(e.loc))
                {
                    result = new ErrorExp();
                    return;
                }
            }

            visit(cast(Expression)e);
        }

        override void visit(FuncExp e)
        {
            //printf("FuncExp::castTo type = %s, t = %s\n", e->type->toChars(), t->toChars());
            FuncExp fe;
            if (e.matchType(t, sc, &fe, 1) > MATCHnomatch)
            {
                result = fe;
                return;
            }
            visit(cast(Expression)e);
        }

        override void visit(CondExp e)
        {
            if (!e.type.equals(t))
            {
                result = new CondExp(e.loc, e.econd, e.e1.castTo(sc, t), e.e2.castTo(sc, t));
                result.type = t;
                return;
            }
            result = e;
        }

        override void visit(CommaExp e)
        {
            Expression e2c = e.e2.castTo(sc, t);
            if (e2c != e.e2)
            {
                result = new CommaExp(e.loc, e.e1, e2c);
                result.type = e2c.type;
            }
            else
            {
                result = e;
                result.type = e.e2.type;
            }
        }

        override void visit(SliceExp e)
        {
            //printf("SliceExp::castTo e = %s, type = %s, t = %s\n", e->toChars(), e->type->toChars(), t->toChars());

            Type tb = t.toBasetype();
            Type typeb = e.type.toBasetype();

            if (e.type.equals(t) || typeb.ty != Tarray ||
                (tb.ty != Tarray && tb.ty != Tsarray))
            {
                visit(cast(Expression)e);
                return;
            }
            if (tb.ty == Tarray)
            {
                if (typeb.nextOf().equivalent(tb.nextOf()))
                {
                    // T[] to const(T)[]
                    result = e.copy();
                    result.type = t;
                }
                else
                {
                    visit(cast(Expression)e);
                }
                return;
            }
            // Handle the cast from Tarray to Tsarray with CT-known slicing
            TypeSArray tsa = cast(TypeSArray)toStaticArrayType(e);
            if (tsa && tsa.size(e.loc) == tb.size(e.loc))
            {
                /* Match if the sarray sizes are equal:
                 *  T[a .. b] to const(T)[b-a]
                 *  T[a .. b] to U[dim] if (T.sizeof*(b-a) == U.sizeof*dim)
                 *
                 * If a SliceExp has Tsarray, it will become lvalue.
                 * That's handled in SliceExp::isLvalue and toLvalue
                 */
                result = e.copy();
                result.type = t;
                return;
            }
            if (tsa && tsa.dim.equals((cast(TypeSArray)tb).dim))
            {
                /* Match if the dimensions are equal
                 * with the implicit conversion of e->e1:
                 *  cast(float[2]) [2.0, 1.0, 0.0][0..2];
                 */
                Type t1b = e.e1.type.toBasetype();
                if (t1b.ty == Tsarray)
                    t1b = tb.nextOf().sarrayOf((cast(TypeSArray)t1b).dim.toInteger());
                else if (t1b.ty == Tarray)
                    t1b = tb.nextOf().arrayOf();
                else if (t1b.ty == Tpointer)
                    t1b = tb.nextOf().pointerTo();
                else
                    assert(0);
                if (e.e1.implicitConvTo(t1b) > MATCHnomatch)
                {
                    Expression e1x = e.e1.implicitCastTo(sc, t1b);
                    assert(e1x.op != TOKerror);
                    e = cast(SliceExp)e.copy();
                    e.e1 = e1x;
                    e.type = t;
                    result = e;
                    return;
                }
            }
            e.error("cannot cast expression %s of type %s to %s", e.toChars(), tsa ? tsa.toChars() : e.type.toChars(), t.toChars());
            result = new ErrorExp();
        }
    }

    scope CastTo v = new CastTo(sc, t);
    e.accept(v);
    return v.result;
}

/****************************************
 * Set type inference target
 *      t       Target type
 *      flag    1: don't put an error when inference fails
 */
extern (C++) Expression inferType(Expression e, Type t, int flag = 0)
{
    extern (C++) final class InferType : Visitor
    {
        alias visit = super.visit;
    public:
        Type t;
        int flag;
        Expression result;

        extern (D) this(Type t, int flag)
        {
            this.t = t;
            this.flag = flag;
        }

        override void visit(Expression e)
        {
            result = e;
        }

        override void visit(ArrayLiteralExp ale)
        {
            Type tb = t.toBasetype();
            if (tb.ty == Tarray || tb.ty == Tsarray)
            {
                Type tn = tb.nextOf();
                if (ale.basis)
                    ale.basis = inferType(ale.basis, tn, flag);
                for (size_t i = 0; i < ale.elements.dim; i++)
                {
                    Expression e = (*ale.elements)[i];
                    if (e)
                    {
                        e = inferType(e, tn, flag);
                        (*ale.elements)[i] = e;
                    }
                }
            }
            result = ale;
        }

        override void visit(AssocArrayLiteralExp aale)
        {
            Type tb = t.toBasetype();
            if (tb.ty == Taarray)
            {
                TypeAArray taa = cast(TypeAArray)tb;
                Type ti = taa.index;
                Type tv = taa.nextOf();
                for (size_t i = 0; i < aale.keys.dim; i++)
                {
                    Expression e = (*aale.keys)[i];
                    if (e)
                    {
                        e = inferType(e, ti, flag);
                        (*aale.keys)[i] = e;
                    }
                }
                for (size_t i = 0; i < aale.values.dim; i++)
                {
                    Expression e = (*aale.values)[i];
                    if (e)
                    {
                        e = inferType(e, tv, flag);
                        (*aale.values)[i] = e;
                    }
                }
            }
            result = aale;
        }

        override void visit(FuncExp fe)
        {
            //printf("FuncExp::inferType('%s'), to=%s\n", fe->type ? fe->type->toChars() : "null", t->toChars());
            if (t.ty == Tdelegate || t.ty == Tpointer && t.nextOf().ty == Tfunction)
            {
                fe.fd.treq = t;
            }
            result = fe;
        }

        override void visit(CondExp ce)
        {
            Type tb = t.toBasetype();
            ce.e1 = inferType(ce.e1, tb, flag);
            ce.e2 = inferType(ce.e2, tb, flag);
            result = ce;
        }
    }

    if (!t)
        return e;
    scope InferType v = new InferType(t, flag);
    e.accept(v);
    return v.result;
}

/****************************************
 * Scale addition/subtraction to/from pointer.
 */
extern (C++) Expression scaleFactor(BinExp be, Scope* sc)
{
    Type t1b = be.e1.type.toBasetype();
    Type t2b = be.e2.type.toBasetype();
    Expression eoff;
    if (t1b.ty == Tpointer && t2b.isintegral())
    {
        // Need to adjust operator by the stride
        // Replace (ptr + int) with (ptr + (int * stride))
        Type t = Type.tptrdiff_t;
        d_uns64 stride = t1b.nextOf().size(be.loc);
        if (!t.equals(t2b))
            be.e2 = be.e2.castTo(sc, t);
        eoff = be.e2;
        be.e2 = new MulExp(be.loc, be.e2, new IntegerExp(Loc(), stride, t));
        be.e2.type = t;
        be.type = be.e1.type;
    }
    else if (t2b.ty == Tpointer && t1b.isintegral())
    {
        // Need to adjust operator by the stride
        // Replace (int + ptr) with (ptr + (int * stride))
        Type t = Type.tptrdiff_t;
        Expression e;
        d_uns64 stride = t2b.nextOf().size(be.loc);
        if (!t.equals(t1b))
            e = be.e1.castTo(sc, t);
        else
            e = be.e1;
        eoff = e;
        e = new MulExp(be.loc, e, new IntegerExp(Loc(), stride, t));
        e.type = t;
        be.type = be.e2.type;
        be.e1 = be.e2;
        be.e2 = e;
    }
    else
        assert(0);
    if (sc.func && !sc.intypeof)
    {
        eoff = eoff.optimize(WANTvalue);
        if (eoff.op == TOKint64 && eoff.toInteger() == 0)
        {
        }
        else if (sc.func.setUnsafe())
        {
            be.error("pointer arithmetic not allowed in @safe functions");
            return new ErrorExp();
        }
    }
    return be;
}

/**************************************
 * Return true if e is an empty array literal with dimensionality
 * equal to or less than type of other array.
 * [], [[]], [[[]]], etc.
 * I.e., make sure that [1,2] is compatible with [],
 * [[1,2]] is compatible with [[]], etc.
 */
extern (C++) bool isVoidArrayLiteral(Expression e, Type other)
{
    while (e.op == TOKarrayliteral && e.type.ty == Tarray && ((cast(ArrayLiteralExp)e).elements.dim == 1))
    {
        auto ale = cast(ArrayLiteralExp)e;
        e = ale.getElement(0);
        if (other.ty == Tsarray || other.ty == Tarray)
            other = other.nextOf();
        else
            return false;
    }
    if (other.ty != Tsarray && other.ty != Tarray)
        return false;
    Type t = e.type;
    return (e.op == TOKarrayliteral && t.ty == Tarray && t.nextOf().ty == Tvoid && (cast(ArrayLiteralExp)e).elements.dim == 0);
}

// used by deduceType()
extern (C++) Type rawTypeMerge(Type t1, Type t2)
{
    if (t1.equals(t2))
        return t1;
    if (t1.equivalent(t2))
        return t1.castMod(MODmerge(t1.mod, t2.mod));

    auto t1b = t1.toBasetype();
    auto t2b = t2.toBasetype();
    if (t1b.equals(t2b))
        return t1b;
    if (t1b.equivalent(t2b))
        return t1b.castMod(MODmerge(t1b.mod, t2b.mod));

    auto ty = cast(TY)impcnvResult[t1b.ty][t2b.ty];
    if (ty != Terror)
        return Type.basic[ty];

    return null;
}

/**************************************
 * Combine types.
 * Output:
 *      *pt     merged type, if *pt is not NULL
 *      *pe1    rewritten e1
 *      *pe2    rewritten e2
 * Returns:
 *      true    success
 *      false   failed
 */
extern (C++) bool typeMerge(Scope* sc, TOK op, Type* pt, Expression* pe1, Expression* pe2)
{
    //printf("typeMerge() %s op %s\n", pe1.toChars(), pe2.toChars());
    MATCH m;
    Expression e1 = *pe1;
    Expression e2 = *pe2;
    Type t1b = e1.type.toBasetype();
    Type t2b = e2.type.toBasetype();
    if (op != TOKquestion || t1b.ty != t2b.ty && (t1b.isTypeBasic() && t2b.isTypeBasic()))
    {
        e1 = integralPromotions(e1, sc);
        e2 = integralPromotions(e2, sc);
    }
    Type t1 = e1.type;
    Type t2 = e2.type;
    assert(t1);
    Type t = t1;
    /* The start type of alias this type recursion.
     * In following case, we should save A, and stop recursion
     * if it appears again.
     *      X -> Y -> [A] -> B -> A -> B -> ...
     */
    Type att1 = null;
    Type att2 = null;
    //if (t1) printf("\tt1 = %s\n", t1->toChars());
    //if (t2) printf("\tt2 = %s\n", t2->toChars());
    debug
    {
        if (!t2)
            printf("\te2 = '%s'\n", e2.toChars());
    }
    assert(t2);
Lagain:
    t1b = t1.toBasetype();
    t2b = t2.toBasetype();
    TY ty = cast(TY)impcnvResult[t1b.ty][t2b.ty];
    if (ty != Terror)
    {
        TY ty1 = cast(TY)impcnvType1[t1b.ty][t2b.ty];
        TY ty2 = cast(TY)impcnvType2[t1b.ty][t2b.ty];
        if (t1b.ty == ty1) // if no promotions
        {
            if (t1.equals(t2))
            {
                t = t1;
                goto Lret;
            }
            if (t1b.equals(t2b))
            {
                t = t1b;
                goto Lret;
            }
        }
        t = Type.basic[ty];
        t1 = Type.basic[ty1];
        t2 = Type.basic[ty2];
        e1 = e1.castTo(sc, t1);
        e2 = e2.castTo(sc, t2);
        //printf("after typeCombine():\n");
        //print();
        //printf("ty = %d, ty1 = %d, ty2 = %d\n", ty, ty1, ty2);
        goto Lret;
    }
    t1 = t1b;
    t2 = t2b;
    if (t1.ty == Ttuple || t2.ty == Ttuple)
        goto Lincompatible;
    if (t1.equals(t2))
    {
        // merging can not result in new enum type
        if (t.ty == Tenum)
            t = t1b;
    }
    else if ((t1.ty == Tpointer && t2.ty == Tpointer) || (t1.ty == Tdelegate && t2.ty == Tdelegate))
    {
        // Bring pointers to compatible type
        Type t1n = t1.nextOf();
        Type t2n = t2.nextOf();
        if (t1n.equals(t2n))
        {
        }
        else if (t1n.ty == Tvoid) // pointers to void are always compatible
            t = t2;
        else if (t2n.ty == Tvoid)
        {
        }
        else if (t1.implicitConvTo(t2))
        {
            goto Lt2;
        }
        else if (t2.implicitConvTo(t1))
        {
            goto Lt1;
        }
        else if (t1n.ty == Tfunction && t2n.ty == Tfunction)
        {
            TypeFunction tf1 = cast(TypeFunction)t1n;
            TypeFunction tf2 = cast(TypeFunction)t2n;
            tf1.purityLevel();
            tf2.purityLevel();
            TypeFunction d = cast(TypeFunction)tf1.syntaxCopy();
            if (tf1.purity != tf2.purity)
                d.purity = PUREimpure;
            assert(d.purity != PUREfwdref);
            d.isnothrow = (tf1.isnothrow && tf2.isnothrow);
            d.isnogc = (tf1.isnogc && tf2.isnogc);
            if (tf1.trust == tf2.trust)
                d.trust = tf1.trust;
            else if (tf1.trust <= TRUSTsystem || tf2.trust <= TRUSTsystem)
                d.trust = TRUSTsystem;
            else
                d.trust = TRUSTtrusted;
            Type tx = null;
            if (t1.ty == Tdelegate)
            {
                tx = new TypeDelegate(d);
            }
            else
                tx = d.pointerTo();
            tx = tx.semantic(e1.loc, sc);
            if (t1.implicitConvTo(tx) && t2.implicitConvTo(tx))
            {
                t = tx;
                e1 = e1.castTo(sc, t);
                e2 = e2.castTo(sc, t);
                goto Lret;
            }
            goto Lincompatible;
        }
        else if (t1n.mod != t2n.mod)
        {
            if (!t1n.isImmutable() && !t2n.isImmutable() && t1n.isShared() != t2n.isShared())
                goto Lincompatible;
            ubyte mod = MODmerge(t1n.mod, t2n.mod);
            t1 = t1n.castMod(mod).pointerTo();
            t2 = t2n.castMod(mod).pointerTo();
            t = t1;
            goto Lagain;
        }
        else if (t1n.ty == Tclass && t2n.ty == Tclass)
        {
            ClassDeclaration cd1 = t1n.isClassHandle();
            ClassDeclaration cd2 = t2n.isClassHandle();
            int offset;
            if (cd1.isBaseOf(cd2, &offset))
            {
                if (offset)
                    e2 = e2.castTo(sc, t);
            }
            else if (cd2.isBaseOf(cd1, &offset))
            {
                t = t2;
                if (offset)
                    e1 = e1.castTo(sc, t);
            }
            else
                goto Lincompatible;
        }
        else
        {
            t1 = t1n.constOf().pointerTo();
            t2 = t2n.constOf().pointerTo();
            if (t1.implicitConvTo(t2))
            {
                goto Lt2;
            }
            else if (t2.implicitConvTo(t1))
            {
                goto Lt1;
            }
            goto Lincompatible;
        }
    }
    else if ((t1.ty == Tsarray || t1.ty == Tarray) && (e2.op == TOKnull && t2.ty == Tpointer && t2.nextOf().ty == Tvoid || e2.op == TOKarrayliteral && t2.ty == Tsarray && t2.nextOf().ty == Tvoid && (cast(TypeSArray)t2).dim.toInteger() == 0 || isVoidArrayLiteral(e2, t1)))
    {
        /*  (T[n] op void*)   => T[]
         *  (T[]  op void*)   => T[]
         *  (T[n] op void[0]) => T[]
         *  (T[]  op void[0]) => T[]
         *  (T[n] op void[])  => T[]
         *  (T[]  op void[])  => T[]
         */
        goto Lx1;
    }
    else if ((t2.ty == Tsarray || t2.ty == Tarray) && (e1.op == TOKnull && t1.ty == Tpointer && t1.nextOf().ty == Tvoid || e1.op == TOKarrayliteral && t1.ty == Tsarray && t1.nextOf().ty == Tvoid && (cast(TypeSArray)t1).dim.toInteger() == 0 || isVoidArrayLiteral(e1, t2)))
    {
        /*  (void*   op T[n]) => T[]
         *  (void*   op T[])  => T[]
         *  (void[0] op T[n]) => T[]
         *  (void[0] op T[])  => T[]
         *  (void[]  op T[n]) => T[]
         *  (void[]  op T[])  => T[]
         */
        goto Lx2;
    }
    else if ((t1.ty == Tsarray || t1.ty == Tarray) && (m = t1.implicitConvTo(t2)) != MATCHnomatch)
    {
        // Bugzilla 7285: Tsarray op [x, y, ...] should to be Tsarray
        // Bugzilla 14737: Tsarray ~ [x, y, ...] should to be Tarray
        if (t1.ty == Tsarray && e2.op == TOKarrayliteral && op != TOKcat)
            goto Lt1;
        if (m == MATCHconst && (op == TOKaddass || op == TOKminass || op == TOKmulass || op == TOKdivass || op == TOKmodass || op == TOKpowass || op == TOKandass || op == TOKorass || op == TOKxorass))
        {
            // Don't make the lvalue const
            t = t2;
            goto Lret;
        }
        goto Lt2;
    }
    else if ((t2.ty == Tsarray || t2.ty == Tarray) && t2.implicitConvTo(t1))
    {
        // Bugzilla 7285 & 14737
        if (t2.ty == Tsarray && e1.op == TOKarrayliteral && op != TOKcat)
            goto Lt2;
        goto Lt1;
    }
    else if ((t1.ty == Tsarray || t1.ty == Tarray || t1.ty == Tpointer) && (t2.ty == Tsarray || t2.ty == Tarray || t2.ty == Tpointer) && t1.nextOf().mod != t2.nextOf().mod)
    {
        /* If one is mutable and the other invariant, then retry
         * with both of them as const
         */
        Type t1n = t1.nextOf();
        Type t2n = t2.nextOf();
        ubyte mod;
        if (e1.op == TOKnull && e2.op != TOKnull)
            mod = t2n.mod;
        else if (e1.op != TOKnull && e2.op == TOKnull)
            mod = t1n.mod;
        else if (!t1n.isImmutable() && !t2n.isImmutable() && t1n.isShared() != t2n.isShared())
            goto Lincompatible;
        else
            mod = MODmerge(t1n.mod, t2n.mod);
        if (t1.ty == Tpointer)
            t1 = t1n.castMod(mod).pointerTo();
        else
            t1 = t1n.castMod(mod).arrayOf();
        if (t2.ty == Tpointer)
            t2 = t2n.castMod(mod).pointerTo();
        else
            t2 = t2n.castMod(mod).arrayOf();
        t = t1;
        goto Lagain;
    }
    else if (t1.ty == Tclass && t2.ty == Tclass)
    {
        if (t1.mod != t2.mod)
        {
            ubyte mod;
            if (e1.op == TOKnull && e2.op != TOKnull)
                mod = t2.mod;
            else if (e1.op != TOKnull && e2.op == TOKnull)
                mod = t1.mod;
            else if (!t1.isImmutable() && !t2.isImmutable() && t1.isShared() != t2.isShared())
                goto Lincompatible;
            else
                mod = MODmerge(t1.mod, t2.mod);
            t1 = t1.castMod(mod);
            t2 = t2.castMod(mod);
            t = t1;
            goto Lagain;
        }
        goto Lcc;
    }
    else if (t1.ty == Tclass || t2.ty == Tclass)
    {
    Lcc:
        while (1)
        {
            MATCH i1 = e2.implicitConvTo(t1);
            MATCH i2 = e1.implicitConvTo(t2);
            if (i1 && i2)
            {
                // We have the case of class vs. void*, so pick class
                if (t1.ty == Tpointer)
                    i1 = MATCHnomatch;
                else if (t2.ty == Tpointer)
                    i2 = MATCHnomatch;
            }
            if (i2)
            {
                e2 = e2.castTo(sc, t2);
                goto Lt2;
            }
            else if (i1)
            {
                e1 = e1.castTo(sc, t1);
                goto Lt1;
            }
            else if (t1.ty == Tclass && t2.ty == Tclass)
            {
                TypeClass tc1 = cast(TypeClass)t1;
                TypeClass tc2 = cast(TypeClass)t2;
                /* Pick 'tightest' type
                 */
                ClassDeclaration cd1 = tc1.sym.baseClass;
                ClassDeclaration cd2 = tc2.sym.baseClass;
                if (cd1 && cd2)
                {
                    t1 = cd1.type.castMod(t1.mod);
                    t2 = cd2.type.castMod(t2.mod);
                }
                else if (cd1)
                    t1 = cd1.type;
                else if (cd2)
                    t2 = cd2.type;
                else
                    goto Lincompatible;
            }
            else if (t1.ty == Tstruct && (cast(TypeStruct)t1).sym.aliasthis)
            {
                if (att1 && e1.type == att1)
                    goto Lincompatible;
                if (!att1 && e1.type.checkAliasThisRec())
                    att1 = e1.type;
                //printf("att tmerge(c || c) e1 = %s\n", e1->type->toChars());
                e1 = resolveAliasThis(sc, e1);
                t1 = e1.type;
                continue;
            }
            else if (t2.ty == Tstruct && (cast(TypeStruct)t2).sym.aliasthis)
            {
                if (att2 && e2.type == att2)
                    goto Lincompatible;
                if (!att2 && e2.type.checkAliasThisRec())
                    att2 = e2.type;
                //printf("att tmerge(c || c) e2 = %s\n", e2->type->toChars());
                e2 = resolveAliasThis(sc, e2);
                t2 = e2.type;
                continue;
            }
            else
                goto Lincompatible;
        }
    }
    else if (t1.ty == Tstruct && t2.ty == Tstruct)
    {
        if (t1.mod != t2.mod)
        {
            if (!t1.isImmutable() && !t2.isImmutable() && t1.isShared() != t2.isShared())
                goto Lincompatible;
            ubyte mod = MODmerge(t1.mod, t2.mod);
            t1 = t1.castMod(mod);
            t2 = t2.castMod(mod);
            t = t1;
            goto Lagain;
        }
        TypeStruct ts1 = cast(TypeStruct)t1;
        TypeStruct ts2 = cast(TypeStruct)t2;
        if (ts1.sym != ts2.sym)
        {
            if (!ts1.sym.aliasthis && !ts2.sym.aliasthis)
                goto Lincompatible;
            MATCH i1 = MATCHnomatch;
            MATCH i2 = MATCHnomatch;
            Expression e1b = null;
            Expression e2b = null;
            if (ts2.sym.aliasthis)
            {
                if (att2 && e2.type == att2)
                    goto Lincompatible;
                if (!att2 && e2.type.checkAliasThisRec())
                    att2 = e2.type;
                //printf("att tmerge(s && s) e2 = %s\n", e2->type->toChars());
                e2b = resolveAliasThis(sc, e2);
                i1 = e2b.implicitConvTo(t1);
            }
            if (ts1.sym.aliasthis)
            {
                if (att1 && e1.type == att1)
                    goto Lincompatible;
                if (!att1 && e1.type.checkAliasThisRec())
                    att1 = e1.type;
                //printf("att tmerge(s && s) e1 = %s\n", e1->type->toChars());
                e1b = resolveAliasThis(sc, e1);
                i2 = e1b.implicitConvTo(t2);
            }
            if (i1 && i2)
                goto Lincompatible;
            if (i1)
                goto Lt1;
            else if (i2)
                goto Lt2;
            if (e1b)
            {
                e1 = e1b;
                t1 = e1b.type.toBasetype();
            }
            if (e2b)
            {
                e2 = e2b;
                t2 = e2b.type.toBasetype();
            }
            t = t1;
            goto Lagain;
        }
    }
    else if (t1.ty == Tstruct || t2.ty == Tstruct)
    {
        if (t1.ty == Tstruct && (cast(TypeStruct)t1).sym.aliasthis)
        {
            if (att1 && e1.type == att1)
                goto Lincompatible;
            if (!att1 && e1.type.checkAliasThisRec())
                att1 = e1.type;
            //printf("att tmerge(s || s) e1 = %s\n", e1->type->toChars());
            e1 = resolveAliasThis(sc, e1);
            t1 = e1.type;
            t = t1;
            goto Lagain;
        }
        if (t2.ty == Tstruct && (cast(TypeStruct)t2).sym.aliasthis)
        {
            if (att2 && e2.type == att2)
                goto Lincompatible;
            if (!att2 && e2.type.checkAliasThisRec())
                att2 = e2.type;
            //printf("att tmerge(s || s) e2 = %s\n", e2->type->toChars());
            e2 = resolveAliasThis(sc, e2);
            t2 = e2.type;
            t = t2;
            goto Lagain;
        }
        goto Lincompatible;
    }
    else if ((e1.op == TOKstring || e1.op == TOKnull) && e1.implicitConvTo(t2))
    {
        goto Lt2;
    }
    else if ((e2.op == TOKstring || e2.op == TOKnull) && e2.implicitConvTo(t1))
    {
        goto Lt1;
    }
    else if (t1.ty == Tsarray && t2.ty == Tsarray && e2.implicitConvTo(t1.nextOf().arrayOf()))
    {
    Lx1:
        t = t1.nextOf().arrayOf(); // T[]
        e1 = e1.castTo(sc, t);
        e2 = e2.castTo(sc, t);
    }
    else if (t1.ty == Tsarray && t2.ty == Tsarray && e1.implicitConvTo(t2.nextOf().arrayOf()))
    {
    Lx2:
        t = t2.nextOf().arrayOf();
        e1 = e1.castTo(sc, t);
        e2 = e2.castTo(sc, t);
    }
    else if (t1.ty == Tvector && t2.ty == Tvector)
    {
        // Bugzilla 13841, all vector types should have no common types between
        // different vectors, even though their sizes are same.
        goto Lincompatible;
    }
    else if (t1.ty == Tvector && t2.ty != Tvector && e2.implicitConvTo(t1))
    {
        e2 = e2.castTo(sc, t1);
        t2 = t1;
        t = t1;
        goto Lagain;
    }
    else if (t2.ty == Tvector && t1.ty != Tvector && e1.implicitConvTo(t2))
    {
        e1 = e1.castTo(sc, t2);
        t1 = t2;
        t = t1;
        goto Lagain;
    }
    else if (t1.isintegral() && t2.isintegral())
    {
        if (t1.ty != t2.ty)
        {
            e1 = integralPromotions(e1, sc);
            e2 = integralPromotions(e2, sc);
            t1 = e1.type;
            t2 = e2.type;
            goto Lagain;
        }
        assert(t1.ty == t2.ty);
        if (!t1.isImmutable() && !t2.isImmutable() && t1.isShared() != t2.isShared())
            goto Lincompatible;
        ubyte mod = MODmerge(t1.mod, t2.mod);
        t1 = t1.castMod(mod);
        t2 = t2.castMod(mod);
        t = t1;
        e1 = e1.castTo(sc, t);
        e2 = e2.castTo(sc, t);
        goto Lagain;
    }
    else if (t1.ty == Tnull && t2.ty == Tnull)
    {
        ubyte mod = MODmerge(t1.mod, t2.mod);
        t = t1.castMod(mod);
        e1 = e1.castTo(sc, t);
        e2 = e2.castTo(sc, t);
        goto Lret;
    }
    else if (t2.ty == Tnull && (t1.ty == Tpointer || t1.ty == Taarray || t1.ty == Tarray))
    {
        goto Lt1;
    }
    else if (t1.ty == Tnull && (t2.ty == Tpointer || t2.ty == Taarray || t2.ty == Tarray))
    {
        goto Lt2;
    }
    else if (t1.ty == Tarray && isBinArrayOp(op) && isArrayOpOperand(e1))
    {
        if (e2.implicitConvTo(t1.nextOf()))
        {
            // T[] op T
            // T[] op cast(T)U
            e2 = e2.castTo(sc, t1.nextOf());
            t = t1.nextOf().arrayOf();
        }
        else if (t1.nextOf().implicitConvTo(e2.type))
        {
            // (cast(T)U)[] op T    (Bugzilla 12780)
            // e1 is left as U[], it will be handled in arrayOp() later.
            t = e2.type.arrayOf();
        }
        else if (t2.ty == Tarray && isArrayOpOperand(e2))
        {
            if (t1.nextOf().implicitConvTo(t2.nextOf()))
            {
                // (cast(T)U)[] op T[]  (Bugzilla 12780)
                // e1 is left as U[], it will be handled in arrayOp() later.
                t = t2.nextOf().arrayOf();
            }
            else if (t2.nextOf().implicitConvTo(t1.nextOf()))
            {
                // T[] op (cast(T)U)[]  (Bugzilla 12780)
                // e2 is left as U[], it will be handled in arrayOp() later.
                t = t1.nextOf().arrayOf();
            }
            else
                goto Lincompatible;
        }
        else
            goto Lincompatible;
    }
    else if (t2.ty == Tarray && isBinArrayOp(op) && isArrayOpOperand(e2))
    {
        if (e1.implicitConvTo(t2.nextOf()))
        {
            // T op T[]
            // cast(T)U op T[]
            e1 = e1.castTo(sc, t2.nextOf());
            t = t2.nextOf().arrayOf();
        }
        else if (t2.nextOf().implicitConvTo(e1.type))
        {
            // T op (cast(T)U)[]    (Bugzilla 12780)
            // e2 is left as U[], it will be handled in arrayOp() later.
            t = e1.type.arrayOf();
        }
        else
            goto Lincompatible;
        //printf("test %s\n", Token::toChars(op));
        e1 = e1.optimize(WANTvalue);
        if (isCommutative(op) && e1.isConst())
        {
            /* Swap operands to minimize number of functions generated
             */
            //printf("swap %s\n", Token::toChars(op));
            Expression tmp = e1;
            e1 = e2;
            e2 = tmp;
        }
    }
    else
    {
    Lincompatible:
        return false;
    }
Lret:
    if (!*pt)
        *pt = t;
    *pe1 = e1;
    *pe2 = e2;
    version (none)
    {
        printf("-typeMerge() %s op %s\n", e1.toChars(), e2.toChars());
        if (e1.type)
            printf("\tt1 = %s\n", e1.type.toChars());
        if (e2.type)
            printf("\tt2 = %s\n", e2.type.toChars());
        printf("\ttype = %s\n", t.toChars());
    }
    //print();
    return true;
Lt1:
    e2 = e2.castTo(sc, t1);
    t = t1;
    goto Lret;
Lt2:
    e1 = e1.castTo(sc, t2);
    t = t2;
    goto Lret;
}

/************************************
 * Bring leaves to common type.
 * Returns ErrorExp if error occurs. otherwise returns NULL.
 */
extern (C++) Expression typeCombine(BinExp be, Scope* sc)
{
    Type t1 = be.e1.type.toBasetype();
    Type t2 = be.e2.type.toBasetype();
    if (be.op == TOKmin || be.op == TOKadd)
    {
        // struct+struct, and class+class are errors
        if (t1.ty == Tstruct && t2.ty == Tstruct)
            goto Lerror;
        else if (t1.ty == Tclass && t2.ty == Tclass)
            goto Lerror;
        else if (t1.ty == Taarray && t2.ty == Taarray)
            goto Lerror;
    }
    if (!typeMerge(sc, be.op, &be.type, &be.e1, &be.e2))
        goto Lerror;
    // If the types have no value, return an error
    if (be.e1.op == TOKerror)
        return be.e1;
    if (be.e2.op == TOKerror)
        return be.e2;
    return null;
Lerror:
    Expression ex = be.incompatibleTypes();
    if (ex.op == TOKerror)
        return ex;
    return new ErrorExp();
}

/***********************************
 * Do integral promotions (convertchk).
 * Don't convert <array of> to <pointer to>
 */
extern (C++) Expression integralPromotions(Expression e, Scope* sc)
{
    //printf("integralPromotions %s %s\n", e->toChars(), e->type->toChars());
    switch (e.type.toBasetype().ty)
    {
    case Tvoid:
        e.error("void has no value");
        return new ErrorExp();
    case Tint8:
    case Tuns8:
    case Tint16:
    case Tuns16:
    case Tbool:
    case Tchar:
    case Twchar:
        e = e.castTo(sc, Type.tint32);
        break;
    case Tdchar:
        e = e.castTo(sc, Type.tuns32);
        break;
    default:
        break;
    }
    return e;
}

/***********************************
 * See if both types are arrays that can be compared
 * for equality. Return true if so.
 * If they are arrays, but incompatible, issue error.
 * This is to enable comparing things like an immutable
 * array with a mutable one.
 */
extern (C++) bool arrayTypeCompatible(Loc loc, Type t1, Type t2)
{
    t1 = t1.toBasetype().merge2();
    t2 = t2.toBasetype().merge2();
    if ((t1.ty == Tarray || t1.ty == Tsarray || t1.ty == Tpointer) && (t2.ty == Tarray || t2.ty == Tsarray || t2.ty == Tpointer))
    {
        if (t1.nextOf().implicitConvTo(t2.nextOf()) < MATCHconst && t2.nextOf().implicitConvTo(t1.nextOf()) < MATCHconst && (t1.nextOf().ty != Tvoid && t2.nextOf().ty != Tvoid))
        {
            error(loc, "array equality comparison type mismatch, %s vs %s", t1.toChars(), t2.toChars());
        }
        return true;
    }
    return false;
}

/***********************************
 * See if both types are arrays that can be compared
 * for equality without any casting. Return true if so.
 * This is to enable comparing things like an immutable
 * array with a mutable one.
 */
extern (C++) bool arrayTypeCompatibleWithoutCasting(Loc loc, Type t1, Type t2)
{
    t1 = t1.toBasetype();
    t2 = t2.toBasetype();
    if ((t1.ty == Tarray || t1.ty == Tsarray || t1.ty == Tpointer) && t2.ty == t1.ty)
    {
        if (t1.nextOf().implicitConvTo(t2.nextOf()) >= MATCHconst || t2.nextOf().implicitConvTo(t1.nextOf()) >= MATCHconst)
            return true;
    }
    return false;
}

/******************************************************************/
/* Determine the integral ranges of an expression.
 * This is used to determine if implicit narrowing conversions will
 * be allowed.
 */
extern (C++) IntRange getIntRange(Expression e)
{
    extern (C++) final class IntRangeVisitor : Visitor
    {
        alias visit = super.visit;
    private:
        static uinteger_t getMask(uinteger_t v)
        {
            // Ref: http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v |= v >> 32;
            return v;
        }

        // The algorithms for &, |, ^ are not yet the best! Sometimes they will produce
        //  not the tightest bound. See
        //      https://github.com/D-Programming-Language/dmd/pull/116
        //  for detail.
        static IntRange unsignedBitwiseAnd(ref const(IntRange) a, ref const(IntRange) b)
        {
            // the DiffMasks stores the mask of bits which are variable in the range.
            uinteger_t aDiffMask = getMask(a.imin.value ^ a.imax.value);
            uinteger_t bDiffMask = getMask(b.imin.value ^ b.imax.value);
            // Since '&' computes the digitwise-minimum, the we could set all varying
            //  digits to 0 to get a lower bound, and set all varying digits to 1 to get
            //  an upper bound.
            IntRange result;
            result.imin.value = (a.imin.value & ~aDiffMask) & (b.imin.value & ~bDiffMask);
            result.imax.value = (a.imax.value | aDiffMask) & (b.imax.value | bDiffMask);
            // Sometimes the upper bound is overestimated. The upper bound will never
            //  exceed the input.
            if (result.imax.value > a.imax.value)
                result.imax.value = a.imax.value;
            if (result.imax.value > b.imax.value)
                result.imax.value = b.imax.value;
            result.imin.negative = result.imax.negative = a.imin.negative && b.imin.negative;
            return result;
        }

        static IntRange unsignedBitwiseOr(ref const(IntRange) a, ref const(IntRange) b)
        {
            // the DiffMasks stores the mask of bits which are variable in the range.
            uinteger_t aDiffMask = getMask(a.imin.value ^ a.imax.value);
            uinteger_t bDiffMask = getMask(b.imin.value ^ b.imax.value);
            // The imax algorithm by Adam D. Ruppe.
            // http://www.digitalmars.com/pnews/read.php?server=news.digitalmars.com&group=digitalmars.D&artnum=108796
            IntRange result;
            result.imin.value = (a.imin.value & ~aDiffMask) | (b.imin.value & ~bDiffMask);
            result.imax.value = a.imax.value | b.imax.value | getMask(a.imax.value & b.imax.value);
            // Sometimes the lower bound is underestimated. The lower bound will never
            //  less than the input.
            if (result.imin.value < a.imin.value)
                result.imin.value = a.imin.value;
            if (result.imin.value < b.imin.value)
                result.imin.value = b.imin.value;
            result.imin.negative = result.imax.negative = a.imin.negative || b.imin.negative;
            return result;
        }

        static IntRange unsignedBitwiseXor(ref const(IntRange) a, ref const(IntRange) b)
        {
            // the DiffMasks stores the mask of bits which are variable in the range.
            uinteger_t aDiffMask = getMask(a.imin.value ^ a.imax.value);
            uinteger_t bDiffMask = getMask(b.imin.value ^ b.imax.value);
            IntRange result;
            result.imin.value = (a.imin.value ^ b.imin.value) & ~(aDiffMask | bDiffMask);
            result.imax.value = (a.imax.value ^ b.imax.value) | (aDiffMask | bDiffMask);
            result.imin.negative = result.imax.negative = a.imin.negative != b.imin.negative;
            return result;
        }

    public:
        IntRange range;

        override void visit(Expression e)
        {
            range = IntRange.fromType(e.type);
        }

        override void visit(IntegerExp e)
        {
            range = IntRange(SignExtendedNumber(e.getInteger()))._cast(e.type);
        }

        override void visit(CastExp e)
        {
            range = getIntRange(e.e1)._cast(e.type);
        }

        override void visit(AddExp e)
        {
            IntRange ir1 = getIntRange(e.e1);
            IntRange ir2 = getIntRange(e.e2);
            range = IntRange(ir1.imin + ir2.imin, ir1.imax + ir2.imax)._cast(e.type);
        }

        override void visit(MinExp e)
        {
            IntRange ir1 = getIntRange(e.e1);
            IntRange ir2 = getIntRange(e.e2);
            range = IntRange(ir1.imin - ir2.imax, ir1.imax - ir2.imin)._cast(e.type);
        }

        override void visit(DivExp e)
        {
            IntRange ir1 = getIntRange(e.e1);
            IntRange ir2 = getIntRange(e.e2);
            // Should we ignore the possibility of div-by-0???
            if (ir2.containsZero())
            {
                visit(cast(Expression)e);
                return;
            }
            // [a,b] / [c,d] = [min (a/c, a/d, b/c, b/d), max (a/c, a/d, b/c, b/d)]
            SignExtendedNumber[4] bdy;
            bdy[0] = ir1.imin / ir2.imin;
            bdy[1] = ir1.imin / ir2.imax;
            bdy[2] = ir1.imax / ir2.imin;
            bdy[3] = ir1.imax / ir2.imax;
            range = IntRange.fromNumbers4(bdy.ptr)._cast(e.type);
        }

        override void visit(MulExp e)
        {
            IntRange ir1 = getIntRange(e.e1);
            IntRange ir2 = getIntRange(e.e2);
            // [a,b] * [c,d] = [min (ac, ad, bc, bd), max (ac, ad, bc, bd)]
            SignExtendedNumber[4] bdy;
            bdy[0] = ir1.imin * ir2.imin;
            bdy[1] = ir1.imin * ir2.imax;
            bdy[2] = ir1.imax * ir2.imin;
            bdy[3] = ir1.imax * ir2.imax;
            range = IntRange.fromNumbers4(bdy.ptr)._cast(e.type);
        }

        override void visit(ModExp e)
        {
            IntRange irNum = getIntRange(e.e1);
            IntRange irDen = getIntRange(e.e2).absNeg();
            /*
             due to the rules of D (C)'s % operator, we need to consider the cases
             separately in different range of signs.

             case 1. [500, 1700] % [7, 23] (numerator is always positive)
             = [0, 22]
             case 2. [-500, 1700] % [7, 23] (numerator can be negative)
             = [-22, 22]
             case 3. [-1700, -500] % [7, 23] (numerator is always negative)
             = [-22, 0]

             the number 22 is the maximum absolute value in the denomator's range. We
             don't care about divide by zero.
             */
            // Modding on 0 is invalid anyway.
            if (!irDen.imin.negative)
            {
                visit(cast(Expression)e);
                return;
            }
            ++irDen.imin;
            irDen.imax = -irDen.imin;
            if (!irNum.imin.negative)
                irNum.imin.value = 0;
            else if (irNum.imin < irDen.imin)
                irNum.imin = irDen.imin;
            if (irNum.imax.negative)
            {
                irNum.imax.negative = false;
                irNum.imax.value = 0;
            }
            else if (irNum.imax > irDen.imax)
                irNum.imax = irDen.imax;
            range = irNum._cast(e.type);
        }

        override void visit(AndExp e)
        {
            IntRange ir1 = getIntRange(e.e1);
            IntRange ir2 = getIntRange(e.e2);
            IntRange ir1neg, ir1pos, ir2neg, ir2pos;
            bool has1neg, has1pos, has2neg, has2pos;
            ir1.splitBySign(ir1neg, has1neg, ir1pos, has1pos);
            ir2.splitBySign(ir2neg, has2neg, ir2pos, has2pos);
            IntRange result;
            bool hasResult = false;
            if (has1pos && has2pos)
                result.unionOrAssign(unsignedBitwiseAnd(ir1pos, ir2pos), hasResult);
            if (has1pos && has2neg)
                result.unionOrAssign(unsignedBitwiseAnd(ir1pos, ir2neg), hasResult);
            if (has1neg && has2pos)
                result.unionOrAssign(unsignedBitwiseAnd(ir1neg, ir2pos), hasResult);
            if (has1neg && has2neg)
                result.unionOrAssign(unsignedBitwiseAnd(ir1neg, ir2neg), hasResult);
            assert(hasResult);
            range = result._cast(e.type);
        }

        override void visit(OrExp e)
        {
            IntRange ir1 = getIntRange(e.e1);
            IntRange ir2 = getIntRange(e.e2);
            IntRange ir1neg, ir1pos, ir2neg, ir2pos;
            bool has1neg, has1pos, has2neg, has2pos;
            ir1.splitBySign(ir1neg, has1neg, ir1pos, has1pos);
            ir2.splitBySign(ir2neg, has2neg, ir2pos, has2pos);
            IntRange result;
            bool hasResult = false;
            if (has1pos && has2pos)
                result.unionOrAssign(unsignedBitwiseOr(ir1pos, ir2pos), hasResult);
            if (has1pos && has2neg)
                result.unionOrAssign(unsignedBitwiseOr(ir1pos, ir2neg), hasResult);
            if (has1neg && has2pos)
                result.unionOrAssign(unsignedBitwiseOr(ir1neg, ir2pos), hasResult);
            if (has1neg && has2neg)
                result.unionOrAssign(unsignedBitwiseOr(ir1neg, ir2neg), hasResult);
            assert(hasResult);
            range = result._cast(e.type);
        }

        override void visit(XorExp e)
        {
            IntRange ir1 = getIntRange(e.e1);
            IntRange ir2 = getIntRange(e.e2);
            IntRange ir1neg, ir1pos, ir2neg, ir2pos;
            bool has1neg, has1pos, has2neg, has2pos;
            ir1.splitBySign(ir1neg, has1neg, ir1pos, has1pos);
            ir2.splitBySign(ir2neg, has2neg, ir2pos, has2pos);
            IntRange result;
            bool hasResult = false;
            if (has1pos && has2pos)
                result.unionOrAssign(unsignedBitwiseXor(ir1pos, ir2pos), hasResult);
            if (has1pos && has2neg)
                result.unionOrAssign(unsignedBitwiseXor(ir1pos, ir2neg), hasResult);
            if (has1neg && has2pos)
                result.unionOrAssign(unsignedBitwiseXor(ir1neg, ir2pos), hasResult);
            if (has1neg && has2neg)
                result.unionOrAssign(unsignedBitwiseXor(ir1neg, ir2neg), hasResult);
            assert(hasResult);
            range = result._cast(e.type);
        }

        override void visit(ShlExp e)
        {
            IntRange ir1 = getIntRange(e.e1);
            IntRange ir2 = getIntRange(e.e2);
            if (ir2.imin.negative)
                ir2 = IntRange(SignExtendedNumber(0), SignExtendedNumber(64));
            SignExtendedNumber lower = ir1.imin << (ir1.imin.negative ? ir2.imax : ir2.imin);
            SignExtendedNumber upper = ir1.imax << (ir1.imax.negative ? ir2.imin : ir2.imax);
            range = IntRange(lower, upper)._cast(e.type);
        }

        override void visit(ShrExp e)
        {
            IntRange ir1 = getIntRange(e.e1);
            IntRange ir2 = getIntRange(e.e2);
            if (ir2.imin.negative)
                ir2 = IntRange(SignExtendedNumber(0), SignExtendedNumber(64));
            SignExtendedNumber lower = ir1.imin >> (ir1.imin.negative ? ir2.imin : ir2.imax);
            SignExtendedNumber upper = ir1.imax >> (ir1.imax.negative ? ir2.imax : ir2.imin);
            range = IntRange(lower, upper)._cast(e.type);
        }

        override void visit(UshrExp e)
        {
            IntRange ir1 = getIntRange(e.e1).castUnsigned(e.e1.type);
            IntRange ir2 = getIntRange(e.e2);
            if (ir2.imin.negative)
                ir2 = IntRange(SignExtendedNumber(0), SignExtendedNumber(64));
            range = IntRange(ir1.imin >> ir2.imax, ir1.imax >> ir2.imin)._cast(e.type);
        }

        override void visit(AssignExp e)
        {
            range = getIntRange(e.e2)._cast(e.type);
        }

        override void visit(CondExp e)
        {
            // No need to check e->econd; assume caller has called optimize()
            IntRange ir1 = getIntRange(e.e1);
            IntRange ir2 = getIntRange(e.e2);
            range = ir1.unionWith(ir2)._cast(e.type);
        }

        override void visit(VarExp e)
        {
            Expression ie;
            VarDeclaration vd = e.var.isVarDeclaration();
            if (vd && vd.range)
                range = vd.range._cast(e.type);
            else if (vd && vd._init && !vd.type.isMutable() && (ie = vd.getConstInitializer()) !is null)
                ie.accept(this);
            else
                visit(cast(Expression)e);
        }

        override void visit(CommaExp e)
        {
            e.e2.accept(this);
        }

        override void visit(ComExp e)
        {
            IntRange ir = getIntRange(e.e1);
            range = IntRange(SignExtendedNumber(~ir.imax.value, !ir.imax.negative), SignExtendedNumber(~ir.imin.value, !ir.imin.negative))._cast(e.type);
        }

        override void visit(NegExp e)
        {
            IntRange ir = getIntRange(e.e1);
            range = IntRange(-ir.imax, -ir.imin)._cast(e.type);
        }
    }

    scope IntRangeVisitor v = new IntRangeVisitor();
    e.accept(v);
    return v.range;
}
