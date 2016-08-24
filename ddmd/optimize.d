// Compiler implementation of the D programming language
// Copyright (c) 1999-2015 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.optimize;

import core.stdc.stdio;

import ddmd.constfold;
import ddmd.ctfeexpr;
import ddmd.dclass;
import ddmd.declaration;
import ddmd.expression;
import ddmd.globals;
import ddmd.init;
import ddmd.mtype;
import ddmd.root.ctfloat;
import ddmd.sideeffect;
import ddmd.tokens;
import ddmd.visitor;

/*************************************
 * If variable has a const initializer,
 * return that initializer.
 */
extern (C++) Expression expandVar(int result, VarDeclaration v)
{
    //printf("expandVar(result = %d, v = %p, %s)\n", result, v, v ? v.toChars() : "null");
    Expression e = null;
    if (!v)
        return e;
    if (!v.originalType && v._scope) // semantic() not yet run
        v.semantic(v._scope);
    if (v.isConst() || v.isImmutable() || v.storage_class & STCmanifest)
    {
        if (!v.type)
        {
            return e;
        }
        Type tb = v.type.toBasetype();
        if (v.storage_class & STCmanifest || v.type.toBasetype().isscalar() || ((result & WANTexpand) && (tb.ty != Tsarray && tb.ty != Tstruct)))
        {
            if (v._init)
            {
                if (v.inuse)
                {
                    if (v.storage_class & STCmanifest)
                    {
                        v.error("recursive initialization of constant");
                        goto Lerror;
                    }
                    goto L1;
                }
                Expression ei = v.getConstInitializer();
                if (!ei)
                {
                    if (v.storage_class & STCmanifest)
                    {
                        v.error("enum cannot be initialized with %s", v._init.toChars());
                        goto Lerror;
                    }
                    goto L1;
                }
                if (ei.op == TOKconstruct || ei.op == TOKblit)
                {
                    AssignExp ae = cast(AssignExp)ei;
                    ei = ae.e2;
                    if (ei.isConst() == 1)
                    {
                    }
                    else if (ei.op == TOKstring)
                    {
                        // Bugzilla 14459: We should not constfold the string literal
                        // if it's typed as a C string, because the value expansion
                        // will drop the pointer identity.
                        if (!(result & WANTexpand) && ei.type.toBasetype().ty == Tpointer)
                            goto L1;
                    }
                    else
                        goto L1;
                    if (ei.type == v.type)
                    {
                        // const variable initialized with const expression
                    }
                    else if (ei.implicitConvTo(v.type) >= MATCHconst)
                    {
                        // const var initialized with non-const expression
                        ei = ei.implicitCastTo(null, v.type);
                        ei = ei.semantic(null);
                    }
                    else
                        goto L1;
                }
                else if (!(v.storage_class & STCmanifest) && ei.isConst() != 1 && ei.op != TOKstring && ei.op != TOKaddress)
                {
                    goto L1;
                }
                if (!ei.type)
                {
                    goto L1;
                }
                else
                {
                    // Should remove the copy() operation by
                    // making all mods to expressions copy-on-write
                    e = ei.copy();
                }
            }
            else
            {
                version (all)
                {
                    goto L1;
                }
                else
                {
                    // BUG: what if const is initialized in constructor?
                    e = v.type.defaultInit();
                    e.loc = e1.loc;
                }
            }
            if (e.type != v.type)
            {
                e = e.castTo(null, v.type);
            }
            v.inuse++;
            e = e.optimize(result);
            v.inuse--;
        }
    }
L1:
    //if (e) printf("\te = %p, %s, e->type = %d, %s\n", e, e->toChars(), e->type->ty, e->type->toChars());
    return e;
Lerror:
    return new ErrorExp();
}

extern (C++) Expression fromConstInitializer(int result, Expression e1)
{
    //printf("fromConstInitializer(result = %x, %s)\n", result, e1.toChars());
    //static int xx; if (xx++ == 10) assert(0);
    Expression e = e1;
    if (e1.op == TOKvar)
    {
        VarExp ve = cast(VarExp)e1;
        VarDeclaration v = ve.var.isVarDeclaration();
        e = expandVar(result, v);
        if (e)
        {
            // If it is a comma expression involving a declaration, we mustn't
            // perform a copy -- we'd get two declarations of the same variable.
            // See bugzilla 4465.
            if (e.op == TOKcomma && (cast(CommaExp)e).e1.op == TOKdeclaration)
                e = e1;
            else if (e.type != e1.type && e1.type && e1.type.ty != Tident)
            {
                // Type 'paint' operation
                e = e.copy();
                e.type = e1.type;
            }
            e.loc = e1.loc;
        }
        else
        {
            e = e1;
        }
    }
    return e;
}

extern (C++) Expression Expression_optimize(Expression e, int result, bool keepLvalue)
{
    extern (C++) final class OptimizeVisitor : Visitor
    {
        alias visit = super.visit;
    public:
        int result;
        bool keepLvalue;
        Expression ret;

        extern (D) this(int result, bool keepLvalue)
        {
            this.result = result;
            this.keepLvalue = keepLvalue;
        }

        bool expOptimize(ref Expression e, int flags, bool keepLvalue = false)
        {
            if (!e)
                return false;
            Expression ex = e.optimize(flags, keepLvalue);
            if (ex.op == TOKerror)
            {
                ret = ex; // store error result
                return true;
            }
            else
            {
                e = ex; // modify original
                return false;
            }
        }

        bool unaOptimize(UnaExp e, int flags)
        {
            return expOptimize(e.e1, flags);
        }

        bool binOptimize(BinExp e, int flags)
        {
            expOptimize(e.e1, flags);
            expOptimize(e.e2, flags);
            return ret.op == TOKerror;
        }

        override void visit(Expression e)
        {
            //printf("Expression::optimize(result = x%x) %s\n", result, e->toChars());
        }

        override void visit(VarExp e)
        {
            if (keepLvalue)
            {
                VarDeclaration v = e.var.isVarDeclaration();
                if (v && !(v.storage_class & STCmanifest))
                    return;
            }
            ret = fromConstInitializer(result, e);
        }

        override void visit(TupleExp e)
        {
            expOptimize(e.e0, WANTvalue);
            for (size_t i = 0; i < e.exps.dim; i++)
            {
                expOptimize((*e.exps)[i], WANTvalue);
            }
        }

        override void visit(ArrayLiteralExp e)
        {
            if (e.elements)
            {
                expOptimize(e.basis, result & WANTexpand);
                for (size_t i = 0; i < e.elements.dim; i++)
                {
                    expOptimize((*e.elements)[i], result & WANTexpand);
                }
            }
        }

        override void visit(AssocArrayLiteralExp e)
        {
            assert(e.keys.dim == e.values.dim);
            for (size_t i = 0; i < e.keys.dim; i++)
            {
                expOptimize((*e.keys)[i], result & WANTexpand);
                expOptimize((*e.values)[i], result & WANTexpand);
            }
        }

        override void visit(StructLiteralExp e)
        {
            if (e.stageflags & stageOptimize)
                return;
            int old = e.stageflags;
            e.stageflags |= stageOptimize;
            if (e.elements)
            {
                for (size_t i = 0; i < e.elements.dim; i++)
                {
                    expOptimize((*e.elements)[i], result & WANTexpand);
                }
            }
            e.stageflags = old;
        }

        override void visit(UnaExp e)
        {
            //printf("UnaExp::optimize() %s\n", e->toChars());
            if (unaOptimize(e, result))
                return;
        }

        override void visit(NegExp e)
        {
            if (unaOptimize(e, result))
                return;
            if (e.e1.isConst() == 1)
            {
                ret = Neg(e.type, e.e1).copy();
            }
        }

        override void visit(ComExp e)
        {
            if (unaOptimize(e, result))
                return;
            if (e.e1.isConst() == 1)
            {
                ret = Com(e.type, e.e1).copy();
            }
        }

        override void visit(NotExp e)
        {
            if (unaOptimize(e, result))
                return;
            if (e.e1.isConst() == 1)
            {
                ret = Not(e.type, e.e1).copy();
            }
        }

        override void visit(BoolExp e)
        {
            if (unaOptimize(e, result))
                return;
            if (e.e1.isConst() == 1)
            {
                ret = Bool(e.type, e.e1).copy();
            }
        }

        override void visit(SymOffExp e)
        {
            assert(e.var);
        }

        override void visit(AddrExp e)
        {
            //printf("AddrExp::optimize(result = %d) %s\n", result, e->toChars());
            /* Rewrite &(a,b) as (a,&b)
             */
            if (e.e1.op == TOKcomma)
            {
                CommaExp ce = cast(CommaExp)e.e1;
                auto ae = new AddrExp(e.loc, ce.e2);
                ae.type = e.type;
                ret = new CommaExp(ce.loc, ce.e1, ae);
                ret.type = e.type;
                ret = ret.optimize(result);
                return;
            }
            // Keep lvalue-ness
            if (expOptimize(e.e1, result, true))
                return;
            // Convert &*ex to ex
            if (e.e1.op == TOKstar)
            {
                Expression ex = (cast(PtrExp)e.e1).e1;
                if (e.type.equals(ex.type))
                    ret = ex;
                else if (e.type.toBasetype().equivalent(ex.type.toBasetype()))
                {
                    ret = ex.copy();
                    ret.type = e.type;
                }
                return;
            }
            if (e.e1.op == TOKvar)
            {
                VarExp ve = cast(VarExp)e.e1;
                if (!ve.var.isOut() && !ve.var.isRef() && !ve.var.isImportedSymbol())
                {
                    ret = new SymOffExp(e.loc, ve.var, 0, ve.hasOverloads);
                    ret.type = e.type;
                    return;
                }
            }
            if (e.e1.op == TOKindex)
            {
                // Convert &array[n] to &array+n
                IndexExp ae = cast(IndexExp)e.e1;
                if (ae.e2.op == TOKint64 && ae.e1.op == TOKvar)
                {
                    sinteger_t index = ae.e2.toInteger();
                    VarExp ve = cast(VarExp)ae.e1;
                    if (ve.type.ty == Tsarray && !ve.var.isImportedSymbol())
                    {
                        TypeSArray ts = cast(TypeSArray)ve.type;
                        sinteger_t dim = ts.dim.toInteger();
                        if (index < 0 || index >= dim)
                        {
                            e.error("array index %lld is out of bounds [0..%lld]", index, dim);
                            ret = new ErrorExp();
                            return;
                        }
                        ret = new SymOffExp(e.loc, ve.var, index * ts.nextOf().size());
                        ret.type = e.type;
                        return;
                    }
                }
            }
        }

        override void visit(PtrExp e)
        {
            //printf("PtrExp::optimize(result = x%x) %s\n", result, e->toChars());
            if (expOptimize(e.e1, result))
                return;
            // Convert *&ex to ex
            // But only if there is no type punning involved
            if (e.e1.op == TOKaddress)
            {
                Expression ex = (cast(AddrExp)e.e1).e1;
                if (e.type.equals(ex.type))
                    ret = ex;
                else if (e.type.toBasetype().equivalent(ex.type.toBasetype()))
                {
                    ret = ex.copy();
                    ret.type = e.type;
                }
            }
            if (keepLvalue)
                return;
            // Constant fold *(&structliteral + offset)
            if (e.e1.op == TOKadd)
            {
                Expression ex = Ptr(e.type, e.e1).copy();
                if (!CTFEExp.isCantExp(ex))
                {
                    ret = ex;
                    return;
                }
            }
            if (e.e1.op == TOKsymoff)
            {
                SymOffExp se = cast(SymOffExp)e.e1;
                VarDeclaration v = se.var.isVarDeclaration();
                Expression ex = expandVar(result, v);
                if (ex && ex.op == TOKstructliteral)
                {
                    StructLiteralExp sle = cast(StructLiteralExp)ex;
                    ex = sle.getField(e.type, cast(uint)se.offset);
                    if (ex && !CTFEExp.isCantExp(ex))
                    {
                        ret = ex;
                        return;
                    }
                }
            }
        }

        override void visit(DotVarExp e)
        {
            //printf("DotVarExp::optimize(result = x%x) %s\n", result, e->toChars());
            if (expOptimize(e.e1, result))
                return;
            if (keepLvalue)
                return;
            Expression ex = e.e1;
            if (ex.op == TOKvar)
            {
                VarExp ve = cast(VarExp)ex;
                VarDeclaration v = ve.var.isVarDeclaration();
                ex = expandVar(result, v);
            }
            if (ex && ex.op == TOKstructliteral)
            {
                StructLiteralExp sle = cast(StructLiteralExp)ex;
                VarDeclaration vf = e.var.isVarDeclaration();
                if (vf && !vf.overlapped)
                {
                    /* Bugzilla 13021: Prevent optimization if vf has overlapped fields.
                     */
                    ex = sle.getField(e.type, vf.offset);
                    if (ex && !CTFEExp.isCantExp(ex))
                    {
                        ret = ex;
                        return;
                    }
                }
            }
        }

        override void visit(NewExp e)
        {
            expOptimize(e.thisexp, WANTvalue);
            // Optimize parameters
            if (e.newargs)
            {
                for (size_t i = 0; i < e.newargs.dim; i++)
                {
                    expOptimize((*e.newargs)[i], WANTvalue);
                }
            }
            if (e.arguments)
            {
                for (size_t i = 0; i < e.arguments.dim; i++)
                {
                    expOptimize((*e.arguments)[i], WANTvalue);
                }
            }
        }

        override void visit(CallExp e)
        {
            //printf("CallExp::optimize(result = %d) %s\n", result, e->toChars());
            // Optimize parameters with keeping lvalue-ness
            if (expOptimize(e.e1, result))
                return;
            if (e.arguments)
            {
                Type t1 = e.e1.type.toBasetype();
                if (t1.ty == Tdelegate)
                    t1 = t1.nextOf();
                assert(t1.ty == Tfunction);
                TypeFunction tf = cast(TypeFunction)t1;
                for (size_t i = 0; i < e.arguments.dim; i++)
                {
                    Parameter p = Parameter.getNth(tf.parameters, i);
                    bool keep = p && (p.storageClass & (STCref | STCout)) != 0;
                    expOptimize((*e.arguments)[i], WANTvalue, keep);
                }
            }
        }

        override void visit(CastExp e)
        {
            //printf("CastExp::optimize(result = %d) %s\n", result, e->toChars());
            //printf("from %s to %s\n", e->type->toChars(), e->to->toChars());
            //printf("from %s\n", e->type->toChars());
            //printf("e1->type %s\n", e->e1->type->toChars());
            //printf("type = %p\n", e->type);
            assert(e.type);
            TOK op1 = e.e1.op;
            Expression e1old = e.e1;
            if (expOptimize(e.e1, result))
                return;
            e.e1 = fromConstInitializer(result, e.e1);
            if (e.e1 == e1old && e.e1.op == TOKarrayliteral && e.type.toBasetype().ty == Tpointer && e.e1.type.toBasetype().ty != Tsarray)
            {
                // Casting this will result in the same expression, and
                // infinite loop because of Expression::implicitCastTo()
                return; // no change
            }
            if ((e.e1.op == TOKstring || e.e1.op == TOKarrayliteral) && (e.type.ty == Tpointer || e.type.ty == Tarray) && e.e1.type.toBasetype().nextOf().size() == e.type.nextOf().size())
            {
                // Bugzilla 12937: If target type is void array, trying to paint
                // e->e1 with that type will cause infinite recursive optimization.
                if (e.type.nextOf().ty == Tvoid)
                    return;
                ret = e.e1.castTo(null, e.type);
                //printf(" returning1 %s\n", ret->toChars());
                return;
            }
            if (e.e1.op == TOKstructliteral && e.e1.type.implicitConvTo(e.type) >= MATCHconst)
            {
                //printf(" returning2 %s\n", e->e1->toChars());
            L1:
                // Returning e1 with changing its type
                ret = (e1old == e.e1 ? e.e1.copy() : e.e1);
                ret.type = e.type;
                return;
            }
            /* The first test here is to prevent infinite loops
             */
            if (op1 != TOKarrayliteral && e.e1.op == TOKarrayliteral)
            {
                ret = e.e1.castTo(null, e.to);
                return;
            }
            if (e.e1.op == TOKnull && (e.type.ty == Tpointer || e.type.ty == Tclass || e.type.ty == Tarray))
            {
                //printf(" returning3 %s\n", e->e1->toChars());
                goto L1;
            }
            if (e.type.ty == Tclass && e.e1.type.ty == Tclass)
            {
                // See if we can remove an unnecessary cast
                ClassDeclaration cdfrom = e.e1.type.isClassHandle();
                ClassDeclaration cdto = e.type.isClassHandle();
                int offset;
                if (cdto.isBaseOf(cdfrom, &offset) && offset == 0)
                {
                    //printf(" returning4 %s\n", e->e1->toChars());
                    goto L1;
                }
            }
            // We can convert 'head const' to mutable
            if (e.to.mutableOf().constOf().equals(e.e1.type.mutableOf().constOf()))
            {
                //printf(" returning5 %s\n", e->e1->toChars());
                goto L1;
            }
            if (e.e1.isConst())
            {
                if (e.e1.op == TOKsymoff)
                {
                    if (e.type.size() == e.e1.type.size() && e.type.toBasetype().ty != Tsarray)
                    {
                        goto L1;
                    }
                    return;
                }
                if (e.to.toBasetype().ty != Tvoid)
                {
                    if (e.e1.type.equals(e.type) && e.type.equals(e.to))
                        ret = e.e1;
                    else
                        ret = Cast(e.loc, e.type, e.to, e.e1).copy();
                }
            }
            //printf(" returning6 %s\n", ret->toChars());
        }

        override void visit(BinExp e)
        {
            //printf("BinExp::optimize(result = %d) %s\n", result, e->toChars());
            // don't replace const variable with its initializer in e1
            bool e2only = (e.op == TOKconstruct || e.op == TOKblit);
            if (e2only ? expOptimize(e.e2, result) : binOptimize(e, result))
                return;
            if (e.op == TOKshlass || e.op == TOKshrass || e.op == TOKushrass)
            {
                if (e.e2.isConst() == 1)
                {
                    sinteger_t i2 = e.e2.toInteger();
                    d_uns64 sz = e.e1.type.size() * 8;
                    if (i2 < 0 || i2 >= sz)
                    {
                        e.error("shift assign by %lld is outside the range 0..%llu", i2, cast(ulong)sz - 1);
                        ret = new ErrorExp();
                        return;
                    }
                }
            }
        }

        override void visit(AddExp e)
        {
            //printf("AddExp::optimize(%s)\n", e->toChars());
            if (binOptimize(e, result))
                return;
            if (e.e1.isConst() && e.e2.isConst())
            {
                if (e.e1.op == TOKsymoff && e.e2.op == TOKsymoff)
                    return;
                ret = Add(e.loc, e.type, e.e1, e.e2).copy();
            }
        }

        override void visit(MinExp e)
        {
            if (binOptimize(e, result))
                return;
            if (e.e1.isConst() && e.e2.isConst())
            {
                if (e.e2.op == TOKsymoff)
                    return;
                ret = Min(e.loc, e.type, e.e1, e.e2).copy();
            }
        }

        override void visit(MulExp e)
        {
            //printf("MulExp::optimize(result = %d) %s\n", result, e->toChars());
            if (binOptimize(e, result))
                return;
            if (e.e1.isConst() == 1 && e.e2.isConst() == 1)
            {
                ret = Mul(e.loc, e.type, e.e1, e.e2).copy();
            }
        }

        override void visit(DivExp e)
        {
            //printf("DivExp::optimize(%s)\n", e->toChars());
            if (binOptimize(e, result))
                return;
            if (e.e1.isConst() == 1 && e.e2.isConst() == 1)
            {
                ret = Div(e.loc, e.type, e.e1, e.e2).copy();
            }
        }

        override void visit(ModExp e)
        {
            if (binOptimize(e, result))
                return;
            if (e.e1.isConst() == 1 && e.e2.isConst() == 1)
            {
                ret = Mod(e.loc, e.type, e.e1, e.e2).copy();
            }
        }

        void shift_optimize(BinExp e, UnionExp function(Loc, Type, Expression, Expression) shift)
        {
            if (binOptimize(e, result))
                return;
            if (e.e2.isConst() == 1)
            {
                sinteger_t i2 = e.e2.toInteger();
                d_uns64 sz = e.e1.type.size() * 8;
                if (i2 < 0 || i2 >= sz)
                {
                    e.error("shift by %lld is outside the range 0..%llu", i2, cast(ulong)sz - 1);
                    ret = new ErrorExp();
                    return;
                }
                if (e.e1.isConst() == 1)
                    ret = (*shift)(e.loc, e.type, e.e1, e.e2).copy();
            }
        }

        override void visit(ShlExp e)
        {
            //printf("ShlExp::optimize(result = %d) %s\n", result, e->toChars());
            shift_optimize(e, &Shl);
        }

        override void visit(ShrExp e)
        {
            //printf("ShrExp::optimize(result = %d) %s\n", result, e->toChars());
            shift_optimize(e, &Shr);
        }

        override void visit(UshrExp e)
        {
            //printf("UshrExp::optimize(result = %d) %s\n", result, toChars());
            shift_optimize(e, &Ushr);
        }

        override void visit(AndExp e)
        {
            if (binOptimize(e, result))
                return;
            if (e.e1.isConst() == 1 && e.e2.isConst() == 1)
                ret = And(e.loc, e.type, e.e1, e.e2).copy();
        }

        override void visit(OrExp e)
        {
            if (binOptimize(e, result))
                return;
            if (e.e1.isConst() == 1 && e.e2.isConst() == 1)
                ret = Or(e.loc, e.type, e.e1, e.e2).copy();
        }

        override void visit(XorExp e)
        {
            if (binOptimize(e, result))
                return;
            if (e.e1.isConst() == 1 && e.e2.isConst() == 1)
                ret = Xor(e.loc, e.type, e.e1, e.e2).copy();
        }

        override void visit(PowExp e)
        {
            if (binOptimize(e, result))
                return;
            // Replace 1 ^^ x or 1.0^^x by (x, 1)
            if ((e.e1.op == TOKint64 && e.e1.toInteger() == 1) || (e.e1.op == TOKfloat64 && e.e1.toReal() == real_t(1)))
            {
                ret = new CommaExp(e.loc, e.e2, e.e1);
                return;
            }
            // Replace -1 ^^ x by (x&1) ? -1 : 1, where x is integral
            if (e.e2.type.isintegral() && e.e1.op == TOKint64 && cast(sinteger_t)e.e1.toInteger() == -1)
            {
                Type resultType = e.type;
                ret = new AndExp(e.loc, e.e2, new IntegerExp(e.loc, 1, e.e2.type));
                ret = new CondExp(e.loc, ret, new IntegerExp(e.loc, -1, resultType), new IntegerExp(e.loc, 1, resultType));
                return;
            }
            // Replace x ^^ 0 or x^^0.0 by (x, 1)
            if ((e.e2.op == TOKint64 && e.e2.toInteger() == 0) || (e.e2.op == TOKfloat64 && e.e2.toReal() == real_t(0)))
            {
                if (e.e1.type.isintegral())
                    ret = new IntegerExp(e.loc, 1, e.e1.type);
                else
                    ret = new RealExp(e.loc, real_t(1), e.e1.type);
                ret = new CommaExp(e.loc, e.e1, ret);
                return;
            }
            // Replace x ^^ 1 or x^^1.0 by (x)
            if ((e.e2.op == TOKint64 && e.e2.toInteger() == 1) || (e.e2.op == TOKfloat64 && e.e2.toReal() == real_t(1)))
            {
                ret = e.e1;
                return;
            }
            // Replace x ^^ -1.0 by (1.0 / x)
            if ((e.e2.op == TOKfloat64 && e.e2.toReal() == real_t(-1)))
            {
                ret = new DivExp(e.loc, new RealExp(e.loc, real_t(1), e.e2.type), e.e1);
                return;
            }
            // All other negative integral powers are illegal
            if (e.e1.type.isintegral() && (e.e2.op == TOKint64) && cast(sinteger_t)e.e2.toInteger() < 0)
            {
                e.error("cannot raise %s to a negative integer power. Did you mean (cast(real)%s)^^%s ?", e.e1.type.toBasetype().toChars(), e.e1.toChars(), e.e2.toChars());
                ret = new ErrorExp();
                return;
            }
            // If e2 *could* have been an integer, make it one.
            if (e.e2.op == TOKfloat64)
            {
                version (all)
                {
                    // Work around redundant REX.W prefix breaking Valgrind
                    // when built with affected versions of DMD.
                    // https://issues.dlang.org/show_bug.cgi?id=14952
                    // This can be removed once compiling with DMD 2.068 or
                    // older is no longer supported.
                    const r = e.e2.toReal();
                    if (r == real_t(cast(sinteger_t)r))
                        e.e2 = new IntegerExp(e.loc, e.e2.toInteger(), Type.tint64);
                }
                else
                {
                    if (e.e2.toReal() == cast(sinteger_t)e.e2.toReal())
                        e.e2 = new IntegerExp(e.loc, e.e2.toInteger(), Type.tint64);
                }
            }
            if (e.e1.isConst() == 1 && e.e2.isConst() == 1)
            {
                Expression ex = Pow(e.loc, e.type, e.e1, e.e2).copy();
                if (!CTFEExp.isCantExp(ex))
                {
                    ret = ex;
                    return;
                }
            }
            // (2 ^^ n) ^^ p -> 1 << n * p
            if (e.e1.op == TOKint64 && e.e1.toInteger() > 0 && !((e.e1.toInteger() - 1) & e.e1.toInteger()) && e.e2.type.isintegral() && e.e2.type.isunsigned())
            {
                dinteger_t i = e.e1.toInteger();
                dinteger_t mul = 1;
                while ((i >>= 1) > 1)
                    mul++;
                Expression shift = new MulExp(e.loc, e.e2, new IntegerExp(e.loc, mul, e.e2.type));
                shift.type = e.e2.type;
                shift = shift.castTo(null, Type.tshiftcnt);
                ret = new ShlExp(e.loc, new IntegerExp(e.loc, 1, e.e1.type), shift);
                ret.type = e.type;
                return;
            }
        }

        override void visit(CommaExp e)
        {
            //printf("CommaExp::optimize(result = %d) %s\n", result, e->toChars());
            // Comma needs special treatment, because it may
            // contain compiler-generated declarations. We can interpret them, but
            // otherwise we must NOT attempt to constant-fold them.
            // In particular, if the comma returns a temporary variable, it needs
            // to be an lvalue (this is particularly important for struct constructors)
            expOptimize(e.e1, WANTvalue);
            expOptimize(e.e2, result, keepLvalue);
            if (ret.op == TOKerror)
                return;
            if (!e.e1 || e.e1.op == TOKint64 || e.e1.op == TOKfloat64 || !hasSideEffect(e.e1))
            {
                ret = e.e2;
                if (ret)
                    ret.type = e.type;
            }
            //printf("-CommaExp::optimize(result = %d) %s\n", result, e->e->toChars());
        }

        override void visit(ArrayLengthExp e)
        {
            //printf("ArrayLengthExp::optimize(result = %d) %s\n", result, e->toChars());
            if (unaOptimize(e, WANTexpand))
                return;
            // CTFE interpret static immutable arrays (to get better diagnostics)
            if (e.e1.op == TOKvar)
            {
                VarDeclaration v = (cast(VarExp)e.e1).var.isVarDeclaration();
                if (v && (v.storage_class & STCstatic) && (v.storage_class & STCimmutable) && v._init)
                {
                    if (Expression ci = v.getConstInitializer())
                        e.e1 = ci;
                }
            }
            if (e.e1.op == TOKstring || e.e1.op == TOKarrayliteral || e.e1.op == TOKassocarrayliteral || e.e1.type.toBasetype().ty == Tsarray)
            {
                ret = ArrayLength(e.type, e.e1).copy();
            }
        }

        override void visit(EqualExp e)
        {
            //printf("EqualExp::optimize(result = %x) %s\n", result, e->toChars());
            if (binOptimize(e, WANTvalue))
                return;
            Expression e1 = fromConstInitializer(result, e.e1);
            Expression e2 = fromConstInitializer(result, e.e2);
            if (e1.op == TOKerror)
            {
                ret = e1;
                return;
            }
            if (e2.op == TOKerror)
            {
                ret = e2;
                return;
            }
            ret = Equal(e.op, e.loc, e.type, e1, e2).copy();
            if (CTFEExp.isCantExp(ret))
                ret = e;
        }

        override void visit(IdentityExp e)
        {
            //printf("IdentityExp::optimize(result = %d) %s\n", result, e->toChars());
            if (binOptimize(e, WANTvalue))
                return;
            if ((e.e1.isConst() && e.e2.isConst()) || (e.e1.op == TOKnull && e.e2.op == TOKnull))
            {
                ret = Identity(e.op, e.loc, e.type, e.e1, e.e2).copy();
                if (CTFEExp.isCantExp(ret))
                    ret = e;
            }
        }

        /* It is possible for constant folding to change an array expression of
         * unknown length, into one where the length is known.
         * If the expression 'arr' is a literal, set lengthVar to be its length.
         */
        static void setLengthVarIfKnown(VarDeclaration lengthVar, Expression arr)
        {
            if (!lengthVar)
                return;
            if (lengthVar._init && !lengthVar._init.isVoidInitializer())
                return; // we have previously calculated the length
            size_t len;
            if (arr.op == TOKstring)
                len = (cast(StringExp)arr).len;
            else if (arr.op == TOKarrayliteral)
                len = (cast(ArrayLiteralExp)arr).elements.dim;
            else
            {
                Type t = arr.type.toBasetype();
                if (t.ty == Tsarray)
                    len = cast(size_t)(cast(TypeSArray)t).dim.toInteger();
                else
                    return; // we don't know the length yet
            }
            Expression dollar = new IntegerExp(Loc(), len, Type.tsize_t);
            lengthVar._init = new ExpInitializer(Loc(), dollar);
            lengthVar.storage_class |= STCstatic | STCconst;
        }

        override void visit(IndexExp e)
        {
            //printf("IndexExp::optimize(result = %d) %s\n", result, e->toChars());
            if (expOptimize(e.e1, result & WANTexpand))
                return;
            Expression ex = fromConstInitializer(result, e.e1);
            // We might know $ now
            setLengthVarIfKnown(e.lengthVar, ex);
            if (expOptimize(e.e2, WANTvalue))
                return;
            if (keepLvalue)
                return;
            ret = Index(e.type, ex, e.e2).copy();
            if (CTFEExp.isCantExp(ret))
                ret = e;
        }

        override void visit(SliceExp e)
        {
            //printf("SliceExp::optimize(result = %d) %s\n", result, e->toChars());
            if (expOptimize(e.e1, result & WANTexpand))
                return;
            if (!e.lwr)
            {
                if (e.e1.op == TOKstring)
                {
                    // Convert slice of string literal into dynamic array
                    Type t = e.e1.type.toBasetype();
                    if (Type tn = t.nextOf())
                        ret = e.e1.castTo(null, tn.arrayOf());
                }
            }
            else
            {
                e.e1 = fromConstInitializer(result, e.e1);
                // We might know $ now
                setLengthVarIfKnown(e.lengthVar, e.e1);
                expOptimize(e.lwr, WANTvalue);
                expOptimize(e.upr, WANTvalue);
                if (ret.op == TOKerror)
                    return;
                ret = Slice(e.type, e.e1, e.lwr, e.upr).copy();
                if (CTFEExp.isCantExp(ret))
                    ret = e;
            }
            // Bugzilla 14649: We need to leave the slice form so it might be
            // a part of array operation.
            // Assume that the backend codegen will handle the form `e[]`
            // as an equal to `e` itself.
            if (ret.op == TOKstring)
            {
                e.e1 = ret;
                e.lwr = null;
                e.upr = null;
                ret = e;
            }
            //printf("-SliceExp::optimize() %s\n", ret->toChars());
        }

        override void visit(AndAndExp e)
        {
            //printf("AndAndExp::optimize(%d) %s\n", result, e->toChars());
            if (expOptimize(e.e1, WANTvalue))
                return;
            if (e.e1.isBool(false))
            {
                // Replace with (e1, false)
                ret = new IntegerExp(e.loc, 0, Type.tbool);
                ret = Expression.combine(e.e1, ret);
                if (e.type.toBasetype().ty == Tvoid)
                {
                    ret = new CastExp(e.loc, ret, Type.tvoid);
                    ret.type = e.type;
                }
                ret = ret.optimize(result);
                return;
            }
            if (expOptimize(e.e2, WANTvalue))
                return;
            if (e.e1.isConst())
            {
                if (e.e2.isConst())
                {
                    bool n1 = e.e1.isBool(true);
                    bool n2 = e.e2.isBool(true);
                    ret = new IntegerExp(e.loc, n1 && n2, e.type);
                }
                else if (e.e1.isBool(true))
                {
                    if (e.type.toBasetype().ty == Tvoid)
                        ret = e.e2;
                    else
                        ret = new BoolExp(e.loc, e.e2, e.type);
                }
            }
        }

        override void visit(OrOrExp e)
        {
            //printf("OrOrExp::optimize(%d) %s\n", result, e->toChars());
            if (expOptimize(e.e1, WANTvalue))
                return;
            if (e.e1.isBool(true))
            {
                // Replace with (e1, true)
                ret = new IntegerExp(e.loc, 1, Type.tbool);
                ret = Expression.combine(e.e1, ret);
                if (e.type.toBasetype().ty == Tvoid)
                {
                    ret = new CastExp(e.loc, ret, Type.tvoid);
                    ret.type = e.type;
                }
                ret = ret.optimize(result);
                return;
            }
            if (expOptimize(e.e2, WANTvalue))
                return;
            if (e.e1.isConst())
            {
                if (e.e2.isConst())
                {
                    bool n1 = e.e1.isBool(true);
                    bool n2 = e.e2.isBool(true);
                    ret = new IntegerExp(e.loc, n1 || n2, e.type);
                }
                else if (e.e1.isBool(false))
                {
                    if (e.type.toBasetype().ty == Tvoid)
                        ret = e.e2;
                    else
                        ret = new BoolExp(e.loc, e.e2, e.type);
                }
            }
        }

        override void visit(CmpExp e)
        {
            //printf("CmpExp::optimize() %s\n", e->toChars());
            if (binOptimize(e, WANTvalue))
                return;
            Expression e1 = fromConstInitializer(result, e.e1);
            Expression e2 = fromConstInitializer(result, e.e2);
            ret = Cmp(e.op, e.loc, e.type, e1, e2).copy();
            if (CTFEExp.isCantExp(ret))
                ret = e;
        }

        override void visit(CatExp e)
        {
            //printf("CatExp::optimize(%d) %s\n", result, e->toChars());
            if (binOptimize(e, result))
                return;
            if (e.e1.op == TOKcat)
            {
                // Bugzilla 12798: optimize ((expr ~ str1) ~ str2)
                CatExp ce1 = cast(CatExp)e.e1;
                scope CatExp cex = new CatExp(e.loc, ce1.e2, e.e2);
                cex.type = e.type;
                Expression ex = cex.optimize(result);
                if (ex != cex)
                {
                    e.e1 = ce1.e1;
                    e.e2 = ex;
                }
            }
            // optimize "str"[] -> "str"
            if (e.e1.op == TOKslice)
            {
                SliceExp se1 = cast(SliceExp)e.e1;
                if (se1.e1.op == TOKstring && !se1.lwr)
                    e.e1 = se1.e1;
            }
            if (e.e2.op == TOKslice)
            {
                SliceExp se2 = cast(SliceExp)e.e2;
                if (se2.e1.op == TOKstring && !se2.lwr)
                    e.e2 = se2.e1;
            }
            ret = Cat(e.type, e.e1, e.e2).copy();
            if (CTFEExp.isCantExp(ret))
                ret = e;
        }

        override void visit(CondExp e)
        {
            if (expOptimize(e.econd, WANTvalue))
                return;
            if (e.econd.isBool(true))
                ret = e.e1.optimize(result, keepLvalue);
            else if (e.econd.isBool(false))
                ret = e.e2.optimize(result, keepLvalue);
            else
            {
                expOptimize(e.e1, result, keepLvalue);
                expOptimize(e.e2, result, keepLvalue);
            }
        }
    }

    scope OptimizeVisitor v = new OptimizeVisitor(result, keepLvalue);
    v.ret = e;
    e.accept(v);
    return v.ret;
}
