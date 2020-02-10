/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/opover.d, _opover.d)
 * Documentation:  https://dlang.org/phobos/dmd_opover.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/opover.d
 */

module dmd.opover;

import core.stdc.stdio;
import dmd.aggregate;
import dmd.aliasthis;
import dmd.arraytypes;
import dmd.dclass;
import dmd.declaration;
import dmd.dscope;
import dmd.dstruct;
import dmd.dsymbol;
import dmd.dtemplate;
import dmd.errors;
import dmd.expression;
import dmd.expressionsem;
import dmd.func;
import dmd.globals;
import dmd.id;
import dmd.identifier;
import dmd.mtype;
import dmd.statement;
import dmd.tokens;
import dmd.typesem;
import dmd.visitor;

/***********************************
 * Determine if operands of binary op can be reversed
 * to fit operator overload.
 */
bool isCommutative(TOK op)
{
    switch (op)
    {
    case TOK.add:
    case TOK.mul:
    case TOK.and:
    case TOK.or:
    case TOK.xor:
    // EqualExp
    case TOK.equal:
    case TOK.notEqual:
    // CmpExp
    case TOK.lessThan:
    case TOK.lessOrEqual:
    case TOK.greaterThan:
    case TOK.greaterOrEqual:
        return true;
    default:
        break;
    }
    return false;
}

/***********************************
 * Get Identifier for operator overload.
 */
private Identifier opId(Expression e)
{
    switch (e.op)
    {
    case TOK.uadd:                      return Id.uadd;
    case TOK.negate:                    return Id.neg;
    case TOK.tilde:                     return Id.com;
    case TOK.cast_:                     return Id._cast;
    case TOK.in_:                       return Id.opIn;
    case TOK.plusPlus:                  return Id.postinc;
    case TOK.minusMinus:                return Id.postdec;
    case TOK.add:                       return Id.add;
    case TOK.min:                       return Id.sub;
    case TOK.mul:                       return Id.mul;
    case TOK.div:                       return Id.div;
    case TOK.mod:                       return Id.mod;
    case TOK.pow:                       return Id.pow;
    case TOK.leftShift:                 return Id.shl;
    case TOK.rightShift:                return Id.shr;
    case TOK.unsignedRightShift:        return Id.ushr;
    case TOK.and:                       return Id.iand;
    case TOK.or:                        return Id.ior;
    case TOK.xor:                       return Id.ixor;
    case TOK.concatenate:               return Id.cat;
    case TOK.assign:                    return Id.assign;
    case TOK.addAssign:                 return Id.addass;
    case TOK.minAssign:                 return Id.subass;
    case TOK.mulAssign:                 return Id.mulass;
    case TOK.divAssign:                 return Id.divass;
    case TOK.modAssign:                 return Id.modass;
    case TOK.powAssign:                 return Id.powass;
    case TOK.leftShiftAssign:           return Id.shlass;
    case TOK.rightShiftAssign:          return Id.shrass;
    case TOK.unsignedRightShiftAssign:  return Id.ushrass;
    case TOK.andAssign:                 return Id.andass;
    case TOK.orAssign:                  return Id.orass;
    case TOK.xorAssign:                 return Id.xorass;
    case TOK.concatenateAssign:         return Id.catass;
    case TOK.equal:                     return Id.eq;
    case TOK.lessThan:
    case TOK.lessOrEqual:
    case TOK.greaterThan:
    case TOK.greaterOrEqual:            return Id.cmp;
    case TOK.array:                     return Id.index;
    case TOK.star:                      return Id.opStar;
    default:                            assert(0);
    }
}

/***********************************
 * Get Identifier for reverse operator overload,
 * `null` if not supported for this operator.
 */
private Identifier opId_r(Expression e)
{
    switch (e.op)
    {
    case TOK.in_:               return Id.opIn_r;
    case TOK.add:               return Id.add_r;
    case TOK.min:               return Id.sub_r;
    case TOK.mul:               return Id.mul_r;
    case TOK.div:               return Id.div_r;
    case TOK.mod:               return Id.mod_r;
    case TOK.pow:               return Id.pow_r;
    case TOK.leftShift:         return Id.shl_r;
    case TOK.rightShift:        return Id.shr_r;
    case TOK.unsignedRightShift:return Id.ushr_r;
    case TOK.and:               return Id.iand_r;
    case TOK.or:                return Id.ior_r;
    case TOK.xor:               return Id.ixor_r;
    case TOK.concatenate:       return Id.cat_r;
    default:                    return null;
    }
}

/*******************************************
 * Helper function to turn operator into template argument list
 */
Objects* opToArg(Scope* sc, TOK op)
{
    /* Remove the = from op=
     */
    switch (op)
    {
    case TOK.addAssign:
        op = TOK.add;
        break;
    case TOK.minAssign:
        op = TOK.min;
        break;
    case TOK.mulAssign:
        op = TOK.mul;
        break;
    case TOK.divAssign:
        op = TOK.div;
        break;
    case TOK.modAssign:
        op = TOK.mod;
        break;
    case TOK.andAssign:
        op = TOK.and;
        break;
    case TOK.orAssign:
        op = TOK.or;
        break;
    case TOK.xorAssign:
        op = TOK.xor;
        break;
    case TOK.leftShiftAssign:
        op = TOK.leftShift;
        break;
    case TOK.rightShiftAssign:
        op = TOK.rightShift;
        break;
    case TOK.unsignedRightShiftAssign:
        op = TOK.unsignedRightShift;
        break;
    case TOK.concatenateAssign:
        op = TOK.concatenate;
        break;
    case TOK.powAssign:
        op = TOK.pow;
        break;
    default:
        break;
    }
    Expression e = new StringExp(Loc.initial, Token.toString(op));
    e = e.expressionSemantic(sc);
    auto tiargs = new Objects();
    tiargs.push(e);
    return tiargs;
}

// Try alias this on first operand
private Expression checkAliasThisForLhs(AggregateDeclaration ad, Scope* sc, BinExp e)
{
    if (!ad || !ad.aliasthis)
        return null;

    /* Rewrite (e1 op e2) as:
     *      (e1.aliasthis op e2)
     */
    if (e.att1 && e.e1.type == e.att1)
        return null;
    //printf("att %s e1 = %s\n", Token::toChars(e.op), e.e1.type.toChars());
    Expression e1 = new DotIdExp(e.loc, e.e1, ad.aliasthis.ident);
    BinExp be = cast(BinExp)e.copy();
    if (!be.att1 && e.e1.type.checkAliasThisRec())
        be.att1 = e.e1.type;
    be.e1 = e1;

    Expression result;
    if (be.op == TOK.concatenateAssign)
        result = be.op_overload(sc);
    else
        result = be.trySemantic(sc);

    return result;
}

// Try alias this on second operand
private Expression checkAliasThisForRhs(AggregateDeclaration ad, Scope* sc, BinExp e)
{
    if (!ad || !ad.aliasthis)
        return null;
    /* Rewrite (e1 op e2) as:
     *      (e1 op e2.aliasthis)
     */
    if (e.att2 && e.e2.type == e.att2)
        return null;
    //printf("att %s e2 = %s\n", Token::toChars(e.op), e.e2.type.toChars());
    Expression e2 = new DotIdExp(e.loc, e.e2, ad.aliasthis.ident);
    BinExp be = cast(BinExp)e.copy();
    if (!be.att2 && e.e2.type.checkAliasThisRec())
        be.att2 = e.e2.type;
    be.e2 = e2;

    Expression result;
    if (be.op == TOK.concatenateAssign)
        result = be.op_overload(sc);
    else
        result = be.trySemantic(sc);

    return result;
}

/************************************
 * Operator overload.
 * Check for operator overload, if so, replace
 * with function call.
 * Params:
 *      e = expression with operator
 *      sc = context
 *      pop = if not null, is set to the operator that was actually overloaded,
 *            which may not be `e.op`. Happens when operands are reversed to
 *            to match an overload
 * Returns:
 *      `null` if not an operator overload,
 *      otherwise the lowered expression
 */
Expression op_overload(Expression e, Scope* sc, TOK* pop = null)
{
    extern (C++) final class OpOverload : Visitor
    {
        alias visit = Visitor.visit;
    public:
        Scope* sc;
        TOK* pop;
        Expression result;

        extern (D) this(Scope* sc, TOK* pop)
        {
            this.sc = sc;
            this.pop = pop;
        }

        override void visit(Expression e)
        {
            assert(0);
        }

        override void visit(UnaExp e)
        {
            //printf("UnaExp::op_overload() (%s)\n", e.toChars());
            if (e.e1.op == TOK.array)
            {
                ArrayExp ae = cast(ArrayExp)e.e1;
                ae.e1 = ae.e1.expressionSemantic(sc);
                ae.e1 = resolveProperties(sc, ae.e1);
                Expression ae1old = ae.e1;
                const(bool) maybeSlice = (ae.arguments.dim == 0 || ae.arguments.dim == 1 && (*ae.arguments)[0].op == TOK.interval);
                IntervalExp ie = null;
                if (maybeSlice && ae.arguments.dim)
                {
                    assert((*ae.arguments)[0].op == TOK.interval);
                    ie = cast(IntervalExp)(*ae.arguments)[0];
                }
                while (true)
                {
                    if (ae.e1.op == TOK.error)
                    {
                        result = ae.e1;
                        return;
                    }
                    Expression e0 = null;
                    Expression ae1save = ae.e1;
                    ae.lengthVar = null;
                    Type t1b = ae.e1.type.toBasetype();
                    AggregateDeclaration ad = isAggregate(t1b);
                    if (!ad)
                        break;
                    if (search_function(ad, Id.opIndexUnary))
                    {
                        // Deal with $
                        result = resolveOpDollar(sc, ae, &e0);
                        if (!result) // op(a[i..j]) might be: a.opSliceUnary!(op)(i, j)
                            goto Lfallback;
                        if (result.op == TOK.error)
                            return;
                        /* Rewrite op(a[arguments]) as:
                         *      a.opIndexUnary!(op)(arguments)
                         */
                        Expressions* a = ae.arguments.copy();
                        Objects* tiargs = opToArg(sc, e.op);
                        result = new DotTemplateInstanceExp(e.loc, ae.e1, Id.opIndexUnary, tiargs);
                        result = new CallExp(e.loc, result, a);
                        if (maybeSlice) // op(a[]) might be: a.opSliceUnary!(op)()
                            result = result.trySemantic(sc);
                        else
                            result = result.expressionSemantic(sc);
                        if (result)
                        {
                            result = Expression.combine(e0, result);
                            return;
                        }
                    }
                Lfallback:
                    if (maybeSlice && search_function(ad, Id.opSliceUnary))
                    {
                        // Deal with $
                        result = resolveOpDollar(sc, ae, ie, &e0);
                        if (result.op == TOK.error)
                            return;
                        /* Rewrite op(a[i..j]) as:
                         *      a.opSliceUnary!(op)(i, j)
                         */
                        auto a = new Expressions();
                        if (ie)
                        {
                            a.push(ie.lwr);
                            a.push(ie.upr);
                        }
                        Objects* tiargs = opToArg(sc, e.op);
                        result = new DotTemplateInstanceExp(e.loc, ae.e1, Id.opSliceUnary, tiargs);
                        result = new CallExp(e.loc, result, a);
                        result = result.expressionSemantic(sc);
                        result = Expression.combine(e0, result);
                        return;
                    }
                    // Didn't find it. Forward to aliasthis
                    if (ad.aliasthis && t1b != ae.att1)
                    {
                        if (!ae.att1 && t1b.checkAliasThisRec())
                            ae.att1 = t1b;
                        /* Rewrite op(a[arguments]) as:
                         *      op(a.aliasthis[arguments])
                         */
                        ae.e1 = resolveAliasThis(sc, ae1save, true);
                        if (ae.e1)
                            continue;
                    }
                    break;
                }
                ae.e1 = ae1old; // recovery
                ae.lengthVar = null;
            }
            e.e1 = e.e1.expressionSemantic(sc);
            e.e1 = resolveProperties(sc, e.e1);
            if (e.e1.op == TOK.error)
            {
                result = e.e1;
                return;
            }
            AggregateDeclaration ad = isAggregate(e.e1.type);
            if (ad)
            {
                Dsymbol fd = null;
                /* Rewrite as:
                 *      e1.opUnary!(op)()
                 */
                fd = search_function(ad, Id.opUnary);
                if (fd)
                {
                    Objects* tiargs = opToArg(sc, e.op);
                    result = new DotTemplateInstanceExp(e.loc, e.e1, fd.ident, tiargs);
                    result = new CallExp(e.loc, result);
                    result = result.expressionSemantic(sc);
                    return;
                }
                // D1-style operator overloads, deprecated
                if (e.op != TOK.prePlusPlus && e.op != TOK.preMinusMinus)
                {
                    auto id = opId(e);
                    fd = search_function(ad, id);
                    if (fd)
                    {
                        // @@@DEPRECATED_2.094@@@.
                        // Deprecated in 2.088
                        // Make an error in 2.094
                        e.deprecation("`%s` is deprecated.  Use `opUnary(string op)() if (op == \"%s\")` instead.", id.toChars(), Token.toChars(e.op));
                        // Rewrite +e1 as e1.add()
                        result = build_overload(e.loc, sc, e.e1, null, fd);
                        return;
                    }
                }
                // Didn't find it. Forward to aliasthis
                if (ad.aliasthis && e.e1.type != e.att1)
                {
                    /* Rewrite op(e1) as:
                     *      op(e1.aliasthis)
                     */
                    //printf("att una %s e1 = %s\n", Token::toChars(op), this.e1.type.toChars());
                    Expression e1 = new DotIdExp(e.loc, e.e1, ad.aliasthis.ident);
                    UnaExp ue = cast(UnaExp)e.copy();
                    if (!ue.att1 && e.e1.type.checkAliasThisRec())
                        ue.att1 = e.e1.type;
                    ue.e1 = e1;
                    result = ue.trySemantic(sc);
                    return;
                }
            }
        }

        override void visit(ArrayExp ae)
        {
            //printf("ArrayExp::op_overload() (%s)\n", ae.toChars());
            ae.e1 = ae.e1.expressionSemantic(sc);
            ae.e1 = resolveProperties(sc, ae.e1);
            Expression ae1old = ae.e1;
            const(bool) maybeSlice = (ae.arguments.dim == 0 || ae.arguments.dim == 1 && (*ae.arguments)[0].op == TOK.interval);
            IntervalExp ie = null;
            if (maybeSlice && ae.arguments.dim)
            {
                assert((*ae.arguments)[0].op == TOK.interval);
                ie = cast(IntervalExp)(*ae.arguments)[0];
            }
            while (true)
            {
                if (ae.e1.op == TOK.error)
                {
                    result = ae.e1;
                    return;
                }
                Expression e0 = null;
                Expression ae1save = ae.e1;
                ae.lengthVar = null;
                Type t1b = ae.e1.type.toBasetype();
                AggregateDeclaration ad = isAggregate(t1b);
                if (!ad)
                {
                    // If the non-aggregate expression ae.e1 is indexable or sliceable,
                    // convert it to the corresponding concrete expression.
                    if (isIndexableNonAggregate(t1b) || ae.e1.op == TOK.type)
                    {
                        // Convert to SliceExp
                        if (maybeSlice)
                        {
                            result = new SliceExp(ae.loc, ae.e1, ie);
                            result = result.expressionSemantic(sc);
                            return;
                        }
                        // Convert to IndexExp
                        if (ae.arguments.dim == 1)
                        {
                            result = new IndexExp(ae.loc, ae.e1, (*ae.arguments)[0]);
                            result = result.expressionSemantic(sc);
                            return;
                        }
                    }
                    break;
                }
                if (search_function(ad, Id.index))
                {
                    // Deal with $
                    result = resolveOpDollar(sc, ae, &e0);
                    if (!result) // a[i..j] might be: a.opSlice(i, j)
                        goto Lfallback;
                    if (result.op == TOK.error)
                        return;
                    /* Rewrite e1[arguments] as:
                     *      e1.opIndex(arguments)
                     */
                    Expressions* a = ae.arguments.copy();
                    result = new DotIdExp(ae.loc, ae.e1, Id.index);
                    result = new CallExp(ae.loc, result, a);
                    if (maybeSlice) // a[] might be: a.opSlice()
                        result = result.trySemantic(sc);
                    else
                        result = result.expressionSemantic(sc);
                    if (result)
                    {
                        result = Expression.combine(e0, result);
                        return;
                    }
                }
            Lfallback:
                if (maybeSlice && ae.e1.op == TOK.type)
                {
                    result = new SliceExp(ae.loc, ae.e1, ie);
                    result = result.expressionSemantic(sc);
                    result = Expression.combine(e0, result);
                    return;
                }
                if (maybeSlice && search_function(ad, Id.slice))
                {
                    // Deal with $
                    result = resolveOpDollar(sc, ae, ie, &e0);
                    if (result.op == TOK.error)
                        return;
                    /* Rewrite a[i..j] as:
                     *      a.opSlice(i, j)
                     */
                    auto a = new Expressions();
                    if (ie)
                    {
                        a.push(ie.lwr);
                        a.push(ie.upr);
                    }
                    result = new DotIdExp(ae.loc, ae.e1, Id.slice);
                    result = new CallExp(ae.loc, result, a);
                    result = result.expressionSemantic(sc);
                    result = Expression.combine(e0, result);
                    return;
                }
                // Didn't find it. Forward to aliasthis
                if (ad.aliasthis && t1b != ae.att1)
                {
                    if (!ae.att1 && t1b.checkAliasThisRec())
                        ae.att1 = t1b;
                    //printf("att arr e1 = %s\n", this.e1.type.toChars());
                    /* Rewrite op(a[arguments]) as:
                     *      op(a.aliasthis[arguments])
                     */
                    ae.e1 = resolveAliasThis(sc, ae1save, true);
                    if (ae.e1)
                        continue;
                }
                break;
            }
            ae.e1 = ae1old; // recovery
            ae.lengthVar = null;
        }

        /***********************************************
         * This is mostly the same as UnaryExp::op_overload(), but has
         * a different rewrite.
         */
        override void visit(CastExp e)
        {
            //printf("CastExp::op_overload() (%s)\n", e.toChars());
            AggregateDeclaration ad = isAggregate(e.e1.type);
            if (ad)
            {
                Dsymbol fd = null;
                /* Rewrite as:
                 *      e1.opCast!(T)()
                 */
                fd = search_function(ad, Id._cast);
                if (fd)
                {
                    version (all)
                    {
                        // Backwards compatibility with D1 if opCast is a function, not a template
                        if (fd.isFuncDeclaration())
                        {
                            // Rewrite as:  e1.opCast()
                            result = build_overload(e.loc, sc, e.e1, null, fd);
                            return;
                        }
                    }
                    auto tiargs = new Objects();
                    tiargs.push(e.to);
                    result = new DotTemplateInstanceExp(e.loc, e.e1, fd.ident, tiargs);
                    result = new CallExp(e.loc, result);
                    result = result.expressionSemantic(sc);
                    return;
                }
                // Didn't find it. Forward to aliasthis
                if (ad.aliasthis)
                {
                    /* Rewrite op(e1) as:
                     *      op(e1.aliasthis)
                     */
                    Expression e1 = resolveAliasThis(sc, e.e1);
                    result = e.copy();
                    (cast(UnaExp)result).e1 = e1;
                    result = result.op_overload(sc);
                    return;
                }
            }
        }

        override void visit(BinExp e)
        {
            //printf("BinExp::op_overload() (%s)\n", e.toChars());
            Identifier id = opId(e);
            Identifier id_r = opId_r(e);
            Expressions args1;
            Expressions args2;
            int argsset = 0;
            AggregateDeclaration ad1 = isAggregate(e.e1.type);
            AggregateDeclaration ad2 = isAggregate(e.e2.type);
            if (e.op == TOK.assign && ad1 == ad2)
            {
                StructDeclaration sd = ad1.isStructDeclaration();
                if (sd && !sd.hasIdentityAssign)
                {
                    /* This is bitwise struct assignment. */
                    return;
                }
            }
            Dsymbol s = null;
            Dsymbol s_r = null;
            Objects* tiargs = null;
            if (e.op == TOK.plusPlus || e.op == TOK.minusMinus)
            {
                // Bug4099 fix
                if (ad1 && search_function(ad1, Id.opUnary))
                    return;
            }
            if (e.op != TOK.equal && e.op != TOK.notEqual && e.op != TOK.assign && e.op != TOK.plusPlus && e.op != TOK.minusMinus)
            {
                /* Try opBinary and opBinaryRight
                 */
                if (ad1)
                {
                    s = search_function(ad1, Id.opBinary);
                    if (s && !s.isTemplateDeclaration())
                    {
                        e.e1.error("`%s.opBinary` isn't a template", e.e1.toChars());
                        result = new ErrorExp();
                        return;
                    }
                }
                if (ad2)
                {
                    s_r = search_function(ad2, Id.opBinaryRight);
                    if (s_r && !s_r.isTemplateDeclaration())
                    {
                        e.e2.error("`%s.opBinaryRight` isn't a template", e.e2.toChars());
                        result = new ErrorExp();
                        return;
                    }
                    if (s_r && s_r == s) // https://issues.dlang.org/show_bug.cgi?id=12778
                        s_r = null;
                }
                // Set tiargs, the template argument list, which will be the operator string
                if (s || s_r)
                {
                    id = Id.opBinary;
                    id_r = Id.opBinaryRight;
                    tiargs = opToArg(sc, e.op);
                }
            }
            if (!s && !s_r)
            {
                // Try the D1-style operators, deprecated
                if (ad1 && id)
                {
                    s = search_function(ad1, id);
                    if (s && id != Id.assign)
                    {
                        // @@@DEPRECATED_2.094@@@.
                        // Deprecated in 2.088
                        // Make an error in 2.094
                        if (id == Id.postinc || id == Id.postdec)
                            e.deprecation("`%s` is deprecated.  Use `opUnary(string op)() if (op == \"%s\")` instead.", id.toChars(), Token.toChars(e.op));
                        else
                            e.deprecation("`%s` is deprecated.  Use `opBinary(string op)(...) if (op == \"%s\")` instead.", id.toChars(), Token.toChars(e.op));
                    }
                }
                if (ad2 && id_r)
                {
                    s_r = search_function(ad2, id_r);
                    // https://issues.dlang.org/show_bug.cgi?id=12778
                    // If both x.opBinary(y) and y.opBinaryRight(x) found,
                    // and they are exactly same symbol, x.opBinary(y) should be preferred.
                    if (s_r && s_r == s)
                        s_r = null;
                    if (s_r)
                    {
                        // @@@DEPRECATED_2.094@@@.
                        // Deprecated in 2.088
                        // Make an error in 2.094
                        e.deprecation("`%s` is deprecated.  Use `opBinaryRight(string op)(...) if (op == \"%s\")` instead.", id_r.toChars(), Token.toChars(e.op));
                    }
                }
            }
            if (s || s_r)
            {
                /* Try:
                 *      a.opfunc(b)
                 *      b.opfunc_r(a)
                 * and see which is better.
                 */
                args1.setDim(1);
                args1[0] = e.e1;
                expandTuples(&args1);
                args2.setDim(1);
                args2[0] = e.e2;
                expandTuples(&args2);
                argsset = 1;
                MatchAccumulator m;
                if (s)
                {
                    functionResolve(m, s, e.loc, sc, tiargs, e.e1.type, &args2);
                    if (m.lastf && (m.lastf.errors || m.lastf.semantic3Errors))
                    {
                        result = new ErrorExp();
                        return;
                    }
                }
                FuncDeclaration lastf = m.lastf;
                if (s_r)
                {
                    functionResolve(m, s_r, e.loc, sc, tiargs, e.e2.type, &args1);
                    if (m.lastf && (m.lastf.errors || m.lastf.semantic3Errors))
                    {
                        result = new ErrorExp();
                        return;
                    }
                }
                if (m.count > 1)
                {
                    // Error, ambiguous
                    e.error("overloads `%s` and `%s` both match argument list for `%s`", m.lastf.type.toChars(), m.nextf.type.toChars(), m.lastf.toChars());
                }
                else if (m.last <= MATCH.nomatch)
                {
                    if (tiargs)
                        goto L1;
                    m.lastf = null;
                }
                if (e.op == TOK.plusPlus || e.op == TOK.minusMinus)
                {
                    // Kludge because operator overloading regards e++ and e--
                    // as unary, but it's implemented as a binary.
                    // Rewrite (e1 ++ e2) as e1.postinc()
                    // Rewrite (e1 -- e2) as e1.postdec()
                    result = build_overload(e.loc, sc, e.e1, null, m.lastf ? m.lastf : s);
                }
                else if (lastf && m.lastf == lastf || !s_r && m.last <= MATCH.nomatch)
                {
                    // Rewrite (e1 op e2) as e1.opfunc(e2)
                    result = build_overload(e.loc, sc, e.e1, e.e2, m.lastf ? m.lastf : s);
                }
                else
                {
                    // Rewrite (e1 op e2) as e2.opfunc_r(e1)
                    result = build_overload(e.loc, sc, e.e2, e.e1, m.lastf ? m.lastf : s_r);
                }
                return;
            }
        L1:
            version (all)
            {
                // Retained for D1 compatibility
                if (isCommutative(e.op) && !tiargs)
                {
                    s = null;
                    s_r = null;
                    if (ad1 && id_r)
                    {
                        s_r = search_function(ad1, id_r);
                    }
                    if (ad2 && id)
                    {
                        s = search_function(ad2, id);
                        if (s && s == s_r) // https://issues.dlang.org/show_bug.cgi?id=12778
                            s = null;
                    }
                    if (s || s_r)
                    {
                        /* Try:
                         *  a.opfunc_r(b)
                         *  b.opfunc(a)
                         * and see which is better.
                         */
                        if (!argsset)
                        {
                            args1.setDim(1);
                            args1[0] = e.e1;
                            expandTuples(&args1);
                            args2.setDim(1);
                            args2[0] = e.e2;
                            expandTuples(&args2);
                        }
                        MatchAccumulator m;
                        if (s_r)
                        {
                            functionResolve(m, s_r, e.loc, sc, tiargs, e.e1.type, &args2);
                            if (m.lastf && (m.lastf.errors || m.lastf.semantic3Errors))
                            {
                                result = new ErrorExp();
                                return;
                            }
                        }
                        FuncDeclaration lastf = m.lastf;
                        if (s)
                        {
                            functionResolve(m, s, e.loc, sc, tiargs, e.e2.type, &args1);
                            if (m.lastf && (m.lastf.errors || m.lastf.semantic3Errors))
                            {
                                result = new ErrorExp();
                                return;
                            }
                        }
                        if (m.count > 1)
                        {
                            // Error, ambiguous
                            e.error("overloads `%s` and `%s` both match argument list for `%s`", m.lastf.type.toChars(), m.nextf.type.toChars(), m.lastf.toChars());
                        }
                        else if (m.last <= MATCH.nomatch)
                        {
                            m.lastf = null;
                        }

                        if (lastf && m.lastf == lastf || !s && m.last <= MATCH.nomatch)
                        {
                            // Rewrite (e1 op e2) as e1.opfunc_r(e2)
                            result = build_overload(e.loc, sc, e.e1, e.e2, m.lastf ? m.lastf : s_r);
                        }
                        else
                        {
                            // Rewrite (e1 op e2) as e2.opfunc(e1)
                            result = build_overload(e.loc, sc, e.e2, e.e1, m.lastf ? m.lastf : s);
                        }
                        // When reversing operands of comparison operators,
                        // need to reverse the sense of the op
                        if (pop)
                            *pop = reverseRelation(e.op);
                        return;
                    }
                }
            }

            Expression tempResult;
            if (!(e.op == TOK.assign && ad2 && ad1 == ad2)) // https://issues.dlang.org/show_bug.cgi?id=2943
            {
                result = checkAliasThisForLhs(ad1, sc, e);
                if (result)
                {
                    /* https://issues.dlang.org/show_bug.cgi?id=19441
                     *
                     * alias this may not be used for partial assignment.
                     * If a struct has a single member which is aliased this
                     * directly or aliased to a ref getter function that returns
                     * the mentioned member, then alias this may be
                     * used since the object will be fully initialised.
                     * If the struct is nested, the context pointer is considered
                     * one of the members, hence the `ad1.fields.dim == 2 && ad1.vthis`
                     * condition.
                     */
                    if (e.op != TOK.assign || e.e1.op == TOK.type)
                        return;

                    if (ad1.fields.dim == 1 || (ad1.fields.dim == 2 && ad1.vthis))
                    {
                        auto var = ad1.aliasthis.sym.isVarDeclaration();
                        if (var && var.type == ad1.fields[0].type)
                            return;

                        auto func = ad1.aliasthis.sym.isFuncDeclaration();
                        auto tf = cast(TypeFunction)(func.type);
                        if (tf.isref && ad1.fields[0].type == tf.next)
                            return;
                    }
                    tempResult = result;
                }
            }
            if (!(e.op == TOK.assign && ad1 && ad1 == ad2)) // https://issues.dlang.org/show_bug.cgi?id=2943
            {
                result = checkAliasThisForRhs(ad2, sc, e);
                if (result)
                    return;
            }

            // @@@DEPRECATED_2019-02@@@
            // 1. Deprecation for 1 year
            // 2. Turn to error after
            if (tempResult)
            {
                // move this line where tempResult is assigned to result and turn to error when derecation period is over
                e.deprecation("Cannot use `alias this` to partially initialize variable `%s` of type `%s`. Use `%s`", e.e1.toChars(), ad1.toChars(), (cast(BinExp)tempResult).e1.toChars());
                // delete this line when deprecation period is over
                result = tempResult;
            }
        }

        override void visit(EqualExp e)
        {
            //printf("EqualExp::op_overload() (%s)\n", e.toChars());
            Type t1 = e.e1.type.toBasetype();
            Type t2 = e.e2.type.toBasetype();

            /* Check for array equality.
             */
            if ((t1.ty == Tarray || t1.ty == Tsarray) &&
                (t2.ty == Tarray || t2.ty == Tsarray))
            {
                bool needsDirectEq()
                {
                    Type t1n = t1.nextOf().toBasetype();
                    Type t2n = t2.nextOf().toBasetype();
                    if (((t1n.ty == Tchar || t1n.ty == Twchar || t1n.ty == Tdchar) &&
                         (t2n.ty == Tchar || t2n.ty == Twchar || t2n.ty == Tdchar)) ||
                        (t1n.ty == Tvoid || t2n.ty == Tvoid))
                    {
                        return false;
                    }
                    if (t1n.constOf() != t2n.constOf())
                        return true;

                    Type t = t1n;
                    while (t.toBasetype().nextOf())
                        t = t.nextOf().toBasetype();
                    if (t.ty != Tstruct)
                        return false;

                    if (global.params.useTypeInfo && Type.dtypeinfo)
                        semanticTypeInfo(sc, t);

                    return (cast(TypeStruct)t).sym.hasIdentityEquals;
                }

                if (needsDirectEq() && !(t1.ty == Tarray && t2.ty == Tarray))
                {
                    /* Rewrite as:
                     *      __ArrayEq(e1, e2)
                     */
                    Expression eeq = new IdentifierExp(e.loc, Id.__ArrayEq);
                    result = new CallExp(e.loc, eeq, e.e1, e.e2);
                    if (e.op == TOK.notEqual)
                        result = new NotExp(e.loc, result);
                    result = result.trySemantic(sc); // for better error message
                    if (!result)
                    {
                        e.error("cannot compare `%s` and `%s`", t1.toChars(), t2.toChars());
                        result = new ErrorExp();
                    }
                    return;
                }
            }

            /* Check for class equality with null literal or typeof(null).
             */
            if (t1.ty == Tclass && e.e2.op == TOK.null_ ||
                t2.ty == Tclass && e.e1.op == TOK.null_)
            {
                e.error("use `%s` instead of `%s` when comparing with `null`",
                    Token.toChars(e.op == TOK.equal ? TOK.identity : TOK.notIdentity),
                    Token.toChars(e.op));
                result = new ErrorExp();
                return;
            }
            if (t1.ty == Tclass && t2.ty == Tnull ||
                t1.ty == Tnull && t2.ty == Tclass)
            {
                // Comparing a class with typeof(null) should not call opEquals
                return;
            }

            /* Check for class equality.
             */
            if (t1.ty == Tclass && t2.ty == Tclass)
            {
                ClassDeclaration cd1 = t1.isClassHandle();
                ClassDeclaration cd2 = t2.isClassHandle();
                if (!(cd1.classKind == ClassKind.cpp || cd2.classKind == ClassKind.cpp))
                {
                    /* Rewrite as:
                     *      .object.opEquals(e1, e2)
                     */
                    Expression e1x = e.e1;
                    Expression e2x = e.e2;

                    /* The explicit cast is necessary for interfaces
                     * https://issues.dlang.org/show_bug.cgi?id=4088
                     */
                    Type to = ClassDeclaration.object.getType();
                    if (cd1.isInterfaceDeclaration())
                        e1x = new CastExp(e.loc, e.e1, t1.isMutable() ? to : to.constOf());
                    if (cd2.isInterfaceDeclaration())
                        e2x = new CastExp(e.loc, e.e2, t2.isMutable() ? to : to.constOf());

                    result = new IdentifierExp(e.loc, Id.empty);
                    result = new DotIdExp(e.loc, result, Id.object);
                    result = new DotIdExp(e.loc, result, Id.eq);
                    result = new CallExp(e.loc, result, e1x, e2x);
                    if (e.op == TOK.notEqual)
                        result = new NotExp(e.loc, result);
                    result = result.expressionSemantic(sc);
                    return;
                }
            }

            result = compare_overload(e, sc, Id.eq, null);
            if (result)
            {
                if (result.op == TOK.call && e.op == TOK.notEqual)
                {
                    result = new NotExp(result.loc, result);
                    result = result.expressionSemantic(sc);
                }
                return;
            }

            if (t1.ty == Tarray && t2.ty == Tarray)
                return;

            /* Check for pointer equality.
             */
            if (t1.ty == Tpointer || t2.ty == Tpointer)
            {
                /* Rewrite:
                 *      ptr1 == ptr2
                 * as:
                 *      ptr1 is ptr2
                 *
                 * This is just a rewriting for deterministic AST representation
                 * as the backend input.
                 */
                auto op2 = e.op == TOK.equal ? TOK.identity : TOK.notIdentity;
                result = new IdentityExp(op2, e.loc, e.e1, e.e2);
                result = result.expressionSemantic(sc);
                return;
            }

            /* Check for struct equality without opEquals.
             */
            if (t1.ty == Tstruct && t2.ty == Tstruct)
            {
                auto sd = (cast(TypeStruct)t1).sym;
                if (sd != (cast(TypeStruct)t2).sym)
                    return;

                import dmd.clone : needOpEquals;
                if (!global.params.fieldwise && !needOpEquals(sd))
                {
                    // Use bitwise equality.
                    auto op2 = e.op == TOK.equal ? TOK.identity : TOK.notIdentity;
                    result = new IdentityExp(op2, e.loc, e.e1, e.e2);
                    result = result.expressionSemantic(sc);
                    return;
                }

                /* Do memberwise equality.
                 * https://dlang.org/spec/expression.html#equality_expressions
                 * Rewrite:
                 *      e1 == e2
                 * as:
                 *      e1.tupleof == e2.tupleof
                 *
                 * If sd is a nested struct, and if it's nested in a class, it will
                 * also compare the parent class's equality. Otherwise, compares
                 * the identity of parent context through void*.
                 */
                if (e.att1 && t1 == e.att1) return;
                if (e.att2 && t2 == e.att2) return;

                e = cast(EqualExp)e.copy();
                if (!e.att1) e.att1 = t1;
                if (!e.att2) e.att2 = t2;
                e.e1 = new DotIdExp(e.loc, e.e1, Id._tupleof);
                e.e2 = new DotIdExp(e.loc, e.e2, Id._tupleof);

                auto sc2 = sc.push();
                sc2.flags = (sc2.flags & ~SCOPE.onlysafeaccess) | SCOPE.noaccesscheck;
                result = e.expressionSemantic(sc2);
                sc2.pop();

                /* https://issues.dlang.org/show_bug.cgi?id=15292
                 * if the rewrite result is same with the original,
                 * the equality is unresolvable because it has recursive definition.
                 */
                if (result.op == e.op &&
                    (cast(EqualExp)result).e1.type.toBasetype() == t1)
                {
                    e.error("cannot compare `%s` because its auto generated member-wise equality has recursive definition",
                        t1.toChars());
                    result = new ErrorExp();
                }
                return;
            }

            /* Check for tuple equality.
             */
            if (e.e1.op == TOK.tuple && e.e2.op == TOK.tuple)
            {
                auto tup1 = cast(TupleExp)e.e1;
                auto tup2 = cast(TupleExp)e.e2;
                size_t dim = tup1.exps.dim;
                if (dim != tup2.exps.dim)
                {
                    e.error("mismatched tuple lengths, `%d` and `%d`",
                        cast(int)dim, cast(int)tup2.exps.dim);
                    result = new ErrorExp();
                    return;
                }

                if (dim == 0)
                {
                    // zero-length tuple comparison should always return true or false.
                    result = IntegerExp.createBool(e.op == TOK.equal);
                }
                else
                {
                    for (size_t i = 0; i < dim; i++)
                    {
                        auto ex1 = (*tup1.exps)[i];
                        auto ex2 = (*tup2.exps)[i];
                        auto eeq = new EqualExp(e.op, e.loc, ex1, ex2);
                        eeq.att1 = e.att1;
                        eeq.att2 = e.att2;

                        if (!result)
                            result = eeq;
                        else if (e.op == TOK.equal)
                            result = new LogicalExp(e.loc, TOK.andAnd, result, eeq);
                        else
                            result = new LogicalExp(e.loc, TOK.orOr, result, eeq);
                    }
                    assert(result);
                }
                result = Expression.combine(tup1.e0, tup2.e0, result);
                result = result.expressionSemantic(sc);

                return;
            }
        }

        override void visit(CmpExp e)
        {
            //printf("CmpExp:: () (%s)\n", e.toChars());
            result = compare_overload(e, sc, Id.cmp, pop);
        }

        /*********************************
         * Operator overloading for op=
         */
        override void visit(BinAssignExp e)
        {
            //printf("BinAssignExp::op_overload() (%s)\n", e.toChars());
            if (e.e1.op == TOK.array)
            {
                ArrayExp ae = cast(ArrayExp)e.e1;
                ae.e1 = ae.e1.expressionSemantic(sc);
                ae.e1 = resolveProperties(sc, ae.e1);
                Expression ae1old = ae.e1;
                const(bool) maybeSlice = (ae.arguments.dim == 0 || ae.arguments.dim == 1 && (*ae.arguments)[0].op == TOK.interval);
                IntervalExp ie = null;
                if (maybeSlice && ae.arguments.dim)
                {
                    assert((*ae.arguments)[0].op == TOK.interval);
                    ie = cast(IntervalExp)(*ae.arguments)[0];
                }
                while (true)
                {
                    if (ae.e1.op == TOK.error)
                    {
                        result = ae.e1;
                        return;
                    }
                    Expression e0 = null;
                    Expression ae1save = ae.e1;
                    ae.lengthVar = null;
                    Type t1b = ae.e1.type.toBasetype();
                    AggregateDeclaration ad = isAggregate(t1b);
                    if (!ad)
                        break;
                    if (search_function(ad, Id.opIndexOpAssign))
                    {
                        // Deal with $
                        result = resolveOpDollar(sc, ae, &e0);
                        if (!result) // (a[i..j] op= e2) might be: a.opSliceOpAssign!(op)(e2, i, j)
                            goto Lfallback;
                        if (result.op == TOK.error)
                            return;
                        result = e.e2.expressionSemantic(sc);
                        if (result.op == TOK.error)
                            return;
                        e.e2 = result;
                        /* Rewrite a[arguments] op= e2 as:
                         *      a.opIndexOpAssign!(op)(e2, arguments)
                         */
                        Expressions* a = ae.arguments.copy();
                        a.insert(0, e.e2);
                        Objects* tiargs = opToArg(sc, e.op);
                        result = new DotTemplateInstanceExp(e.loc, ae.e1, Id.opIndexOpAssign, tiargs);
                        result = new CallExp(e.loc, result, a);
                        if (maybeSlice) // (a[] op= e2) might be: a.opSliceOpAssign!(op)(e2)
                            result = result.trySemantic(sc);
                        else
                            result = result.expressionSemantic(sc);
                        if (result)
                        {
                            result = Expression.combine(e0, result);
                            return;
                        }
                    }
                Lfallback:
                    if (maybeSlice && search_function(ad, Id.opSliceOpAssign))
                    {
                        // Deal with $
                        result = resolveOpDollar(sc, ae, ie, &e0);
                        if (result.op == TOK.error)
                            return;
                        result = e.e2.expressionSemantic(sc);
                        if (result.op == TOK.error)
                            return;
                        e.e2 = result;
                        /* Rewrite (a[i..j] op= e2) as:
                         *      a.opSliceOpAssign!(op)(e2, i, j)
                         */
                        auto a = new Expressions();
                        a.push(e.e2);
                        if (ie)
                        {
                            a.push(ie.lwr);
                            a.push(ie.upr);
                        }
                        Objects* tiargs = opToArg(sc, e.op);
                        result = new DotTemplateInstanceExp(e.loc, ae.e1, Id.opSliceOpAssign, tiargs);
                        result = new CallExp(e.loc, result, a);
                        result = result.expressionSemantic(sc);
                        result = Expression.combine(e0, result);
                        return;
                    }
                    // Didn't find it. Forward to aliasthis
                    if (ad.aliasthis && t1b != ae.att1)
                    {
                        if (!ae.att1 && t1b.checkAliasThisRec())
                            ae.att1 = t1b;
                        /* Rewrite (a[arguments] op= e2) as:
                         *      a.aliasthis[arguments] op= e2
                         */
                        ae.e1 = resolveAliasThis(sc, ae1save, true);
                        if (ae.e1)
                            continue;
                    }
                    break;
                }
                ae.e1 = ae1old; // recovery
                ae.lengthVar = null;
            }
            result = e.binSemanticProp(sc);
            if (result)
                return;
            // Don't attempt 'alias this' if an error occurred
            if (e.e1.type.ty == Terror || e.e2.type.ty == Terror)
            {
                result = new ErrorExp();
                return;
            }
            Identifier id = opId(e);
            Expressions args2;
            AggregateDeclaration ad1 = isAggregate(e.e1.type);
            Dsymbol s = null;
            Objects* tiargs = null;
            /* Try opOpAssign
             */
            if (ad1)
            {
                s = search_function(ad1, Id.opOpAssign);
                if (s && !s.isTemplateDeclaration())
                {
                    e.error("`%s.opOpAssign` isn't a template", e.e1.toChars());
                    result = new ErrorExp();
                    return;
                }
            }
            // Set tiargs, the template argument list, which will be the operator string
            if (s)
            {
                id = Id.opOpAssign;
                tiargs = opToArg(sc, e.op);
            }

            // Try D1-style operator overload, deprecated
            if (!s && ad1 && id)
            {
                s = search_function(ad1, id);
                if (s)
                {
                    // @@@DEPRECATED_2.094@@@.
                    // Deprecated in 2.088
                    // Make an error in 2.094
                    scope char[] op = Token.toString(e.op).dup;
                    op[$-1] = '\0'; // remove trailing `=`
                    e.deprecation("`%s` is deprecated.  Use `opOpAssign(string op)(...) if (op == \"%s\")` instead.", id.toChars(), op.ptr);
                }
            }

            if (s)
            {
                /* Try:
                 *      a.opOpAssign(b)
                 */
                args2.setDim(1);
                args2[0] = e.e2;
                expandTuples(&args2);
                MatchAccumulator m;
                if (s)
                {
                    functionResolve(m, s, e.loc, sc, tiargs, e.e1.type, &args2);
                    if (m.lastf && (m.lastf.errors || m.lastf.semantic3Errors))
                    {
                        result = new ErrorExp();
                        return;
                    }
                }
                if (m.count > 1)
                {
                    // Error, ambiguous
                    e.error("overloads `%s` and `%s` both match argument list for `%s`", m.lastf.type.toChars(), m.nextf.type.toChars(), m.lastf.toChars());
                }
                else if (m.last <= MATCH.nomatch)
                {
                    if (tiargs)
                        goto L1;
                    m.lastf = null;
                }
                // Rewrite (e1 op e2) as e1.opOpAssign(e2)
                result = build_overload(e.loc, sc, e.e1, e.e2, m.lastf ? m.lastf : s);
                return;
            }
        L1:
            result = checkAliasThisForLhs(ad1, sc, e);
            if (result || !s) // no point in trying Rhs alias-this if there's no overload of any kind in lhs
                return;

            result = checkAliasThisForRhs(isAggregate(e.e2.type), sc, e);
        }
    }

    if (pop)
        *pop = e.op;
    scope OpOverload v = new OpOverload(sc, pop);
    e.accept(v);
    return v.result;
}

/******************************************
 * Common code for overloading of EqualExp and CmpExp
 */
private Expression compare_overload(BinExp e, Scope* sc, Identifier id, TOK* pop)
{
    //printf("BinExp::compare_overload(id = %s) %s\n", id.toChars(), e.toChars());
    AggregateDeclaration ad1 = isAggregate(e.e1.type);
    AggregateDeclaration ad2 = isAggregate(e.e2.type);
    Dsymbol s = null;
    Dsymbol s_r = null;
    if (ad1)
    {
        s = search_function(ad1, id);
    }
    if (ad2)
    {
        s_r = search_function(ad2, id);
        if (s == s_r)
            s_r = null;
    }
    Objects* tiargs = null;
    if (s || s_r)
    {
        /* Try:
         *      a.opEquals(b)
         *      b.opEquals(a)
         * and see which is better.
         */
        Expressions args1 = Expressions(1);
        args1[0] = e.e1;
        expandTuples(&args1);
        Expressions args2 = Expressions(1);
        args2[0] = e.e2;
        expandTuples(&args2);
        MatchAccumulator m;
        if (0 && s && s_r)
        {
            printf("s  : %s\n", s.toPrettyChars());
            printf("s_r: %s\n", s_r.toPrettyChars());
        }
        if (s)
        {
            functionResolve(m, s, e.loc, sc, tiargs, e.e1.type, &args2);
            if (m.lastf && (m.lastf.errors || m.lastf.semantic3Errors))
                return new ErrorExp();
        }
        FuncDeclaration lastf = m.lastf;
        int count = m.count;
        if (s_r)
        {
            functionResolve(m, s_r, e.loc, sc, tiargs, e.e2.type, &args1);
            if (m.lastf && (m.lastf.errors || m.lastf.semantic3Errors))
                return new ErrorExp();
        }
        if (m.count > 1)
        {
            /* The following if says "not ambiguous" if there's one match
             * from s and one from s_r, in which case we pick s.
             * This doesn't follow the spec, but is a workaround for the case
             * where opEquals was generated from templates and we cannot figure
             * out if both s and s_r came from the same declaration or not.
             * The test case is:
             *   import std.typecons;
             *   void main() {
             *    assert(tuple("has a", 2u) == tuple("has a", 1));
             *   }
             */
            if (!(m.lastf == lastf && m.count == 2 && count == 1))
            {
                // Error, ambiguous
                e.error("overloads `%s` and `%s` both match argument list for `%s`", m.lastf.type.toChars(), m.nextf.type.toChars(), m.lastf.toChars());
            }
        }
        else if (m.last <= MATCH.nomatch)
        {
            m.lastf = null;
        }
        Expression result;
        if (lastf && m.lastf == lastf || !s_r && m.last <= MATCH.nomatch)
        {
            // Rewrite (e1 op e2) as e1.opfunc(e2)
            result = build_overload(e.loc, sc, e.e1, e.e2, m.lastf ? m.lastf : s);
        }
        else
        {
            // Rewrite (e1 op e2) as e2.opfunc_r(e1)
            result = build_overload(e.loc, sc, e.e2, e.e1, m.lastf ? m.lastf : s_r);
            // When reversing operands of comparison operators,
            // need to reverse the sense of the op
            if (pop)
                *pop = reverseRelation(e.op);
        }
        return result;
    }
    /*
     * https://issues.dlang.org/show_bug.cgi?id=16657
     * at this point, no matching opEquals was found for structs,
     * so we should not follow the alias this comparison code.
     */
    if ((e.op == TOK.equal || e.op == TOK.notEqual) && ad1 == ad2)
        return null;
    Expression result = checkAliasThisForLhs(ad1, sc, e);
    return result ? result : checkAliasThisForRhs(isAggregate(e.e2.type), sc, e);
}

/***********************************
 * Utility to build a function call out of this reference and argument.
 */
Expression build_overload(const ref Loc loc, Scope* sc, Expression ethis, Expression earg, Dsymbol d)
{
    assert(d);
    Expression e;
    Declaration decl = d.isDeclaration();
    if (decl)
        e = new DotVarExp(loc, ethis, decl, false);
    else
        e = new DotIdExp(loc, ethis, d.ident);
    e = new CallExp(loc, e, earg);
    e = e.expressionSemantic(sc);
    return e;
}

/***************************************
 * Search for function funcid in aggregate ad.
 */
Dsymbol search_function(ScopeDsymbol ad, Identifier funcid)
{
    Dsymbol s = ad.search(Loc.initial, funcid);
    if (s)
    {
        //printf("search_function: s = '%s'\n", s.kind());
        Dsymbol s2 = s.toAlias();
        //printf("search_function: s2 = '%s'\n", s2.kind());
        FuncDeclaration fd = s2.isFuncDeclaration();
        if (fd && fd.type.ty == Tfunction)
            return fd;
        TemplateDeclaration td = s2.isTemplateDeclaration();
        if (td)
            return td;
    }
    return null;
}

/**************************************
 * Figure out what is being foreach'd over by looking at the ForeachAggregate.
 * Params:
 *      sc = context
 *      isForeach = true for foreach, false for foreach_reverse
 *      feaggr = ForeachAggregate
 *      sapply = set to function opApply/opApplyReverse, or delegate, or null.
 *               Overload resolution is not done.
 * Returns:
 *      true if successfully figured it out; feaggr updated with semantic analysis.
 *      false for failed, which is an error.
 */
bool inferForeachAggregate(Scope* sc, bool isForeach, ref Expression feaggr, out Dsymbol sapply)
{
    //printf("inferForeachAggregate(%s)\n", feaggr.toChars());
    bool sliced;
    Type att = null;
    auto aggr = feaggr;
    while (1)
    {
        aggr = aggr.expressionSemantic(sc);
        aggr = resolveProperties(sc, aggr);
        aggr = aggr.optimize(WANTvalue);
        if (!aggr.type || aggr.op == TOK.error)
            return false;
        Type tab = aggr.type.toBasetype();
        switch (tab.ty)
        {
        case Tarray:            // https://dlang.org/spec/statement.html#foreach_over_arrays
        case Tsarray:           // https://dlang.org/spec/statement.html#foreach_over_arrays
        case Ttuple:            // https://dlang.org/spec/statement.html#foreach_over_tuples
        case Taarray:           // https://dlang.org/spec/statement.html#foreach_over_associative_arrays
            break;

        case Tclass:
        case Tstruct:
        {
            AggregateDeclaration ad = (tab.ty == Tclass) ? (cast(TypeClass)tab).sym
                                                         : (cast(TypeStruct)tab).sym;
            if (!sliced)
            {
                sapply = search_function(ad, isForeach ? Id.apply : Id.applyReverse);
                if (sapply)
                {
                    // https://dlang.org/spec/statement.html#foreach_over_struct_and_classes
                    // opApply aggregate
                    break;
                }
                if (feaggr.op != TOK.type)
                {
                    /* See if rewriting `aggr` to `aggr[]` will work
                     */
                    Expression rinit = new ArrayExp(aggr.loc, feaggr);
                    rinit = rinit.trySemantic(sc);
                    if (rinit) // if it worked
                    {
                        aggr = rinit;
                        sliced = true;  // only try it once
                        continue;
                    }
                }
            }
            if (ad.search(Loc.initial, isForeach ? Id.Ffront : Id.Fback))
            {
                // https://dlang.org/spec/statement.html#foreach-with-ranges
                // range aggregate
                break;
            }
            if (ad.aliasthis)
            {
                if (att == tab)         // error, circular alias this
                    return false;
                if (!att && tab.checkAliasThisRec())
                    att = tab;
                aggr = resolveAliasThis(sc, aggr);
                continue;
            }
            return false;
        }

        case Tdelegate:        // https://dlang.org/spec/statement.html#foreach_over_delegates
            if (aggr.op == TOK.delegate_)
            {
                sapply = (cast(DelegateExp)aggr).func;
            }
            break;

        case Terror:
            break;

        default:
            return false;
        }
        feaggr = aggr;
        return true;
    }
    assert(0);
}

/*****************************************
 * Given array of foreach parameters and an aggregate type,
 * find best opApply overload,
 * if any of the parameter types are missing, attempt to infer
 * them from the aggregate type.
 * Params:
 *      fes = the foreach statement
 *      sc = context
 *      sapply = null or opApply or delegate
 * Returns:
 *      false for errors
 */
bool inferApplyArgTypes(ForeachStatement fes, Scope* sc, ref Dsymbol sapply)
{
    if (!fes.parameters || !fes.parameters.dim)
        return false;
    if (sapply) // prefer opApply
    {
        foreach (Parameter p; *fes.parameters)
        {
            if (p.type)
            {
                p.type = p.type.typeSemantic(fes.loc, sc);
                p.type = p.type.addStorageClass(p.storageClass);
            }
        }

        // Determine ethis for sapply
        Expression ethis;
        Type tab = fes.aggr.type.toBasetype();
        if (tab.ty == Tclass || tab.ty == Tstruct)
            ethis = fes.aggr;
        else
        {
            assert(tab.ty == Tdelegate && fes.aggr.op == TOK.delegate_);
            ethis = (cast(DelegateExp)fes.aggr).e1;
        }

        /* Look for like an
         *  int opApply(int delegate(ref Type [, ...]) dg);
         * overload
         */
        if (FuncDeclaration fd = sapply.isFuncDeclaration())
        {
            auto fdapply = findBestOpApplyMatch(ethis, fd, fes.parameters);
            if (fdapply)
            {
                // Fill in any missing types on foreach parameters[]
                matchParamsToOpApply(cast(TypeFunction)fdapply.type, fes.parameters, true);
                sapply = fdapply;
                return true;
            }
            return false;
        }
        return sapply !is null;
    }

    Parameter p = (*fes.parameters)[0];
    Type taggr = fes.aggr.type;
    assert(taggr);
    Type tab = taggr.toBasetype();
    switch (tab.ty)
    {
    case Tarray:
    case Tsarray:
    case Ttuple:
        if (fes.parameters.dim == 2)
        {
            if (!p.type)
            {
                p.type = Type.tsize_t; // key type
                p.type = p.type.addStorageClass(p.storageClass);
            }
            p = (*fes.parameters)[1];
        }
        if (!p.type && tab.ty != Ttuple)
        {
            p.type = tab.nextOf(); // value type
            p.type = p.type.addStorageClass(p.storageClass);
        }
        break;

    case Taarray:
        {
            TypeAArray taa = cast(TypeAArray)tab;
            if (fes.parameters.dim == 2)
            {
                if (!p.type)
                {
                    p.type = taa.index; // key type
                    p.type = p.type.addStorageClass(p.storageClass);
                    if (p.storageClass & STC.ref_) // key must not be mutated via ref
                        p.type = p.type.addMod(MODFlags.const_);
                }
                p = (*fes.parameters)[1];
            }
            if (!p.type)
            {
                p.type = taa.next; // value type
                p.type = p.type.addStorageClass(p.storageClass);
            }
            break;
        }

    case Tclass:
    case Tstruct:
    {
        AggregateDeclaration ad = (tab.ty == Tclass) ? (cast(TypeClass)tab).sym
                                                     : (cast(TypeStruct)tab).sym;
        if (fes.parameters.dim == 1)
        {
            if (!p.type)
            {
                /* Look for a front() or back() overload
                 */
                Identifier id = (fes.op == TOK.foreach_) ? Id.Ffront : Id.Fback;
                Dsymbol s = ad.search(Loc.initial, id);
                FuncDeclaration fd = s ? s.isFuncDeclaration() : null;
                if (fd)
                {
                    // Resolve inout qualifier of front type
                    p.type = fd.type.nextOf();
                    if (p.type)
                    {
                        p.type = p.type.substWildTo(tab.mod);
                        p.type = p.type.addStorageClass(p.storageClass);
                    }
                }
                else if (s && s.isTemplateDeclaration())
                {
                }
                else if (s && s.isDeclaration())
                    p.type = (cast(Declaration)s).type;
                else
                    break;
            }
            break;
        }
        break;
    }

    case Tdelegate:
        if (!matchParamsToOpApply(cast(TypeFunction)tab.nextOf(), fes.parameters, true))
            return false;
        break;

    default:
        break; // ignore error, caught later
    }
    return true;
}

/*********************************************
 * Find best overload match on fstart given ethis and parameters[].
 * Params:
 *      ethis = expression to use for `this`
 *      fstart = opApply or foreach delegate
 *      parameters = ForeachTypeList (i.e. foreach parameters)
 * Returns:
 *      best match if there is one, null if error
 */
private FuncDeclaration findBestOpApplyMatch(Expression ethis, FuncDeclaration fstart, Parameters* parameters)
{
    MOD mod = ethis.type.mod;
    MATCH match = MATCH.nomatch;
    FuncDeclaration fd_best;
    FuncDeclaration fd_ambig;

    overloadApply(fstart, (Dsymbol s)
    {
        auto f = s.isFuncDeclaration();
        if (!f)
            return 0;           // continue
        auto tf = cast(TypeFunction)f.type;
        MATCH m = MATCH.exact;
        if (f.isThis())
        {
            if (!MODimplicitConv(mod, tf.mod))
                m = MATCH.nomatch;
            else if (mod != tf.mod)
                m = MATCH.constant;
        }
        if (!matchParamsToOpApply(tf, parameters, false))
            m = MATCH.nomatch;
        if (m > match)
        {
            fd_best = f;
            fd_ambig = null;
            match = m;
        }
        else if (m == match && m > MATCH.nomatch)
        {
            assert(fd_best);
            /* Ignore covariant matches, as later on it can be redone
             * after the opApply delegate has its attributes inferred.
             */
            if (tf.covariant(fd_best.type) != 1 &&
                fd_best.type.covariant(tf) != 1)
                fd_ambig = f;                           // not covariant, so ambiguous
        }
        return 0;               // continue
    });

    if (fd_ambig)
    {
        .error(ethis.loc, "`%s.%s` matches more than one declaration:\n`%s`:     `%s`\nand:\n`%s`:     `%s`",
            ethis.toChars(), fstart.ident.toChars(),
            fd_best.loc.toChars(), fd_best.type.toChars(),
            fd_ambig.loc.toChars(), fd_ambig.type.toChars());
        return null;
    }

    return fd_best;
}

/******************************
 * Determine if foreach parameters match opApply parameters.
 * Infer missing foreach parameter types from type of opApply delegate.
 * Params:
 *      tf = type of opApply or delegate
 *      parameters = foreach parameters
 *      infer = infer missing parameter types
 * Returns:
 *      true for match for this function
 *      false for no match for this function
 */
private bool matchParamsToOpApply(TypeFunction tf, Parameters* parameters, bool infer)
{
    enum nomatch = false;

    /* opApply/delegate has exactly one parameter, and that parameter
     * is a delegate that looks like:
     *     int opApply(int delegate(ref Type [, ...]) dg);
     */
    if (tf.parameterList.length != 1)
        return nomatch;

    /* Get the type of opApply's dg parameter
     */
    Parameter p0 = tf.parameterList[0];
    if (p0.type.ty != Tdelegate)
        return nomatch;
    TypeFunction tdg = cast(TypeFunction)p0.type.nextOf();
    assert(tdg.ty == Tfunction);

    /* We now have tdg, the type of the delegate.
     * tdg's parameters must match that of the foreach arglist (i.e. parameters).
     * Fill in missing types in parameters.
     */
    const nparams = tdg.parameterList.length;
    if (nparams == 0 || nparams != parameters.dim || tdg.parameterList.varargs != VarArg.none)
        return nomatch; // parameter mismatch

    foreach (u, p; *parameters)
    {
        Parameter param = tdg.parameterList[u];
        if (p.type)
        {
            if (!p.type.equals(param.type))
                return nomatch;
        }
        else if (infer)
        {
            p.type = param.type;
            p.type = p.type.addStorageClass(p.storageClass);
        }
    }
    return true;
}

/**
 * Reverse relational operator, eg >= becomes <=
 * Note this is not negation.
 * Params:
 *      op = comparison operator to reverse
 * Returns:
 *      reverse of op
 */
private TOK reverseRelation(TOK op) pure
{
    switch (op)
    {
        case TOK.greaterOrEqual:  op = TOK.lessOrEqual;    break;
        case TOK.greaterThan:     op = TOK.lessThan;       break;
        case TOK.lessOrEqual:     op = TOK.greaterOrEqual; break;
        case TOK.lessThan:        op = TOK.greaterThan;    break;
        default:                  break;
    }
    return op;
}
