/**
 * Semantic analysis of expressions.
 *
 * Specification: ($LINK2 https://dlang.org/spec/expression.html, Expressions)
 *
 * Copyright:   Copyright (C) 1999-2020 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/expressionsem.d, _expressionsem.d)
 * Documentation:  https://dlang.org/phobos/dmd_expressionsem.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/expressionsem.d
 */

module dmd.expressionsem;

import core.stdc.stdio;

import dmd.access;
import dmd.aggregate;
import dmd.aliasthis;
import dmd.arrayop;
import dmd.arraytypes;
import dmd.attrib;
import dmd.astcodegen;
import dmd.canthrow;
import dmd.chkformat;
import dmd.ctorflow;
import dmd.dscope;
import dmd.dsymbol;
import dmd.declaration;
import dmd.dclass;
import dmd.dcast;
import dmd.delegatize;
import dmd.denum;
import dmd.dimport;
import dmd.dinterpret;
import dmd.dmangle;
import dmd.dmodule;
import dmd.dstruct;
import dmd.dsymbolsem;
import dmd.dtemplate;
import dmd.errors;
import dmd.escape;
import dmd.expression;
import dmd.func;
import dmd.globals;
import dmd.hdrgen;
import dmd.id;
import dmd.identifier;
import dmd.imphint;
import dmd.init;
import dmd.initsem;
import dmd.inline;
import dmd.intrange;
import dmd.mtype;
import dmd.nspace;
import dmd.opover;
import dmd.optimize;
import dmd.parse;
import dmd.root.ctfloat;
import dmd.root.file;
import dmd.root.filename;
import dmd.root.outbuffer;
import dmd.root.rootobject;
import dmd.root.string;
import dmd.semantic2;
import dmd.semantic3;
import dmd.sideeffect;
import dmd.safe;
import dmd.target;
import dmd.tokens;
import dmd.traits;
import dmd.typesem;
import dmd.typinf;
import dmd.utf;
import dmd.utils;
import dmd.visitor;

version (IN_LLVM) import gen.dpragma;

enum LOGSEMANTIC = false;

/********************************************************
 * Perform semantic analysis and CTFE on expressions to produce
 * a string.
 * Params:
 *      buf = append generated string to buffer
 *      sc = context
 *      exps = array of Expressions
 * Returns:
 *      true on error
 */
bool expressionsToString(ref OutBuffer buf, Scope* sc, Expressions* exps)
{
    if (!exps)
        return false;

    foreach (ex; *exps)
    {
        if (!ex)
            continue;
        auto sc2 = sc.startCTFE();
        auto e2 = ex.expressionSemantic(sc2);
        auto e3 = resolveProperties(sc2, e2);
        sc2.endCTFE();

        // allowed to contain types as well as expressions
        auto e4 = ctfeInterpretForPragmaMsg(e3);
        if (!e4 || e4.op == TOK.error)
            return true;

        // expand tuple
        if (auto te = e4.isTupleExp())
        {
            if (expressionsToString(buf, sc, te.exps))
                return true;
            continue;
        }
        // char literals exp `.toStringExp` return `null` but we cant override it
        // because in most contexts we don't want the conversion to succeed.
        IntegerExp ie = e4.isIntegerExp();
        const ty = (ie && ie.type) ? ie.type.ty : Terror;
        if (ty.isSomeChar)
        {
            auto tsa = new TypeSArray(ie.type, new IntegerExp(1));
            e4 = new ArrayLiteralExp(ex.loc, tsa, ie);
        }

        if (StringExp se = e4.toStringExp())
            buf.writestring(se.toUTF8(sc).peekString());
        else
            buf.writestring(e4.toString());
    }
    return false;
}


/***********************************************************
 * Resolve `exp` as a compile-time known string.
 * Params:
 *  sc  = scope
 *  exp = Expression which expected as a string
 *  s   = What the string is expected for, will be used in error diagnostic.
 * Returns:
 *  String literal, or `null` if error happens.
 */
StringExp semanticString(Scope *sc, Expression exp, const char* s)
{
    sc = sc.startCTFE();
    exp = exp.expressionSemantic(sc);
    exp = resolveProperties(sc, exp);
    sc = sc.endCTFE();

    if (exp.op == TOK.error)
        return null;

    auto e = exp;
    if (exp.type.isString())
    {
        e = e.ctfeInterpret();
        if (e.op == TOK.error)
            return null;
    }

    auto se = e.toStringExp();
    if (!se)
    {
        exp.error("`string` expected for %s, not `(%s)` of type `%s`",
            s, exp.toChars(), exp.type.toChars());
        return null;
    }
    return se;
}

private Expression extractOpDollarSideEffect(Scope* sc, UnaExp ue)
{
    Expression e0;
    Expression e1 = Expression.extractLast(ue.e1, e0);
    // https://issues.dlang.org/show_bug.cgi?id=12585
    // Extract the side effect part if ue.e1 is comma.

    if ((sc.flags & SCOPE.ctfe) ? hasSideEffect(e1) : !isTrivialExp(e1)) // match logic in extractSideEffect()
    {
        /* Even if opDollar is needed, 'e1' should be evaluate only once. So
         * Rewrite:
         *      e1.opIndex( ... use of $ ... )
         *      e1.opSlice( ... use of $ ... )
         * as:
         *      (ref __dop = e1, __dop).opIndex( ... __dop.opDollar ...)
         *      (ref __dop = e1, __dop).opSlice( ... __dop.opDollar ...)
         */
        e1 = extractSideEffect(sc, "__dop", e0, e1, false);
        assert(e1.op == TOK.variable);
        VarExp ve = cast(VarExp)e1;
        ve.var.storage_class |= STC.exptemp;     // lifetime limited to expression
    }
    ue.e1 = e1;
    return e0;
}

/**************************************
 * Runs semantic on ae.arguments. Declares temporary variables
 * if '$' was used.
 */
Expression resolveOpDollar(Scope* sc, ArrayExp ae, Expression* pe0)
{
    assert(!ae.lengthVar);
    *pe0 = null;
    AggregateDeclaration ad = isAggregate(ae.e1.type);
    Dsymbol slice = search_function(ad, Id.slice);
    //printf("slice = %s %s\n", slice.kind(), slice.toChars());
    foreach (i, e; *ae.arguments)
    {
        if (i == 0)
            *pe0 = extractOpDollarSideEffect(sc, ae);

        if (e.op == TOK.interval && !(slice && slice.isTemplateDeclaration()))
        {
        Lfallback:
            if (ae.arguments.dim == 1)
                return null;
            ae.error("multi-dimensional slicing requires template `opSlice`");
            return new ErrorExp();
        }
        //printf("[%d] e = %s\n", i, e.toChars());

        // Create scope for '$' variable for this dimension
        auto sym = new ArrayScopeSymbol(sc, ae);
        sym.parent = sc.scopesym;
        sc = sc.push(sym);
        ae.lengthVar = null; // Create it only if required
        ae.currentDimension = i; // Dimension for $, if required

        e = e.expressionSemantic(sc);
        e = resolveProperties(sc, e);

        if (ae.lengthVar && sc.func)
        {
            // If $ was used, declare it now
            Expression de = new DeclarationExp(ae.loc, ae.lengthVar);
            de = de.expressionSemantic(sc);
            *pe0 = Expression.combine(*pe0, de);
        }
        sc = sc.pop();

        if (e.op == TOK.interval)
        {
            IntervalExp ie = cast(IntervalExp)e;

            auto tiargs = new Objects();
            Expression edim = new IntegerExp(ae.loc, i, Type.tsize_t);
            edim = edim.expressionSemantic(sc);
            tiargs.push(edim);

            auto fargs = new Expressions(2);
            (*fargs)[0] = ie.lwr;
            (*fargs)[1] = ie.upr;

            uint xerrors = global.startGagging();
            sc = sc.push();
            FuncDeclaration fslice = resolveFuncCall(ae.loc, sc, slice, tiargs, ae.e1.type, fargs, FuncResolveFlag.quiet);
            sc = sc.pop();
            global.endGagging(xerrors);
            if (!fslice)
                goto Lfallback;

            e = new DotTemplateInstanceExp(ae.loc, ae.e1, slice.ident, tiargs);
            e = new CallExp(ae.loc, e, fargs);
            e = e.expressionSemantic(sc);
        }

        if (!e.type)
        {
            ae.error("`%s` has no value", e.toChars());
            e = new ErrorExp();
        }
        if (e.op == TOK.error)
            return e;

        (*ae.arguments)[i] = e;
    }
    return ae;
}

/**************************************
 * Runs semantic on se.lwr and se.upr. Declares a temporary variable
 * if '$' was used.
 * Returns:
 *      ae, or ErrorExp if errors occurred
 */
Expression resolveOpDollar(Scope* sc, ArrayExp ae, IntervalExp ie, Expression* pe0)
{
    //assert(!ae.lengthVar);
    if (!ie)
        return ae;

    VarDeclaration lengthVar = ae.lengthVar;
    bool errors = false;

    // create scope for '$'
    auto sym = new ArrayScopeSymbol(sc, ae);
    sym.parent = sc.scopesym;
    sc = sc.push(sym);

    Expression sem(Expression e)
    {
        e = e.expressionSemantic(sc);
        e = resolveProperties(sc, e);
        if (!e.type)
        {
            ae.error("`%s` has no value", e.toChars());
            errors = true;
        }
        return e;
    }

    ie.lwr = sem(ie.lwr);
    ie.upr = sem(ie.upr);

    if (lengthVar != ae.lengthVar && sc.func)
    {
        // If $ was used, declare it now
        Expression de = new DeclarationExp(ae.loc, ae.lengthVar);
        de = de.expressionSemantic(sc);
        *pe0 = Expression.combine(*pe0, de);
    }

    sc = sc.pop();

    return errors ? new ErrorExp() : ae;
}

/******************************
 * Perform semantic() on an array of Expressions.
 */
bool arrayExpressionSemantic(Expressions* exps, Scope* sc, bool preserveErrors = false)
{
    bool err = false;
    if (exps)
    {
        foreach (ref e; *exps)
        {
            if (e)
            {
                auto e2 = e.expressionSemantic(sc);
                if (e2.op == TOK.error)
                    err = true;
                if (preserveErrors || e2.op != TOK.error)
                    e = e2;
            }
        }
    }
    return err;
}

/******************************
 * Check the tail CallExp is really property function call.
 * Bugs:
 * This doesn't appear to do anything.
 */
private bool checkPropertyCall(Expression e)
{
    e = lastComma(e);

    if (e.op == TOK.call)
    {
        CallExp ce = cast(CallExp)e;
        TypeFunction tf;
        if (ce.f)
        {
            tf = cast(TypeFunction)ce.f.type;
            /* If a forward reference to ce.f, try to resolve it
             */
            if (!tf.deco && ce.f.semanticRun < PASS.semanticdone)
            {
                ce.f.dsymbolSemantic(null);
                tf = cast(TypeFunction)ce.f.type;
            }
        }
        else if (ce.e1.type.ty == Tfunction)
            tf = cast(TypeFunction)ce.e1.type;
        else if (ce.e1.type.ty == Tdelegate)
            tf = cast(TypeFunction)ce.e1.type.nextOf();
        else if (ce.e1.type.ty == Tpointer && ce.e1.type.nextOf().ty == Tfunction)
            tf = cast(TypeFunction)ce.e1.type.nextOf();
        else
            assert(0);
    }
    return false;
}

/******************************
 * Find symbol in accordance with the UFCS name look up rule
 */
private Expression searchUFCS(Scope* sc, UnaExp ue, Identifier ident)
{
    //printf("searchUFCS(ident = %s)\n", ident.toChars());
    Loc loc = ue.loc;

    // TODO: merge with Scope.search.searchScopes()
    Dsymbol searchScopes(int flags)
    {
        Dsymbol s = null;
        for (Scope* scx = sc; scx; scx = scx.enclosing)
        {
            if (!scx.scopesym)
                continue;
            if (scx.scopesym.isModule())
                flags |= SearchUnqualifiedModule;    // tell Module.search() that SearchLocalsOnly is to be obeyed
            s = scx.scopesym.search(loc, ident, flags);
            if (s)
            {
                // overload set contains only module scope symbols.
                if (s.isOverloadSet())
                    break;
                // selective/renamed imports also be picked up
                if (AliasDeclaration ad = s.isAliasDeclaration())
                {
                    if (ad._import)
                        break;
                }
                // See only module scope symbols for UFCS target.
                Dsymbol p = s.toParent2();
                if (p && p.isModule())
                    break;
            }
            s = null;

            // Stop when we hit a module, but keep going if that is not just under the global scope
            if (scx.scopesym.isModule() && !(scx.enclosing && !scx.enclosing.enclosing))
                break;
        }
        return s;
    }

    int flags = 0;
    Dsymbol s;

    if (sc.flags & SCOPE.ignoresymbolvisibility)
        flags |= IgnoreSymbolVisibility;

    // First look in local scopes
    s = searchScopes(flags | SearchLocalsOnly);
    if (!s)
    {
        // Second look in imported modules
        s = searchScopes(flags | SearchImportsOnly);
    }

    if (!s)
        return ue.e1.type.Type.getProperty(sc, loc, ident, 0);

    FuncDeclaration f = s.isFuncDeclaration();
    if (f)
    {
        TemplateDeclaration td = getFuncTemplateDecl(f);
        if (td)
        {
            if (td.overroot)
                td = td.overroot;
            s = td;
        }
    }

    if (ue.op == TOK.dotTemplateInstance)
    {
        DotTemplateInstanceExp dti = cast(DotTemplateInstanceExp)ue;
        auto ti = new TemplateInstance(loc, s.ident, dti.ti.tiargs);
        if (!ti.updateTempDecl(sc, s))
            return new ErrorExp();
        return new ScopeExp(loc, ti);
    }
    else
    {
        //printf("-searchUFCS() %s\n", s.toChars());
        return new DsymbolExp(loc, s);
    }
}

/******************************
 * Pull out callable entity with UFCS.
 */
private Expression resolveUFCS(Scope* sc, CallExp ce)
{
    Loc loc = ce.loc;
    Expression eleft;
    Expression e;

    if (ce.e1.op == TOK.dotIdentifier)
    {
        DotIdExp die = cast(DotIdExp)ce.e1;
        Identifier ident = die.ident;

        Expression ex = die.semanticX(sc);
        if (ex != die)
        {
            ce.e1 = ex;
            return null;
        }
        eleft = die.e1;

        Type t = eleft.type.toBasetype();
        if (t.ty == Tarray || t.ty == Tsarray || t.ty == Tnull || (t.isTypeBasic() && t.ty != Tvoid))
        {
            /* Built-in types and arrays have no callable properties, so do shortcut.
             * It is necessary in: e.init()
             */
        }
        else if (t.ty == Taarray)
        {
            if (ident == Id.remove)
            {
                /* Transform:
                 *  aa.remove(arg) into delete aa[arg]
                 */
                if (!ce.arguments || ce.arguments.dim != 1)
                {
                    ce.error("expected key as argument to `aa.remove()`");
                    return new ErrorExp();
                }
                if (!eleft.type.isMutable())
                {
                    ce.error("cannot remove key from `%s` associative array `%s`", MODtoChars(t.mod), eleft.toChars());
                    return new ErrorExp();
                }
                Expression key = (*ce.arguments)[0];
                key = key.expressionSemantic(sc);
                key = resolveProperties(sc, key);

                TypeAArray taa = cast(TypeAArray)t;
                key = key.implicitCastTo(sc, taa.index);

                if (key.checkValue() || key.checkSharedAccess(sc))
                    return new ErrorExp();

                semanticTypeInfo(sc, taa.index);

                return new RemoveExp(loc, eleft, key);
            }
        }
        else
        {
            if (Expression ey = die.semanticY(sc, 1))
            {
                if (ey.op == TOK.error)
                    return ey;
                ce.e1 = ey;
                if (isDotOpDispatch(ey))
                {
                    uint errors = global.startGagging();
                    e = ce.syntaxCopy().expressionSemantic(sc);
                    if (!global.endGagging(errors))
                        return e;

                    // even opDispatch and UFCS must have valid arguments,
                    // so now that we've seen indication of a problem,
                    // check them for issues.
                    Expressions* originalArguments = Expression.arraySyntaxCopy(ce.arguments);

                    if (arrayExpressionSemantic(originalArguments, sc))
                        return new ErrorExp();

                    /* fall down to UFCS */
                }
                else
                    return null;
            }
        }

        /* https://issues.dlang.org/show_bug.cgi?id=13953
         *
         * If a struct has an alias this to an associative array
         * and remove is used on a struct instance, we have to
         * check first if there is a remove function that can be called
         * on the struct. If not we must check the alias this.
         *
         * struct A
         * {
         *      string[string] a;
         *      alias a this;
         * }
         *
         * void fun()
         * {
         *      A s;
         *      s.remove("foo");
         * }
         */
        const errors = global.startGagging();
        e = searchUFCS(sc, die, ident);
        // if there were any errors and the identifier was remove
        if (global.endGagging(errors))
        {
            if (ident == Id.remove)
            {
                // check alias this
                Expression alias_e = resolveAliasThis(sc, die.e1, 1);
                if (alias_e && alias_e != die.e1)
                {
                    die.e1 = alias_e;
                    CallExp ce2 = cast(CallExp)ce.syntaxCopy();
                    ce2.e1 = die;
                    e = cast(CallExp)ce2.trySemantic(sc);
                    if (e)
                        return e;
                }
            }
            // if alias this did not work out, print the initial errors
            searchUFCS(sc, die, ident);
        }
    }
    else if (ce.e1.op == TOK.dotTemplateInstance)
    {
        DotTemplateInstanceExp dti = cast(DotTemplateInstanceExp)ce.e1;
        if (Expression ey = dti.semanticY(sc, 1))
        {
            ce.e1 = ey;
            return null;
        }
        eleft = dti.e1;
        e = searchUFCS(sc, dti, dti.ti.name);
    }
    else
        return null;

    // Rewrite
    ce.e1 = e;
    if (!ce.arguments)
        ce.arguments = new Expressions();
    ce.arguments.shift(eleft);

    return null;
}

/******************************
 * Pull out property with UFCS.
 */
private Expression resolveUFCSProperties(Scope* sc, Expression e1, Expression e2 = null)
{
    Loc loc = e1.loc;
    Expression eleft;
    Expression e;

    if (e1.op == TOK.dotIdentifier)
    {
        DotIdExp die = cast(DotIdExp)e1;
        eleft = die.e1;
        e = searchUFCS(sc, die, die.ident);
    }
    else if (e1.op == TOK.dotTemplateInstance)
    {
        DotTemplateInstanceExp dti;
        dti = cast(DotTemplateInstanceExp)e1;
        eleft = dti.e1;
        e = searchUFCS(sc, dti, dti.ti.name);
    }
    else
        return null;

    if (e is null)
        return null;

    // Rewrite
    if (e2)
    {
        // run semantic without gagging
        e2 = e2.expressionSemantic(sc);

        /* f(e1) = e2
         */
        Expression ex = e.copy();
        auto a1 = new Expressions(1);
        (*a1)[0] = eleft;
        ex = new CallExp(loc, ex, a1);
        auto e1PassSemantic = ex.trySemantic(sc);

        /* f(e1, e2)
         */
        auto a2 = new Expressions(2);
        (*a2)[0] = eleft;
        (*a2)[1] = e2;
        e = new CallExp(loc, e, a2);
        e = e.trySemantic(sc);
        if (!e1PassSemantic && !e)
        {
            /* https://issues.dlang.org/show_bug.cgi?id=20448
             *
             * If both versions have failed to pass semantic,
             * f(e1) = e2 gets priority in error printing
             * because f might be a templated function that
             * failed to instantiate and we have to print
             * the instantiation errors.
             */
            return e1.expressionSemantic(sc);
        }
        else if (ex && !e)
        {
            checkPropertyCall(ex);
            ex = new AssignExp(loc, ex, e2);
            return ex.expressionSemantic(sc);
        }
        else
        {
            // strict setter prints errors if fails
            e = e.expressionSemantic(sc);
        }
        checkPropertyCall(e);
        return e;
    }
    else
    {
        /* f(e1)
         */
        auto arguments = new Expressions(1);
        (*arguments)[0] = eleft;
        e = new CallExp(loc, e, arguments);
        e = e.expressionSemantic(sc);
        checkPropertyCall(e);
        return e.expressionSemantic(sc);
    }
}

/******************************
 * If e1 is a property function (template), resolve it.
 */
Expression resolvePropertiesOnly(Scope* sc, Expression e1)
{
    //printf("e1 = %s %s\n", Token::toChars(e1.op), e1.toChars());

    Expression handleOverloadSet(OverloadSet os)
    {
        assert(os);
        foreach (s; os.a)
        {
            auto fd = s.isFuncDeclaration();
            auto td = s.isTemplateDeclaration();
            if (fd)
            {
                if ((cast(TypeFunction)fd.type).isproperty)
                    return resolveProperties(sc, e1);
            }
            else if (td && td.onemember && (fd = td.onemember.isFuncDeclaration()) !is null)
            {
                if ((cast(TypeFunction)fd.type).isproperty ||
                    (fd.storage_class2 & STC.property) ||
                    (td._scope.stc & STC.property))
                    return resolveProperties(sc, e1);
            }
        }
        return e1;
    }

    Expression handleTemplateDecl(TemplateDeclaration td)
    {
        assert(td);
        if (td.onemember)
        {
            if (auto fd = td.onemember.isFuncDeclaration())
            {
                if ((cast(TypeFunction)fd.type).isproperty ||
                    (fd.storage_class2 & STC.property) ||
                    (td._scope.stc & STC.property))
                    return resolveProperties(sc, e1);
            }
        }
        return e1;
    }

    Expression handleFuncDecl(FuncDeclaration fd)
    {
        assert(fd);
        if ((cast(TypeFunction)fd.type).isproperty)
            return resolveProperties(sc, e1);
        return e1;
    }

    if (auto de = e1.isDotExp())
    {
        if (auto os = de.e2.isOverExp())
            return handleOverloadSet(os.vars);
    }
    else if (auto oe = e1.isOverExp())
        return handleOverloadSet(oe.vars);
    else if (auto dti = e1.isDotTemplateInstanceExp())
    {
        if (dti.ti.tempdecl)
            if (auto td = dti.ti.tempdecl.isTemplateDeclaration())
                return handleTemplateDecl(td);
    }
    else if (auto dte = e1.isDotTemplateExp())
        return handleTemplateDecl(dte.td);
    else if (e1.op == TOK.scope_)
    {
        Dsymbol s = (cast(ScopeExp)e1).sds;
        TemplateInstance ti = s.isTemplateInstance();
        if (ti && !ti.semanticRun && ti.tempdecl)
            if (auto td = ti.tempdecl.isTemplateDeclaration())
                return handleTemplateDecl(td);
    }
    else if (e1.op == TOK.template_)
        return handleTemplateDecl((cast(TemplateExp)e1).td);
    else if (e1.op == TOK.dotVariable && e1.type.ty == Tfunction)
    {
        DotVarExp dve = cast(DotVarExp)e1;
        return handleFuncDecl(dve.var.isFuncDeclaration());
    }
    else if (e1.op == TOK.variable && e1.type && e1.type.ty == Tfunction && (sc.intypeof || !(cast(VarExp)e1).var.needThis()))
        return handleFuncDecl((cast(VarExp)e1).var.isFuncDeclaration());
    return e1;
}

/****************************************
 * Turn symbol `s` into the expression it represents.
 *
 * Params:
 *      s = symbol to resolve
 *      loc = location of use of `s`
 *      sc = context
 *      hasOverloads = applies if `s` represents a function.
 *          true means it's overloaded and will be resolved later,
 *          false means it's the exact function symbol.
 * Returns:
 *      `s` turned into an expression, `ErrorExp` if an error occurred
 */
Expression symbolToExp(Dsymbol s, const ref Loc loc, Scope *sc, bool hasOverloads)
{
    static if (LOGSEMANTIC)
    {
        printf("DsymbolExp::resolve(%s %s)\n", s.kind(), s.toChars());
    }

Lagain:
    Expression e;

    //printf("DsymbolExp:: %p '%s' is a symbol\n", this, toChars());
    //printf("s = '%s', s.kind = '%s'\n", s.toChars(), s.kind());
    Dsymbol olds = s;
    Declaration d = s.isDeclaration();
    if (d && (d.storage_class & STC.templateparameter))
    {
        s = s.toAlias();
    }
    else
    {
        if (!s.isFuncDeclaration()) // functions are checked after overloading
        {
            s.checkDeprecated(loc, sc);
            if (d)
                d.checkDisabled(loc, sc);
        }

        // https://issues.dlang.org/show_bug.cgi?id=12023
        // if 's' is a tuple variable, the tuple is returned.
        s = s.toAlias();

        //printf("s = '%s', s.kind = '%s', s.needThis() = %p\n", s.toChars(), s.kind(), s.needThis());
        if (s != olds && !s.isFuncDeclaration())
        {
            s.checkDeprecated(loc, sc);
            if (d)
                d.checkDisabled(loc, sc);
        }
    }

    if (auto em = s.isEnumMember())
    {
        return em.getVarExp(loc, sc);
    }
    if (auto v = s.isVarDeclaration())
    {
        //printf("Identifier '%s' is a variable, type '%s'\n", s.toChars(), v.type.toChars());
        if (sc.intypeof == 1 && !v.inuse)
            v.dsymbolSemantic(sc);
        if (!v.type ||                  // during variable type inference
            !v.type.deco && v.inuse)    // during variable type semantic
        {
            if (v.inuse)    // variable type depends on the variable itself
                error(loc, "circular reference to %s `%s`", v.kind(), v.toPrettyChars());
            else            // variable type cannot be determined
                error(loc, "forward reference to %s `%s`", v.kind(), v.toPrettyChars());
            return new ErrorExp();
        }
        if (v.type.ty == Terror)
            return new ErrorExp();

        if ((v.storage_class & STC.manifest) && v._init)
        {
            if (v.inuse)
            {
                error(loc, "circular initialization of %s `%s`", v.kind(), v.toPrettyChars());
                return new ErrorExp();
            }
            e = v.expandInitializer(loc);
            v.inuse++;
            e = e.expressionSemantic(sc);
            v.inuse--;
            return e;
        }

        // Change the ancestor lambdas to delegate before hasThis(sc) call.
        if (v.checkNestedReference(sc, loc))
            return new ErrorExp();

        if (v.needThis() && hasThis(sc))
            e = new DotVarExp(loc, new ThisExp(loc), v);
        else
            e = new VarExp(loc, v);
        e = e.expressionSemantic(sc);
        return e;
    }
    if (auto fld = s.isFuncLiteralDeclaration())
    {
        //printf("'%s' is a function literal\n", fld.toChars());
        e = new FuncExp(loc, fld);
        return e.expressionSemantic(sc);
    }
    if (auto f = s.isFuncDeclaration())
    {
        f = f.toAliasFunc();
        if (!f.functionSemantic())
            return new ErrorExp();

        if (!hasOverloads && f.checkForwardRef(loc))
            return new ErrorExp();

        auto fd = s.isFuncDeclaration();
        fd.type = f.type;
        return new VarExp(loc, fd, hasOverloads);
    }
    if (OverDeclaration od = s.isOverDeclaration())
    {
        e = new VarExp(loc, od, true);
        e.type = Type.tvoid;
        return e;
    }
    if (OverloadSet o = s.isOverloadSet())
    {
        //printf("'%s' is an overload set\n", o.toChars());
        return new OverExp(loc, o);
    }

    if (Import imp = s.isImport())
    {
        if (!imp.pkg)
        {
            .error(loc, "forward reference of import `%s`", imp.toChars());
            return new ErrorExp();
        }
        auto ie = new ScopeExp(loc, imp.pkg);
        return ie.expressionSemantic(sc);
    }
    if (Package pkg = s.isPackage())
    {
        auto ie = new ScopeExp(loc, pkg);
        return ie.expressionSemantic(sc);
    }
    if (Module mod = s.isModule())
    {
        auto ie = new ScopeExp(loc, mod);
        return ie.expressionSemantic(sc);
    }
    if (Nspace ns = s.isNspace())
    {
        auto ie = new ScopeExp(loc, ns);
        return ie.expressionSemantic(sc);
    }

    if (Type t = s.getType())
    {
        return (new TypeExp(loc, t)).expressionSemantic(sc);
    }

    if (TupleDeclaration tup = s.isTupleDeclaration())
    {
        if (tup.needThis() && hasThis(sc))
            e = new DotVarExp(loc, new ThisExp(loc), tup);
        else
            e = new TupleExp(loc, tup);
        e = e.expressionSemantic(sc);
        return e;
    }

    if (TemplateInstance ti = s.isTemplateInstance())
    {
        ti.dsymbolSemantic(sc);
        if (!ti.inst || ti.errors)
            return new ErrorExp();
        s = ti.toAlias();
        if (!s.isTemplateInstance())
            goto Lagain;
        e = new ScopeExp(loc, ti);
        e = e.expressionSemantic(sc);
        return e;
    }
    if (TemplateDeclaration td = s.isTemplateDeclaration())
    {
        Dsymbol p = td.toParentLocal();
        FuncDeclaration fdthis = hasThis(sc);
        AggregateDeclaration ad = p ? p.isAggregateDeclaration() : null;
        if (fdthis && ad && fdthis.isMemberLocal() == ad && (td._scope.stc & STC.static_) == 0)
        {
            e = new DotTemplateExp(loc, new ThisExp(loc), td);
        }
        else
            e = new TemplateExp(loc, td);
        e = e.expressionSemantic(sc);
        return e;
    }

    .error(loc, "%s `%s` is not a variable", s.kind(), s.toChars());
    return new ErrorExp();
}

/*************************************************************
 * Given var, get the
 * right `this` pointer if var is in an outer class, but our
 * existing `this` pointer is in an inner class.
 * Params:
 *      loc = location to use for error messages
 *      sc = context
 *      ad = struct or class we need the correct `this` for
 *      e1 = existing `this`
 *      var = the specific member of ad we're accessing
 *      flag = if true, return `null` instead of throwing an error
 * Returns:
 *      Expression representing the `this` for the var
 */
private Expression getRightThis(const ref Loc loc, Scope* sc, AggregateDeclaration ad, Expression e1, Dsymbol var, int flag = 0)
{
    //printf("\ngetRightThis(e1 = %s, ad = %s, var = %s)\n", e1.toChars(), ad.toChars(), var.toChars());
L1:
    Type t = e1.type.toBasetype();
    //printf("e1.type = %s, var.type = %s\n", e1.type.toChars(), var.type.toChars());

    if (e1.op == TOK.objcClassReference)
    {
        // We already have an Objective-C class reference, just use that as 'this'.
        return e1;
    }
    else if (ad && ad.isClassDeclaration && ad.isClassDeclaration.classKind == ClassKind.objc &&
             var.isFuncDeclaration && var.isFuncDeclaration.isStatic &&
             var.isFuncDeclaration.selector)
    {
        return new ObjcClassReferenceExp(e1.loc, cast(ClassDeclaration) ad);
    }

    /* Access of a member which is a template parameter in dual-scope scenario
     * class A { inc(alias m)() { ++m; } } // `m` needs `this` of `B`
     * class B {int m; inc() { new A().inc!m(); } }
     */
    if (e1.op == TOK.this_)
    {
        FuncDeclaration f = hasThis(sc);
        if (f && f.isThis2)
        {
            if (f.followInstantiationContext(ad))
            {
                e1 = new VarExp(loc, f.vthis);
                e1 = new PtrExp(loc, e1);
                e1 = new IndexExp(loc, e1, IntegerExp.literal!1);
                e1 = getThisSkipNestedFuncs(loc, sc, f.toParent2(), ad, e1, t, var);
                if (e1.op == TOK.error)
                    return e1;
                goto L1;
            }
        }
    }

    /* If e1 is not the 'this' pointer for ad
     */
    if (ad &&
        !(t.ty == Tpointer && t.nextOf().ty == Tstruct && (cast(TypeStruct)t.nextOf()).sym == ad) &&
        !(t.ty == Tstruct && (cast(TypeStruct)t).sym == ad))
    {
        ClassDeclaration cd = ad.isClassDeclaration();
        ClassDeclaration tcd = t.isClassHandle();

        /* e1 is the right this if ad is a base class of e1
         */
        if (!cd || !tcd || !(tcd == cd || cd.isBaseOf(tcd, null)))
        {
            /* Only classes can be inner classes with an 'outer'
             * member pointing to the enclosing class instance
             */
            if (tcd && tcd.isNested())
            {
                /* e1 is the 'this' pointer for an inner class: tcd.
                 * Rewrite it as the 'this' pointer for the outer class.
                 */
                auto vthis = tcd.followInstantiationContext(ad) ? tcd.vthis2 : tcd.vthis;
                e1 = new DotVarExp(loc, e1, vthis);
                e1.type = vthis.type;
                e1.type = e1.type.addMod(t.mod);
                // Do not call ensureStaticLinkTo()
                //e1 = e1.semantic(sc);

                // Skip up over nested functions, and get the enclosing
                // class type.
                e1 = getThisSkipNestedFuncs(loc, sc, tcd.toParentP(ad), ad, e1, t, var);
                if (e1.op == TOK.error)
                    return e1;
                goto L1;
            }

            /* Can't find a path from e1 to ad
             */
            if (flag)
                return null;
            e1.error("`this` for `%s` needs to be type `%s` not type `%s`", var.toChars(), ad.toChars(), t.toChars());
            return new ErrorExp();
        }
    }
    return e1;
}

/***************************************
 * Pull out any properties.
 */
private Expression resolvePropertiesX(Scope* sc, Expression e1, Expression e2 = null)
{
    //printf("resolvePropertiesX, e1 = %s %s, e2 = %s\n", Token.toChars(e1.op), e1.toChars(), e2 ? e2.toChars() : null);
    Loc loc = e1.loc;

    OverloadSet os;
    Dsymbol s;
    Objects* tiargs;
    Type tthis;
    if (e1.op == TOK.dot)
    {
        DotExp de = cast(DotExp)e1;
        if (de.e2.op == TOK.overloadSet)
        {
            tiargs = null;
            tthis = de.e1.type;
            os = (cast(OverExp)de.e2).vars;
            goto Los;
        }
    }
    else if (e1.op == TOK.overloadSet)
    {
        tiargs = null;
        tthis = null;
        os = (cast(OverExp)e1).vars;
    Los:
        assert(os);
        FuncDeclaration fd = null;
        if (e2)
        {
            e2 = e2.expressionSemantic(sc);
            if (e2.op == TOK.error)
                return new ErrorExp();
            e2 = resolveProperties(sc, e2);

            Expressions a;
            a.push(e2);

            for (size_t i = 0; i < os.a.dim; i++)
            {
                if (FuncDeclaration f = resolveFuncCall(loc, sc, os.a[i], tiargs, tthis, &a, FuncResolveFlag.quiet))
                {
                    if (f.errors)
                        return new ErrorExp();
                    fd = f;
                    assert(fd.type.ty == Tfunction);
                }
            }
            if (fd)
            {
                Expression e = new CallExp(loc, e1, e2);
                return e.expressionSemantic(sc);
            }
        }
        {
            for (size_t i = 0; i < os.a.dim; i++)
            {
                if (FuncDeclaration f = resolveFuncCall(loc, sc, os.a[i], tiargs, tthis, null, FuncResolveFlag.quiet))
                {
                    if (f.errors)
                        return new ErrorExp();
                    fd = f;
                    assert(fd.type.ty == Tfunction);
                    TypeFunction tf = cast(TypeFunction)fd.type;
                    if (!tf.isref && e2)
                    {
                        error(loc, "%s is not an lvalue", e1.toChars());
                        return new ErrorExp();
                    }
                }
            }
            if (fd)
            {
                Expression e = new CallExp(loc, e1);
                if (e2)
                    e = new AssignExp(loc, e, e2);
                return e.expressionSemantic(sc);
            }
        }
        if (e2)
            goto Leprop;
    }
    else if (e1.op == TOK.dotTemplateInstance)
    {
        DotTemplateInstanceExp dti = cast(DotTemplateInstanceExp)e1;
        if (!dti.findTempDecl(sc))
            goto Leprop;
        if (!dti.ti.semanticTiargs(sc))
            goto Leprop;
        tiargs = dti.ti.tiargs;
        tthis = dti.e1.type;
        if ((os = dti.ti.tempdecl.isOverloadSet()) !is null)
            goto Los;
        if ((s = dti.ti.tempdecl) !is null)
            goto Lfd;
    }
    else if (e1.op == TOK.dotTemplateDeclaration)
    {
        DotTemplateExp dte = cast(DotTemplateExp)e1;
        s = dte.td;
        tiargs = null;
        tthis = dte.e1.type;
        goto Lfd;
    }
    else if (e1.op == TOK.scope_)
    {
        s = (cast(ScopeExp)e1).sds;
        TemplateInstance ti = s.isTemplateInstance();
        if (ti && !ti.semanticRun && ti.tempdecl)
        {
            //assert(ti.needsTypeInference(sc));
            if (!ti.semanticTiargs(sc))
                goto Leprop;
            tiargs = ti.tiargs;
            tthis = null;
            if ((os = ti.tempdecl.isOverloadSet()) !is null)
                goto Los;
            if ((s = ti.tempdecl) !is null)
                goto Lfd;
        }
    }
    else if (e1.op == TOK.template_)
    {
        s = (cast(TemplateExp)e1).td;
        tiargs = null;
        tthis = null;
        goto Lfd;
    }
    else if (e1.op == TOK.dotVariable && e1.type && e1.type.toBasetype().ty == Tfunction)
    {
        DotVarExp dve = cast(DotVarExp)e1;
        s = dve.var.isFuncDeclaration();
        tiargs = null;
        tthis = dve.e1.type;
        goto Lfd;
    }
    else if (e1.op == TOK.variable && e1.type && e1.type.toBasetype().ty == Tfunction)
    {
        s = (cast(VarExp)e1).var.isFuncDeclaration();
        tiargs = null;
        tthis = null;
    Lfd:
        assert(s);
        if (e2)
        {
            e2 = e2.expressionSemantic(sc);
            if (e2.op == TOK.error)
                return new ErrorExp();
            e2 = resolveProperties(sc, e2);

            Expressions a;
            a.push(e2);

            FuncDeclaration fd = resolveFuncCall(loc, sc, s, tiargs, tthis, &a, FuncResolveFlag.quiet);
            if (fd && fd.type)
            {
                if (fd.errors)
                    return new ErrorExp();
                assert(fd.type.ty == Tfunction);
                Expression e = new CallExp(loc, e1, e2);
                return e.expressionSemantic(sc);
            }
        }
        {
            FuncDeclaration fd = resolveFuncCall(loc, sc, s, tiargs, tthis, null, FuncResolveFlag.quiet);
            if (fd && fd.type)
            {
                if (fd.errors)
                    return new ErrorExp();
                assert(fd.type.ty == Tfunction);
                TypeFunction tf = cast(TypeFunction)fd.type;
                if (!e2 || tf.isref)
                {
                    Expression e = new CallExp(loc, e1);
                    if (e2)
                        e = new AssignExp(loc, e, e2);
                    return e.expressionSemantic(sc);
                }
            }
        }
        if (FuncDeclaration fd = s.isFuncDeclaration())
        {
            // Keep better diagnostic message for invalid property usage of functions
            assert(fd.type.ty == Tfunction);
            Expression e = new CallExp(loc, e1, e2);
            return e.expressionSemantic(sc);
        }
        if (e2)
            goto Leprop;
    }
    if (e1.op == TOK.variable)
    {
        VarExp ve = cast(VarExp)e1;
        VarDeclaration v = ve.var.isVarDeclaration();
        if (v && ve.checkPurity(sc, v))
            return new ErrorExp();
    }
    if (e2)
        return null;

    if (e1.type && e1.op != TOK.type) // function type is not a property
    {
        /* Look for e1 being a lazy parameter; rewrite as delegate call
         * only if the symbol wasn't already treated as a delegate
         */
        auto ve = e1.isVarExp();
        if (ve && ve.var.storage_class & STC.lazy_ && !ve.delegateWasExtracted)
        {
                Expression e = new CallExp(loc, e1);
                return e.expressionSemantic(sc);
        }
        else if (e1.op == TOK.dotVariable)
        {
            // Check for reading overlapped pointer field in @safe code.
            if (checkUnsafeAccess(sc, e1, true, true))
                return new ErrorExp();
        }
        else if (e1.op == TOK.dot)
        {
            e1.error("expression has no value");
            return new ErrorExp();
        }
        else if (e1.op == TOK.call)
        {
            CallExp ce = cast(CallExp)e1;
            // Check for reading overlapped pointer field in @safe code.
            if (checkUnsafeAccess(sc, ce.e1, true, true))
                return new ErrorExp();
        }
    }

    if (!e1.type)
    {
        error(loc, "cannot resolve type for %s", e1.toChars());
        e1 = new ErrorExp();
    }
    return e1;

Leprop:
    error(loc, "not a property %s", e1.toChars());
    return new ErrorExp();
}

extern (C++) Expression resolveProperties(Scope* sc, Expression e)
{
    //printf("resolveProperties(%s)\n", e.toChars());
    e = resolvePropertiesX(sc, e);
    if (e.checkRightThis(sc))
        return new ErrorExp();
    return e;
}

/****************************************
 * The common type is determined by applying ?: to each pair.
 * Output:
 *      exps[]  properties resolved, implicitly cast to common type, rewritten in place
 *      *pt     if pt is not NULL, set to the common type
 * Returns:
 *      true    a semantic error was detected
 */
private bool arrayExpressionToCommonType(Scope* sc, Expressions* exps, Type* pt)
{
    /* Still have a problem with:
     *  ubyte[][] = [ cast(ubyte[])"hello", [1]];
     * which works if the array literal is initialized top down with the ubyte[][]
     * type, but fails with this function doing bottom up typing.
     */

    //printf("arrayExpressionToCommonType()\n");
    scope IntegerExp integerexp = IntegerExp.literal!0;
    scope CondExp condexp = new CondExp(Loc.initial, integerexp, null, null);

    Type t0 = null;
    Expression e0 = null;
    size_t j0 = ~0;
    bool foundType;

    for (size_t i = 0; i < exps.dim; i++)
    {
        Expression e = (*exps)[i];
        if (!e)
            continue;

        e = resolveProperties(sc, e);
        if (!e.type)
        {
            e.error("`%s` has no value", e.toChars());
            t0 = Type.terror;
            continue;
        }
        if (e.op == TOK.type)
        {
            foundType = true; // do not break immediately, there might be more errors
            e.checkValue(); // report an error "type T has no value"
            t0 = Type.terror;
            continue;
        }
        if (e.type.ty == Tvoid)
        {
            // void expressions do not concur to the determination of the common
            // type.
            continue;
        }
        if (checkNonAssignmentArrayOp(e))
        {
            t0 = Type.terror;
            continue;
        }

        e = doCopyOrMove(sc, e);

        if (!foundType && t0 && !t0.equals(e.type))
        {
            /* This applies ?: to merge the types. It's backwards;
             * ?: should call this function to merge types.
             */
            condexp.type = null;
            condexp.e1 = e0;
            condexp.e2 = e;
            condexp.loc = e.loc;
            Expression ex = condexp.expressionSemantic(sc);
            if (ex.op == TOK.error)
                e = ex;
            else
            {
                (*exps)[j0] = condexp.e1;
                e = condexp.e2;
            }
        }
        j0 = i;
        e0 = e;
        t0 = e.type;
        if (e.op != TOK.error)
            (*exps)[i] = e;
    }

    if (!t0)
        t0 = Type.tvoid; // [] is typed as void[]
    else if (t0.ty != Terror)
    {
        for (size_t i = 0; i < exps.dim; i++)
        {
            Expression e = (*exps)[i];
            if (!e)
                continue;

            e = e.implicitCastTo(sc, t0);
            //assert(e.op != TOK.error);
            if (e.op == TOK.error)
            {
                /* https://issues.dlang.org/show_bug.cgi?id=13024
                 * a workaround for the bug in typeMerge -
                 * it should paint e1 and e2 by deduced common type,
                 * but doesn't in this particular case.
                 */
                t0 = Type.terror;
                break;
            }
            (*exps)[i] = e;
        }
    }
    if (pt)
        *pt = t0;

    return (t0 == Type.terror);
}

private Expression opAssignToOp(const ref Loc loc, TOK op, Expression e1, Expression e2)
{
    Expression e;
    switch (op)
    {
    case TOK.addAssign:
        e = new AddExp(loc, e1, e2);
        break;

    case TOK.minAssign:
        e = new MinExp(loc, e1, e2);
        break;

    case TOK.mulAssign:
        e = new MulExp(loc, e1, e2);
        break;

    case TOK.divAssign:
        e = new DivExp(loc, e1, e2);
        break;

    case TOK.modAssign:
        e = new ModExp(loc, e1, e2);
        break;

    case TOK.andAssign:
        e = new AndExp(loc, e1, e2);
        break;

    case TOK.orAssign:
        e = new OrExp(loc, e1, e2);
        break;

    case TOK.xorAssign:
        e = new XorExp(loc, e1, e2);
        break;

    case TOK.leftShiftAssign:
        e = new ShlExp(loc, e1, e2);
        break;

    case TOK.rightShiftAssign:
        e = new ShrExp(loc, e1, e2);
        break;

    case TOK.unsignedRightShiftAssign:
        e = new UshrExp(loc, e1, e2);
        break;

    default:
        assert(0);
    }
    return e;
}

/*********************
 * Rewrite:
 *    array.length op= e2
 * as:
 *    array.length = array.length op e2
 * or:
 *    auto tmp = &array;
 *    (*tmp).length = (*tmp).length op e2
 */
private Expression rewriteOpAssign(BinExp exp)
{
    Expression e;

    assert(exp.e1.op == TOK.arrayLength);
    ArrayLengthExp ale = cast(ArrayLengthExp)exp.e1;
    if (ale.e1.op == TOK.variable)
    {
        e = opAssignToOp(exp.loc, exp.op, ale, exp.e2);
        e = new AssignExp(exp.loc, ale.syntaxCopy(), e);
    }
    else
    {
        /*    auto tmp = &array;
         *    (*tmp).length = (*tmp).length op e2
         */
        auto tmp = copyToTemp(0, "__arraylength", new AddrExp(ale.loc, ale.e1));

        Expression e1 = new ArrayLengthExp(ale.loc, new PtrExp(ale.loc, new VarExp(ale.loc, tmp)));
        Expression elvalue = e1.syntaxCopy();
        e = opAssignToOp(exp.loc, exp.op, e1, exp.e2);
        e = new AssignExp(exp.loc, elvalue, e);
        e = new CommaExp(exp.loc, new DeclarationExp(ale.loc, tmp), e);
    }
    return e;
}

/****************************************
 * Preprocess arguments to function.
 * Input:
 *      reportErrors    whether or not to report errors here.  Some callers are not
 *                      checking actual function params, so they'll do their own error reporting
 * Output:
 *      exps[]  tuples expanded, properties resolved, rewritten in place
 * Returns:
 *      true    a semantic error occurred
 */
private bool preFunctionParameters(Scope* sc, Expressions* exps, const bool reportErrors = true)
{
    bool err = false;
    if (exps)
    {
        expandTuples(exps);

        for (size_t i = 0; i < exps.dim; i++)
        {
            Expression arg = (*exps)[i];
            arg = resolveProperties(sc, arg);
            if (arg.op == TOK.type)
            {
                // for static alias this: https://issues.dlang.org/show_bug.cgi?id=17684
                arg = resolveAliasThis(sc, arg);

                if (arg.op == TOK.type)
                {
                    if (reportErrors)
                    {
                        arg.error("cannot pass type `%s` as a function argument", arg.toChars());
                        arg = new ErrorExp();
                    }
                    err = true;
                }
            }
            else if (arg.type.toBasetype().ty == Tfunction)
            {
                if (reportErrors)
                {
                    arg.error("cannot pass function `%s` as a function argument", arg.toChars());
                    arg = new ErrorExp();
                }
                err = true;
            }
            else if (checkNonAssignmentArrayOp(arg))
            {
                arg = new ErrorExp();
                err = true;
            }
            (*exps)[i] = arg;
        }
    }
    return err;
}

/********************************************
 * Issue an error if default construction is disabled for type t.
 * Default construction is required for arrays and 'out' parameters.
 * Returns:
 *      true    an error was issued
 */
private bool checkDefCtor(Loc loc, Type t)
{
    t = t.baseElemOf();
    if (t.ty == Tstruct)
    {
        StructDeclaration sd = (cast(TypeStruct)t).sym;
        if (sd.noDefaultCtor)
        {
            sd.error(loc, "default construction is disabled");
            return true;
        }
    }
    return false;
}

/****************************************
 * Now that we know the exact type of the function we're calling,
 * the arguments[] need to be adjusted:
 *      1. implicitly convert argument to the corresponding parameter type
 *      2. add default arguments for any missing arguments
 *      3. do default promotions on arguments corresponding to ...
 *      4. add hidden _arguments[] argument
 *      5. call copy constructor for struct value arguments
 * Params:
 *      loc       = location of function call
 *      sc        = context
 *      tf        = type of the function
 *      ethis     = `this` argument, `null` if none or not known
 *      tthis     = type of `this` argument, `null` if no `this` argument
 *      arguments = array of actual arguments to function call
 *      fd        = the function being called, `null` if called indirectly
 *      prettype  = set to return type of function
 *      peprefix  = set to expression to execute before `arguments[]` are evaluated, `null` if none
 * Returns:
 *      true    errors happened
 */
private bool functionParameters(const ref Loc loc, Scope* sc,
    TypeFunction tf, Expression ethis, Type tthis, Expressions* arguments, FuncDeclaration fd,
    Type* prettype, Expression* peprefix)
{
    //printf("functionParameters() %s\n", fd ? fd.toChars() : "");
    assert(arguments);
    assert(fd || tf.next);
    size_t nargs = arguments ? arguments.dim : 0;
    const size_t nparams = tf.parameterList.length;
    const olderrors = global.errors;
    bool err = false;
    *prettype = Type.terror;
    Expression eprefix = null;
    *peprefix = null;

    if (nargs > nparams && tf.parameterList.varargs == VarArg.none)
    {
        error(loc, "expected %llu arguments, not %llu for non-variadic function type `%s`", cast(ulong)nparams, cast(ulong)nargs, tf.toChars());
        return true;
    }

    // If inferring return type, and semantic3() needs to be run if not already run
    if (!tf.next && fd.inferRetType)
    {
        fd.functionSemantic();
    }
    else if (fd && fd.parent)
    {
        TemplateInstance ti = fd.parent.isTemplateInstance();
        if (ti && ti.tempdecl)
        {
            fd.functionSemantic3();
        }
    }
    const isCtorCall = fd && fd.needThis() && fd.isCtorDeclaration();

    const size_t n = (nargs > nparams) ? nargs : nparams; // n = max(nargs, nparams)

    /* If the function return type has wildcards in it, we'll need to figure out the actual type
     * based on the actual argument types.
     * Start with the `this` argument, later on merge into wildmatch the mod bits of the rest
     * of the arguments.
     */
    MOD wildmatch = (tthis && !isCtorCall) ? tthis.Type.deduceWild(tf, false) : 0;

    bool done = false;
    foreach (const i; 0 .. n)
    {
        Expression arg = (i < nargs) ? (*arguments)[i] : null;

        if (i < nparams)
        {
            bool errorArgs()
            {
                error(loc, "expected %llu function arguments, not %llu", cast(ulong)nparams, cast(ulong)nargs);
                return true;
            }

            Parameter p = tf.parameterList[i];
            const bool isRef = (p.storageClass & (STC.ref_ | STC.out_)) != 0;

            if (!arg)
            {
                if (!p.defaultArg)
                {
                    if (tf.parameterList.varargs == VarArg.typesafe && i + 1 == nparams)
                        goto L2;
                    return errorArgs();
                }
                arg = p.defaultArg;
                arg = inlineCopy(arg, sc);
                // __FILE__, __LINE__, __MODULE__, __FUNCTION__, and __PRETTY_FUNCTION__
                arg = arg.resolveLoc(loc, sc);
                arguments.push(arg);
                nargs++;
            }
            else
            {
                if (arg.op == TOK.default_)
                {
                    arg = arg.resolveLoc(loc, sc);
                    (*arguments)[i] = arg;
                }
            }


            if (isRef && !p.type.isConst && !p.type.isImmutable
                && (p.storageClass & STC.const_) != STC.const_
                && (p.storageClass & STC.immutable_) != STC.immutable_
                && checkIfIsStructLiteralDotExpr(arg))
                    break;

            if (tf.parameterList.varargs == VarArg.typesafe && i + 1 == nparams) // https://dlang.org/spec/function.html#variadic
            {
                //printf("\t\tvarargs == 2, p.type = '%s'\n", p.type.toChars());
                {
                    MATCH m;
                    if ((m = arg.implicitConvTo(p.type)) > MATCH.nomatch)
                    {
                        if (p.type.nextOf() && arg.implicitConvTo(p.type.nextOf()) >= m)
                            goto L2;
                        else if (nargs != nparams)
                            return errorArgs();
                        goto L1;
                    }
                }
            L2:
                Type tb = p.type.toBasetype();
                switch (tb.ty)
                {
                case Tsarray:
                case Tarray:
                    {
                        /* Create a static array variable v of type arg.type:
                         *  T[dim] __arrayArg = [ arguments[i], ..., arguments[nargs-1] ];
                         *
                         * The array literal in the initializer of the hidden variable
                         * is now optimized.
                         * https://issues.dlang.org/show_bug.cgi?id=2356
                         */
                        Type tbn = (cast(TypeArray)tb).next;    // array element type
                        Type tret = p.isLazyArray();

                        auto elements = new Expressions(nargs - i);
                        foreach (u; 0 .. elements.dim)
                        {
                            Expression a = (*arguments)[i + u];
                            if (tret && a.implicitConvTo(tret))
                            {
                                // p is a lazy array of delegates, tret is return type of the delegates
                                a = a.implicitCastTo(sc, tret)
                                     .optimize(WANTvalue)
                                     .toDelegate(tret, sc);
                            }
                            else
                                a = a.implicitCastTo(sc, tbn);
                            a = a.addDtorHook(sc);
                            (*elements)[u] = a;
                        }
                        // https://issues.dlang.org/show_bug.cgi?id=14395
                        // Convert to a static array literal, or its slice.
                        arg = new ArrayLiteralExp(loc, tbn.sarrayOf(nargs - i), elements);
                        if (tb.ty == Tarray)
                        {
                            arg = new SliceExp(loc, arg, null, null);
                            arg.type = p.type;
                        }
                        break;
                    }
                case Tclass:
                    {
                        /* Set arg to be:
                         *      new Tclass(arg0, arg1, ..., argn)
                         */
                        auto args = new Expressions(nargs - i);
                        foreach (u; i .. nargs)
                            (*args)[u - i] = (*arguments)[u];
                        arg = new NewExp(loc, null, null, p.type, args);
                        break;
                    }
                default:
                    if (!arg)
                    {
                        error(loc, "not enough arguments");
                        return true;
                    }
                    break;
                }
                arg = arg.expressionSemantic(sc);
                //printf("\targ = '%s'\n", arg.toChars());
                arguments.setDim(i + 1);
                (*arguments)[i] = arg;
                nargs = i + 1;
                done = true;
            }

        L1:
            if (!(p.storageClass & STC.lazy_ && p.type.ty == Tvoid))
            {

                if (ubyte wm = arg.type.deduceWild(p.type, isRef))
                {
                    wildmatch = wildmatch ? MODmerge(wildmatch, wm) : wm;
                    //printf("[%d] p = %s, a = %s, wm = %d, wildmatch = %d\n", i, p.type.toChars(), arg.type.toChars(), wm, wildmatch);
                }
            }
        }
        if (done)
            break;
    }
    if ((wildmatch == MODFlags.mutable || wildmatch == MODFlags.immutable_) &&
        tf.next && tf.next.hasWild() &&
        (tf.isref || !tf.next.implicitConvTo(tf.next.immutableOf())))
    {
        bool errorInout(MOD wildmatch)
        {
            const(char)* s = wildmatch == MODFlags.mutable ? "mutable" : MODtoChars(wildmatch);
            error(loc, "modify `inout` to `%s` is not allowed inside `inout` function", s);
            return true;
        }

        if (fd)
        {
            /* If the called function may return the reference to
             * outer inout data, it should be rejected.
             *
             * void foo(ref inout(int) x) {
             *   ref inout(int) bar(inout(int)) { return x; }
             *   struct S {
             *      ref inout(int) bar() inout { return x; }
             *      ref inout(int) baz(alias a)() inout { return x; }
             *   }
             *   bar(int.init) = 1;  // bad!
             *   S().bar() = 1;      // bad!
             * }
             * void test() {
             *   int a;
             *   auto s = foo(a);
             *   s.baz!a() = 1;      // bad!
             * }
             *
             */
            bool checkEnclosingWild(Dsymbol s)
            {
                bool checkWild(Dsymbol s)
                {
                    if (!s)
                        return false;
                    if (auto ad = s.isAggregateDeclaration())
                    {
                        if (ad.isNested())
                            return checkEnclosingWild(s);
                    }
                    else if (auto ff = s.isFuncDeclaration())
                    {
                        if ((cast(TypeFunction)ff.type).iswild)
                            return errorInout(wildmatch);

                        if (ff.isNested() || ff.isThis())
                            return checkEnclosingWild(s);
                    }
                    return false;
                }

                Dsymbol ctx0 = s.toParent2();
                Dsymbol ctx1 = s.toParentLocal();
                if (checkWild(ctx0))
                    return true;
                if (ctx0 != ctx1)
                    return checkWild(ctx1);
                return false;
            }
            if ((fd.isThis() || fd.isNested()) && checkEnclosingWild(fd))
                return true;
        }
        else if (tf.isWild())
            return errorInout(wildmatch);
    }

    Expression firstArg = ((tf.next && tf.next.ty == Tvoid || isCtorCall) &&
                           tthis &&
                           tthis.isMutable() && tthis.toBasetype().ty == Tstruct &&
                           tthis.hasPointers())
                          ? ethis : null;

    assert(nargs >= nparams);
    foreach (const i, arg; (*arguments)[0 .. nargs])
    {
        assert(arg);
        if (i < nparams)
        {
            Parameter p = tf.parameterList[i];
            Type targ = arg.type;               // keep original type for isCopyable() because alias this
                                                // resolution may hide an uncopyable type

            if (!(p.storageClass & STC.lazy_ && p.type.ty == Tvoid))
            {
                Type tprm = p.type.hasWild()
                    ? p.type.substWildTo(wildmatch)
                    : p.type;

                const hasCopyCtor = (arg.type.ty == Tstruct) && (cast(TypeStruct)arg.type).sym.hasCopyCtor;
                const typesMatch = arg.type.mutableOf().unSharedOf().equals(tprm.mutableOf().unSharedOf());
                if (!((hasCopyCtor && typesMatch) || tprm.equals(arg.type)))
                {
                    //printf("arg.type = %s, p.type = %s\n", arg.type.toChars(), p.type.toChars());
                    arg = arg.implicitCastTo(sc, tprm);
                    arg = arg.optimize(WANTvalue, (p.storageClass & (STC.ref_ | STC.out_)) != 0);
                }
            }
            if (p.storageClass & STC.ref_)
            {
                if (global.params.rvalueRefParam &&
                    !arg.isLvalue() &&
                    targ.isCopyable())
                {   /* allow rvalues to be passed to ref parameters by copying
                     * them to a temp, then pass the temp as the argument
                     */
                    auto v = copyToTemp(0, "__rvalue", arg);
                    Expression ev = new DeclarationExp(arg.loc, v);
                    ev = new CommaExp(arg.loc, ev, new VarExp(arg.loc, v));
                    arg = ev.expressionSemantic(sc);
                }
                arg = arg.toLvalue(sc, arg);

                // Look for mutable misaligned pointer, etc., in @safe mode
                err |= checkUnsafeAccess(sc, arg, false, true);
            }
            else if (p.storageClass & STC.out_)
            {
                Type t = arg.type;
                if (!t.isMutable() || !t.isAssignable()) // check blit assignable
                {
                    arg.error("cannot modify struct `%s` with immutable members", arg.toChars());
                    err = true;
                }
                else
                {
                    // Look for misaligned pointer, etc., in @safe mode
                    err |= checkUnsafeAccess(sc, arg, false, true);
                    err |= checkDefCtor(arg.loc, t); // t must be default constructible
                }
                arg = arg.toLvalue(sc, arg);
            }
            else if (p.storageClass & STC.lazy_)
            {
                // Convert lazy argument to a delegate
                auto t = (p.type.ty == Tvoid) ? p.type : arg.type;
                arg = toDelegate(arg, t, sc);
            }
            //printf("arg: %s\n", arg.toChars());
            //printf("type: %s\n", arg.type.toChars());
            //printf("param: %s\n", p.toChars());

            if (firstArg && p.storageClass & STC.return_)
            {
                /* Argument value can be assigned to firstArg.
                 * Check arg to see if it matters.
                 */
                if (global.params.vsafe)
                    err |= checkParamArgumentReturn(sc, firstArg, arg, false);
            }
            else if (tf.parameterEscapes(tthis, p))
            {
                /* Argument value can escape from the called function.
                 * Check arg to see if it matters.
                 */
                if (global.params.vsafe)
                    err |= checkParamArgumentEscape(sc, fd, p, arg, false, false);
            }
            else
            {
                /* Argument value cannot escape from the called function.
                 */
                Expression a = arg;
                if (a.op == TOK.cast_)
                    a = (cast(CastExp)a).e1;

                ArrayLiteralExp ale;
                if (p.type.toBasetype().ty == Tarray && !(p.storageClass & STC.return_) &&
                    (ale = a.isArrayLiteralExp()) !is null)
                {
                    // allocate the array literal as temporary static array on the stack
                    ale.type = ale.type.nextOf().sarrayOf(ale.elements ? ale.elements.length : 0);
                    auto tmp = copyToTemp(0, "__arrayliteral_on_stack", ale);
                    auto declareTmp = new DeclarationExp(ale.loc, tmp);
                    auto castToSlice = new CastExp(ale.loc, new VarExp(ale.loc, tmp), p.type);
                    arg = CommaExp.combine(declareTmp, castToSlice);
                    arg = arg.expressionSemantic(sc);
                }
                else if (a.op == TOK.function_)
                {
                    /* Function literals can only appear once, so if this
                     * appearance was scoped, there cannot be any others.
                     */
                    FuncExp fe = cast(FuncExp)a;
                    fe.fd.tookAddressOf = 0;
                }
                else if (a.op == TOK.delegate_)
                {
                    /* For passing a delegate to a scoped parameter,
                     * this doesn't count as taking the address of it.
                     * We only worry about 'escaping' references to the function.
                     */
                    DelegateExp de = cast(DelegateExp)a;
                    if (de.e1.op == TOK.variable)
                    {
                        VarExp ve = cast(VarExp)de.e1;
                        FuncDeclaration f = ve.var.isFuncDeclaration();
                        if (f)
                        {
                            if (f.tookAddressOf)
                                --f.tookAddressOf;
                            //printf("--tookAddressOf = %d\n", f.tookAddressOf);
                        }
                    }
                }
            }
            if (!(p.storageClass & (STC.ref_ | STC.out_)))
                err |= arg.checkSharedAccess(sc);

            arg = arg.optimize(WANTvalue, (p.storageClass & (STC.ref_ | STC.out_)) != 0);

            /* Determine if this parameter is the "first reference" parameter through which
             * later "return" arguments can be stored.
             */
            if (i == 0 && !tthis && p.storageClass & (STC.ref_ | STC.out_) && p.type &&
                (tf.next && tf.next.ty == Tvoid || isCtorCall))
            {
                Type tb = p.type.baseElemOf();
                if (tb.isMutable() && tb.hasPointers())
                {
                    firstArg = arg;
                }
            }
        }
        else
        {
            // These will be the trailing ... arguments
            // If not D linkage, do promotions
            // IN_LLVM: don't do promotions on intrinsics
            if (tf.linkage != LINK.d && !(IN_LLVM && fd && DtoIsIntrinsic(fd)))
            {
                // Promote bytes, words, etc., to ints
                arg = integralPromotions(arg, sc);

                // Promote floats to doubles
                switch (arg.type.ty)
                {
                case Tfloat32:
                    arg = arg.castTo(sc, Type.tfloat64);
                    break;

                case Timaginary32:
                    arg = arg.castTo(sc, Type.timaginary64);
                    break;

                default:
                    break;
                }
                if (tf.parameterList.varargs == VarArg.variadic)
                {
                    const(char)* p = tf.linkage == LINK.c ? "extern(C)" : "extern(C++)";
                    if (arg.type.ty == Tarray)
                    {
                        arg.error("cannot pass dynamic arrays to `%s` vararg functions", p);
                        err = true;
                    }
                    if (arg.type.ty == Tsarray)
                    {
                        arg.error("cannot pass static arrays to `%s` vararg functions", p);
                        err = true;
                    }
                }
            }

            // Do not allow types that need destructors
            if (arg.type.needsDestruction())
            {
                arg.error("cannot pass types that need destruction as variadic arguments");
                err = true;
            }

            // Convert static arrays to dynamic arrays
            // BUG: I don't think this is right for D2
            Type tb = arg.type.toBasetype();
            if (tb.ty == Tsarray)
            {
                TypeSArray ts = cast(TypeSArray)tb;
                Type ta = ts.next.arrayOf();
                if (ts.size(arg.loc) == 0)
                    arg = new NullExp(arg.loc, ta);
                else
                    arg = arg.castTo(sc, ta);
            }
            if (tb.ty == Tstruct)
            {
                //arg = callCpCtor(sc, arg);
            }
            // Give error for overloaded function addresses
            if (arg.op == TOK.symbolOffset)
            {
                SymOffExp se = cast(SymOffExp)arg;
                if (se.hasOverloads && !se.var.isFuncDeclaration().isUnique())
                {
                    arg.error("function `%s` is overloaded", arg.toChars());
                    err = true;
                }
            }
            err |= arg.checkValue();
            err |= arg.checkSharedAccess(sc);
            arg = arg.optimize(WANTvalue);
        }
        (*arguments)[i] = arg;
    }

    /* If calling C scanf(), printf(), or any variants, check the format string against the arguments
     */
    const isVa_list = tf.parameterList.varargs == VarArg.none;
    if (fd && fd.flags & FUNCFLAG.printf)
    {
        if (auto se = (*arguments)[nparams - 1 - isVa_list].isStringExp())
        {
            checkPrintfFormat(se.loc, se.peekString(), (*arguments)[nparams .. nargs], isVa_list);
        }
    }
    else if (fd && fd.flags & FUNCFLAG.scanf)
    {
        if (auto se = (*arguments)[nparams - 1 - isVa_list].isStringExp())
        {
            checkScanfFormat(se.loc, se.peekString(), (*arguments)[nparams .. nargs], isVa_list);
        }
    }
    else
    {
        // TODO: not checking the "v" functions yet (for those, check format string only, not args)
    }

    /* Remaining problems:
     * 1. order of evaluation - some function push L-to-R, others R-to-L. Until we resolve what array assignment does (which is
     *    implemented by calling a function) we'll defer this for now.
     * 2. value structs (or static arrays of them) that need to be copy constructed
     * 3. value structs (or static arrays of them) that have destructors, and subsequent arguments that may throw before the
     *    function gets called (functions normally destroy their parameters)
     * 2 and 3 are handled by doing the argument construction in 'eprefix' so that if a later argument throws, they are cleaned
     * up properly. Pushing arguments on the stack then cannot fail.
     */
    {
        /* TODO: tackle problem 1)
         */
        const bool leftToRight = true; // TODO: something like !fd.isArrayOp
        if (!leftToRight)
            assert(nargs == nparams); // no variadics for RTL order, as they would probably be evaluated LTR and so add complexity

        const ptrdiff_t start = (leftToRight ? 0 : cast(ptrdiff_t)nargs - 1);
        const ptrdiff_t end = (leftToRight ? cast(ptrdiff_t)nargs : -1);
        const ptrdiff_t step = (leftToRight ? 1 : -1);

        /* Compute indices of last throwing argument and first arg needing destruction.
         * Used to not set up destructors unless an arg needs destruction on a throw
         * in a later argument.
         */
        ptrdiff_t lastthrow = -1;
        ptrdiff_t firstdtor = -1;
        for (ptrdiff_t i = start; i != end; i += step)
        {
            Expression arg = (*arguments)[i];
            if (canThrow(arg, sc.func, false))
                lastthrow = i;
            if (firstdtor == -1 && arg.type.needsDestruction())
            {
                Parameter p = (i >= nparams ? null : tf.parameterList[i]);
                if (!(p && (p.storageClass & (STC.lazy_ | STC.ref_ | STC.out_))))
                    firstdtor = i;
            }
        }

        /* Does problem 3) apply to this call?
         */
        const bool needsPrefix = (firstdtor >= 0 && lastthrow >= 0
            && (lastthrow - firstdtor) * step > 0);

        /* If so, initialize 'eprefix' by declaring the gate
         */
        VarDeclaration gate = null;
        if (needsPrefix)
        {
            // eprefix => bool __gate [= false]
            Identifier idtmp = Identifier.generateId("__gate");
            gate = new VarDeclaration(loc, Type.tbool, idtmp, null);
            gate.storage_class |= STC.temp | STC.ctfe | STC.volatile_;
            gate.dsymbolSemantic(sc);

            auto ae = new DeclarationExp(loc, gate);
            eprefix = ae.expressionSemantic(sc);
        }

        for (ptrdiff_t i = start; i != end; i += step)
        {
            Expression arg = (*arguments)[i];

            Parameter parameter = (i >= nparams ? null : tf.parameterList[i]);
            const bool isRef = (parameter && (parameter.storageClass & (STC.ref_ | STC.out_)));
            const bool isLazy = (parameter && (parameter.storageClass & STC.lazy_));

            /* Skip lazy parameters
             */
            if (isLazy)
                continue;

            /* Do we have a gate? Then we have a prefix and we're not yet past the last throwing arg.
             * Declare a temporary variable for this arg and append that declaration to 'eprefix',
             * which will implicitly take care of potential problem 2) for this arg.
             * 'eprefix' will therefore finally contain all args up to and including the last
             * potentially throwing arg, excluding all lazy parameters.
             */
            if (gate)
            {
                const bool needsDtor = (!isRef && arg.type.needsDestruction() && i != lastthrow);

                /* Declare temporary 'auto __pfx = arg' (needsDtor) or 'auto __pfy = arg' (!needsDtor)
                 */
                auto tmp = copyToTemp(0,
                    needsDtor ? "__pfx" : "__pfy",
                    !isRef ? arg : arg.addressOf());
                tmp.dsymbolSemantic(sc);

                /* Modify the destructor so it only runs if gate==false, i.e.,
                 * only if there was a throw while constructing the args
                 */
                if (!needsDtor)
                {
                    if (tmp.edtor)
                    {
                        assert(i == lastthrow);
                        tmp.edtor = null;
                    }
                }
                else
                {
                    // edtor => (__gate || edtor)
                    assert(tmp.edtor);
                    Expression e = tmp.edtor;
                    e = new LogicalExp(e.loc, TOK.orOr, new VarExp(e.loc, gate), e);
                    tmp.edtor = e.expressionSemantic(sc);
                    //printf("edtor: %s\n", tmp.edtor.toChars());
                }

                // eprefix => (eprefix, auto __pfx/y = arg)
                auto ae = new DeclarationExp(loc, tmp);
                eprefix = Expression.combine(eprefix, ae.expressionSemantic(sc));

                // arg => __pfx/y
                arg = new VarExp(loc, tmp);
                arg = arg.expressionSemantic(sc);
                if (isRef)
                {
                    arg = new PtrExp(loc, arg);
                    arg = arg.expressionSemantic(sc);
                }

                /* Last throwing arg? Then finalize eprefix => (eprefix, gate = true),
                 * i.e., disable the dtors right after constructing the last throwing arg.
                 * From now on, the callee will take care of destructing the args because
                 * the args are implicitly moved into function parameters.
                 *
                 * Set gate to null to let the next iterations know they don't need to
                 * append to eprefix anymore.
                 */
                if (i == lastthrow)
                {
                    auto e = new AssignExp(gate.loc, new VarExp(gate.loc, gate), IntegerExp.createBool(true));
                    eprefix = Expression.combine(eprefix, e.expressionSemantic(sc));
                    gate = null;
                }
            }
            else
            {
                /* No gate, no prefix to append to.
                 * Handle problem 2) by calling the copy constructor for value structs
                 * (or static arrays of them) if appropriate.
                 */
                Type tv = arg.type.baseElemOf();
                if (!isRef && tv.ty == Tstruct)
                    arg = doCopyOrMove(sc, arg, parameter ? parameter.type : null);
            }

            (*arguments)[i] = arg;
        }
    }
    //if (eprefix) printf("eprefix: %s\n", eprefix.toChars());

    /* Test compliance with DIP1021
     */
    if (global.params.useDIP1021 &&
        tf.trust != TRUST.system && tf.trust != TRUST.trusted)
        err |= checkMutableArguments(sc, fd, tf, ethis, arguments, false);

    // If D linkage and variadic, add _arguments[] as first argument
    if (!IN_LLVM && tf.isDstyleVariadic())
    {
        assert(arguments.dim >= nparams);

        auto args = new Parameters(arguments.dim - nparams);
        for (size_t i = 0; i < arguments.dim - nparams; i++)
        {
            auto arg = new Parameter(STC.in_, (*arguments)[nparams + i].type, null, null, null);
            (*args)[i] = arg;
        }
        auto tup = new TypeTuple(args);
        Expression e = (new TypeidExp(loc, tup)).expressionSemantic(sc);
        arguments.insert(0, e);
    }

    /* Determine function return type: tret
     */
    Type tret = tf.next;
    if (isCtorCall)
    {
        //printf("[%s] fd = %s %s, %d %d %d\n", loc.toChars(), fd.toChars(), fd.type.toChars(),
        //    wildmatch, tf.isWild(), fd.isReturnIsolated());
        if (!tthis)
        {
            assert(sc.intypeof || global.errors);
            tthis = fd.isThis().type.addMod(fd.type.mod);
        }
        if (tf.isWild() && !fd.isReturnIsolated())
        {
            if (wildmatch)
                tret = tret.substWildTo(wildmatch);
            int offset;
            if (!tret.implicitConvTo(tthis) && !(MODimplicitConv(tret.mod, tthis.mod) && tret.isBaseOf(tthis, &offset) && offset == 0))
            {
                const(char)* s1 = tret.isNaked() ? " mutable" : tret.modToChars();
                const(char)* s2 = tthis.isNaked() ? " mutable" : tthis.modToChars();
                .error(loc, "`inout` constructor `%s` creates%s object, not%s", fd.toPrettyChars(), s1, s2);
                err = true;
            }
        }
        tret = tthis;
    }
    else if (wildmatch && tret)
    {
        /* Adjust function return type based on wildmatch
         */
        //printf("wildmatch = x%x, tret = %s\n", wildmatch, tret.toChars());
        tret = tret.substWildTo(wildmatch);
    }

    *prettype = tret;
    *peprefix = eprefix;
    return (err || olderrors != global.errors);
}

/**
 * Determines whether a symbol represents a module or package
 * (Used as a helper for is(type == module) and is(type == package))
 *
 * Params:
 *  sym = the symbol to be checked
 *
 * Returns:
 *  the symbol which `sym` represents (or `null` if it doesn't represent a `Package`)
 */
Package resolveIsPackage(Dsymbol sym)
{
    Package pkg;
    if (Import imp = sym.isImport())
    {
        if (imp.pkg is null)
        {
            .error(sym.loc, "Internal Compiler Error: unable to process forward-referenced import `%s`",
                    imp.toChars());
            assert(0);
        }
        pkg = imp.pkg;
    }
    else if (auto mod = sym.isModule())
        pkg = mod.isPackageFile ? mod.pkg : sym.isPackage();
    else
        pkg = sym.isPackage();
    if (pkg)
        pkg.resolvePKGunknown();
    return pkg;
}

private Module loadStdMath()
{
    __gshared Import impStdMath = null;
    if (!impStdMath)
    {
        auto a = new Identifiers();
        a.push(Id.std);
        auto s = new Import(Loc.initial, a, Id.math, null, false);
        // Module.load will call fatal() if there's no std.math available.
        // Gag the error here, pushing the error handling to the caller.
        uint errors = global.startGagging();
        s.load(null);
        if (s.mod)
        {
            s.mod.importAll(null);
            s.mod.dsymbolSemantic(null);
        }
        global.endGagging(errors);
        impStdMath = s;
    }
    return impStdMath.mod;
}

private extern (C++) final class ExpressionSemanticVisitor : Visitor
{
    alias visit = Visitor.visit;

    Scope* sc;
    Expression result;

    this(Scope* sc)
    {
        this.sc = sc;
    }

    private void setError()
    {
        result = new ErrorExp();
    }

    /**************************
     * Semantically analyze Expression.
     * Determine types, fold constants, etc.
     */
    override void visit(Expression e)
    {
        static if (LOGSEMANTIC)
        {
            printf("Expression::semantic() %s\n", e.toChars());
        }
        if (e.type)
            e.type = e.type.typeSemantic(e.loc, sc);
        else
            e.type = Type.tvoid;
        result = e;
    }

    override void visit(IntegerExp e)
    {
        assert(e.type);
        if (e.type.ty == Terror)
            return setError();

        assert(e.type.deco);
        e.setInteger(e.getInteger());
        result = e;
    }

    override void visit(RealExp e)
    {
        if (!e.type)
            e.type = Type.tfloat64;
        else
            e.type = e.type.typeSemantic(e.loc, sc);
        result = e;
    }

    override void visit(ComplexExp e)
    {
        if (!e.type)
            e.type = Type.tcomplex80;
        else
            e.type = e.type.typeSemantic(e.loc, sc);
        result = e;
    }

    override void visit(IdentifierExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("IdentifierExp::semantic('%s')\n", exp.ident.toChars());
        }
        if (exp.type) // This is used as the dummy expression
        {
            result = exp;
            return;
        }

        Dsymbol scopesym;
        Dsymbol s = sc.search(exp.loc, exp.ident, &scopesym);
        if (s)
        {
            if (s.errors)
                return setError();

            Expression e;

            /* See if the symbol was a member of an enclosing 'with'
             */
            WithScopeSymbol withsym = scopesym.isWithScopeSymbol();
            if (withsym && withsym.withstate.wthis)
            {
                /* Disallow shadowing
                 */
                // First find the scope of the with
                Scope* scwith = sc;
                while (scwith.scopesym != scopesym)
                {
                    scwith = scwith.enclosing;
                    assert(scwith);
                }
                // Look at enclosing scopes for symbols with the same name,
                // in the same function
                for (Scope* scx = scwith; scx && scx.func == scwith.func; scx = scx.enclosing)
                {
                    Dsymbol s2;
                    if (scx.scopesym && scx.scopesym.symtab && (s2 = scx.scopesym.symtab.lookup(s.ident)) !is null && s != s2)
                    {
                        exp.error("with symbol `%s` is shadowing local symbol `%s`", s.toPrettyChars(), s2.toPrettyChars());
                        return setError();
                    }
                }
                s = s.toAlias();

                // Same as wthis.ident
                //  TODO: DotIdExp.semantic will find 'ident' from 'wthis' again.
                //  The redudancy should be removed.
                e = new VarExp(exp.loc, withsym.withstate.wthis);
                e = new DotIdExp(exp.loc, e, exp.ident);
                e = e.expressionSemantic(sc);
            }
            else
            {
                if (withsym)
                {
                    if (auto t = withsym.withstate.exp.isTypeExp())
                    {
                        e = new TypeExp(exp.loc, t.type);
                        e = new DotIdExp(exp.loc, e, exp.ident);
                        result = e.expressionSemantic(sc);
                        return;
                    }
                }

                /* If f is really a function template,
                 * then replace f with the function template declaration.
                 */
                FuncDeclaration f = s.isFuncDeclaration();
                if (f)
                {
                    TemplateDeclaration td = getFuncTemplateDecl(f);
                    if (td)
                    {
                        if (td.overroot) // if not start of overloaded list of TemplateDeclaration's
                            td = td.overroot; // then get the start
                        e = new TemplateExp(exp.loc, td, f);
                        e = e.expressionSemantic(sc);
                        result = e;
                        return;
                    }
                }

                if (global.params.fixAliasThis)
                {
                    ExpressionDsymbol expDsym = scopesym.isExpressionDsymbol();
                    if (expDsym)
                    {
                        //printf("expDsym = %s\n", expDsym.exp.toChars());
                        result = expDsym.exp.expressionSemantic(sc);
                        return;
                    }
                }
                // Haven't done overload resolution yet, so pass 1
                e = symbolToExp(s, exp.loc, sc, true);
            }
            result = e;
            return;
        }

        if (!global.params.fixAliasThis && hasThis(sc))
        {
            for (AggregateDeclaration ad = sc.getStructClassScope(); ad;)
            {
                if (ad.aliasthis)
                {
                    Expression e;
                    e = new ThisExp(exp.loc);
                    e = new DotIdExp(exp.loc, e, ad.aliasthis.ident);
                    e = new DotIdExp(exp.loc, e, exp.ident);
                    e = e.trySemantic(sc);
                    if (e)
                    {
                        result = e;
                        return;
                    }
                }

                auto cd = ad.isClassDeclaration();
                if (cd && cd.baseClass && cd.baseClass != ClassDeclaration.object)
                {
                    ad = cd.baseClass;
                    continue;
                }
                break;
            }
        }

        if (exp.ident == Id.ctfe)
        {
            if (sc.flags & SCOPE.ctfe)
            {
                exp.error("variable `__ctfe` cannot be read at compile time");
                return setError();
            }

            // Create the magic __ctfe bool variable
            auto vd = new VarDeclaration(exp.loc, Type.tbool, Id.ctfe, null);
            vd.storage_class |= STC.temp;
            vd.semanticRun = PASS.semanticdone;
            Expression e = new VarExp(exp.loc, vd);
            e = e.expressionSemantic(sc);
            result = e;
            return;
        }

        // If we've reached this point and are inside a with() scope then we may
        // try one last attempt by checking whether the 'wthis' object supports
        // dynamic dispatching via opDispatch.
        // This is done by rewriting this expression as wthis.ident.
        // The innermost with() scope of the hierarchy to satisfy the condition
        // above wins.
        // https://issues.dlang.org/show_bug.cgi?id=6400
        for (Scope* sc2 = sc; sc2; sc2 = sc2.enclosing)
        {
            if (!sc2.scopesym)
                continue;

            if (auto ss = sc2.scopesym.isWithScopeSymbol())
            {
                if (ss.withstate.wthis)
                {
                    Expression e;
                    e = new VarExp(exp.loc, ss.withstate.wthis);
                    e = new DotIdExp(exp.loc, e, exp.ident);
                    e = e.trySemantic(sc);
                    if (e)
                    {
                        result = e;
                        return;
                    }
                }
                // Try Type.opDispatch (so the static version)
                else if (ss.withstate.exp && ss.withstate.exp.op == TOK.type)
                {
                    if (Type t = ss.withstate.exp.isTypeExp().type)
                    {
                        Expression e;
                        e = new TypeExp(exp.loc, t);
                        e = new DotIdExp(exp.loc, e, exp.ident);
                        e = e.trySemantic(sc);
                        if (e)
                        {
                            result = e;
                            return;
                        }
                    }
                }
            }
        }

        /* Look for what user might have meant
         */
        if (const n = importHint(exp.ident.toString()))
            exp.error("`%s` is not defined, perhaps `import %.*s;` is needed?", exp.ident.toChars(), cast(int)n.length, n.ptr);
        else if (auto s2 = sc.search_correct(exp.ident))
            exp.error("undefined identifier `%s`, did you mean %s `%s`?", exp.ident.toChars(), s2.kind(), s2.toChars());
        else if (const p = Scope.search_correct_C(exp.ident))
            exp.error("undefined identifier `%s`, did you mean `%s`?", exp.ident.toChars(), p);
        else
            exp.error("undefined identifier `%s`", exp.ident.toChars());

        result = new ErrorExp();
    }

    override void visit(DsymbolExp e)
    {
        result = symbolToExp(e.s, e.loc, sc, e.hasOverloads);
    }

    override void visit(ThisExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("ThisExp::semantic()\n");
        }
        if (e.type)
        {
            result = e;
            return;
        }

        FuncDeclaration fd = hasThis(sc); // fd is the uplevel function with the 'this' variable
        AggregateDeclaration ad;

        /* Special case for typeof(this) and typeof(super) since both
         * should work even if they are not inside a non-static member function
         */
        if (!fd && sc.intypeof == 1)
        {
            // Find enclosing struct or class
            for (Dsymbol s = sc.getStructClassScope(); 1; s = s.parent)
            {
                if (!s)
                {
                    e.error("`%s` is not in a class or struct scope", e.toChars());
                    goto Lerr;
                }
                ClassDeclaration cd = s.isClassDeclaration();
                if (cd)
                {
                    e.type = cd.type;
                    result = e;
                    return;
                }
                StructDeclaration sd = s.isStructDeclaration();
                if (sd)
                {
                    e.type = sd.type;
                    result = e;
                    return;
                }
            }
        }
        if (!fd)
            goto Lerr;

        assert(fd.vthis);
        e.var = fd.vthis;
        assert(e.var.parent);
        ad = fd.isMemberLocal();
        if (!ad)
            ad = fd.isMember2();
        assert(ad);
        e.type = ad.type.addMod(e.var.type.mod);

        if (e.var.checkNestedReference(sc, e.loc))
            return setError();

        result = e;
        return;

    Lerr:
        e.error("`this` is only defined in non-static member functions, not `%s`", sc.parent.toChars());
        result = new ErrorExp();
    }

    override void visit(SuperExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("SuperExp::semantic('%s')\n", e.toChars());
        }
        if (e.type)
        {
            result = e;
            return;
        }

        FuncDeclaration fd = hasThis(sc);
        ClassDeclaration cd;
        Dsymbol s;

        /* Special case for typeof(this) and typeof(super) since both
         * should work even if they are not inside a non-static member function
         */
        if (!fd && sc.intypeof == 1)
        {
            // Find enclosing class
            for (s = sc.getStructClassScope(); 1; s = s.parent)
            {
                if (!s)
                {
                    e.error("`%s` is not in a class scope", e.toChars());
                    goto Lerr;
                }
                cd = s.isClassDeclaration();
                if (cd)
                {
                    cd = cd.baseClass;
                    if (!cd)
                    {
                        e.error("class `%s` has no `super`", s.toChars());
                        goto Lerr;
                    }
                    e.type = cd.type;
                    result = e;
                    return;
                }
            }
        }
        if (!fd)
            goto Lerr;

        e.var = fd.vthis;
        assert(e.var && e.var.parent);

        s = fd.toParentDecl();
        if (s.isTemplateDeclaration()) // allow inside template constraint
            s = s.toParent();
        assert(s);
        cd = s.isClassDeclaration();
        //printf("parent is %s %s\n", fd.toParent().kind(), fd.toParent().toChars());
        if (!cd)
            goto Lerr;
        if (!cd.baseClass)
        {
            e.error("no base class for `%s`", cd.toChars());
            e.type = cd.type.addMod(e.var.type.mod);
        }
        else
        {
            e.type = cd.baseClass.type;
            e.type = e.type.castMod(e.var.type.mod);
        }

        if (e.var.checkNestedReference(sc, e.loc))
            return setError();

        result = e;
        return;

    Lerr:
        e.error("`super` is only allowed in non-static class member functions");
        result = new ErrorExp();
    }

    override void visit(NullExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("NullExp::semantic('%s')\n", e.toChars());
        }
        // NULL is the same as (void *)0
        if (e.type)
        {
            result = e;
            return;
        }
        e.type = Type.tnull;
        result = e;
    }

    override void visit(StringExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("StringExp::semantic() %s\n", e.toChars());
        }
        if (e.type)
        {
            result = e;
            return;
        }

        OutBuffer buffer;
        size_t newlen = 0;
        size_t u;
        dchar c;

        switch (e.postfix)
        {
        case 'd':
            for (u = 0; u < e.len;)
            {
                if (const p = utf_decodeChar(e.peekString(), u, c))
                {
                    e.error("%.*s", cast(int)p.length, p.ptr);
                    return setError();
                }
                else
                {
                    buffer.write4(c);
                    newlen++;
                }
            }
            buffer.write4(0);
            e.setData(buffer.extractData(), newlen, 4);
            e.type = new TypeDArray(Type.tdchar.immutableOf());
            e.committed = 1;
            break;

        case 'w':
            for (u = 0; u < e.len;)
            {
                if (const p = utf_decodeChar(e.peekString(), u, c))
                {
                    e.error("%.*s", cast(int)p.length, p.ptr);
                    return setError();
                }
                else
                {
                    buffer.writeUTF16(c);
                    newlen++;
                    if (c >= 0x10000)
                        newlen++;
                }
            }
            buffer.writeUTF16(0);
            e.setData(buffer.extractData(), newlen, 2);
            e.type = new TypeDArray(Type.twchar.immutableOf());
            e.committed = 1;
            break;

        case 'c':
            e.committed = 1;
            goto default;

        default:
            e.type = new TypeDArray(Type.tchar.immutableOf());
            break;
        }
        e.type = e.type.typeSemantic(e.loc, sc);
        //type = type.immutableOf();
        //printf("type = %s\n", type.toChars());

        result = e;
    }

    override void visit(TupleExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("+TupleExp::semantic(%s)\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (exp.e0)
            exp.e0 = exp.e0.expressionSemantic(sc);

        // Run semantic() on each argument
        bool err = false;
        for (size_t i = 0; i < exp.exps.dim; i++)
        {
            Expression e = (*exp.exps)[i];
            e = e.expressionSemantic(sc);
            if (!e.type)
            {
                exp.error("`%s` has no value", e.toChars());
                err = true;
            }
            else if (e.op == TOK.error)
                err = true;
            else
                (*exp.exps)[i] = e;
        }
        if (err)
            return setError();

        expandTuples(exp.exps);

        exp.type = new TypeTuple(exp.exps);
        exp.type = exp.type.typeSemantic(exp.loc, sc);
        //printf("-TupleExp::semantic(%s)\n", toChars());
        result = exp;
    }

    override void visit(ArrayLiteralExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("ArrayLiteralExp::semantic('%s')\n", e.toChars());
        }
        if (e.type)
        {
            result = e;
            return;
        }

        /* Perhaps an empty array literal [ ] should be rewritten as null?
         */

        if (e.basis)
            e.basis = e.basis.expressionSemantic(sc);
        if (arrayExpressionSemantic(e.elements, sc) || (e.basis && e.basis.op == TOK.error))
            return setError();

        expandTuples(e.elements);

        Type t0;
        if (e.basis)
            e.elements.push(e.basis);
        bool err = arrayExpressionToCommonType(sc, e.elements, &t0);
        if (e.basis)
            e.basis = e.elements.pop();
        if (err)
            return setError();

        e.type = t0.arrayOf();
        e.type = e.type.typeSemantic(e.loc, sc);

        /* Disallow array literals of type void being used.
         */
        if (e.elements.dim > 0 && t0.ty == Tvoid)
        {
            e.error("`%s` of type `%s` has no value", e.toChars(), e.type.toChars());
            return setError();
        }

        if (global.params.useTypeInfo && Type.dtypeinfo)
            semanticTypeInfo(sc, e.type);

        result = e;
    }

    override void visit(AssocArrayLiteralExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("AssocArrayLiteralExp::semantic('%s')\n", e.toChars());
        }
        if (e.type)
        {
            result = e;
            return;
        }

        // Run semantic() on each element
        bool err_keys = arrayExpressionSemantic(e.keys, sc);
        bool err_vals = arrayExpressionSemantic(e.values, sc);
        if (err_keys || err_vals)
            return setError();

        expandTuples(e.keys);
        expandTuples(e.values);
        if (e.keys.dim != e.values.dim)
        {
            e.error("number of keys is %u, must match number of values %u", e.keys.dim, e.values.dim);
            return setError();
        }

        Type tkey = null;
        Type tvalue = null;
        err_keys = arrayExpressionToCommonType(sc, e.keys, &tkey);
        err_vals = arrayExpressionToCommonType(sc, e.values, &tvalue);
        if (err_keys || err_vals)
            return setError();

        if (tkey == Type.terror || tvalue == Type.terror)
            return setError();

        e.type = new TypeAArray(tvalue, tkey);
        e.type = e.type.typeSemantic(e.loc, sc);

        semanticTypeInfo(sc, e.type);

        if (global.params.vsafe)
        {
            if (checkAssocArrayLiteralEscape(sc, e, false))
                return setError();
        }

        result = e;
    }

    override void visit(StructLiteralExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("StructLiteralExp::semantic('%s')\n", e.toChars());
        }
        if (e.type)
        {
            result = e;
            return;
        }

        e.sd.size(e.loc);
        if (e.sd.sizeok != Sizeok.done)
            return setError();

        // run semantic() on each element
        if (arrayExpressionSemantic(e.elements, sc))
            return setError();

        expandTuples(e.elements);

        /* Fit elements[] to the corresponding type of field[].
         */
        if (!e.sd.fit(e.loc, sc, e.elements, e.stype))
            return setError();

        /* Fill out remainder of elements[] with default initializers for fields[]
         */
        if (!e.sd.fill(e.loc, e.elements, false))
        {
            /* An error in the initializer needs to be recorded as an error
             * in the enclosing function or template, since the initializer
             * will be part of the stuct declaration.
             */
            global.increaseErrorCount();
            return setError();
        }

        if (checkFrameAccess(e.loc, sc, e.sd, e.elements.dim))
            return setError();

        e.type = e.stype ? e.stype : e.sd.type;
        result = e;
    }

    override void visit(TypeExp exp)
    {
        if (exp.type.ty == Terror)
            return setError();

        //printf("TypeExp::semantic(%s)\n", type.toChars());
        Expression e;
        Type t;
        Dsymbol s;

        dmd.typesem.resolve(exp.type, exp.loc, sc, &e, &t, &s, true);
        if (e)
        {
            // `(Type)` is actually `(var)` so if `(var)` is a member requiring `this`
            // then rewrite as `(this.var)` in case it would be followed by a DotVar
            // to fix https://issues.dlang.org/show_bug.cgi?id=9490
            VarExp ve = e.isVarExp();
            if (ve && ve.var && exp.parens && !ve.var.isStatic() && !(sc.stc & STC.static_) &&
                sc.func && sc.func.needThis && ve.var.toParent2().isAggregateDeclaration())
            {
                // printf("apply fix for issue 9490: add `this.` to `%s`...\n", e.toChars());
                e = new DotVarExp(exp.loc, new ThisExp(exp.loc), ve.var, false);
            }
            //printf("e = %s %s\n", Token::toChars(e.op), e.toChars());
            e = e.expressionSemantic(sc);
        }
        else if (t)
        {
            //printf("t = %d %s\n", t.ty, t.toChars());
            exp.type = t.typeSemantic(exp.loc, sc);
            e = exp;
        }
        else if (s)
        {
            //printf("s = %s %s\n", s.kind(), s.toChars());
            e = symbolToExp(s, exp.loc, sc, true);
        }
        else
            assert(0);

        if (global.params.vcomplex)
            exp.type.checkComplexTransition(exp.loc, sc);

        result = e;
    }

    override void visit(ScopeExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("+ScopeExp::semantic(%p '%s')\n", exp, exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        ScopeDsymbol sds2 = exp.sds;
        TemplateInstance ti = sds2.isTemplateInstance();
        while (ti)
        {
            WithScopeSymbol withsym;
            if (!ti.findTempDecl(sc, &withsym) || !ti.semanticTiargs(sc))
                return setError();
            if (withsym && withsym.withstate.wthis)
            {
                Expression e = new VarExp(exp.loc, withsym.withstate.wthis);
                e = new DotTemplateInstanceExp(exp.loc, e, ti);
                result = e.expressionSemantic(sc);
                return;
            }
            if (ti.needsTypeInference(sc))
            {
                if (TemplateDeclaration td = ti.tempdecl.isTemplateDeclaration())
                {
                    Dsymbol p = td.toParentLocal();
                    FuncDeclaration fdthis = hasThis(sc);
                    AggregateDeclaration ad = p ? p.isAggregateDeclaration() : null;
                    if (fdthis && ad && fdthis.isMemberLocal() == ad && (td._scope.stc & STC.static_) == 0)
                    {
                        Expression e = new DotTemplateInstanceExp(exp.loc, new ThisExp(exp.loc), ti.name, ti.tiargs);
                        result = e.expressionSemantic(sc);
                        return;
                    }
                }
                else if (OverloadSet os = ti.tempdecl.isOverloadSet())
                {
                    FuncDeclaration fdthis = hasThis(sc);
                    AggregateDeclaration ad = os.parent.isAggregateDeclaration();
                    if (fdthis && ad && fdthis.isMemberLocal() == ad)
                    {
                        Expression e = new DotTemplateInstanceExp(exp.loc, new ThisExp(exp.loc), ti.name, ti.tiargs);
                        result = e.expressionSemantic(sc);
                        return;
                    }
                }
                // ti is an instance which requires IFTI.
                exp.sds = ti;
                exp.type = Type.tvoid;
                result = exp;
                return;
            }
            ti.dsymbolSemantic(sc);
            if (!ti.inst || ti.errors)
                return setError();

            Dsymbol s = ti.toAlias();
            if (s == ti)
            {
                exp.sds = ti;
                exp.type = Type.tvoid;
                result = exp;
                return;
            }
            sds2 = s.isScopeDsymbol();
            if (sds2)
            {
                ti = sds2.isTemplateInstance();
                //printf("+ sds2 = %s, '%s'\n", sds2.kind(), sds2.toChars());
                continue;
            }

            if (auto v = s.isVarDeclaration())
            {
                if (!v.type)
                {
                    exp.error("forward reference of %s `%s`", v.kind(), v.toChars());
                    return setError();
                }
                if ((v.storage_class & STC.manifest) && v._init)
                {
                    /* When an instance that will be converted to a constant exists,
                     * the instance representation "foo!tiargs" is treated like a
                     * variable name, and its recursive appearance check (note that
                     * it's equivalent with a recursive instantiation of foo) is done
                     * separately from the circular initialization check for the
                     * eponymous enum variable declaration.
                     *
                     *  template foo(T) {
                     *    enum bool foo = foo;    // recursive definition check (v.inuse)
                     *  }
                     *  template bar(T) {
                     *    enum bool bar = bar!T;  // recursive instantiation check (ti.inuse)
                     *  }
                     */
                    if (ti.inuse)
                    {
                        exp.error("recursive expansion of %s `%s`", ti.kind(), ti.toPrettyChars());
                        return setError();
                    }
                    v.checkDeprecated(exp.loc, sc);
                    auto e = v.expandInitializer(exp.loc);
                    ti.inuse++;
                    e = e.expressionSemantic(sc);
                    ti.inuse--;
                    result = e;
                    return;
                }
            }

            //printf("s = %s, '%s'\n", s.kind(), s.toChars());
            auto e = symbolToExp(s, exp.loc, sc, true);
            //printf("-1ScopeExp::semantic()\n");
            result = e;
            return;
        }

        //printf("sds2 = %s, '%s'\n", sds2.kind(), sds2.toChars());
        //printf("\tparent = '%s'\n", sds2.parent.toChars());
        sds2.dsymbolSemantic(sc);

        // (Aggregate|Enum)Declaration
        if (auto t = sds2.getType())
        {
            result = (new TypeExp(exp.loc, t)).expressionSemantic(sc);
            return;
        }

        if (auto td = sds2.isTemplateDeclaration())
        {
            result = (new TemplateExp(exp.loc, td)).expressionSemantic(sc);
            return;
        }

        exp.sds = sds2;
        exp.type = Type.tvoid;
        //printf("-2ScopeExp::semantic() %s\n", toChars());
        result = exp;
    }

    override void visit(NewExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("NewExp::semantic() %s\n", exp.toChars());
            if (exp.thisexp)
                printf("\tthisexp = %s\n", exp.thisexp.toChars());
            printf("\tnewtype: %s\n", exp.newtype.toChars());
        }
        if (exp.type) // if semantic() already run
        {
            result = exp;
            return;
        }

        //for error messages if the argument in [] is not convertible to size_t
        const originalNewtype = exp.newtype;

        // https://issues.dlang.org/show_bug.cgi?id=11581
        // With the syntax `new T[edim]` or `thisexp.new T[edim]`,
        // T should be analyzed first and edim should go into arguments iff it's
        // not a tuple.
        Expression edim = null;
        if (!exp.arguments && exp.newtype.ty == Tsarray)
        {
            edim = (cast(TypeSArray)exp.newtype).dim;
            exp.newtype = (cast(TypeNext)exp.newtype).next;
        }

        ClassDeclaration cdthis = null;
        if (exp.thisexp)
        {
            exp.thisexp = exp.thisexp.expressionSemantic(sc);
            if (exp.thisexp.op == TOK.error)
                return setError();

            cdthis = exp.thisexp.type.isClassHandle();
            if (!cdthis)
            {
                exp.error("`this` for nested class must be a class type, not `%s`", exp.thisexp.type.toChars());
                return setError();
            }

            sc = sc.push(cdthis);
            exp.type = exp.newtype.typeSemantic(exp.loc, sc);
            sc = sc.pop();
        }
        else
        {
            exp.type = exp.newtype.typeSemantic(exp.loc, sc);
        }
        if (exp.type.ty == Terror)
            return setError();

        if (edim)
        {
            if (exp.type.toBasetype().ty == Ttuple)
            {
                // --> new T[edim]
                exp.type = new TypeSArray(exp.type, edim);
                exp.type = exp.type.typeSemantic(exp.loc, sc);
                if (exp.type.ty == Terror)
                    return setError();
            }
            else
            {
                // --> new T[](edim)
                exp.arguments = new Expressions();
                exp.arguments.push(edim);
                exp.type = exp.type.arrayOf();
            }
        }

        exp.newtype = exp.type; // in case type gets cast to something else
        Type tb = exp.type.toBasetype();
        //printf("tb: %s, deco = %s\n", tb.toChars(), tb.deco);
        if (arrayExpressionSemantic(exp.newargs, sc) ||
            preFunctionParameters(sc, exp.newargs))
        {
            return setError();
        }
        if (arrayExpressionSemantic(exp.arguments, sc))
        {
            return setError();
        }
        //https://issues.dlang.org/show_bug.cgi?id=20547
        //exp.arguments are the "parameters" to [], not to a real function
        //so the errors that come from preFunctionParameters are misleading
        if (originalNewtype.ty == Tsarray)
        {
            if (preFunctionParameters(sc, exp.arguments, false))
            {
                exp.error("cannot create a `%s` with `new`", originalNewtype.toChars());
                return setError();
            }
        }
        else if (preFunctionParameters(sc, exp.arguments))
        {
            return setError();
        }

        if (exp.thisexp && tb.ty != Tclass)
        {
            exp.error("`.new` is only for allocating nested classes, not `%s`", tb.toChars());
            return setError();
        }

        const size_t nargs = exp.arguments ? exp.arguments.dim : 0;
        Expression newprefix = null;

        if (tb.ty == Tclass)
        {
            auto cd = (cast(TypeClass)tb).sym;
            cd.size(exp.loc);
            if (cd.sizeok != Sizeok.done)
                return setError();
            if (!cd.ctor)
                cd.ctor = cd.searchCtor();
            if (cd.noDefaultCtor && !nargs && !cd.defaultCtor)
            {
                exp.error("default construction is disabled for type `%s`", cd.type.toChars());
                return setError();
            }

            if (cd.isInterfaceDeclaration())
            {
                exp.error("cannot create instance of interface `%s`", cd.toChars());
                return setError();
            }

            if (cd.isAbstract())
            {
                exp.error("cannot create instance of abstract class `%s`", cd.toChars());
                for (size_t i = 0; i < cd.vtbl.dim; i++)
                {
                    FuncDeclaration fd = cd.vtbl[i].isFuncDeclaration();
                    if (fd && fd.isAbstract())
                    {
                        errorSupplemental(exp.loc, "function `%s` is not implemented",
                            fd.toFullSignature());
                    }
                }
                return setError();
            }
            // checkDeprecated() is already done in newtype.typeSemantic().

            if (cd.isNested())
            {
                /* We need a 'this' pointer for the nested class.
                 * Ensure we have the right one.
                 */
                Dsymbol s = cd.toParentLocal();

                //printf("cd isNested, parent = %s '%s'\n", s.kind(), s.toPrettyChars());
                if (auto cdn = s.isClassDeclaration())
                {
                    if (!cdthis)
                    {
                        // Supply an implicit 'this' and try again
                        exp.thisexp = new ThisExp(exp.loc);
                        for (Dsymbol sp = sc.parent; 1; sp = sp.toParentLocal())
                        {
                            if (!sp)
                            {
                                exp.error("outer class `%s` `this` needed to `new` nested class `%s`",
                                    cdn.toChars(), cd.toChars());
                                return setError();
                            }
                            ClassDeclaration cdp = sp.isClassDeclaration();
                            if (!cdp)
                                continue;
                            if (cdp == cdn || cdn.isBaseOf(cdp, null))
                                break;
                            // Add a '.outer' and try again
                            exp.thisexp = new DotIdExp(exp.loc, exp.thisexp, Id.outer);
                        }

                        exp.thisexp = exp.thisexp.expressionSemantic(sc);
                        if (exp.thisexp.op == TOK.error)
                            return setError();
                        cdthis = exp.thisexp.type.isClassHandle();
                    }
                    if (cdthis != cdn && !cdn.isBaseOf(cdthis, null))
                    {
                        //printf("cdthis = %s\n", cdthis.toChars());
                        exp.error("`this` for nested class must be of type `%s`, not `%s`",
                            cdn.toChars(), exp.thisexp.type.toChars());
                        return setError();
                    }
                    if (!MODimplicitConv(exp.thisexp.type.mod, exp.newtype.mod))
                    {
                        exp.error("nested type `%s` should have the same or weaker constancy as enclosing type `%s`",
                            exp.newtype.toChars(), exp.thisexp.type.toChars());
                        return setError();
                    }
                }
                else if (exp.thisexp)
                {
                    exp.error("`.new` is only for allocating nested classes");
                    return setError();
                }
                else if (auto fdn = s.isFuncDeclaration())
                {
                    // make sure the parent context fdn of cd is reachable from sc
                    if (!ensureStaticLinkTo(sc.parent, fdn))
                    {
                        exp.error("outer function context of `%s` is needed to `new` nested class `%s`",
                            fdn.toPrettyChars(), cd.toPrettyChars());
                        return setError();
                    }
                }
                else
                    assert(0);
            }
            else if (exp.thisexp)
            {
                exp.error("`.new` is only for allocating nested classes");
                return setError();
            }

            if (cd.vthis2)
            {
                if (AggregateDeclaration ad2 = cd.isMember2())
                {
                    Expression te = new ThisExp(exp.loc).expressionSemantic(sc);
                    if (te.op != TOK.error)
                        te = getRightThis(exp.loc, sc, ad2, te, cd);
                    if (te.op == TOK.error)
                    {
                        exp.error("need `this` of type `%s` needed to `new` nested class `%s`", ad2.toChars(), cd.toChars());
                        return setError();
                    }
                }
            }

            if (cd.aggNew)
            {
                // Prepend the size argument to newargs[]
                Expression e = new IntegerExp(exp.loc, cd.size(exp.loc), Type.tsize_t);
                if (!exp.newargs)
                    exp.newargs = new Expressions();
                exp.newargs.shift(e);

                FuncDeclaration f = resolveFuncCall(exp.loc, sc, cd.aggNew, null, tb, exp.newargs, FuncResolveFlag.standard);
                if (!f || f.errors)
                    return setError();

                checkFunctionAttributes(exp, sc, f);
                checkAccess(cd, exp.loc, sc, f);

                TypeFunction tf = cast(TypeFunction)f.type;
                Type rettype;
                if (functionParameters(exp.loc, sc, tf, null, null, exp.newargs, f, &rettype, &newprefix))
                    return setError();

                exp.allocator = f.isNewDeclaration();
                assert(exp.allocator);
            }
            else
            {
                if (exp.newargs && exp.newargs.dim)
                {
                    exp.error("no allocator for `%s`", cd.toChars());
                    return setError();
                }
            }

            if (cd.ctor)
            {
                FuncDeclaration f = resolveFuncCall(exp.loc, sc, cd.ctor, null, tb, exp.arguments, FuncResolveFlag.standard);
                if (!f || f.errors)
                    return setError();

                checkFunctionAttributes(exp, sc, f);
                checkAccess(cd, exp.loc, sc, f);

                TypeFunction tf = cast(TypeFunction)f.type;
                if (!exp.arguments)
                    exp.arguments = new Expressions();
                if (functionParameters(exp.loc, sc, tf, null, exp.type, exp.arguments, f, &exp.type, &exp.argprefix))
                    return setError();

                exp.member = f.isCtorDeclaration();
                assert(exp.member);
            }
            else
            {
                if (nargs)
                {
                    exp.error("no constructor for `%s`", cd.toChars());
                    return setError();
                }

                // https://issues.dlang.org/show_bug.cgi?id=19941
                // Run semantic on all field initializers to resolve any forward
                // references. This is the same as done for structs in sd.fill().
                for (ClassDeclaration c = cd; c; c = c.baseClass)
                {
                    foreach (v; c.fields)
                    {
                        if (v.inuse || v._scope is null || v._init is null ||
                            v._init.isVoidInitializer())
                            continue;
                        v.inuse++;
                        v._init = v._init.initializerSemantic(v._scope, v.type, INITinterpret);
                        v.inuse--;
                    }
                }
            }
        }
        else if (tb.ty == Tstruct)
        {
            auto sd = (cast(TypeStruct)tb).sym;
            sd.size(exp.loc);
            if (sd.sizeok != Sizeok.done)
                return setError();
            if (!sd.ctor)
                sd.ctor = sd.searchCtor();
            if (sd.noDefaultCtor && !nargs)
            {
                exp.error("default construction is disabled for type `%s`", sd.type.toChars());
                return setError();
            }
            // checkDeprecated() is already done in newtype.typeSemantic().

            if (sd.aggNew)
            {
                // Prepend the uint size argument to newargs[]
                Expression e = new IntegerExp(exp.loc, sd.size(exp.loc), Type.tsize_t);
                if (!exp.newargs)
                    exp.newargs = new Expressions();
                exp.newargs.shift(e);

                FuncDeclaration f = resolveFuncCall(exp.loc, sc, sd.aggNew, null, tb, exp.newargs, FuncResolveFlag.standard);
                if (!f || f.errors)
                    return setError();

                checkFunctionAttributes(exp, sc, f);
                checkAccess(sd, exp.loc, sc, f);

                TypeFunction tf = cast(TypeFunction)f.type;
                Type rettype;
                if (functionParameters(exp.loc, sc, tf, null, null, exp.newargs, f, &rettype, &newprefix))
                    return setError();

                exp.allocator = f.isNewDeclaration();
                assert(exp.allocator);
            }
            else
            {
                if (exp.newargs && exp.newargs.dim)
                {
                    exp.error("no allocator for `%s`", sd.toChars());
                    return setError();
                }
            }

            if (sd.ctor && nargs)
            {
                FuncDeclaration f = resolveFuncCall(exp.loc, sc, sd.ctor, null, tb, exp.arguments, FuncResolveFlag.standard);
                if (!f || f.errors)
                    return setError();

                checkFunctionAttributes(exp, sc, f);
                checkAccess(sd, exp.loc, sc, f);

                TypeFunction tf = cast(TypeFunction)f.type;
                if (!exp.arguments)
                    exp.arguments = new Expressions();
                if (functionParameters(exp.loc, sc, tf, null, exp.type, exp.arguments, f, &exp.type, &exp.argprefix))
                    return setError();

                exp.member = f.isCtorDeclaration();
                assert(exp.member);

                if (checkFrameAccess(exp.loc, sc, sd, sd.fields.dim))
                    return setError();
            }
            else
            {
                if (!exp.arguments)
                    exp.arguments = new Expressions();

                if (!sd.fit(exp.loc, sc, exp.arguments, tb))
                    return setError();

                if (!sd.fill(exp.loc, exp.arguments, false))
                    return setError();

                if (checkFrameAccess(exp.loc, sc, sd, exp.arguments ? exp.arguments.dim : 0))
                    return setError();

                /* Since a `new` allocation may escape, check each of the arguments for escaping
                 */
                if (global.params.vsafe)
                {
                    foreach (arg; *exp.arguments)
                    {
                        if (arg && checkNewEscape(sc, arg, false))
                            return setError();
                    }
                }
            }

            exp.type = exp.type.pointerTo();
        }
        else if (tb.ty == Tarray && nargs)
        {
            Type tn = tb.nextOf().baseElemOf();
            Dsymbol s = tn.toDsymbol(sc);
            AggregateDeclaration ad = s ? s.isAggregateDeclaration() : null;
            if (ad && ad.noDefaultCtor)
            {
                exp.error("default construction is disabled for type `%s`", tb.nextOf().toChars());
                return setError();
            }
            for (size_t i = 0; i < nargs; i++)
            {
                if (tb.ty != Tarray)
                {
                    exp.error("too many arguments for array");
                    return setError();
                }

                Expression arg = (*exp.arguments)[i];
                arg = resolveProperties(sc, arg);
                arg = arg.implicitCastTo(sc, Type.tsize_t);
                if (arg.op == TOK.error)
                    return setError();
                arg = arg.optimize(WANTvalue);
                if (arg.op == TOK.int64 && cast(sinteger_t)arg.toInteger() < 0)
                {
                    exp.error("negative array index `%s`", arg.toChars());
                    return setError();
                }
                (*exp.arguments)[i] = arg;
                tb = (cast(TypeDArray)tb).next.toBasetype();
            }
        }
        else if (tb.isscalar())
        {
            if (!nargs)
            {
            }
            else if (nargs == 1)
            {
                Expression e = (*exp.arguments)[0];
                e = e.implicitCastTo(sc, tb);
                (*exp.arguments)[0] = e;
            }
            else
            {
                exp.error("more than one argument for construction of `%s`", exp.type.toChars());
                return setError();
            }

            exp.type = exp.type.pointerTo();
        }
        else
        {
            exp.error("cannot create a `%s` with `new`", exp.type.toChars());
            return setError();
        }

        //printf("NewExp: '%s'\n", toChars());
        //printf("NewExp:type '%s'\n", type.toChars());
        semanticTypeInfo(sc, exp.type);

        if (newprefix)
        {
            result = Expression.combine(newprefix, exp);
            return;
        }
        result = exp;
    }

    override void visit(NewAnonClassExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("NewAnonClassExp::semantic() %s\n", e.toChars());
            //printf("thisexp = %p\n", thisexp);
            //printf("type: %s\n", type.toChars());
        }

        Expression d = new DeclarationExp(e.loc, e.cd);
        sc = sc.push(); // just create new scope
        sc.flags &= ~SCOPE.ctfe; // temporary stop CTFE
        d = d.expressionSemantic(sc);
        sc = sc.pop();

        if (!e.cd.errors && sc.intypeof && !sc.parent.inNonRoot())
        {
            ScopeDsymbol sds = sc.tinst ? cast(ScopeDsymbol)sc.tinst : sc._module;
            if (!sds.members)
                sds.members = new Dsymbols();
            sds.members.push(e.cd);
        }

        Expression n = new NewExp(e.loc, e.thisexp, e.newargs, e.cd.type, e.arguments);

        Expression c = new CommaExp(e.loc, d, n);
        result = c.expressionSemantic(sc);
    }

    override void visit(SymOffExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("SymOffExp::semantic('%s')\n", e.toChars());
        }
        //var.dsymbolSemantic(sc);
        if (!e.type)
            e.type = e.var.type.pointerTo();

        if (auto v = e.var.isVarDeclaration())
        {
            if (v.checkNestedReference(sc, e.loc))
                return setError();
        }
        else if (auto f = e.var.isFuncDeclaration())
        {
            if (f.checkNestedReference(sc, e.loc))
                return setError();
        }

        result = e;
    }

    override void visit(VarExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("VarExp::semantic(%s)\n", e.toChars());
        }

        auto vd = e.var.isVarDeclaration();
        auto fd = e.var.isFuncDeclaration();

        if (fd)
        {
            //printf("L%d fd = %s\n", __LINE__, f.toChars());
            if (!fd.functionSemantic())
                return setError();
        }

        if (!e.type)
            e.type = e.var.type;
        if (e.type && !e.type.deco)
        {
            auto decl = e.var.isDeclaration();
            if (decl)
                decl.inuse++;
            e.type = e.type.typeSemantic(e.loc, sc);
            if (decl)
                decl.inuse--;
        }

        /* Fix for 1161 doesn't work because it causes protection
         * problems when instantiating imported templates passing private
         * variables as alias template parameters.
         */
        //checkAccess(loc, sc, NULL, var);

        if (vd)
        {
            if (vd.checkNestedReference(sc, e.loc))
                return setError();

            // https://issues.dlang.org/show_bug.cgi?id=12025
            // If the variable is not actually used in runtime code,
            // the purity violation error is redundant.
            //checkPurity(sc, vd);
        }
        else if (fd)
        {
            // TODO: If fd isn't yet resolved its overload, the checkNestedReference
            // call would cause incorrect validation.
            // Maybe here should be moved in CallExp, or AddrExp for functions.
            if (fd.checkNestedReference(sc, e.loc))
                return setError();
        }
        else if (auto od = e.var.isOverDeclaration())
        {
            e.type = Type.tvoid; // ambiguous type?
        }

        result = e;
    }

    override void visit(FuncExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("FuncExp::semantic(%s)\n", exp.toChars());
            if (exp.fd.treq)
                printf("  treq = %s\n", exp.fd.treq.toChars());
        }

        if (exp.type)
        {
            result = exp;
            return;
        }

        Expression e = exp;
        uint olderrors;

        sc = sc.push(); // just create new scope
        sc.flags &= ~SCOPE.ctfe; // temporary stop CTFE
        sc.protection = Prot(Prot.Kind.public_); // https://issues.dlang.org/show_bug.cgi?id=12506

        /* fd.treq might be incomplete type,
            * so should not semantic it.
            * void foo(T)(T delegate(int) dg){}
            * foo(a=>a); // in IFTI, treq == T delegate(int)
            */
        //if (fd.treq)
        //    fd.treq = fd.treq.dsymbolSemantic(loc, sc);

        exp.genIdent(sc);

        // Set target of return type inference
        if (exp.fd.treq && !exp.fd.type.nextOf())
        {
            TypeFunction tfv = null;
            if (exp.fd.treq.ty == Tdelegate || (exp.fd.treq.ty == Tpointer && exp.fd.treq.nextOf().ty == Tfunction))
                tfv = cast(TypeFunction)exp.fd.treq.nextOf();
            if (tfv)
            {
                TypeFunction tfl = cast(TypeFunction)exp.fd.type;
                tfl.next = tfv.nextOf();
            }
        }

        //printf("td = %p, treq = %p\n", td, fd.treq);
        if (exp.td)
        {
            assert(exp.td.parameters && exp.td.parameters.dim);
            exp.td.dsymbolSemantic(sc);
            exp.type = Type.tvoid; // temporary type

            if (exp.fd.treq) // defer type determination
            {
                FuncExp fe;
                if (exp.matchType(exp.fd.treq, sc, &fe) > MATCH.nomatch)
                    e = fe;
                else
                    e = new ErrorExp();
            }
            goto Ldone;
        }

        olderrors = global.errors;
        exp.fd.dsymbolSemantic(sc);
        if (olderrors == global.errors)
        {
            exp.fd.semantic2(sc);
            if (olderrors == global.errors)
                exp.fd.semantic3(sc);
        }
        if (olderrors != global.errors)
        {
            if (exp.fd.type && exp.fd.type.ty == Tfunction && !exp.fd.type.nextOf())
                (cast(TypeFunction)exp.fd.type).next = Type.terror;
            e = new ErrorExp();
            goto Ldone;
        }

        // Type is a "delegate to" or "pointer to" the function literal
        if ((exp.fd.isNested() && exp.fd.tok == TOK.delegate_) || (exp.tok == TOK.reserved && exp.fd.treq && exp.fd.treq.ty == Tdelegate))
        {
            exp.type = new TypeDelegate(exp.fd.type);
            exp.type = exp.type.typeSemantic(exp.loc, sc);

            exp.fd.tok = TOK.delegate_;
        }
        else
        {
            exp.type = new TypePointer(exp.fd.type);
            exp.type = exp.type.typeSemantic(exp.loc, sc);
            //type = fd.type.pointerTo();

            /* A lambda expression deduced to function pointer might become
                * to a delegate literal implicitly.
                *
                *   auto foo(void function() fp) { return 1; }
                *   assert(foo({}) == 1);
                *
                * So, should keep fd.tok == TOKreserve if fd.treq == NULL.
                */
            if (exp.fd.treq && exp.fd.treq.ty == Tpointer)
            {
                // change to non-nested
                exp.fd.tok = TOK.function_;
                exp.fd.vthis = null;
            }
        }
        exp.fd.tookAddressOf++;

    Ldone:
        sc = sc.pop();
        result = e;
    }

    /**
     * Perform semantic analysis on function literals
     *
     * Test the following construct:
     * ---
     * (x, y, z) { return x + y + z; }(42, 84, 1992);
     * ---
     */
    Expression callExpSemantic(FuncExp exp, Scope* sc, Expressions* arguments)
    {
        if ((!exp.type || exp.type == Type.tvoid) && exp.td && arguments && arguments.dim)
        {
            for (size_t k = 0; k < arguments.dim; k++)
            {
                Expression checkarg = (*arguments)[k];
                if (checkarg.op == TOK.error)
                    return checkarg;
            }

            exp.genIdent(sc);

            assert(exp.td.parameters && exp.td.parameters.dim);
            exp.td.dsymbolSemantic(sc);

            TypeFunction tfl = cast(TypeFunction)exp.fd.type;
            size_t dim = tfl.parameterList.length;
            if (arguments.dim < dim)
            {
                // Default arguments are always typed, so they don't need inference.
                Parameter p = tfl.parameterList[arguments.dim];
                if (p.defaultArg)
                    dim = arguments.dim;
            }

            if ((tfl.parameterList.varargs == VarArg.none && arguments.dim > dim) ||
                arguments.dim < dim)
            {
                OutBuffer buf;
                foreach (idx, ref arg; *arguments)
                    buf.printf("%s%s", (idx ? ", ".ptr : "".ptr), arg.type.toChars());
                exp.error("function literal `%s%s` is not callable using argument types `(%s)`",
                          exp.fd.toChars(), parametersTypeToChars(tfl.parameterList),
                          buf.peekChars());
                exp.errorSupplemental("too %s arguments, expected `%d`, got `%d`",
                                      arguments.dim < dim ? "few".ptr : "many".ptr,
                                      cast(int)dim, cast(int)arguments.dim);
                return new ErrorExp();
            }

            auto tiargs = new Objects();
            tiargs.reserve(exp.td.parameters.dim);

            for (size_t i = 0; i < exp.td.parameters.dim; i++)
            {
                TemplateParameter tp = (*exp.td.parameters)[i];
                for (size_t u = 0; u < dim; u++)
                {
                    Parameter p = tfl.parameterList[u];
                    if (p.type.ty == Tident && (cast(TypeIdentifier)p.type).ident == tp.ident)
                    {
                        Expression e = (*arguments)[u];
                        tiargs.push(e.type);
                        u = dim; // break inner loop
                    }
                }
            }

            auto ti = new TemplateInstance(exp.loc, exp.td, tiargs);
            return (new ScopeExp(exp.loc, ti)).expressionSemantic(sc);
        }
        return exp.expressionSemantic(sc);
    }

    override void visit(CallExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("CallExp::semantic() %s\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return; // semantic() already run
        }

        Objects* tiargs = null; // initial list of template arguments
        Expression ethis = null;
        Type tthis = null;
        Expression e1org = exp.e1;

        if (exp.e1.op == TOK.comma)
        {
            /* Rewrite (a,b)(args) as (a,(b(args)))
             */
            auto ce = cast(CommaExp)exp.e1;
            exp.e1 = ce.e2;
            ce.e2 = exp;
            result = ce.expressionSemantic(sc);
            return;
        }
        if (exp.e1.op == TOK.delegate_)
        {
            DelegateExp de = cast(DelegateExp)exp.e1;
            exp.e1 = new DotVarExp(de.loc, de.e1, de.func, de.hasOverloads);
            visit(exp);
            return;
        }
        if (exp.e1.op == TOK.function_)
        {
            if (arrayExpressionSemantic(exp.arguments, sc) || preFunctionParameters(sc, exp.arguments))
                return setError();

            // Run e1 semantic even if arguments have any errors
            FuncExp fe = cast(FuncExp)exp.e1;
            exp.e1 = callExpSemantic(fe, sc, exp.arguments);
            if (exp.e1.op == TOK.error)
            {
                result = exp.e1;
                return;
            }
        }

        if (Expression ex = resolveUFCS(sc, exp))
        {
            result = ex;
            return;
        }

        /* This recognizes:
         *  foo!(tiargs)(funcargs)
         */
        if (exp.e1.op == TOK.scope_)
        {
            ScopeExp se = cast(ScopeExp)exp.e1;
            TemplateInstance ti = se.sds.isTemplateInstance();
            if (ti)
            {
                /* Attempt to instantiate ti. If that works, go with it.
                 * If not, go with partial explicit specialization.
                 */
                WithScopeSymbol withsym;
                if (!ti.findTempDecl(sc, &withsym) || !ti.semanticTiargs(sc))
                    return setError();
                if (withsym && withsym.withstate.wthis)
                {
                    exp.e1 = new VarExp(exp.e1.loc, withsym.withstate.wthis);
                    exp.e1 = new DotTemplateInstanceExp(exp.e1.loc, exp.e1, ti);
                    goto Ldotti;
                }
                if (ti.needsTypeInference(sc, 1))
                {
                    /* Go with partial explicit specialization
                     */
                    tiargs = ti.tiargs;
                    assert(ti.tempdecl);
                    if (TemplateDeclaration td = ti.tempdecl.isTemplateDeclaration())
                        exp.e1 = new TemplateExp(exp.loc, td);
                    else if (OverDeclaration od = ti.tempdecl.isOverDeclaration())
                        exp.e1 = new VarExp(exp.loc, od);
                    else
                        exp.e1 = new OverExp(exp.loc, ti.tempdecl.isOverloadSet());
                }
                else
                {
                    Expression e1x = exp.e1.expressionSemantic(sc);
                    if (e1x.op == TOK.error)
                    {
                        result = e1x;
                        return;
                    }
                    exp.e1 = e1x;
                }
            }
        }

        /* This recognizes:
         *  expr.foo!(tiargs)(funcargs)
         */
    Ldotti:
        if (exp.e1.op == TOK.dotTemplateInstance && !exp.e1.type)
        {
            DotTemplateInstanceExp se = cast(DotTemplateInstanceExp)exp.e1;
            TemplateInstance ti = se.ti;
            {
                /* Attempt to instantiate ti. If that works, go with it.
                 * If not, go with partial explicit specialization.
                 */
                if (!se.findTempDecl(sc) || !ti.semanticTiargs(sc))
                    return setError();
                if (ti.needsTypeInference(sc, 1))
                {
                    /* Go with partial explicit specialization
                     */
                    tiargs = ti.tiargs;
                    assert(ti.tempdecl);
                    if (TemplateDeclaration td = ti.tempdecl.isTemplateDeclaration())
                        exp.e1 = new DotTemplateExp(exp.loc, se.e1, td);
                    else if (OverDeclaration od = ti.tempdecl.isOverDeclaration())
                    {
                        exp.e1 = new DotVarExp(exp.loc, se.e1, od, true);
                    }
                    else
                        exp.e1 = new DotExp(exp.loc, se.e1, new OverExp(exp.loc, ti.tempdecl.isOverloadSet()));
                }
                else
                {
                    Expression e1x = exp.e1.expressionSemantic(sc);
                    if (e1x.op == TOK.error)
                    {
                        result = e1x;
                        return;
                    }
                    exp.e1 = e1x;
                }
            }
        }

    Lagain:
        //printf("Lagain: %s\n", toChars());
        exp.f = null;
        if (exp.e1.op == TOK.this_ || exp.e1.op == TOK.super_)
        {
            // semantic() run later for these
        }
        else
        {
            if (exp.e1.op == TOK.dotIdentifier)
            {
                DotIdExp die = cast(DotIdExp)exp.e1;
                exp.e1 = die.expressionSemantic(sc);
                /* Look for e1 having been rewritten to expr.opDispatch!(string)
                 * We handle such earlier, so go back.
                 * Note that in the rewrite, we carefully did not run semantic() on e1
                 */
                if (exp.e1.op == TOK.dotTemplateInstance && !exp.e1.type)
                {
                    goto Ldotti;
                }
            }
            else
            {
                __gshared int nest;
                if (++nest > global.recursionLimit)
                {
                    exp.error("recursive evaluation of `%s`", exp.toChars());
                    --nest;
                    return setError();
                }
                Expression ex = unaSemantic(exp, sc);
                --nest;
                if (ex)
                {
                    result = ex;
                    return;
                }
            }

            /* Look for e1 being a lazy parameter
             */
            if (exp.e1.op == TOK.variable)
            {
                VarExp ve = cast(VarExp)exp.e1;
                if (ve.var.storage_class & STC.lazy_)
                {
                    // lazy parameters can be called without violating purity and safety
                    Type tw = ve.var.type;
                    Type tc = ve.var.type.substWildTo(MODFlags.const_);
                    auto tf = new TypeFunction(ParameterList(), tc, LINK.d, STC.safe | STC.pure_);
                    (tf = cast(TypeFunction)tf.typeSemantic(exp.loc, sc)).next = tw; // hack for bug7757
                    auto t = new TypeDelegate(tf);
                    ve.type = t.typeSemantic(exp.loc, sc);
                }
                VarDeclaration v = ve.var.isVarDeclaration();
                if (v && ve.checkPurity(sc, v))
                    return setError();
            }

            if (exp.e1.op == TOK.symbolOffset && (cast(SymOffExp)exp.e1).hasOverloads)
            {
                SymOffExp se = cast(SymOffExp)exp.e1;
                exp.e1 = new VarExp(se.loc, se.var, true);
                exp.e1 = exp.e1.expressionSemantic(sc);
            }
            else if (exp.e1.op == TOK.dot)
            {
                DotExp de = cast(DotExp)exp.e1;

                if (de.e2.op == TOK.overloadSet)
                {
                    ethis = de.e1;
                    tthis = de.e1.type;
                    exp.e1 = de.e2;
                }
            }
            else if (exp.e1.op == TOK.star && exp.e1.type.ty == Tfunction)
            {
                // Rewrite (*fp)(arguments) to fp(arguments)
                exp.e1 = (cast(PtrExp)exp.e1).e1;
            }
        }

        Type t1 = exp.e1.type ? exp.e1.type.toBasetype() : null;

        if (exp.e1.op == TOK.error)
        {
            result = exp.e1;
            return;
        }
        if (arrayExpressionSemantic(exp.arguments, sc) || preFunctionParameters(sc, exp.arguments))
            return setError();

        // Check for call operator overload
        if (t1)
        {
            if (t1.ty == Tstruct)
            {
                auto sd = (cast(TypeStruct)t1).sym;
                sd.size(exp.loc); // Resolve forward references to construct object
                if (sd.sizeok != Sizeok.done)
                    return setError();
                if (!sd.ctor)
                    sd.ctor = sd.searchCtor();
                /* If `sd.ctor` is a generated copy constructor, this means that it
                   is the single constructor that this struct has. In order to not
                   disable default construction, the ctor is nullified. The side effect
                   of this is that the generated copy constructor cannot be called
                   explicitly, but that is ok, because when calling a constructor the
                   default constructor should have priority over the generated copy
                   constructor.
                */
                if (sd.ctor)
                {
                    auto ctor = sd.ctor.isCtorDeclaration();
                    if (ctor && ctor.isCpCtor && ctor.generated)
                        sd.ctor = null;
                }

                // First look for constructor
                if (exp.e1.op == TOK.type && sd.ctor)
                {
                    if (!sd.noDefaultCtor && !(exp.arguments && exp.arguments.dim))
                        goto Lx;

                    auto sle = new StructLiteralExp(exp.loc, sd, null, exp.e1.type);
                    if (!sd.fill(exp.loc, sle.elements, true))
                        return setError();
                    if (checkFrameAccess(exp.loc, sc, sd, sle.elements.dim))
                        return setError();

                    // https://issues.dlang.org/show_bug.cgi?id=14556
                    // Set concrete type to avoid further redundant semantic().
                    sle.type = exp.e1.type;

                    /* Constructor takes a mutable object, so don't use
                     * the immutable initializer symbol.
                     */
                    sle.useStaticInit = false;

                    Expression e = sle;
                    if (auto cf = sd.ctor.isCtorDeclaration())
                    {
                        e = new DotVarExp(exp.loc, e, cf, true);
                    }
                    else if (auto td = sd.ctor.isTemplateDeclaration())
                    {
                        e = new DotIdExp(exp.loc, e, td.ident);
                    }
                    else if (auto os = sd.ctor.isOverloadSet())
                    {
                        e = new DotExp(exp.loc, e, new OverExp(exp.loc, os));
                    }
                    else
                        assert(0);
                    e = new CallExp(exp.loc, e, exp.arguments);
                    e = e.expressionSemantic(sc);
                    result = e;
                    return;
                }
                // No constructor, look for overload of opCall
                if (search_function(sd, Id.call))
                    goto L1;
                // overload of opCall, therefore it's a call
                if (exp.e1.op != TOK.type)
                {
                    if (sd.aliasthis && exp.e1.type != exp.att1)
                    {
                        if (!exp.att1 && exp.e1.type.checkAliasThisRec())
                            exp.att1 = exp.e1.type;
                        exp.e1 = resolveAliasThis(sc, exp.e1);
                        goto Lagain;
                    }
                    exp.error("%s `%s` does not overload ()", sd.kind(), sd.toChars());
                    return setError();
                }

                /* It's a struct literal
                 */
            Lx:
                Expression e = new StructLiteralExp(exp.loc, sd, exp.arguments, exp.e1.type);
                e = e.expressionSemantic(sc);
                result = e;
                return;
            }
            else if (t1.ty == Tclass)
            {
            L1:
                // Rewrite as e1.call(arguments)
                Expression e = new DotIdExp(exp.loc, exp.e1, Id.call);
                e = new CallExp(exp.loc, e, exp.arguments);
                e = e.expressionSemantic(sc);
                result = e;
                return;
            }
            else if (exp.e1.op == TOK.type && t1.isscalar())
            {
                Expression e;

                // Make sure to use the the enum type itself rather than its
                // base type
                // https://issues.dlang.org/show_bug.cgi?id=16346
                if (exp.e1.type.ty == Tenum)
                {
                    t1 = exp.e1.type;
                }

                if (!exp.arguments || exp.arguments.dim == 0)
                {
                    e = t1.defaultInitLiteral(exp.loc);
                }
                else if (exp.arguments.dim == 1)
                {
                    e = (*exp.arguments)[0];
                    e = e.implicitCastTo(sc, t1);
                    e = new CastExp(exp.loc, e, t1);
                }
                else
                {
                    exp.error("more than one argument for construction of `%s`", t1.toChars());
                    return setError();
                }
                e = e.expressionSemantic(sc);
                result = e;
                return;
            }
        }

        static FuncDeclaration resolveOverloadSet(Loc loc, Scope* sc,
            OverloadSet os, Objects* tiargs, Type tthis, Expressions* arguments)
        {
            FuncDeclaration f = null;
            foreach (s; os.a)
            {
                if (tiargs && s.isFuncDeclaration())
                    continue;
                if (auto f2 = resolveFuncCall(loc, sc, s, tiargs, tthis, arguments, FuncResolveFlag.quiet))
                {
                    if (f2.errors)
                        return null;
                    if (f)
                    {
                        /* Error if match in more than one overload set,
                         * even if one is a 'better' match than the other.
                         */
                        ScopeDsymbol.multiplyDefined(loc, f, f2);
                    }
                    else
                        f = f2;
                }
            }
            if (!f)
                .error(loc, "no overload matches for `%s`", os.toChars());
            else if (f.errors)
                f = null;
            return f;
        }

        bool isSuper = false;
        if (exp.e1.op == TOK.dotVariable && t1.ty == Tfunction || exp.e1.op == TOK.dotTemplateDeclaration)
        {
            UnaExp ue = cast(UnaExp)exp.e1;

            Expression ue1 = ue.e1;
            Expression ue1old = ue1; // need for 'right this' check
            VarDeclaration v;
            if (ue1.op == TOK.variable && (v = (cast(VarExp)ue1).var.isVarDeclaration()) !is null && v.needThis())
            {
                ue.e1 = new TypeExp(ue1.loc, ue1.type);
                ue1 = null;
            }

            DotVarExp dve;
            DotTemplateExp dte;
            Dsymbol s;
            if (exp.e1.op == TOK.dotVariable)
            {
                dve = cast(DotVarExp)exp.e1;
                dte = null;
                s = dve.var;
                tiargs = null;
            }
            else
            {
                dve = null;
                dte = cast(DotTemplateExp)exp.e1;
                s = dte.td;
            }

            // Do overload resolution
            exp.f = resolveFuncCall(exp.loc, sc, s, tiargs, ue1 ? ue1.type : null, exp.arguments, FuncResolveFlag.standard);
            if (!exp.f || exp.f.errors || exp.f.type.ty == Terror)
                return setError();

            if (exp.f.interfaceVirtual)
            {
                /* Cast 'this' to the type of the interface, and replace f with the interface's equivalent
                 */
                auto b = exp.f.interfaceVirtual;
                auto ad2 = b.sym;
                ue.e1 = ue.e1.castTo(sc, ad2.type.addMod(ue.e1.type.mod));
                ue.e1 = ue.e1.expressionSemantic(sc);
                ue1 = ue.e1;
                auto vi = exp.f.findVtblIndex(&ad2.vtbl, cast(int)ad2.vtbl.dim);
                assert(vi >= 0);
                exp.f = ad2.vtbl[vi].isFuncDeclaration();
                assert(exp.f);
            }
            if (exp.f.needThis())
            {
                AggregateDeclaration ad = exp.f.toParentLocal().isAggregateDeclaration();
                ue.e1 = getRightThis(exp.loc, sc, ad, ue.e1, exp.f);
                if (ue.e1.op == TOK.error)
                {
                    result = ue.e1;
                    return;
                }
                ethis = ue.e1;
                tthis = ue.e1.type;
                if (!(exp.f.type.ty == Tfunction && (cast(TypeFunction)exp.f.type).isscope))
                {
                    if (global.params.vsafe && checkParamArgumentEscape(sc, exp.f, null, ethis, false, false))
                        return setError();
                }
            }

            /* Cannot call public functions from inside invariant
             * (because then the invariant would have infinite recursion)
             */
            if (sc.func && sc.func.isInvariantDeclaration() && ue.e1.op == TOK.this_ && exp.f.addPostInvariant())
            {
                exp.error("cannot call `public`/`export` function `%s` from invariant", exp.f.toChars());
                return setError();
            }

            checkFunctionAttributes(exp, sc, exp.f);
            checkAccess(exp.loc, sc, ue.e1, exp.f);
            if (!exp.f.needThis())
            {
                exp.e1 = Expression.combine(ue.e1, new VarExp(exp.loc, exp.f, false));
            }
            else
            {
                if (ue1old.checkRightThis(sc))
                    return setError();
                if (exp.e1.op == TOK.dotVariable)
                {
                    dve.var = exp.f;
                    exp.e1.type = exp.f.type;
                }
                else
                {
                    exp.e1 = new DotVarExp(exp.loc, dte.e1, exp.f, false);
                    exp.e1 = exp.e1.expressionSemantic(sc);
                    if (exp.e1.op == TOK.error)
                        return setError();
                    ue = cast(UnaExp)exp.e1;
                }
                version (none)
                {
                    printf("ue.e1 = %s\n", ue.e1.toChars());
                    printf("f = %s\n", exp.f.toChars());
                    printf("t = %s\n", t.toChars());
                    printf("e1 = %s\n", exp.e1.toChars());
                    printf("e1.type = %s\n", exp.e1.type.toChars());
                }

                // See if we need to adjust the 'this' pointer
                AggregateDeclaration ad = exp.f.isThis();
                ClassDeclaration cd = ue.e1.type.isClassHandle();
                if (ad && cd && ad.isClassDeclaration())
                {
                    if (ue.e1.op == TOK.dotType)
                    {
                        ue.e1 = (cast(DotTypeExp)ue.e1).e1;
                        exp.directcall = true;
                    }
                    else if (ue.e1.op == TOK.super_)
                        exp.directcall = true;
                    else if ((cd.storage_class & STC.final_) != 0) // https://issues.dlang.org/show_bug.cgi?id=14211
                        exp.directcall = true;

                    if (ad != cd)
                    {
                        ue.e1 = ue.e1.castTo(sc, ad.type.addMod(ue.e1.type.mod));
                        ue.e1 = ue.e1.expressionSemantic(sc);
                    }
                }
            }
            // If we've got a pointer to a function then deference it
            // https://issues.dlang.org/show_bug.cgi?id=16483
            if (exp.e1.type.ty == Tpointer && exp.e1.type.nextOf().ty == Tfunction)
            {
                Expression e = new PtrExp(exp.loc, exp.e1);
                e.type = exp.e1.type.nextOf();
                exp.e1 = e;
            }
            t1 = exp.e1.type;
        }
        else if (exp.e1.op == TOK.super_ || exp.e1.op == TOK.this_)
        {
            auto ad = sc.func ? sc.func.isThis() : null;
            auto cd = ad ? ad.isClassDeclaration() : null;

            isSuper = exp.e1.op == TOK.super_;
            if (isSuper)
            {
                // Base class constructor call
                if (!cd || !cd.baseClass || !sc.func.isCtorDeclaration())
                {
                    exp.error("super class constructor call must be in a constructor");
                    return setError();
                }
                if (!cd.baseClass.ctor)
                {
                    exp.error("no super class constructor for `%s`", cd.baseClass.toChars());
                    return setError();
                }
            }
            else
            {
                // `this` call expression must be inside a
                // constructor
                if (!ad || !sc.func.isCtorDeclaration())
                {
                    exp.error("constructor call must be in a constructor");
                    return setError();
                }

                // https://issues.dlang.org/show_bug.cgi?id=18719
                // If `exp` is a call expression to another constructor
                // then it means that all struct/class fields will be
                // initialized after this call.
                foreach (ref field; sc.ctorflow.fieldinit)
                {
                    field.csx |= CSX.this_ctor;
                }
            }

            if (!sc.intypeof && !(sc.ctorflow.callSuper & CSX.halt))
            {
                if (sc.inLoop || sc.ctorflow.callSuper & CSX.label)
                    exp.error("constructor calls not allowed in loops or after labels");
                if (sc.ctorflow.callSuper & (CSX.super_ctor | CSX.this_ctor))
                    exp.error("multiple constructor calls");
                if ((sc.ctorflow.callSuper & CSX.return_) && !(sc.ctorflow.callSuper & CSX.any_ctor))
                    exp.error("an earlier `return` statement skips constructor");
                sc.ctorflow.callSuper |= CSX.any_ctor | (isSuper ? CSX.super_ctor : CSX.this_ctor);
            }

            tthis = ad.type.addMod(sc.func.type.mod);
            auto ctor = isSuper ? cd.baseClass.ctor : ad.ctor;
            if (auto os = ctor.isOverloadSet())
                exp.f = resolveOverloadSet(exp.loc, sc, os, null, tthis, exp.arguments);
            else
                exp.f = resolveFuncCall(exp.loc, sc, ctor, null, tthis, exp.arguments, FuncResolveFlag.standard);

            if (!exp.f || exp.f.errors)
                return setError();

            checkFunctionAttributes(exp, sc, exp.f);
            checkAccess(exp.loc, sc, null, exp.f);

            exp.e1 = new DotVarExp(exp.e1.loc, exp.e1, exp.f, false);
            exp.e1 = exp.e1.expressionSemantic(sc);
            t1 = exp.e1.type;

            // BUG: this should really be done by checking the static
            // call graph
            if (exp.f == sc.func)
            {
                exp.error("cyclic constructor call");
                return setError();
            }
        }
        else if (exp.e1.op == TOK.overloadSet)
        {
            auto os = (cast(OverExp)exp.e1).vars;
            exp.f = resolveOverloadSet(exp.loc, sc, os, tiargs, tthis, exp.arguments);
            if (!exp.f)
                return setError();
            if (ethis)
                exp.e1 = new DotVarExp(exp.loc, ethis, exp.f, false);
            else
                exp.e1 = new VarExp(exp.loc, exp.f, false);
            goto Lagain;
        }
        else if (!t1)
        {
            exp.error("function expected before `()`, not `%s`", exp.e1.toChars());
            return setError();
        }
        else if (t1.ty == Terror)
        {
            return setError();
        }
        else if (t1.ty != Tfunction)
        {
            TypeFunction tf;
            const(char)* p;
            Dsymbol s;
            exp.f = null;
            if (exp.e1.op == TOK.function_)
            {
                // function literal that direct called is always inferred.
                assert((cast(FuncExp)exp.e1).fd);
                exp.f = (cast(FuncExp)exp.e1).fd;
                tf = cast(TypeFunction)exp.f.type;
                p = "function literal";
            }
            else if (t1.ty == Tdelegate)
            {
                TypeDelegate td = cast(TypeDelegate)t1;
                assert(td.next.ty == Tfunction);
                tf = cast(TypeFunction)td.next;
                p = "delegate";
            }
            else if (t1.ty == Tpointer && (cast(TypePointer)t1).next.ty == Tfunction)
            {
                tf = cast(TypeFunction)(cast(TypePointer)t1).next;
                p = "function pointer";
            }
            else if (exp.e1.op == TOK.dotVariable && (cast(DotVarExp)exp.e1).var.isOverDeclaration())
            {
                DotVarExp dve = cast(DotVarExp)exp.e1;
                exp.f = resolveFuncCall(exp.loc, sc, dve.var, tiargs, dve.e1.type, exp.arguments, FuncResolveFlag.overloadOnly);
                if (!exp.f)
                    return setError();
                if (exp.f.needThis())
                {
                    dve.var = exp.f;
                    dve.type = exp.f.type;
                    dve.hasOverloads = false;
                    goto Lagain;
                }
                exp.e1 = new VarExp(dve.loc, exp.f, false);
                Expression e = new CommaExp(exp.loc, dve.e1, exp);
                result = e.expressionSemantic(sc);
                return;
            }
            else if (exp.e1.op == TOK.variable && (cast(VarExp)exp.e1).var.isOverDeclaration())
            {
                s = (cast(VarExp)exp.e1).var;
                goto L2;
            }
            else if (exp.e1.op == TOK.template_)
            {
                s = (cast(TemplateExp)exp.e1).td;
            L2:
                exp.f = resolveFuncCall(exp.loc, sc, s, tiargs, null, exp.arguments, FuncResolveFlag.standard);
                if (!exp.f || exp.f.errors)
                    return setError();
                if (exp.f.needThis())
                {
                    if (hasThis(sc))
                    {
                        // Supply an implicit 'this', as in
                        //    this.ident
                        exp.e1 = new DotVarExp(exp.loc, (new ThisExp(exp.loc)).expressionSemantic(sc), exp.f, false);
                        goto Lagain;
                    }
                    else if (isNeedThisScope(sc, exp.f))
                    {
                        exp.error("need `this` for `%s` of type `%s`", exp.f.toChars(), exp.f.type.toChars());
                        return setError();
                    }
                }
                exp.e1 = new VarExp(exp.e1.loc, exp.f, false);
                goto Lagain;
            }
            else
            {
                exp.error("function expected before `()`, not `%s` of type `%s`", exp.e1.toChars(), exp.e1.type.toChars());
                return setError();
            }

            const(char)* failMessage;
            Expression[] fargs = exp.arguments ? (*exp.arguments)[] : null;
            if (!tf.callMatch(null, fargs, 0, &failMessage, sc))
            {
                OutBuffer buf;
                buf.writeByte('(');
                argExpTypesToCBuffer(&buf, exp.arguments);
                buf.writeByte(')');
                if (tthis)
                    tthis.modToBuffer(&buf);

                //printf("tf = %s, args = %s\n", tf.deco, (*arguments)[0].type.deco);
                .error(exp.loc, "%s `%s%s` is not callable using argument types `%s`",
                    p, exp.e1.toChars(), parametersTypeToChars(tf.parameterList), buf.peekChars());
                if (failMessage)
                    errorSupplemental(exp.loc, "%s", failMessage);
                return setError();
            }
            // Purity and safety check should run after testing arguments matching
            if (exp.f)
            {
                exp.checkPurity(sc, exp.f);
                exp.checkSafety(sc, exp.f);
                exp.checkNogc(sc, exp.f);
                if (exp.f.checkNestedReference(sc, exp.loc))
                    return setError();
            }
            else if (sc.func && sc.intypeof != 1 && !(sc.flags & (SCOPE.ctfe | SCOPE.debug_)))
            {
                bool err = false;
                if (!tf.purity && sc.func.setImpure())
                {
                    exp.error("`pure` %s `%s` cannot call impure %s `%s`",
                        sc.func.kind(), sc.func.toPrettyChars(), p, exp.e1.toChars());
                    err = true;
                }
                if (!tf.isnogc && sc.func.setGC())
                {
                    exp.error("`@nogc` %s `%s` cannot call non-@nogc %s `%s`",
                        sc.func.kind(), sc.func.toPrettyChars(), p, exp.e1.toChars());
                    err = true;
                }
                if (tf.trust <= TRUST.system && sc.func.setUnsafe())
                {
                    exp.error("`@safe` %s `%s` cannot call `@system` %s `%s`",
                        sc.func.kind(), sc.func.toPrettyChars(), p, exp.e1.toChars());
                    err = true;
                }
                if (err)
                    return setError();
            }

            if (t1.ty == Tpointer)
            {
                Expression e = new PtrExp(exp.loc, exp.e1);
                e.type = tf;
                exp.e1 = e;
            }
            t1 = tf;
        }
        else if (exp.e1.op == TOK.variable)
        {
            // Do overload resolution
            VarExp ve = cast(VarExp)exp.e1;

            exp.f = ve.var.isFuncDeclaration();
            assert(exp.f);
            tiargs = null;

            if (exp.f.overnext)
                exp.f = resolveFuncCall(exp.loc, sc, exp.f, tiargs, null, exp.arguments, FuncResolveFlag.overloadOnly);
            else
            {
                exp.f = exp.f.toAliasFunc();
                TypeFunction tf = cast(TypeFunction)exp.f.type;
                const(char)* failMessage;
                Expression[] fargs = exp.arguments ? (*exp.arguments)[] : null;
                if (!tf.callMatch(null, fargs, 0, &failMessage, sc))
                {
                    OutBuffer buf;
                    buf.writeByte('(');
                    argExpTypesToCBuffer(&buf, exp.arguments);
                    buf.writeByte(')');

                    //printf("tf = %s, args = %s\n", tf.deco, (*arguments)[0].type.deco);
                    .error(exp.loc, "%s `%s%s` is not callable using argument types `%s`",
                        exp.f.kind(), exp.f.toPrettyChars(), parametersTypeToChars(tf.parameterList), buf.peekChars());
                    if (failMessage)
                        errorSupplemental(exp.loc, "%s", failMessage);
                    exp.f = null;
                }
            }
            if (!exp.f || exp.f.errors)
                return setError();

            if (exp.f.needThis())
            {
                // Change the ancestor lambdas to delegate before hasThis(sc) call.
                if (exp.f.checkNestedReference(sc, exp.loc))
                    return setError();

                if (hasThis(sc))
                {
                    // Supply an implicit 'this', as in
                    //    this.ident
                    exp.e1 = new DotVarExp(exp.loc, (new ThisExp(exp.loc)).expressionSemantic(sc), ve.var);
                    // Note: we cannot use f directly, because further overload resolution
                    // through the supplied 'this' may cause different result.
                    goto Lagain;
                }
                else if (isNeedThisScope(sc, exp.f))
                {
                    exp.error("need `this` for `%s` of type `%s`", exp.f.toChars(), exp.f.type.toChars());
                    return setError();
                }
            }

            checkFunctionAttributes(exp, sc, exp.f);
            checkAccess(exp.loc, sc, null, exp.f);
            if (exp.f.checkNestedReference(sc, exp.loc))
                return setError();

            ethis = null;
            tthis = null;

            if (ve.hasOverloads)
            {
                exp.e1 = new VarExp(ve.loc, exp.f, false);
                exp.e1.type = exp.f.type;
            }
            t1 = exp.f.type;
        }
        assert(t1.ty == Tfunction);

        Expression argprefix;
        if (!exp.arguments)
            exp.arguments = new Expressions();
        if (functionParameters(exp.loc, sc, cast(TypeFunction)t1, ethis, tthis, exp.arguments, exp.f, &exp.type, &argprefix))
            return setError();

        if (!exp.type)
        {
            exp.e1 = e1org; // https://issues.dlang.org/show_bug.cgi?id=10922
                        // avoid recursive expression printing
            exp.error("forward reference to inferred return type of function call `%s`", exp.toChars());
            return setError();
        }

        if (exp.f && exp.f.tintro)
        {
            Type t = exp.type;
            int offset = 0;
            TypeFunction tf = cast(TypeFunction)exp.f.tintro;
            if (tf.next.isBaseOf(t, &offset) && offset)
            {
                exp.type = tf.next;
                result = Expression.combine(argprefix, exp.castTo(sc, t));
                return;
            }
        }

        // Handle the case of a direct lambda call
        if (exp.f && exp.f.isFuncLiteralDeclaration() && sc.func && !sc.intypeof)
        {
            exp.f.tookAddressOf = 0;
        }

        result = Expression.combine(argprefix, exp);

        if (isSuper)
        {
            auto ad = sc.func ? sc.func.isThis() : null;
            auto cd = ad ? ad.isClassDeclaration() : null;
            if (cd && cd.classKind == ClassKind.cpp && exp.f && !exp.f.fbody)
            {
                // if super is defined in C++, it sets the vtable pointer to the base class
                // so we have to restore it, but still return 'this' from super() call:
                // (auto __vptrTmp = this.__vptr, auto __superTmp = super()), (this.__vptr = __vptrTmp, __superTmp)
                Loc loc = exp.loc;

                auto vptr = new DotIdExp(loc, new ThisExp(loc), Id.__vptr);
                auto vptrTmpDecl = copyToTemp(0, "__vptrTmp", vptr);
                auto declareVptrTmp = new DeclarationExp(loc, vptrTmpDecl);

                auto superTmpDecl = copyToTemp(0, "__superTmp", result);
                auto declareSuperTmp = new DeclarationExp(loc, superTmpDecl);

                auto declareTmps = new CommaExp(loc, declareVptrTmp, declareSuperTmp);

                auto restoreVptr = new AssignExp(loc, vptr.syntaxCopy(), new VarExp(loc, vptrTmpDecl));

                Expression e = new CommaExp(loc, declareTmps, new CommaExp(loc, restoreVptr, new VarExp(loc, superTmpDecl)));
                result = e.expressionSemantic(sc);
            }
        }

        // declare dual-context container
        if (exp.f && exp.f.isThis2 && !sc.intypeof && sc.func)
        {
            // check access to second `this`
            if (AggregateDeclaration ad2 = exp.f.isMember2())
            {
                Expression te = new ThisExp(exp.loc).expressionSemantic(sc);
                if (te.op != TOK.error)
                    te = getRightThis(exp.loc, sc, ad2, te, exp.f);
                if (te.op == TOK.error)
                {
                    exp.error("need `this` of type `%s` to call function `%s`", ad2.toChars(), exp.f.toChars());
                    return setError();
                }
            }
            VarDeclaration vthis2 = makeThis2Argument(exp.loc, sc, exp.f);            exp.vthis2 = vthis2;
            Expression de = new DeclarationExp(exp.loc, vthis2);
            result = Expression.combine(de, result);
            result = result.expressionSemantic(sc);
        }
    }

    override void visit(DeclarationExp e)
    {
        if (e.type)
        {
            result = e;
            return;
        }
        static if (LOGSEMANTIC)
        {
            printf("DeclarationExp::semantic() %s\n", e.toChars());
        }

        uint olderrors = global.errors;

        /* This is here to support extern(linkage) declaration,
         * where the extern(linkage) winds up being an AttribDeclaration
         * wrapper.
         */
        Dsymbol s = e.declaration;

        while (1)
        {
            AttribDeclaration ad = s.isAttribDeclaration();
            if (ad)
            {
                if (ad.decl && ad.decl.dim == 1)
                {
                    s = (*ad.decl)[0];
                    continue;
                }
            }
            break;
        }

        VarDeclaration v = s.isVarDeclaration();
        if (v)
        {
            // Do semantic() on initializer first, so:
            //      int a = a;
            // will be illegal.
            e.declaration.dsymbolSemantic(sc);
            s.parent = sc.parent;
        }

        //printf("inserting '%s' %p into sc = %p\n", s.toChars(), s, sc);
        // Insert into both local scope and function scope.
        // Must be unique in both.
        if (s.ident)
        {
            if (!sc.insert(s))
            {
                e.error("declaration `%s` is already defined", s.toPrettyChars());
                return setError();
            }
            else if (sc.func)
            {
                // https://issues.dlang.org/show_bug.cgi?id=11720
                // include Dataseg variables
                if ((s.isFuncDeclaration() ||
                     s.isAggregateDeclaration() ||
                     s.isEnumDeclaration() ||
                     v && v.isDataseg()) && !sc.func.localsymtab.insert(s))
                {
                    // https://issues.dlang.org/show_bug.cgi?id=18266
                    // set parent so that type semantic does not assert
                    s.parent = sc.parent;
                    Dsymbol originalSymbol = sc.func.localsymtab.lookup(s.ident);
                    assert(originalSymbol);
                    e.error("declaration `%s` is already defined in another scope in `%s` at line `%d`", s.toPrettyChars(), sc.func.toChars(), originalSymbol.loc.linnum);
                    return setError();
                }
                else
                {
                    // Disallow shadowing
                    for (Scope* scx = sc.enclosing; scx && (scx.func == sc.func || (scx.func && sc.func.fes)); scx = scx.enclosing)
                    {
                        Dsymbol s2;
                        if (scx.scopesym && scx.scopesym.symtab && (s2 = scx.scopesym.symtab.lookup(s.ident)) !is null && s != s2)
                        {
                            // allow STC.local symbols to be shadowed
                            // TODO: not really an optimal design
                            auto decl = s2.isDeclaration();
                            if (!decl || !(decl.storage_class & STC.local))
                            {
                                if (sc.func.fes)
                                {
                                    e.deprecation("%s `%s` is shadowing %s `%s`. Rename the `foreach` variable.", s.kind(), s.ident.toChars(), s2.kind(), s2.toPrettyChars());
                                }
                                else
                                {
                                    e.error("%s `%s` is shadowing %s `%s`", s.kind(), s.ident.toChars(), s2.kind(), s2.toPrettyChars());
                                    return setError();
                                }
                            }
                        }
                    }
                }
            }
        }
        if (!s.isVarDeclaration())
        {
            Scope* sc2 = sc;
            if (sc2.stc & (STC.pure_ | STC.nothrow_ | STC.nogc))
                sc2 = sc.push();
            sc2.stc &= ~(STC.pure_ | STC.nothrow_ | STC.nogc);
            e.declaration.dsymbolSemantic(sc2);
            if (sc2 != sc)
                sc2.pop();
            s.parent = sc.parent;
        }
        if (global.errors == olderrors)
        {
            e.declaration.semantic2(sc);
            if (global.errors == olderrors)
            {
                e.declaration.semantic3(sc);
            }
        }
        // todo: error in declaration should be propagated.

        e.type = Type.tvoid;
        result = e;
    }

    override void visit(TypeidExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("TypeidExp::semantic() %s\n", exp.toChars());
        }
        Type ta = isType(exp.obj);
        Expression ea = isExpression(exp.obj);
        Dsymbol sa = isDsymbol(exp.obj);
        //printf("ta %p ea %p sa %p\n", ta, ea, sa);

        if (ta)
        {
            dmd.typesem.resolve(ta, exp.loc, sc, &ea, &ta, &sa, true);
        }

        if (ea)
        {
            if (auto sym = getDsymbol(ea))
                ea = symbolToExp(sym, exp.loc, sc, false);
            else
                ea = ea.expressionSemantic(sc);
            ea = resolveProperties(sc, ea);
            ta = ea.type;
            if (ea.op == TOK.type)
                ea = null;
        }

        if (!ta)
        {
            //printf("ta %p ea %p sa %p\n", ta, ea, sa);
            exp.error("no type for `typeid(%s)`", ea ? ea.toChars() : (sa ? sa.toChars() : ""));
            return setError();
        }

        if (global.params.vcomplex)
            ta.checkComplexTransition(exp.loc, sc);

        Expression e;
        auto tb = ta.toBasetype();
        if (ea && tb.ty == Tclass)
        {
            if (tb.toDsymbol(sc).isClassDeclaration().classKind == ClassKind.cpp)
            {
                error(exp.loc, "Runtime type information is not supported for `extern(C++)` classes");
                e = new ErrorExp();
            }
            else if (!Type.typeinfoclass)
            {
                error(exp.loc, "`object.TypeInfo_Class` could not be found, but is implicitly used");
                e = new ErrorExp();
            }
            else
            {
                /* Get the dynamic type, which is .classinfo
                */
                ea = ea.expressionSemantic(sc);
                e = new TypeidExp(ea.loc, ea);
                e.type = Type.typeinfoclass.type;
            }
        }
        else if (ta.ty == Terror)
        {
            e = new ErrorExp();
        }
        else
        {
            // Handle this in the glue layer
            e = new TypeidExp(exp.loc, ta);
            e.type = getTypeInfoType(exp.loc, ta, sc);

            semanticTypeInfo(sc, ta);

            if (ea)
            {
                e = new CommaExp(exp.loc, ea, e); // execute ea
                e = e.expressionSemantic(sc);
            }
        }
        result = e;
    }

    override void visit(TraitsExp e)
    {
        result = semanticTraits(e, sc);
    }

    override void visit(HaltExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("HaltExp::semantic()\n");
        }
        e.type = Type.tvoid;
        result = e;
    }

    override void visit(IsExp e)
    {
        /* is(targ id tok tspec)
         * is(targ id :  tok2)
         * is(targ id == tok2)
         */
        static if (LOGSEMANTIC)
        {
            printf("IsExp::semantic(%s)\n", e.toChars());
        }
        if (e.id && !(sc.flags & SCOPE.condition))
        {
            e.error("can only declare type aliases within `static if` conditionals or `static assert`s");
            return setError();
        }

        Type tded = null;
        if (e.tok2 == TOK.package_ || e.tok2 == TOK.module_) // These is() expressions are special because they can work on modules, not just types.
        {
            const oldErrors = global.startGagging();
            Dsymbol sym = e.targ.toDsymbol(sc);
            global.endGagging(oldErrors);

            if (sym is null)
                goto Lno;
            Package p = resolveIsPackage(sym);
            if (p is null)
                goto Lno;
            if (e.tok2 == TOK.package_ && p.isModule()) // Note that isModule() will return null for package modules because they're not actually instances of Module.
                goto Lno;
            else if(e.tok2 == TOK.module_ && !(p.isModule() || p.isPackageMod()))
                goto Lno;
            tded = e.targ;
            goto Lyes;
        }

        {
            Scope* sc2 = sc.copy(); // keep sc.flags
            sc2.tinst = null;
            sc2.minst = null;
            sc2.flags |= SCOPE.fullinst;
            Type t = e.targ.trySemantic(e.loc, sc2);
            sc2.pop();
            if (!t) // errors, so condition is false
                goto Lno;
            e.targ = t;
        }

        if (e.tok2 != TOK.reserved)
        {
            switch (e.tok2)
            {
            case TOK.struct_:
                if (e.targ.ty != Tstruct)
                    goto Lno;
                if ((cast(TypeStruct)e.targ).sym.isUnionDeclaration())
                    goto Lno;
                tded = e.targ;
                break;

            case TOK.union_:
                if (e.targ.ty != Tstruct)
                    goto Lno;
                if (!(cast(TypeStruct)e.targ).sym.isUnionDeclaration())
                    goto Lno;
                tded = e.targ;
                break;

            case TOK.class_:
                if (e.targ.ty != Tclass)
                    goto Lno;
                if ((cast(TypeClass)e.targ).sym.isInterfaceDeclaration())
                    goto Lno;
                tded = e.targ;
                break;

            case TOK.interface_:
                if (e.targ.ty != Tclass)
                    goto Lno;
                if (!(cast(TypeClass)e.targ).sym.isInterfaceDeclaration())
                    goto Lno;
                tded = e.targ;
                break;

            case TOK.const_:
                if (!e.targ.isConst())
                    goto Lno;
                tded = e.targ;
                break;

            case TOK.immutable_:
                if (!e.targ.isImmutable())
                    goto Lno;
                tded = e.targ;
                break;

            case TOK.shared_:
                if (!e.targ.isShared())
                    goto Lno;
                tded = e.targ;
                break;

            case TOK.inout_:
                if (!e.targ.isWild())
                    goto Lno;
                tded = e.targ;
                break;

            case TOK.super_:
                // If class or interface, get the base class and interfaces
                if (e.targ.ty != Tclass)
                    goto Lno;
                else
                {
                    ClassDeclaration cd = (cast(TypeClass)e.targ).sym;
                    auto args = new Parameters();
                    args.reserve(cd.baseclasses.dim);
                    if (cd.semanticRun < PASS.semanticdone)
                        cd.dsymbolSemantic(null);
                    for (size_t i = 0; i < cd.baseclasses.dim; i++)
                    {
                        BaseClass* b = (*cd.baseclasses)[i];
                        args.push(new Parameter(STC.in_, b.type, null, null, null));
                    }
                    tded = new TypeTuple(args);
                }
                break;

            case TOK.enum_:
                if (e.targ.ty != Tenum)
                    goto Lno;
                if (e.id)
                    tded = (cast(TypeEnum)e.targ).sym.getMemtype(e.loc);
                else
                    tded = e.targ;

                if (tded.ty == Terror)
                    return setError();
                break;

            case TOK.delegate_:
                if (e.targ.ty != Tdelegate)
                    goto Lno;
                tded = (cast(TypeDelegate)e.targ).next; // the underlying function type
                break;

            case TOK.function_:
            case TOK.parameters:
                {
                    if (e.targ.ty != Tfunction)
                        goto Lno;
                    tded = e.targ;

                    /* Generate tuple from function parameter types.
                     */
                    assert(tded.ty == Tfunction);
                    auto tdedf = tded.isTypeFunction();
                    size_t dim = tdedf.parameterList.length;
                    auto args = new Parameters();
                    args.reserve(dim);
                    for (size_t i = 0; i < dim; i++)
                    {
                        Parameter arg = tdedf.parameterList[i];
                        assert(arg && arg.type);
                        /* If one of the default arguments was an error,
                           don't return an invalid tuple
                         */
                        if (e.tok2 == TOK.parameters && arg.defaultArg && arg.defaultArg.op == TOK.error)
                            return setError();
                        args.push(new Parameter(arg.storageClass, arg.type, (e.tok2 == TOK.parameters) ? arg.ident : null, (e.tok2 == TOK.parameters) ? arg.defaultArg : null, arg.userAttribDecl));
                    }
                    tded = new TypeTuple(args);
                    break;
                }
            case TOK.return_:
                /* Get the 'return type' for the function,
                 * delegate, or pointer to function.
                 */
                if (e.targ.ty == Tfunction)
                    tded = (cast(TypeFunction)e.targ).next;
                else if (e.targ.ty == Tdelegate)
                {
                    tded = (cast(TypeDelegate)e.targ).next;
                    tded = (cast(TypeFunction)tded).next;
                }
                else if (e.targ.ty == Tpointer && (cast(TypePointer)e.targ).next.ty == Tfunction)
                {
                    tded = (cast(TypePointer)e.targ).next;
                    tded = (cast(TypeFunction)tded).next;
                }
                else
                    goto Lno;
                break;

            case TOK.argumentTypes:
                /* Generate a type tuple of the equivalent types used to determine if a
                 * function argument of this type can be passed in registers.
                 * The results of this are highly platform dependent, and intended
                 * primarly for use in implementing va_arg().
                 */
                tded = target.toArgTypes(e.targ);
                if (!tded)
                    goto Lno;
                // not valid for a parameter
                break;

            case TOK.vector:
                if (e.targ.ty != Tvector)
                    goto Lno;
                tded = (cast(TypeVector)e.targ).basetype;
                break;

            default:
                assert(0);
            }

            // https://issues.dlang.org/show_bug.cgi?id=18753
            if (tded)
                goto Lyes;
            goto Lno;
        }
        else if (e.tspec && !e.id && !(e.parameters && e.parameters.dim))
        {
            /* Evaluate to true if targ matches tspec
             * is(targ == tspec)
             * is(targ : tspec)
             */
            e.tspec = e.tspec.typeSemantic(e.loc, sc);
            //printf("targ  = %s, %s\n", targ.toChars(), targ.deco);
            //printf("tspec = %s, %s\n", tspec.toChars(), tspec.deco);

            if (e.tok == TOK.colon)
            {
                if (e.targ.implicitConvTo(e.tspec))
                    goto Lyes;
                else
                    goto Lno;
            }
            else /* == */
            {
                if (e.targ.equals(e.tspec))
                    goto Lyes;
                else
                    goto Lno;
            }
        }
        else if (e.tspec)
        {
            /* Evaluate to true if targ matches tspec.
             * If true, declare id as an alias for the specialized type.
             * is(targ == tspec, tpl)
             * is(targ : tspec, tpl)
             * is(targ id == tspec)
             * is(targ id : tspec)
             * is(targ id == tspec, tpl)
             * is(targ id : tspec, tpl)
             */
            Identifier tid = e.id ? e.id : Identifier.generateId("__isexp_id");
            e.parameters.insert(0, new TemplateTypeParameter(e.loc, tid, null, null));

            Objects dedtypes = Objects(e.parameters.dim);
            dedtypes.zero();

            MATCH m = deduceType(e.targ, sc, e.tspec, e.parameters, &dedtypes, null, 0, e.tok == TOK.equal);
            //printf("targ: %s\n", targ.toChars());
            //printf("tspec: %s\n", tspec.toChars());
            if (m <= MATCH.nomatch || (m != MATCH.exact && e.tok == TOK.equal))
            {
                goto Lno;
            }
            else
            {
                tded = cast(Type)dedtypes[0];
                if (!tded)
                    tded = e.targ;
                Objects tiargs = Objects(1);
                tiargs[0] = e.targ;

                /* Declare trailing parameters
                 */
                for (size_t i = 1; i < e.parameters.dim; i++)
                {
                    TemplateParameter tp = (*e.parameters)[i];
                    Declaration s = null;

                    m = tp.matchArg(e.loc, sc, &tiargs, i, e.parameters, &dedtypes, &s);
                    if (m <= MATCH.nomatch)
                        goto Lno;
                    s.dsymbolSemantic(sc);
                    if (!sc.insert(s))
                        e.error("declaration `%s` is already defined", s.toChars());

                    unSpeculative(sc, s);
                }
                goto Lyes;
            }
        }
        else if (e.id)
        {
            /* Declare id as an alias for type targ. Evaluate to true
             * is(targ id)
             */
            tded = e.targ;
            goto Lyes;
        }

    Lyes:
        if (e.id)
        {
            Dsymbol s;
            Tuple tup = isTuple(tded);
            if (tup)
                s = new TupleDeclaration(e.loc, e.id, &tup.objects);
            else
                s = new AliasDeclaration(e.loc, e.id, tded);
            s.dsymbolSemantic(sc);

            /* The reason for the !tup is unclear. It fails Phobos unittests if it is not there.
             * More investigation is needed.
             */
            if (!tup && !sc.insert(s))
                e.error("declaration `%s` is already defined", s.toChars());

            unSpeculative(sc, s);
        }
        //printf("Lyes\n");
        result = IntegerExp.createBool(true);
        return;

    Lno:
        //printf("Lno\n");
        result = IntegerExp.createBool(false);
    }

    override void visit(BinAssignExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (checkIfIsStructLiteralDotExpr(exp.e1))
            return setError();
        if (exp.e1.op == TOK.arrayLength)
        {
            // arr.length op= e2;
            e = rewriteOpAssign(exp);
            e = e.expressionSemantic(sc);
            result = e;
            return;
        }
        if (exp.e1.op == TOK.slice || exp.e1.type.ty == Tarray || exp.e1.type.ty == Tsarray)
        {
            if (checkNonAssignmentArrayOp(exp.e1))
                return setError();

            if (exp.e1.op == TOK.slice)
                (cast(SliceExp)exp.e1).arrayop = true;

            // T[] op= ...
            if (exp.e2.implicitConvTo(exp.e1.type.nextOf()))
            {
                // T[] op= T
                exp.e2 = exp.e2.castTo(sc, exp.e1.type.nextOf());
            }
            else if (Expression ex = typeCombine(exp, sc))
            {
                result = ex;
                return;
            }
            exp.type = exp.e1.type;
            result = arrayOp(exp, sc);
            return;
        }

        exp.e1 = exp.e1.expressionSemantic(sc);
        exp.e1 = exp.e1.optimize(WANTvalue);
        exp.e1 = exp.e1.modifiableLvalue(sc, exp.e1);
        exp.type = exp.e1.type;

        if (auto ad = isAggregate(exp.e1.type))
        {
            if (const s = search_function(ad, Id.opOpAssign))
            {
                error(exp.loc, "none of the `opOpAssign` overloads of `%s` are callable for `%s` of type `%s`", ad.toChars(), exp.e1.toChars(), exp.e1.type.toChars());
                return setError();
            }
        }
        if (exp.e1.checkScalar() ||
            exp.e1.checkReadModifyWrite(exp.op, exp.e2) ||
            exp.e1.checkSharedAccess(sc))
            return setError();

        int arith = (exp.op == TOK.addAssign || exp.op == TOK.minAssign || exp.op == TOK.mulAssign || exp.op == TOK.divAssign || exp.op == TOK.modAssign || exp.op == TOK.powAssign);
        int bitwise = (exp.op == TOK.andAssign || exp.op == TOK.orAssign || exp.op == TOK.xorAssign);
        int shift = (exp.op == TOK.leftShiftAssign || exp.op == TOK.rightShiftAssign || exp.op == TOK.unsignedRightShiftAssign);

        if (bitwise && exp.type.toBasetype().ty == Tbool)
            exp.e2 = exp.e2.implicitCastTo(sc, exp.type);
        else if (exp.checkNoBool())
            return setError();

        if ((exp.op == TOK.addAssign || exp.op == TOK.minAssign) && exp.e1.type.toBasetype().ty == Tpointer && exp.e2.type.toBasetype().isintegral())
        {
            result = scaleFactor(exp, sc);
            return;
        }

        if (Expression ex = typeCombine(exp, sc))
        {
            result = ex;
            return;
        }

        if (arith && (exp.checkArithmeticBin() || exp.checkSharedAccessBin(sc)))
            return setError();
        if ((bitwise || shift) && (exp.checkIntegralBin() || exp.checkSharedAccessBin(sc)))
            return setError();

        if (shift)
        {
            if (exp.e2.type.toBasetype().ty != Tvector)
                exp.e2 = exp.e2.castTo(sc, Type.tshiftcnt);
        }

        if (!target.isVectorOpSupported(exp.type.toBasetype(), exp.op, exp.e2.type.toBasetype()))
        {
            result = exp.incompatibleTypes();
            return;
        }

        if (exp.e1.op == TOK.error || exp.e2.op == TOK.error)
            return setError();

        e = exp.checkOpAssignTypes(sc);
        if (e.op == TOK.error)
        {
            result = e;
            return;
        }

        assert(e.op == TOK.assign || e == exp);
        result = (cast(BinExp)e).reorderSettingAAElem(sc);
    }

    private Expression compileIt(CompileExp exp)
    {
        OutBuffer buf;
        if (expressionsToString(buf, sc, exp.exps))
            return null;

        uint errors = global.errors;
        const len = buf.length;
        const str = buf.extractChars()[0 .. len];
        scope p = new Parser!ASTCodegen(exp.loc, sc._module, str, false);
        p.nextToken();
        //printf("p.loc.linnum = %d\n", p.loc.linnum);

        Expression e = p.parseExpression();
        if (global.errors != errors)
            return null;

        if (p.token.value != TOK.endOfFile)
        {
            exp.error("incomplete mixin expression `%s`", str.ptr);
            return null;
        }
        return e;
    }

    override void visit(CompileExp exp)
    {
        /* https://dlang.org/spec/expression.html#mixin_expressions
         */

        static if (LOGSEMANTIC)
        {
            printf("CompileExp::semantic('%s')\n", exp.toChars());
        }

        auto e = compileIt(exp);
        if (!e)
            return setError();
        result = e.expressionSemantic(sc);
    }

    override void visit(ImportExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("ImportExp::semantic('%s')\n", e.toChars());
        }

        auto se = semanticString(sc, e.e1, "file name argument");
        if (!se)
            return setError();
        se = se.toUTF8(sc);

        auto namez = se.toStringz().ptr;
        if (!global.filePath)
        {
            e.error("need `-J` switch to import text file `%s`", namez);
            return setError();
        }

        /* Be wary of CWE-22: Improper Limitation of a Pathname to a Restricted Directory
         * ('Path Traversal') attacks.
         * http://cwe.mitre.org/data/definitions/22.html
         */

        auto name = FileName.safeSearchPath(global.filePath, namez);
        if (!name)
        {
            e.error("file `%s` cannot be found or not in a path specified with `-J`", se.toChars());
            return setError();
        }

        sc._module.contentImportedFiles.push(name);
        if (global.params.verbose)
        {
            const slice = se.peekString();
            message("file      %.*s\t(%s)", cast(int)slice.length, slice.ptr, name);
        }
        if (global.params.moduleDeps !is null)
        {
            OutBuffer* ob = global.params.moduleDeps;
            Module imod = sc.instantiatingModule();

            if (!global.params.moduleDepsFile)
                ob.writestring("depsFile ");
            ob.writestring(imod.toPrettyChars());
            ob.writestring(" (");
            escapePath(ob, imod.srcfile.toChars());
            ob.writestring(") : ");
            if (global.params.moduleDepsFile)
                ob.writestring("string : ");
            ob.write(se.peekString());
            ob.writestring(" (");
            escapePath(ob, name);
            ob.writestring(")");
            ob.writenl();
        }

        {
            auto readResult = File.read(name);
            if (!readResult.success)
            {
                e.error("cannot read file `%s`", name);
                return setError();
            }
            else
            {
                // take ownership of buffer (probably leaking)
                auto data = readResult.extractSlice();
                se = new StringExp(e.loc, data);
            }
        }
        result = se.expressionSemantic(sc);
    }

    override void visit(AssertExp exp)
    {
        // https://dlang.org/spec/expression.html#assert_expressions
        static if (LOGSEMANTIC)
        {
            printf("AssertExp::semantic('%s')\n", exp.toChars());
        }

        const generateMsg = !exp.msg && global.params.checkAction == CHECKACTION.context;
        Expression temporariesPrefix;

        if (generateMsg)
        // no message - use assert expression as msg
        {
            /*
            {
              auto a = e1, b = e2;
              assert(a == b, _d_assert_fail!"=="(a, b));
            }()
            */

            /*
            Stores the result of an operand expression into a temporary
            if necessary, e.g. if it is an impure fuction call containing side
            effects as in https://issues.dlang.org/show_bug.cgi?id=20114

            Params:
                op = an expression which may require a temporary (added to
                     `temporariesPrefix`: `auto tmp = op`) and will be replaced
                     by `tmp` if necessary

            Returns: (possibly replaced) `op`
            */
            Expression maybePromoteToTmp(ref Expression op)
            {
                op = op.expressionSemantic(sc);
                op = resolveProperties(sc, op);
                if (op.hasSideEffect)
                {
                    const stc = op.isLvalue() ? STC.ref_ : 0;
                    auto tmp = copyToTemp(stc, "__assertOp", op);
                    tmp.dsymbolSemantic(sc);

                    auto decl = new DeclarationExp(op.loc, tmp);
                    temporariesPrefix = Expression.combine(temporariesPrefix, decl);

                    op = new VarExp(op.loc, tmp);
                    op = op.expressionSemantic(sc);
                }
                return op;
            }

            // if the assert condition is a mixin expression, try to compile it
            if (auto ce = exp.e1.isCompileExp())
            {
                if (auto e1 = compileIt(ce))
                    exp.e1 = e1;
            }

            const tok = exp.e1.op;
            bool isEqualsCallExpression;
            if (tok == TOK.call)
            {
                const callExp = cast(CallExp) exp.e1;

                // https://issues.dlang.org/show_bug.cgi?id=20331
                // callExp.f may be null if the assert contains a call to
                // a function pointer or literal
                if (const callExpFunc = callExp.f)
                {
                    const callExpIdent = callExpFunc.ident;
                    isEqualsCallExpression = callExpIdent == Id.__equals ||
                                             callExpIdent == Id.eq;
                }
            }
            if (tok == TOK.equal || tok == TOK.notEqual ||
                tok == TOK.lessThan || tok == TOK.greaterThan ||
                tok == TOK.lessOrEqual || tok == TOK.greaterOrEqual ||
                tok == TOK.identity || tok == TOK.notIdentity ||
                tok == TOK.in_ ||
                isEqualsCallExpression)
            {
                if (!verifyHookExist(exp.loc, *sc, Id._d_assert_fail, "generating assert messages"))
                    return setError();

                auto es = new Expressions(2);
                auto tiargs = new Objects(3);
                Loc loc = exp.e1.loc;

                if (isEqualsCallExpression)
                {
                    auto callExp = cast(CallExp) exp.e1;
                    auto args = callExp.arguments;

                    // structs with opEquals get rewritten to a DotVarExp:
                    // a.opEquals(b)
                    // https://issues.dlang.org/show_bug.cgi?id=20100
                    if (args.length == 1)
                    {
                        auto dv = callExp.e1.isDotVarExp();
                        assert(dv);

                        // runtime args
                        (*es)[0] = maybePromoteToTmp(dv.e1);
                        (*es)[1] = maybePromoteToTmp((*args)[0]);
                    }
                    else
                    {
                        // runtime args
                        (*es)[0] = maybePromoteToTmp((*args)[0]);
                        (*es)[1] = maybePromoteToTmp((*args)[1]);
                    }
                }
                else
                {
                    auto binExp = cast(EqualExp) exp.e1;

                    // runtime args
                    (*es)[0] = maybePromoteToTmp(binExp.e1);
                    (*es)[1] = maybePromoteToTmp(binExp.e2);
                }

                // template args
                Expression comp = new StringExp(loc, isEqualsCallExpression ? "==" : Token.toString(exp.e1.op));
                comp = comp.expressionSemantic(sc);
                (*tiargs)[0] = comp;
                (*tiargs)[1] = (*es)[0].type;
                (*tiargs)[2] = (*es)[1].type;

                Expression __assertFail = new IdentifierExp(exp.loc, Id.empty);
                auto assertFail = new DotIdExp(loc, __assertFail, Id.object);

                auto dt = new DotTemplateInstanceExp(loc, assertFail, Id._d_assert_fail, tiargs);
                auto ec = CallExp.create(Loc.initial, dt, es);
                exp.msg = ec;
            }
            else
            {
                OutBuffer buf;
                buf.printf("%s failed", exp.toChars());
                exp.msg = new StringExp(Loc.initial, buf.extractSlice());
            }
        }

        if (Expression ex = unaSemantic(exp, sc))
        {
            result = ex;
            return;
        }

        exp.e1 = resolveProperties(sc, exp.e1);
        // BUG: see if we can do compile time elimination of the Assert
        exp.e1 = exp.e1.optimize(WANTvalue);
        exp.e1 = exp.e1.toBoolean(sc);

        if (exp.msg)
        {
            exp.msg = expressionSemantic(exp.msg, sc);
            exp.msg = resolveProperties(sc, exp.msg);
            exp.msg = exp.msg.implicitCastTo(sc, Type.tchar.constOf().arrayOf());
            exp.msg = exp.msg.optimize(WANTvalue);
            checkParamArgumentEscape(sc, null, null, exp.msg, true, false);
        }

        if (exp.e1.op == TOK.error)
        {
            result = exp.e1;
            return;
        }
        if (exp.msg && exp.msg.op == TOK.error)
        {
            result = exp.msg;
            return;
        }

        auto f1 = checkNonAssignmentArrayOp(exp.e1);
        auto f2 = exp.msg && checkNonAssignmentArrayOp(exp.msg);
        if (f1 || f2)
            return setError();

        if (exp.e1.isBool(false))
        {
            /* This is an `assert(0)` which means halt program execution
             */
            FuncDeclaration fd = sc.parent.isFuncDeclaration();
            if (fd)
                fd.hasReturnExp |= 4;
            sc.ctorflow.orCSX(CSX.halt);

            if (global.params.useAssert == CHECKENABLE.off)
            {
                Expression e = new HaltExp(exp.loc);
                e = e.expressionSemantic(sc);
                result = e;
                return;
            }
        }

        exp.type = Type.tvoid;

        result = !temporariesPrefix
            ? exp
            : Expression.combine(temporariesPrefix, exp).expressionSemantic(sc);
    }

    override void visit(DotIdExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("DotIdExp::semantic(this = %p, '%s')\n", exp, exp.toChars());
            //printf("e1.op = %d, '%s'\n", e1.op, Token::toChars(e1.op));
        }
        Expression e = exp.semanticY(sc, 1);
        if (e && isDotOpDispatch(e))
        {
            uint errors = global.startGagging();
            e = resolvePropertiesX(sc, e);
            if (global.endGagging(errors))
                e = null; /* fall down to UFCS */
            else
            {
                result = e;
                return;
            }
        }
        if (!e) // if failed to find the property
        {
            /* If ident is not a valid property, rewrite:
             *   e1.ident
             * as:
             *   .ident(e1)
             */
            e = resolveUFCSProperties(sc, exp);
        }
        result = e;
    }

    override void visit(DotTemplateExp e)
    {
        if (Expression ex = unaSemantic(e, sc))
        {
            result = ex;
            return;
        }
        result = e;
    }

    override void visit(DotVarExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("DotVarExp::semantic('%s')\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        exp.var = exp.var.toAlias().isDeclaration();

        exp.e1 = exp.e1.expressionSemantic(sc);

        if (auto tup = exp.var.isTupleDeclaration())
        {
            /* Replace:
             *  e1.tuple(a, b, c)
             * with:
             *  tuple(e1.a, e1.b, e1.c)
             */
            Expression e0;
            Expression ev = sc.func ? extractSideEffect(sc, "__tup", e0, exp.e1) : exp.e1;

            auto exps = new Expressions();
            exps.reserve(tup.objects.dim);
            for (size_t i = 0; i < tup.objects.dim; i++)
            {
                RootObject o = (*tup.objects)[i];
                Expression e;
                Declaration var;
                if (o.dyncast() == DYNCAST.expression)
                {
                    e = cast(Expression)o;
                    if (auto se = e.isDsymbolExp())
                        var = se.s.isDeclaration();
                    else if (auto ve = e.isVarExp())
                        if (!ve.var.isFuncDeclaration())
                            // Exempt functions for backwards compatibility reasons.
                            // See: https://issues.dlang.org/show_bug.cgi?id=20470#c1
                            var = ve.var;
                }
                else if (o.dyncast() == DYNCAST.dsymbol)
                {
                    Dsymbol s = cast(Dsymbol) o;
                    Declaration d = s.isDeclaration();
                    if (!d || d.isFuncDeclaration())
                        // Exempt functions for backwards compatibility reasons.
                        // See: https://issues.dlang.org/show_bug.cgi?id=20470#c1
                        e = new DsymbolExp(exp.loc, s);
                    else
                        var = d;
                }
                else if (o.dyncast() == DYNCAST.type)
                {
                    e = new TypeExp(exp.loc, cast(Type)o);
                }
                else
                {
                    exp.error("`%s` is not an expression", o.toChars());
                    return setError();
                }
                if (var)
                    e = new DotVarExp(exp.loc, ev, var);
                exps.push(e);
            }

            Expression e = new TupleExp(exp.loc, e0, exps);
            e = e.expressionSemantic(sc);
            result = e;
            return;
        }

        exp.e1 = exp.e1.addDtorHook(sc);

        Type t1 = exp.e1.type;

        if (FuncDeclaration fd = exp.var.isFuncDeclaration())
        {
            // for functions, do checks after overload resolution
            if (!fd.functionSemantic())
                return setError();

            /* https://issues.dlang.org/show_bug.cgi?id=13843
             * If fd obviously has no overloads, we should
             * normalize AST, and it will give a chance to wrap fd with FuncExp.
             */
            if ((fd.isNested() && !fd.isThis()) || fd.isFuncLiteralDeclaration())
            {
                // (e1, fd)
                auto e = symbolToExp(fd, exp.loc, sc, false);
                result = Expression.combine(exp.e1, e);
                return;
            }

            exp.type = fd.type;
            assert(exp.type);
        }
        else if (OverDeclaration od = exp.var.isOverDeclaration())
        {
            exp.type = Type.tvoid; // ambiguous type?
        }
        else
        {
            exp.type = exp.var.type;
            if (!exp.type && global.errors) // var is goofed up, just return error.
                return setError();
            assert(exp.type);

            if (t1.ty == Tpointer)
                t1 = t1.nextOf();

            exp.type = exp.type.addMod(t1.mod);

            Dsymbol vparent = exp.var.toParent();
            AggregateDeclaration ad = vparent ? vparent.isAggregateDeclaration() : null;
            if (Expression e1x = getRightThis(exp.loc, sc, ad, exp.e1, exp.var, 1))
                exp.e1 = e1x;
            else
            {
                /* Later checkRightThis will report correct error for invalid field variable access.
                 */
                Expression e = new VarExp(exp.loc, exp.var);
                e = e.expressionSemantic(sc);
                result = e;
                return;
            }
            checkAccess(exp.loc, sc, exp.e1, exp.var);

            VarDeclaration v = exp.var.isVarDeclaration();
            if (v && (v.isDataseg() || (v.storage_class & STC.manifest)))
            {
                Expression e = expandVar(WANTvalue, v);
                if (e)
                {
                    result = e;
                    return;
                }
            }

            if (v && (v.isDataseg() || // fix https://issues.dlang.org/show_bug.cgi?id=8238
                      (!v.needThis() && v.semanticRun > PASS.init)))  // fix https://issues.dlang.org/show_bug.cgi?id=17258
            {
                // (e1, v)
                checkAccess(exp.loc, sc, exp.e1, v);
                Expression e = new VarExp(exp.loc, v);
                e = new CommaExp(exp.loc, exp.e1, e);
                e = e.expressionSemantic(sc);
                result = e;
                return;
            }
        }
        //printf("-DotVarExp::semantic('%s')\n", toChars());
        result = exp;
    }

    override void visit(DotTemplateInstanceExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("DotTemplateInstanceExp::semantic('%s')\n", exp.toChars());
        }
        // Indicate we need to resolve by UFCS.
        Expression e = exp.semanticY(sc, 1);
        if (!e)
            e = resolveUFCSProperties(sc, exp);
        result = e;
    }

    override void visit(DelegateExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("DelegateExp::semantic('%s')\n", e.toChars());
        }
        if (e.type)
        {
            result = e;
            return;
        }

        e.e1 = e.e1.expressionSemantic(sc);

        e.type = new TypeDelegate(e.func.type);
        e.type = e.type.typeSemantic(e.loc, sc);

        FuncDeclaration f = e.func.toAliasFunc();
        AggregateDeclaration ad = f.toParentLocal().isAggregateDeclaration();
        if (f.needThis())
            e.e1 = getRightThis(e.loc, sc, ad, e.e1, f);
        if (e.e1.op == TOK.error)
            return setError();

        /* A delegate takes the address of e.e1 in order to set the .ptr field
         * https://issues.dlang.org/show_bug.cgi?id=18575
         */
        if (global.params.vsafe && e.e1.type.toBasetype().ty == Tstruct)
        {
            if (auto v = expToVariable(e.e1))
            {
                if (!checkAddressVar(sc, e, v))
                    return setError();
            }
        }

        if (f.type.ty == Tfunction)
        {
            TypeFunction tf = cast(TypeFunction)f.type;
            if (!MODmethodConv(e.e1.type.mod, f.type.mod))
            {
                OutBuffer thisBuf, funcBuf;
                MODMatchToBuffer(&thisBuf, e.e1.type.mod, tf.mod);
                MODMatchToBuffer(&funcBuf, tf.mod, e.e1.type.mod);
                e.error("%smethod `%s` is not callable using a %s`%s`",
                    funcBuf.peekChars(), f.toPrettyChars(), thisBuf.peekChars(), e.e1.toChars());
                return setError();
            }
        }
        if (ad && ad.isClassDeclaration() && ad.type != e.e1.type)
        {
            // A downcast is required for interfaces
            // https://issues.dlang.org/show_bug.cgi?id=3706
            e.e1 = new CastExp(e.loc, e.e1, ad.type);
            e.e1 = e.e1.expressionSemantic(sc);
        }
        result = e;
        // declare dual-context container
        if (f.isThis2 && !sc.intypeof && sc.func)
        {
            // check access to second `this`
            if (AggregateDeclaration ad2 = f.isMember2())
            {
                Expression te = new ThisExp(e.loc).expressionSemantic(sc);
                if (te.op != TOK.error)
                    te = getRightThis(e.loc, sc, ad2, te, f);
                if (te.op == TOK.error)
                {
                    e.error("need `this` of type `%s` to make delegate from function `%s`", ad2.toChars(), f.toChars());
                    return setError();
                }
            }
            VarDeclaration vthis2 = makeThis2Argument(e.loc, sc, f);
            e.vthis2 = vthis2;
            Expression de = new DeclarationExp(e.loc, vthis2);
            result = Expression.combine(de, result);
            result = result.expressionSemantic(sc);
        }
    }

    override void visit(DotTypeExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("DotTypeExp::semantic('%s')\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (auto e = unaSemantic(exp, sc))
        {
            result = e;
            return;
        }

        exp.type = exp.sym.getType().addMod(exp.e1.type.mod);
        result = exp;
    }

    override void visit(AddrExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("AddrExp::semantic('%s')\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = unaSemantic(exp, sc))
        {
            result = ex;
            return;
        }

        int wasCond = exp.e1.op == TOK.question;

        if (exp.e1.op == TOK.dotTemplateInstance)
        {
            DotTemplateInstanceExp dti = cast(DotTemplateInstanceExp)exp.e1;
            TemplateInstance ti = dti.ti;
            {
                //assert(ti.needsTypeInference(sc));
                ti.dsymbolSemantic(sc);
                if (!ti.inst || ti.errors) // if template failed to expand
                    return setError();

                Dsymbol s = ti.toAlias();
                FuncDeclaration f = s.isFuncDeclaration();
                if (f)
                {
                    exp.e1 = new DotVarExp(exp.e1.loc, dti.e1, f);
                    exp.e1 = exp.e1.expressionSemantic(sc);
                }
            }
        }
        else if (exp.e1.op == TOK.scope_)
        {
            TemplateInstance ti = (cast(ScopeExp)exp.e1).sds.isTemplateInstance();
            if (ti)
            {
                //assert(ti.needsTypeInference(sc));
                ti.dsymbolSemantic(sc);
                if (!ti.inst || ti.errors) // if template failed to expand
                    return setError();

                Dsymbol s = ti.toAlias();
                FuncDeclaration f = s.isFuncDeclaration();
                if (f)
                {
                    exp.e1 = new VarExp(exp.e1.loc, f);
                    exp.e1 = exp.e1.expressionSemantic(sc);
                }
            }
        }
        /* https://issues.dlang.org/show_bug.cgi?id=809
         *
         * If the address of a lazy variable is taken,
         * the expression is rewritten so that the type
         * of it is the delegate type. This means that
         * the symbol is not going to represent a call
         * to the delegate anymore, but rather, the
         * actual symbol.
         */
        if (auto ve = exp.e1.isVarExp())
        {
            if (ve.var.storage_class & STC.lazy_)
            {
                exp.e1 = exp.e1.expressionSemantic(sc);
                exp.e1 = resolveProperties(sc, exp.e1);
                if (auto callExp = exp.e1.isCallExp())
                {
                    if (callExp.e1.type.toBasetype().ty == Tdelegate)
                    {
                        /* https://issues.dlang.org/show_bug.cgi?id=20551
                         *
                         * Cannot take address of lazy parameter in @safe code
                         * because it might end up being a pointer to undefined
                         * memory.
                         */
                        if (sc.func && !sc.intypeof && !(sc.flags & SCOPE.debug_) && sc.func.setUnsafe())
                        {
                            exp.error("cannot take address of lazy parameter `%s` in `@safe` function `%s`",
                                     ve.toChars(), sc.func.toChars());
                            setError();
                        }
                        else
                        {
                            VarExp ve2 = callExp.e1.isVarExp();
                            ve2.delegateWasExtracted = true;
                            ve2.var.storage_class |= STC.scope_;
                            result = ve2;
                        }
                        return;
                    }
                }
            }
        }

        exp.e1 = exp.e1.toLvalue(sc, null);
        if (exp.e1.op == TOK.error)
        {
            result = exp.e1;
            return;
        }
        if (checkNonAssignmentArrayOp(exp.e1))
            return setError();

        if (!exp.e1.type)
        {
            exp.error("cannot take address of `%s`", exp.e1.toChars());
            return setError();
        }

        bool hasOverloads;
        if (auto f = isFuncAddress(exp, &hasOverloads))
        {
            if (!hasOverloads && f.checkForwardRef(exp.loc))
                return setError();
        }
        else if (!exp.e1.type.deco)
        {
            if (exp.e1.op == TOK.variable)
            {
                VarExp ve = cast(VarExp)exp.e1;
                Declaration d = ve.var;
                exp.error("forward reference to %s `%s`", d.kind(), d.toChars());
            }
            else
                exp.error("forward reference to `%s`", exp.e1.toChars());
            return setError();
        }

        exp.type = exp.e1.type.pointerTo();

        // See if this should really be a delegate
        if (exp.e1.op == TOK.dotVariable)
        {
            DotVarExp dve = cast(DotVarExp)exp.e1;
            FuncDeclaration f = dve.var.isFuncDeclaration();
            if (f)
            {
                f = f.toAliasFunc(); // FIXME, should see overloads
                                     // https://issues.dlang.org/show_bug.cgi?id=1983
                if (!dve.hasOverloads)
                    f.tookAddressOf++;

                Expression e;
                if (f.needThis())
                    e = new DelegateExp(exp.loc, dve.e1, f, dve.hasOverloads);
                else // It is a function pointer. Convert &v.f() --> (v, &V.f())
                    e = new CommaExp(exp.loc, dve.e1, new AddrExp(exp.loc, new VarExp(exp.loc, f, dve.hasOverloads)));
                e = e.expressionSemantic(sc);
                result = e;
                return;
            }

            // Look for misaligned pointer in @safe mode
            if (checkUnsafeAccess(sc, dve, !exp.type.isMutable(), true))
                return setError();

            if (global.params.vsafe)
            {
                if (VarDeclaration v = expToVariable(dve.e1))
                {
                    if (!checkAddressVar(sc, exp, v))
                        return setError();
                }
            }
        }
        else if (exp.e1.op == TOK.variable)
        {
            VarExp ve = cast(VarExp)exp.e1;
            VarDeclaration v = ve.var.isVarDeclaration();
            if (v)
            {
                if (!checkAddressVar(sc, exp, v))
                    return setError();

                ve.checkPurity(sc, v);
            }
            FuncDeclaration f = ve.var.isFuncDeclaration();
            if (f)
            {
               if (IN_LLVM && DtoIsIntrinsic(f.toAliasFunc()))
               {
                    exp.error("cannot take the address of intrinsic function `%s`", f.toAliasFunc().toChars());
                    result = new ErrorExp();
                    return;
                }
                /* Because nested functions cannot be overloaded,
                 * mark here that we took its address because castTo()
                 * may not be called with an exact match.
                 */
                if (!ve.hasOverloads || (f.isNested() && !f.needThis()))
                    f.tookAddressOf++;
                if (f.isNested() && !f.needThis())
                {
                    if (f.isFuncLiteralDeclaration())
                    {
                        if (!f.FuncDeclaration.isNested())
                        {
                            /* Supply a 'null' for a this pointer if no this is available
                             */
                            Expression e = new DelegateExp(exp.loc, new NullExp(exp.loc, Type.tnull), f, ve.hasOverloads);
                            e = e.expressionSemantic(sc);
                            result = e;
                            return;
                        }
                    }
                    Expression e = new DelegateExp(exp.loc, exp.e1, f, ve.hasOverloads);
                    e = e.expressionSemantic(sc);
                    result = e;
                    return;
                }
                if (f.needThis())
                {
                    if (hasThis(sc))
                    {
                        /* Should probably supply 'this' after overload resolution,
                         * not before.
                         */
                        Expression ethis = new ThisExp(exp.loc);
                        Expression e = new DelegateExp(exp.loc, ethis, f, ve.hasOverloads);
                        e = e.expressionSemantic(sc);
                        result = e;
                        return;
                    }
                    if (sc.func && !sc.intypeof)
                    {
                        if (!(sc.flags & SCOPE.debug_) && sc.func.setUnsafe())
                        {
                            exp.error("`this` reference necessary to take address of member `%s` in `@safe` function `%s`", f.toChars(), sc.func.toChars());
                        }
                    }
                }
            }
        }
        else if ((exp.e1.op == TOK.this_ || exp.e1.op == TOK.super_) && global.params.vsafe)
        {
            if (VarDeclaration v = expToVariable(exp.e1))
            {
                if (!checkAddressVar(sc, exp, v))
                    return setError();
            }
        }
        else if (exp.e1.op == TOK.call)
        {
            CallExp ce = cast(CallExp)exp.e1;
            if (ce.e1.type.ty == Tfunction)
            {
                TypeFunction tf = cast(TypeFunction)ce.e1.type;
                if (tf.isref && sc.func && !sc.intypeof && !(sc.flags & SCOPE.debug_) && sc.func.setUnsafe())
                {
                    exp.error("cannot take address of `ref return` of `%s()` in `@safe` function `%s`",
                        ce.e1.toChars(), sc.func.toChars());
                }
            }
        }
        else if (exp.e1.op == TOK.index)
        {
            /* For:
             *   int[3] a;
             *   &a[i]
             * check 'a' the same as for a regular variable
             */
            if (VarDeclaration v = expToVariable(exp.e1))
            {
                if (global.params.vsafe && !checkAddressVar(sc, exp, v))
                    return setError();

                exp.e1.checkPurity(sc, v);
            }
        }
        else if (wasCond)
        {
            /* a ? b : c was transformed to *(a ? &b : &c), but we still
             * need to do safety checks
             */
            assert(exp.e1.op == TOK.star);
            PtrExp pe = cast(PtrExp)exp.e1;
            assert(pe.e1.op == TOK.question);
            CondExp ce = cast(CondExp)pe.e1;
            assert(ce.e1.op == TOK.address);
            assert(ce.e2.op == TOK.address);

            // Re-run semantic on the address expressions only
            ce.e1.type = null;
            ce.e1 = ce.e1.expressionSemantic(sc);
            ce.e2.type = null;
            ce.e2 = ce.e2.expressionSemantic(sc);
        }
        result = exp.optimize(WANTvalue);
    }

    override void visit(PtrExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("PtrExp::semantic('%s')\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        Type tb = exp.e1.type.toBasetype();
        switch (tb.ty)
        {
        case Tpointer:
            exp.type = (cast(TypePointer)tb).next;
            break;

        case Tsarray:
        case Tarray:
            if (isNonAssignmentArrayOp(exp.e1))
                goto default;
            exp.error("using `*` on an array is no longer supported; use `*(%s).ptr` instead", exp.e1.toChars());
            exp.type = (cast(TypeArray)tb).next;
            exp.e1 = exp.e1.castTo(sc, exp.type.pointerTo());
            break;

        case Terror:
            return setError();

        default:
            exp.error("can only `*` a pointer, not a `%s`", exp.e1.type.toChars());
            goto case Terror;
        }

        if (exp.checkValue() || exp.checkSharedAccess(sc))
            return setError();

        result = exp;
    }

    override void visit(NegExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("NegExp::semantic('%s')\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        fix16997(sc, exp);
        exp.type = exp.e1.type;
        Type tb = exp.type.toBasetype();
        if (tb.ty == Tarray || tb.ty == Tsarray)
        {
            if (!isArrayOpValid(exp.e1))
            {
                result = arrayOpInvalidError(exp);
                return;
            }
            result = exp;
            return;
        }
        if (!target.isVectorOpSupported(tb, exp.op))
        {
            result = exp.incompatibleTypes();
            return;
        }
        if (exp.e1.checkNoBool())
            return setError();
        if (exp.e1.checkArithmetic() ||
            exp.e1.checkSharedAccess(sc))
            return setError();

        result = exp;
    }

    override void visit(UAddExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("UAddExp::semantic('%s')\n", exp.toChars());
        }
        assert(!exp.type);

        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        fix16997(sc, exp);
        if (!target.isVectorOpSupported(exp.e1.type.toBasetype(), exp.op))
        {
            result = exp.incompatibleTypes();
            return;
        }
        if (exp.e1.checkNoBool())
            return setError();
        if (exp.e1.checkArithmetic())
            return setError();
        if (exp.e1.checkSharedAccess(sc))
            return setError();

        result = exp.e1;
    }

    override void visit(ComExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        fix16997(sc, exp);
        exp.type = exp.e1.type;
        Type tb = exp.type.toBasetype();
        if (tb.ty == Tarray || tb.ty == Tsarray)
        {
            if (!isArrayOpValid(exp.e1))
            {
                result = arrayOpInvalidError(exp);
                return;
            }
            result = exp;
            return;
        }
        if (!target.isVectorOpSupported(tb, exp.op))
        {
            result = exp.incompatibleTypes();
            return;
        }
        if (exp.e1.checkNoBool())
            return setError();
        if (exp.e1.checkIntegral() ||
            exp.e1.checkSharedAccess(sc))
            return setError();

        result = exp;
    }

    override void visit(NotExp e)
    {
        if (e.type)
        {
            result = e;
            return;
        }

        e.setNoderefOperand();

        // Note there is no operator overload
        if (Expression ex = unaSemantic(e, sc))
        {
            result = ex;
            return;
        }

        // for static alias this: https://issues.dlang.org/show_bug.cgi?id=17684
        if (e.e1.op == TOK.type)
            e.e1 = resolveAliasThis(sc, e.e1);

        e.e1 = resolveProperties(sc, e.e1);
        e.e1 = e.e1.toBoolean(sc);
        if (e.e1.type == Type.terror)
        {
            result = e.e1;
            return;
        }

        if (!target.isVectorOpSupported(e.e1.type.toBasetype(), e.op))
        {
            result = e.incompatibleTypes();
        }
        // https://issues.dlang.org/show_bug.cgi?id=13910
        // Today NotExp can take an array as its operand.
        if (checkNonAssignmentArrayOp(e.e1))
            return setError();

        e.type = Type.tbool;
        result = e;
    }

    override void visit(DeleteExp exp)
    {
        if (!sc.isDeprecated)
        {
            // @@@DEPRECATED_2019-02@@@
            // 1. Deprecation for 1 year
            // 2. Error for 1 year
            // 3. Removal of keyword, "delete" can be used for other identities
            if (!exp.isRAII)
                deprecation(exp.loc, "The `delete` keyword has been deprecated.  Use `object.destroy()` (and `core.memory.GC.free()` if applicable) instead.");
        }

        if (Expression ex = unaSemantic(exp, sc))
        {
            result = ex;
            return;
        }
        exp.e1 = resolveProperties(sc, exp.e1);
        exp.e1 = exp.e1.modifiableLvalue(sc, null);
        if (exp.e1.op == TOK.error)
        {
            result = exp.e1;
            return;
        }
        exp.type = Type.tvoid;

        AggregateDeclaration ad = null;
        Type tb = exp.e1.type.toBasetype();
        switch (tb.ty)
        {
        case Tclass:
            {
                auto cd = (cast(TypeClass)tb).sym;
                if (cd.isCOMinterface())
                {
                    /* Because COM classes are deleted by IUnknown.Release()
                     */
                    exp.error("cannot `delete` instance of COM interface `%s`", cd.toChars());
                    return setError();
                }
                ad = cd;
                break;
            }
        case Tpointer:
            tb = (cast(TypePointer)tb).next.toBasetype();
            if (tb.ty == Tstruct)
            {
                ad = (cast(TypeStruct)tb).sym;
                semanticTypeInfo(sc, tb);
            }
            break;

        case Tarray:
            {
                Type tv = tb.nextOf().baseElemOf();
                if (tv.ty == Tstruct)
                {
                    ad = (cast(TypeStruct)tv).sym;
                    if (ad.dtor)
                        semanticTypeInfo(sc, ad.type);
                }
                break;
            }
        default:
            exp.error("cannot delete type `%s`", exp.e1.type.toChars());
            return setError();
        }

        bool err = false;
        if (ad)
        {
            if (ad.dtor)
            {
                err |= exp.checkPurity(sc, ad.dtor);
                err |= exp.checkSafety(sc, ad.dtor);
                err |= exp.checkNogc(sc, ad.dtor);
            }
            if (err)
                return setError();
        }

        if (!sc.intypeof && sc.func &&
            !exp.isRAII &&
            !(sc.flags & SCOPE.debug_) &&
            sc.func.setUnsafe())
        {
            exp.error("`%s` is not `@safe` but is used in `@safe` function `%s`", exp.toChars(), sc.func.toChars());
            err = true;
        }
        if (err)
            return setError();

        result = exp;
    }

    override void visit(CastExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("CastExp::semantic('%s')\n", exp.toChars());
        }
        //static int x; assert(++x < 10);
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (exp.to)
        {
            exp.to = exp.to.typeSemantic(exp.loc, sc);
            if (exp.to == Type.terror)
                return setError();

            if (!exp.to.hasPointers())
                exp.setNoderefOperand();

            // When e1 is a template lambda, this cast may instantiate it with
            // the type 'to'.
            exp.e1 = inferType(exp.e1, exp.to);
        }

        if (auto e = unaSemantic(exp, sc))
        {
            result = e;
            return;
        }

        // for static alias this: https://issues.dlang.org/show_bug.cgi?id=17684
        if (exp.e1.op == TOK.type)
            exp.e1 = resolveAliasThis(sc, exp.e1);

        auto e1x = resolveProperties(sc, exp.e1);
        if (e1x.op == TOK.error)
        {
            result = e1x;
            return;
        }
        if (e1x.checkType())
            return setError();
        exp.e1 = e1x;

        if (!exp.e1.type)
        {
            exp.error("cannot cast `%s`", exp.e1.toChars());
            return setError();
        }

        // https://issues.dlang.org/show_bug.cgi?id=19954
        if (exp.e1.type.ty == Ttuple)
        {
            TupleExp te = exp.e1.isTupleExp();
            if (te.exps.dim == 1)
                exp.e1 = (*te.exps)[0];
        }

        // only allow S(x) rewrite if cast specified S explicitly.
        // See https://issues.dlang.org/show_bug.cgi?id=18545
        const bool allowImplicitConstruction = exp.to !is null;

        if (!exp.to) // Handle cast(const) and cast(immutable), etc.
        {
            exp.to = exp.e1.type.castMod(exp.mod);
            exp.to = exp.to.typeSemantic(exp.loc, sc);

            if (exp.to == Type.terror)
                return setError();
        }

        if (exp.to.ty == Ttuple)
        {
            exp.error("cannot cast `%s` to tuple type `%s`", exp.e1.toChars(), exp.to.toChars());
            return setError();
        }

        // cast(void) is used to mark e1 as unused, so it is safe
        if (exp.to.ty == Tvoid)
        {
            exp.type = exp.to;
            result = exp;
            return;
        }

        if (!exp.to.equals(exp.e1.type) && exp.mod == cast(ubyte)~0)
        {
            if (Expression e = exp.op_overload(sc))
            {
                result = e.implicitCastTo(sc, exp.to);
                return;
            }
        }

        Type t1b = exp.e1.type.toBasetype();
        Type tob = exp.to.toBasetype();

        if (allowImplicitConstruction && tob.ty == Tstruct && !tob.equals(t1b))
        {
            /* Look to replace:
             *  cast(S)t
             * with:
             *  S(t)
             */

            // Rewrite as to.call(e1)
            Expression e = new TypeExp(exp.loc, exp.to);
            e = new CallExp(exp.loc, e, exp.e1);
            e = e.trySemantic(sc);
            if (e)
            {
                result = e;
                return;
            }
        }

        if (!t1b.equals(tob) && (t1b.ty == Tarray || t1b.ty == Tsarray))
        {
            if (checkNonAssignmentArrayOp(exp.e1))
                return setError();
        }

        // Look for casting to a vector type
        if (tob.ty == Tvector && t1b.ty != Tvector)
        {
            result = new VectorExp(exp.loc, exp.e1, exp.to);
            result = result.expressionSemantic(sc);
            return;
        }

        Expression ex = exp.e1.castTo(sc, exp.to);
        if (ex.op == TOK.error)
        {
            result = ex;
            return;
        }

        // Check for unsafe casts
        if (!sc.intypeof &&
            !(sc.flags & SCOPE.debug_) &&
            !isSafeCast(ex, t1b, tob) &&
            (!sc.func && sc.stc & STC.safe || sc.func && sc.func.setUnsafe()))
        {
            exp.error("cast from `%s` to `%s` not allowed in safe code", exp.e1.type.toChars(), exp.to.toChars());
            return setError();
        }

        // `object.__ArrayCast` is a rewrite of an old runtime hook `_d_arraycast`. `_d_arraycast` was not built
        // to handle certain casts.  Those casts which `object.__ArrayCast` does not support are filtered out.
        // See `e2ir.toElemCast` for other types of casts.  If `object.__ArrayCast` is improved to support more
        // casts these conditions and potentially some logic in `e2ir.toElemCast` can be removed.
        if (tob.ty == Tarray)
        {
            // https://issues.dlang.org/show_bug.cgi?id=19840
            if (auto ad = isAggregate(t1b))
            {
                if (ad.aliasthis)
                {
                    Expression e = resolveAliasThis(sc, exp.e1);
                    e = new CastExp(exp.loc, e, exp.to);
                    result = e.expressionSemantic(sc);
                    return;
                }
            }

            if(t1b.ty == Tarray && exp.e1.op != TOK.arrayLiteral && (sc.flags & SCOPE.ctfe) == 0)
            {
                auto tFrom = t1b.nextOf();
                auto tTo = tob.nextOf();

                // https://issues.dlang.org/show_bug.cgi?id=20130
                if (exp.e1.op != TOK.string_ || !ex.isStringExp)
                {
                    const uint fromSize = cast(uint)tFrom.size();
                    const uint toSize = cast(uint)tTo.size();

                    // If array element sizes do not match, we must adjust the dimensions
                    if (fromSize != toSize)
                    {
                        if (!verifyHookExist(exp.loc, *sc, Id.__ArrayCast, "casting array of structs"))
                            return setError();

                        // A runtime check is needed in case arrays don't line up.  That check should
                        // be done in the implementation of `object.__ArrayCast`
                        if (toSize == 0 || (fromSize % toSize) != 0)
                        {
                            // lower to `object.__ArrayCast!(TFrom, TTo)(from)`

                            // fully qualify as `object.__ArrayCast`
                            Expression id = new IdentifierExp(exp.loc, Id.empty);
                            auto dotid = new DotIdExp(exp.loc, id, Id.object);

                            auto tiargs = new Objects();
                            tiargs.push(tFrom);
                            tiargs.push(tTo);
                            auto dt = new DotTemplateInstanceExp(exp.loc, dotid, Id.__ArrayCast, tiargs);

                            auto arguments = new Expressions();
                            arguments.push(exp.e1);
                            Expression ce = new CallExp(exp.loc, dt, arguments);

                            result = expressionSemantic(ce, sc);
                            return;
                        }
                    }
                }
            }
        }

        result = ex;
    }

    override void visit(VectorExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("VectorExp::semantic('%s')\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        exp.e1 = exp.e1.expressionSemantic(sc);
        exp.type = exp.to.typeSemantic(exp.loc, sc);
        if (exp.e1.op == TOK.error || exp.type.ty == Terror)
        {
            result = exp.e1;
            return;
        }

        Type tb = exp.type.toBasetype();
        assert(tb.ty == Tvector);
        TypeVector tv = cast(TypeVector)tb;
        Type te = tv.elementType();
        exp.dim = cast(int)(tv.size(exp.loc) / te.size(exp.loc));

        bool checkElem(Expression elem)
        {
            if (elem.isConst() == 1)
                return false;

             exp.error("constant expression expected, not `%s`", elem.toChars());
             return true;
        }

        exp.e1 = exp.e1.optimize(WANTvalue);
        bool res;
        if (exp.e1.op == TOK.arrayLiteral)
        {
            foreach (i; 0 .. exp.dim)
            {
                // Do not stop on first error - check all AST nodes even if error found
                res |= checkElem(exp.e1.isArrayLiteralExp()[i]);
            }
        }
        else if (exp.e1.type.ty == Tvoid)
            checkElem(exp.e1);

        result = res ? new ErrorExp() : exp;
    }

    override void visit(VectorArrayExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("VectorArrayExp::semantic('%s')\n", e.toChars());
        }
        if (!e.type)
        {
            unaSemantic(e, sc);
            e.e1 = resolveProperties(sc, e.e1);

            if (e.e1.op == TOK.error)
            {
                result = e.e1;
                return;
            }
            assert(e.e1.type.ty == Tvector);
            e.type = e.e1.type.isTypeVector().basetype;
        }
        result = e;
    }

    override void visit(SliceExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("SliceExp::semantic('%s')\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        // operator overloading should be handled in ArrayExp already.
        if (Expression ex = unaSemantic(exp, sc))
        {
            result = ex;
            return;
        }
        exp.e1 = resolveProperties(sc, exp.e1);
        if (exp.e1.op == TOK.type && exp.e1.type.ty != Ttuple)
        {
            if (exp.lwr || exp.upr)
            {
                exp.error("cannot slice type `%s`", exp.e1.toChars());
                return setError();
            }
            Expression e = new TypeExp(exp.loc, exp.e1.type.arrayOf());
            result = e.expressionSemantic(sc);
            return;
        }
        if (!exp.lwr && !exp.upr)
        {
            if (exp.e1.op == TOK.arrayLiteral)
            {
                // Convert [a,b,c][] to [a,b,c]
                Type t1b = exp.e1.type.toBasetype();
                Expression e = exp.e1;
                if (t1b.ty == Tsarray)
                {
                    e = e.copy();
                    e.type = t1b.nextOf().arrayOf();
                }
                result = e;
                return;
            }
            if (exp.e1.op == TOK.slice)
            {
                // Convert e[][] to e[]
                SliceExp se = cast(SliceExp)exp.e1;
                if (!se.lwr && !se.upr)
                {
                    result = se;
                    return;
                }
            }
            if (isArrayOpOperand(exp.e1))
            {
                // Convert (a[]+b[])[] to a[]+b[]
                result = exp.e1;
                return;
            }
        }
        if (exp.e1.op == TOK.error)
        {
            result = exp.e1;
            return;
        }
        if (exp.e1.type.ty == Terror)
            return setError();

        Type t1b = exp.e1.type.toBasetype();
        if (t1b.ty == Tpointer)
        {
            if ((cast(TypePointer)t1b).next.ty == Tfunction)
            {
                exp.error("cannot slice function pointer `%s`", exp.e1.toChars());
                return setError();
            }
            if (!exp.lwr || !exp.upr)
            {
                exp.error("need upper and lower bound to slice pointer");
                return setError();
            }
            if (sc.func && !sc.intypeof && !(sc.flags & SCOPE.debug_) && sc.func.setUnsafe())
            {
                exp.error("pointer slicing not allowed in safe functions");
                return setError();
            }
        }
        else if (t1b.ty == Tarray)
        {
        }
        else if (t1b.ty == Tsarray)
        {
            if (!exp.arrayop && global.params.vsafe)
            {
                /* Slicing a static array is like taking the address of it.
                 * Perform checks as if e[] was &e
                 */
                if (VarDeclaration v = expToVariable(exp.e1))
                {
                    if (exp.e1.op == TOK.dotVariable)
                    {
                        DotVarExp dve = cast(DotVarExp)exp.e1;
                        if ((dve.e1.op == TOK.this_ || dve.e1.op == TOK.super_) &&
                            !(v.storage_class & STC.ref_))
                        {
                            // because it's a class
                            v = null;
                        }
                    }

                    if (v && !checkAddressVar(sc, exp, v))
                        return setError();
                }
            }
        }
        else if (t1b.ty == Ttuple)
        {
            if (!exp.lwr && !exp.upr)
            {
                result = exp.e1;
                return;
            }
            if (!exp.lwr || !exp.upr)
            {
                exp.error("need upper and lower bound to slice tuple");
                return setError();
            }
        }
        else if (t1b.ty == Tvector)
        {
            // Convert e1 to corresponding static array
            TypeVector tv1 = cast(TypeVector)t1b;
            t1b = tv1.basetype;
            t1b = t1b.castMod(tv1.mod);
            exp.e1.type = t1b;
        }
        else
        {
            exp.error("`%s` cannot be sliced with `[]`", t1b.ty == Tvoid ? exp.e1.toChars() : t1b.toChars());
            return setError();
        }

        /* Run semantic on lwr and upr.
         */
        Scope* scx = sc;
        if (t1b.ty == Tsarray || t1b.ty == Tarray || t1b.ty == Ttuple)
        {
            // Create scope for 'length' variable
            ScopeDsymbol sym = new ArrayScopeSymbol(sc, exp);
            sym.parent = sc.scopesym;
            sc = sc.push(sym);
        }
        if (exp.lwr)
        {
            if (t1b.ty == Ttuple)
                sc = sc.startCTFE();
            exp.lwr = exp.lwr.expressionSemantic(sc);
            exp.lwr = resolveProperties(sc, exp.lwr);
            if (t1b.ty == Ttuple)
                sc = sc.endCTFE();
            exp.lwr = exp.lwr.implicitCastTo(sc, Type.tsize_t);
        }
        if (exp.upr)
        {
            if (t1b.ty == Ttuple)
                sc = sc.startCTFE();
            exp.upr = exp.upr.expressionSemantic(sc);
            exp.upr = resolveProperties(sc, exp.upr);
            if (t1b.ty == Ttuple)
                sc = sc.endCTFE();
            exp.upr = exp.upr.implicitCastTo(sc, Type.tsize_t);
        }
        if (sc != scx)
            sc = sc.pop();
        if (exp.lwr && exp.lwr.type == Type.terror || exp.upr && exp.upr.type == Type.terror)
            return setError();

        if (t1b.ty == Ttuple)
        {
            exp.lwr = exp.lwr.ctfeInterpret();
            exp.upr = exp.upr.ctfeInterpret();
            uinteger_t i1 = exp.lwr.toUInteger();
            uinteger_t i2 = exp.upr.toUInteger();

            TupleExp te;
            TypeTuple tup;
            size_t length;
            if (exp.e1.op == TOK.tuple) // slicing an expression tuple
            {
                te = cast(TupleExp)exp.e1;
                tup = null;
                length = te.exps.dim;
            }
            else if (exp.e1.op == TOK.type) // slicing a type tuple
            {
                te = null;
                tup = cast(TypeTuple)t1b;
                length = Parameter.dim(tup.arguments);
            }
            else
                assert(0);

            if (i2 < i1 || length < i2)
            {
                exp.error("string slice `[%llu .. %llu]` is out of bounds", i1, i2);
                return setError();
            }

            size_t j1 = cast(size_t)i1;
            size_t j2 = cast(size_t)i2;
            Expression e;
            if (exp.e1.op == TOK.tuple)
            {
                auto exps = new Expressions(j2 - j1);
                for (size_t i = 0; i < j2 - j1; i++)
                {
                    (*exps)[i] = (*te.exps)[j1 + i];
                }
                e = new TupleExp(exp.loc, te.e0, exps);
            }
            else
            {
                auto args = new Parameters();
                args.reserve(j2 - j1);
                for (size_t i = j1; i < j2; i++)
                {
                    Parameter arg = Parameter.getNth(tup.arguments, i);
                    args.push(arg);
                }
                e = new TypeExp(exp.e1.loc, new TypeTuple(args));
            }
            e = e.expressionSemantic(sc);
            result = e;
            return;
        }

        exp.type = t1b.nextOf().arrayOf();
        // Allow typedef[] -> typedef[]
        if (exp.type.equals(t1b))
            exp.type = exp.e1.type;

        // We might know $ now
        setLengthVarIfKnown(exp.lengthVar, t1b);

        if (exp.lwr && exp.upr)
        {
            exp.lwr = exp.lwr.optimize(WANTvalue);
            exp.upr = exp.upr.optimize(WANTvalue);

            IntRange lwrRange = getIntRange(exp.lwr);
            IntRange uprRange = getIntRange(exp.upr);

            if (t1b.ty == Tsarray || t1b.ty == Tarray)
            {
                Expression el = new ArrayLengthExp(exp.loc, exp.e1);
                el = el.expressionSemantic(sc);
                el = el.optimize(WANTvalue);
                if (el.op == TOK.int64)
                {
                    // Array length is known at compile-time. Upper is in bounds if it fits length.
                    dinteger_t length = el.toInteger();
                    auto bounds = IntRange(SignExtendedNumber(0), SignExtendedNumber(length));
                    exp.upperIsInBounds = bounds.contains(uprRange);
                }
                else if (exp.upr.op == TOK.int64 && exp.upr.toInteger() == 0)
                {
                    // Upper slice expression is '0'. Value is always in bounds.
                    exp.upperIsInBounds = true;
                }
                else if (exp.upr.op == TOK.variable && (cast(VarExp)exp.upr).var.ident == Id.dollar)
                {
                    // Upper slice expression is '$'. Value is always in bounds.
                    exp.upperIsInBounds = true;
                }
            }
            else if (t1b.ty == Tpointer)
            {
                exp.upperIsInBounds = true;
            }
            else
                assert(0);

            exp.lowerIsLessThanUpper = (lwrRange.imax <= uprRange.imin);

            //printf("upperIsInBounds = %d lowerIsLessThanUpper = %d\n", exp.upperIsInBounds, exp.lowerIsLessThanUpper);
        }

        result = exp;
    }

    override void visit(ArrayLengthExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("ArrayLengthExp::semantic('%s')\n", e.toChars());
        }
        if (e.type)
        {
            result = e;
            return;
        }

        if (Expression ex = unaSemantic(e, sc))
        {
            result = ex;
            return;
        }
        e.e1 = resolveProperties(sc, e.e1);

        e.type = Type.tsize_t;
        result = e;
    }

    override void visit(ArrayExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("ArrayExp::semantic('%s')\n", exp.toChars());
        }
        assert(!exp.type);
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (isAggregate(exp.e1.type))
            exp.error("no `[]` operator overload for type `%s`", exp.e1.type.toChars());
        else if (exp.e1.op == TOK.type && exp.e1.type.ty != Ttuple)
            exp.error("static array of `%s` with multiple lengths not allowed", exp.e1.type.toChars());
        else if (isIndexableNonAggregate(exp.e1.type))
            exp.error("only one index allowed to index `%s`", exp.e1.type.toChars());
        else
            exp.error("cannot use `[]` operator on expression of type `%s`", exp.e1.type.toChars());

        result = new ErrorExp();
    }

    override void visit(DotExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("DotExp::semantic('%s')\n", exp.toChars());
            if (exp.type)
                printf("\ttype = %s\n", exp.type.toChars());
        }
        exp.e1 = exp.e1.expressionSemantic(sc);
        exp.e2 = exp.e2.expressionSemantic(sc);

        if (exp.e1.op == TOK.type)
        {
            result = exp.e2;
            return;
        }
        if (exp.e2.op == TOK.type)
        {
            result = exp.e2;
            return;
        }
        if (exp.e2.op == TOK.template_)
        {
            auto td = (cast(TemplateExp)exp.e2).td;
            Expression e = new DotTemplateExp(exp.loc, exp.e1, td);
            result = e.expressionSemantic(sc);
            return;
        }
        if (!exp.type || exp.e1.op == TOK.this_)
            exp.type = exp.e2.type;
        result = exp;
    }

    override void visit(CommaExp e)
    {
        if (e.type)
        {
            result = e;
            return;
        }

        // Allow `((a,b),(x,y))`
        if (e.allowCommaExp)
        {
            CommaExp.allow(e.e1);
            CommaExp.allow(e.e2);
        }

        if (Expression ex = binSemanticProp(e, sc))
        {
            result = ex;
            return;
        }
        e.e1 = e.e1.addDtorHook(sc);

        if (checkNonAssignmentArrayOp(e.e1))
            return setError();

        e.type = e.e2.type;
        if (e.type !is Type.tvoid && !e.allowCommaExp && !e.isGenerated)
            e.error("Using the result of a comma expression is not allowed");
        result = e;
    }

    override void visit(IntervalExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("IntervalExp::semantic('%s')\n", e.toChars());
        }
        if (e.type)
        {
            result = e;
            return;
        }

        Expression le = e.lwr;
        le = le.expressionSemantic(sc);
        le = resolveProperties(sc, le);

        Expression ue = e.upr;
        ue = ue.expressionSemantic(sc);
        ue = resolveProperties(sc, ue);

        if (le.op == TOK.error)
        {
            result = le;
            return;
        }
        if (ue.op == TOK.error)
        {
            result = ue;
            return;
        }

        e.lwr = le;
        e.upr = ue;

        e.type = Type.tvoid;
        result = e;
    }

    override void visit(DelegatePtrExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("DelegatePtrExp::semantic('%s')\n", e.toChars());
        }
        if (!e.type)
        {
            unaSemantic(e, sc);
            e.e1 = resolveProperties(sc, e.e1);

            if (e.e1.op == TOK.error)
            {
                result = e.e1;
                return;
            }
            e.type = Type.tvoidptr;
        }
        result = e;
    }

    override void visit(DelegateFuncptrExp e)
    {
        static if (LOGSEMANTIC)
        {
            printf("DelegateFuncptrExp::semantic('%s')\n", e.toChars());
        }
        if (!e.type)
        {
            unaSemantic(e, sc);
            e.e1 = resolveProperties(sc, e.e1);
            if (e.e1.op == TOK.error)
            {
                result = e.e1;
                return;
            }
            e.type = e.e1.type.nextOf().pointerTo();
        }
        result = e;
    }

    override void visit(IndexExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("IndexExp::semantic('%s')\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        // operator overloading should be handled in ArrayExp already.
        if (!exp.e1.type)
            exp.e1 = exp.e1.expressionSemantic(sc);
        assert(exp.e1.type); // semantic() should already be run on it
        if (exp.e1.op == TOK.type && exp.e1.type.ty != Ttuple)
        {
            exp.e2 = exp.e2.expressionSemantic(sc);
            exp.e2 = resolveProperties(sc, exp.e2);
            Type nt;
            if (exp.e2.op == TOK.type)
                nt = new TypeAArray(exp.e1.type, exp.e2.type);
            else
                nt = new TypeSArray(exp.e1.type, exp.e2);
            Expression e = new TypeExp(exp.loc, nt);
            result = e.expressionSemantic(sc);
            return;
        }
        if (exp.e1.op == TOK.error)
        {
            result = exp.e1;
            return;
        }
        if (exp.e1.type.ty == Terror)
            return setError();

        // Note that unlike C we do not implement the int[ptr]

        Type t1b = exp.e1.type.toBasetype();

        if (t1b.ty == Tvector)
        {
            // Convert e1 to corresponding static array
            TypeVector tv1 = cast(TypeVector)t1b;
            t1b = tv1.basetype;
            t1b = t1b.castMod(tv1.mod);
            exp.e1.type = t1b;
        }

        /* Run semantic on e2
         */
        Scope* scx = sc;
        if (t1b.ty == Tsarray || t1b.ty == Tarray || t1b.ty == Ttuple)
        {
            // Create scope for 'length' variable
            ScopeDsymbol sym = new ArrayScopeSymbol(sc, exp);
            sym.parent = sc.scopesym;
            sc = sc.push(sym);
        }
        if (t1b.ty == Ttuple)
            sc = sc.startCTFE();
        exp.e2 = exp.e2.expressionSemantic(sc);
        exp.e2 = resolveProperties(sc, exp.e2);
        if (t1b.ty == Ttuple)
            sc = sc.endCTFE();
        if (exp.e2.op == TOK.tuple)
        {
            TupleExp te = cast(TupleExp)exp.e2;
            if (te.exps && te.exps.dim == 1)
                exp.e2 = Expression.combine(te.e0, (*te.exps)[0]); // bug 4444 fix
        }
        if (sc != scx)
            sc = sc.pop();
        if (exp.e2.type == Type.terror)
            return setError();

        if (checkNonAssignmentArrayOp(exp.e1))
            return setError();

        switch (t1b.ty)
        {
        case Tpointer:
            if ((cast(TypePointer)t1b).next.ty == Tfunction)
            {
                exp.error("cannot index function pointer `%s`", exp.e1.toChars());
                return setError();
            }
            exp.e2 = exp.e2.implicitCastTo(sc, Type.tsize_t);
            if (exp.e2.type == Type.terror)
                return setError();
            exp.e2 = exp.e2.optimize(WANTvalue);
            if (exp.e2.op == TOK.int64 && exp.e2.toInteger() == 0)
            {
            }
            else if (sc.func && !(sc.flags & SCOPE.debug_) && sc.func.setUnsafe())
            {
                exp.error("safe function `%s` cannot index pointer `%s`", sc.func.toPrettyChars(), exp.e1.toChars());
                return setError();
            }
            exp.type = (cast(TypeNext)t1b).next;
            break;

        case Tarray:
            exp.e2 = exp.e2.implicitCastTo(sc, Type.tsize_t);
            if (exp.e2.type == Type.terror)
                return setError();
            exp.type = (cast(TypeNext)t1b).next;
            break;

        case Tsarray:
            {
                exp.e2 = exp.e2.implicitCastTo(sc, Type.tsize_t);
                if (exp.e2.type == Type.terror)
                    return setError();
                exp.type = t1b.nextOf();
                break;
            }
        case Taarray:
            {
                TypeAArray taa = cast(TypeAArray)t1b;
                /* We can skip the implicit conversion if they differ only by
                 * constness
                 * https://issues.dlang.org/show_bug.cgi?id=2684
                 * see also bug https://issues.dlang.org/show_bug.cgi?id=2954 b
                 */
                if (!arrayTypeCompatibleWithoutCasting(exp.e2.type, taa.index))
                {
                    exp.e2 = exp.e2.implicitCastTo(sc, taa.index); // type checking
                    if (exp.e2.type == Type.terror)
                        return setError();
                }

                semanticTypeInfo(sc, taa);

                exp.type = taa.next;
                break;
            }
        case Ttuple:
            {
                exp.e2 = exp.e2.implicitCastTo(sc, Type.tsize_t);
                if (exp.e2.type == Type.terror)
                    return setError();

                exp.e2 = exp.e2.ctfeInterpret();
                uinteger_t index = exp.e2.toUInteger();

                TupleExp te;
                TypeTuple tup;
                size_t length;
                if (exp.e1.op == TOK.tuple)
                {
                    te = cast(TupleExp)exp.e1;
                    tup = null;
                    length = te.exps.dim;
                }
                else if (exp.e1.op == TOK.type)
                {
                    te = null;
                    tup = cast(TypeTuple)t1b;
                    length = Parameter.dim(tup.arguments);
                }
                else
                    assert(0);

                if (length <= index)
                {
                    exp.error("array index `[%llu]` is outside array bounds `[0 .. %llu]`", index, cast(ulong)length);
                    return setError();
                }
                Expression e;
                if (exp.e1.op == TOK.tuple)
                {
                    e = (*te.exps)[cast(size_t)index];
                    e = Expression.combine(te.e0, e);
                }
                else
                    e = new TypeExp(exp.e1.loc, Parameter.getNth(tup.arguments, cast(size_t)index).type);
                result = e;
                return;
            }
        default:
            exp.error("`%s` must be an array or pointer type, not `%s`", exp.e1.toChars(), exp.e1.type.toChars());
            return setError();
        }

        // We might know $ now
        setLengthVarIfKnown(exp.lengthVar, t1b);

        if (t1b.ty == Tsarray || t1b.ty == Tarray)
        {
            Expression el = new ArrayLengthExp(exp.loc, exp.e1);
            el = el.expressionSemantic(sc);
            el = el.optimize(WANTvalue);
            if (el.op == TOK.int64)
            {
                exp.e2 = exp.e2.optimize(WANTvalue);
                dinteger_t length = el.toInteger();
                if (length)
                {
                    auto bounds = IntRange(SignExtendedNumber(0), SignExtendedNumber(length - 1));
                    exp.indexIsInBounds = bounds.contains(getIntRange(exp.e2));
                }
            }
        }

        result = exp;
    }

    override void visit(PostExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("PostExp::semantic('%s')\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemantic(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e1x = resolveProperties(sc, exp.e1);
        if (e1x.op == TOK.error)
        {
            result = e1x;
            return;
        }
        exp.e1 = e1x;

        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (exp.e1.checkReadModifyWrite(exp.op))
            return setError();

        if (exp.e1.op == TOK.slice)
        {
            const(char)* s = exp.op == TOK.plusPlus ? "increment" : "decrement";
            exp.error("cannot post-%s array slice `%s`, use pre-%s instead", s, exp.e1.toChars(), s);
            return setError();
        }

        exp.e1 = exp.e1.optimize(WANTvalue);

        Type t1 = exp.e1.type.toBasetype();
        if (t1.ty == Tclass || t1.ty == Tstruct || exp.e1.op == TOK.arrayLength)
        {
            /* Check for operator overloading,
             * but rewrite in terms of ++e instead of e++
             */

            /* If e1 is not trivial, take a reference to it
             */
            Expression de = null;
            if (exp.e1.op != TOK.variable && exp.e1.op != TOK.arrayLength)
            {
                // ref v = e1;
                auto v = copyToTemp(STC.ref_, "__postref", exp.e1);
                de = new DeclarationExp(exp.loc, v);
                exp.e1 = new VarExp(exp.e1.loc, v);
            }

            /* Rewrite as:
             * auto tmp = e1; ++e1; tmp
             */
            auto tmp = copyToTemp(0, "__pitmp", exp.e1);
            Expression ea = new DeclarationExp(exp.loc, tmp);

            Expression eb = exp.e1.syntaxCopy();
            eb = new PreExp(exp.op == TOK.plusPlus ? TOK.prePlusPlus : TOK.preMinusMinus, exp.loc, eb);

            Expression ec = new VarExp(exp.loc, tmp);

            // Combine de,ea,eb,ec
            if (de)
                ea = new CommaExp(exp.loc, de, ea);
            e = new CommaExp(exp.loc, ea, eb);
            e = new CommaExp(exp.loc, e, ec);
            e = e.expressionSemantic(sc);
            result = e;
            return;
        }

        exp.e1 = exp.e1.modifiableLvalue(sc, exp.e1);

        e = exp;
        if (exp.e1.checkScalar() ||
            exp.e1.checkSharedAccess(sc))
            return setError();
        if (exp.e1.checkNoBool())
            return setError();

        if (exp.e1.type.ty == Tpointer)
            e = scaleFactor(exp, sc);
        else
            exp.e2 = exp.e2.castTo(sc, exp.e1.type);
        e.type = exp.e1.type;
        result = e;
    }

    override void visit(PreExp exp)
    {
        Expression e = exp.op_overload(sc);
        // printf("PreExp::semantic('%s')\n", toChars());
        if (e)
        {
            result = e;
            return;
        }

        // Rewrite as e1+=1 or e1-=1
        if (exp.op == TOK.prePlusPlus)
            e = new AddAssignExp(exp.loc, exp.e1, new IntegerExp(exp.loc, 1, Type.tint32));
        else
            e = new MinAssignExp(exp.loc, exp.e1, new IntegerExp(exp.loc, 1, Type.tint32));
        result = e.expressionSemantic(sc);
    }

    /*
     * Get the expression initializer for a specific struct
     *
     * Params:
     *  sd = the struct for which the expression initializer is needed
     *  loc = the location of the initializer
     *  sc = the scope where the expression is located
     *  t = the type of the expression
     *
     * Returns:
     *  The expression initializer or error expression if any errors occured
     */
    private Expression getInitExp(StructDeclaration sd, Loc loc, Scope* sc, Type t)
    {
        if (sd.zeroInit && !sd.isNested())
        {
            // https://issues.dlang.org/show_bug.cgi?id=14606
            // Always use BlitExp for the special expression: (struct = 0)
            return new IntegerExp(loc, 0, Type.tint32);
        }

        if (sd.isNested())
        {
            auto sle = new StructLiteralExp(loc, sd, null, t);
            if (!sd.fill(loc, sle.elements, true))
                return new ErrorExp();
            if (checkFrameAccess(loc, sc, sd, sle.elements.dim))
                return new ErrorExp();

            sle.type = t;
            return sle;
        }

        return t.defaultInit(loc);
    }

    override void visit(AssignExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("AssignExp::semantic('%s')\n", exp.toChars());
        }
        //printf("exp.e1.op = %d, '%s'\n", exp.e1.op, Token.toChars(exp.e1.op));
        //printf("exp.e2.op = %d, '%s'\n", exp.e2.op, Token.toChars(exp.e2.op));

        void setResult(Expression e, int line = __LINE__)
        {
            //printf("line %d\n", line);
            result = e;
        }

        if (exp.type)
        {
            return setResult(exp);
        }

        Expression e1old = exp.e1;

        if (auto e2comma = exp.e2.isCommaExp())
        {
            if (!e2comma.isGenerated)
                exp.error("Using the result of a comma expression is not allowed");

            /* Rewrite to get rid of the comma from rvalue
             *   e1=(e0,e2) => e0,(e1=e2)
             */
            Expression e0;
            exp.e2 = Expression.extractLast(e2comma, e0);
            Expression e = Expression.combine(e0, exp);
            return setResult(e.expressionSemantic(sc));
        }

        /* Look for operator overloading of a[arguments] = e2.
         * Do it before e1.expressionSemantic() otherwise the ArrayExp will have been
         * converted to unary operator overloading already.
         */
        if (auto ae = exp.e1.isArrayExp())
        {
            Expression res;

            ae.e1 = ae.e1.expressionSemantic(sc);
            ae.e1 = resolveProperties(sc, ae.e1);
            Expression ae1old = ae.e1;

            const(bool) maybeSlice =
                (ae.arguments.dim == 0 ||
                 ae.arguments.dim == 1 && (*ae.arguments)[0].op == TOK.interval);

            IntervalExp ie = null;
            if (maybeSlice && ae.arguments.dim)
            {
                assert((*ae.arguments)[0].op == TOK.interval);
                ie = cast(IntervalExp)(*ae.arguments)[0];
            }
            while (true)
            {
                if (ae.e1.op == TOK.error)
                    return setResult(ae.e1);

                Expression e0 = null;
                Expression ae1save = ae.e1;
                ae.lengthVar = null;

                Type t1b = ae.e1.type.toBasetype();
                AggregateDeclaration ad = isAggregate(t1b);
                if (!ad)
                    break;
                if (search_function(ad, Id.indexass))
                {
                    // Deal with $
                    res = resolveOpDollar(sc, ae, &e0);
                    if (!res) // a[i..j] = e2 might be: a.opSliceAssign(e2, i, j)
                        goto Lfallback;
                    if (res.op == TOK.error)
                        return setResult(res);

                    res = exp.e2.expressionSemantic(sc);
                    if (res.op == TOK.error)
                        return setResult(res);
                    exp.e2 = res;

                    /* Rewrite (a[arguments] = e2) as:
                     *      a.opIndexAssign(e2, arguments)
                     */
                    Expressions* a = ae.arguments.copy();
                    a.insert(0, exp.e2);
                    res = new DotIdExp(exp.loc, ae.e1, Id.indexass);
                    res = new CallExp(exp.loc, res, a);
                    if (maybeSlice) // a[] = e2 might be: a.opSliceAssign(e2)
                        res = res.trySemantic(sc);
                    else
                        res = res.expressionSemantic(sc);
                    if (res)
                        return setResult(Expression.combine(e0, res));
                }

            Lfallback:
                if (maybeSlice && search_function(ad, Id.sliceass))
                {
                    // Deal with $
                    res = resolveOpDollar(sc, ae, ie, &e0);
                    if (res.op == TOK.error)
                        return setResult(res);

                    res = exp.e2.expressionSemantic(sc);
                    if (res.op == TOK.error)
                        return setResult(res);

                    exp.e2 = res;

                    /* Rewrite (a[i..j] = e2) as:
                     *      a.opSliceAssign(e2, i, j)
                     */
                    auto a = new Expressions();
                    a.push(exp.e2);
                    if (ie)
                    {
                        a.push(ie.lwr);
                        a.push(ie.upr);
                    }
                    res = new DotIdExp(exp.loc, ae.e1, Id.sliceass);
                    res = new CallExp(exp.loc, res, a);
                    res = res.expressionSemantic(sc);
                    return setResult(Expression.combine(e0, res));
                }

                // No operator overloading member function found yet, but
                // there might be an alias this to try.
                if (ad.aliasthis && t1b != ae.att1)
                {
                    if (!ae.att1 && t1b.checkAliasThisRec())
                        ae.att1 = t1b;

                    /* Rewrite (a[arguments] op e2) as:
                     *      a.aliasthis[arguments] op e2
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

        /* Run this.e1 semantic.
         */
        {
            Expression e1x = exp.e1;

            /* With UFCS, e.f = value
             * Could mean:
             *      .f(e, value)
             * or:
             *      .f(e) = value
             */
            if (auto dti = e1x.isDotTemplateInstanceExp())
            {
                Expression e = dti.semanticY(sc, 1);
                if (!e)
                {
                    return setResult(resolveUFCSProperties(sc, e1x, exp.e2));
                }

                e1x = e;
            }
            else if (auto die = e1x.isDotIdExp())
            {
                Expression e = die.semanticY(sc, 1);
                if (e && isDotOpDispatch(e))
                {
                    /* https://issues.dlang.org/show_bug.cgi?id=19687
                     *
                     * On this branch, e2 is semantically analyzed in resolvePropertiesX,
                     * but that call is done with gagged errors. That is the only time when
                     * semantic gets ran on e2, that is why the error never gets to be printed.
                     * In order to make sure that UFCS is tried with correct parameters, e2
                     * needs to have semantic ran on it.
                     */
                    exp.e2 = exp.e2.expressionSemantic(sc);
                    uint errors = global.startGagging();
                    e = resolvePropertiesX(sc, e, exp.e2);
                    if (global.endGagging(errors))
                        e = null; /* fall down to UFCS */
                    else
                        return setResult(e);
                }
                if (!e)
                    return setResult(resolveUFCSProperties(sc, e1x, exp.e2));
                e1x = e;
            }
            else
            {
                if (auto se = e1x.isSliceExp())
                    se.arrayop = true;

                e1x = e1x.expressionSemantic(sc);
            }

            /* We have f = value.
             * Could mean:
             *      f(value)
             * or:
             *      f() = value
             */
            if (Expression e = resolvePropertiesX(sc, e1x, exp.e2))
                return setResult(e);

            if (e1x.checkRightThis(sc))
            {
                return setError();
            }
            exp.e1 = e1x;
            assert(exp.e1.type);
        }
        Type t1 = exp.e1.type.toBasetype();

        /* Run this.e2 semantic.
         * Different from other binary expressions, the analysis of e2
         * depends on the result of e1 in assignments.
         */
        {
            Expression e2x = inferType(exp.e2, t1.baseElemOf());
            e2x = e2x.expressionSemantic(sc);
            e2x = resolveProperties(sc, e2x);
            if (e2x.op == TOK.type)
                e2x = resolveAliasThis(sc, e2x); //https://issues.dlang.org/show_bug.cgi?id=17684
            if (e2x.op == TOK.error)
                return setResult(e2x);
            if (e2x.checkValue() || e2x.checkSharedAccess(sc))
                return setError();
            exp.e2 = e2x;
        }

        /* Rewrite tuple assignment as a tuple of assignments.
         */
        {
            Expression e2x = exp.e2;

        Ltupleassign:
            if (exp.e1.op == TOK.tuple && e2x.op == TOK.tuple)
            {
                TupleExp tup1 = cast(TupleExp)exp.e1;
                TupleExp tup2 = cast(TupleExp)e2x;
                size_t dim = tup1.exps.dim;
                Expression e = null;
                if (dim != tup2.exps.dim)
                {
                    exp.error("mismatched tuple lengths, %d and %d", cast(int)dim, cast(int)tup2.exps.dim);
                    return setError();
                }
                if (dim == 0)
                {
                    e = new IntegerExp(exp.loc, 0, Type.tint32);
                    e = new CastExp(exp.loc, e, Type.tvoid); // avoid "has no effect" error
                    e = Expression.combine(tup1.e0, tup2.e0, e);
                }
                else
                {
                    auto exps = new Expressions(dim);
                    for (size_t i = 0; i < dim; i++)
                    {
                        Expression ex1 = (*tup1.exps)[i];
                        Expression ex2 = (*tup2.exps)[i];
                        (*exps)[i] = new AssignExp(exp.loc, ex1, ex2);
                    }
                    e = new TupleExp(exp.loc, Expression.combine(tup1.e0, tup2.e0), exps);
                }
                return setResult(e.expressionSemantic(sc));
            }

            /* Look for form: e1 = e2.aliasthis.
             */
            if (exp.e1.op == TOK.tuple)
            {
                TupleDeclaration td = isAliasThisTuple(e2x);
                if (!td)
                    goto Lnomatch;

                assert(exp.e1.type.ty == Ttuple);
                TypeTuple tt = cast(TypeTuple)exp.e1.type;

                Expression e0;
                Expression ev = extractSideEffect(sc, "__tup", e0, e2x);

                auto iexps = new Expressions();
                iexps.push(ev);
                for (size_t u = 0; u < iexps.dim; u++)
                {
                Lexpand:
                    Expression e = (*iexps)[u];

                    Parameter arg = Parameter.getNth(tt.arguments, u);
                    //printf("[%d] iexps.dim = %d, ", u, iexps.dim);
                    //printf("e = (%s %s, %s), ", Token::tochars[e.op], e.toChars(), e.type.toChars());
                    //printf("arg = (%s, %s)\n", arg.toChars(), arg.type.toChars());

                    if (!arg || !e.type.implicitConvTo(arg.type))
                    {
                        // expand initializer to tuple
                        if (expandAliasThisTuples(iexps, u) != -1)
                        {
                            if (iexps.dim <= u)
                                break;
                            goto Lexpand;
                        }
                        goto Lnomatch;
                    }
                }
                e2x = new TupleExp(e2x.loc, e0, iexps);
                e2x = e2x.expressionSemantic(sc);
                if (e2x.op == TOK.error)
                {
                    result = e2x;
                    return;
                }
                // Do not need to overwrite this.e2
                goto Ltupleassign;
            }
        Lnomatch:
        }

        exp.e1.checkSharedAccess(sc);

        /* Inside constructor, if this is the first assignment of object field,
         * rewrite this to initializing the field.
         */
        if (exp.op == TOK.assign
            && exp.e1.checkModifiable(sc) == Modifiable.initialization)
        {
            //printf("[%s] change to init - %s\n", exp.loc.toChars(), exp.toChars());
            auto t = exp.type;
            exp = new ConstructExp(exp.loc, exp.e1, exp.e2);
            exp.type = t;

            // @@@DEPRECATED_2020-06@@@
            // When removing, alter `checkModifiable` to return the correct value.
            if (sc.func.isStaticCtorDeclaration() && !sc.func.isSharedStaticCtorDeclaration() &&
                exp.e1.type.isImmutable())
            {
                deprecation(exp.loc, "initialization of `immutable` variable from `static this` is deprecated.");
                deprecationSupplemental(exp.loc, "Use `shared static this` instead.");
            }

            // https://issues.dlang.org/show_bug.cgi?id=13515
            // set Index::modifiable flag for complex AA element initialization
            if (auto ie1 = exp.e1.isIndexExp())
            {
                Expression e1x = ie1.markSettingAAElem();
                if (e1x.op == TOK.error)
                {
                    result = e1x;
                    return;
                }
            }
        }
        else if (exp.op == TOK.construct && exp.e1.op == TOK.variable &&
                 (cast(VarExp)exp.e1).var.storage_class & (STC.out_ | STC.ref_))
        {
            exp.memset |= MemorySet.referenceInit;
        }

        /* If it is an assignment from a 'foreign' type,
         * check for operator overloading.
         */
        if (exp.memset & MemorySet.referenceInit)
        {
            // If this is an initialization of a reference,
            // do nothing
        }
        else if (t1.ty == Tstruct)
        {
            auto e1x = exp.e1;
            auto e2x = exp.e2;
            auto sd = (cast(TypeStruct)t1).sym;

            if (exp.op == TOK.construct)
            {
                Type t2 = e2x.type.toBasetype();
                if (t2.ty == Tstruct && sd == (cast(TypeStruct)t2).sym)
                {
                    sd.size(exp.loc);
                    if (sd.sizeok != Sizeok.done)
                        return setError();
                    if (!sd.ctor)
                        sd.ctor = sd.searchCtor();

                    // https://issues.dlang.org/show_bug.cgi?id=15661
                    // Look for the form from last of comma chain.
                    auto e2y = lastComma(e2x);

                    CallExp ce = (e2y.op == TOK.call) ? cast(CallExp)e2y : null;
                    DotVarExp dve = (ce && ce.e1.op == TOK.dotVariable)
                        ? cast(DotVarExp)ce.e1 : null;
                    if (sd.ctor && ce && dve && dve.var.isCtorDeclaration() &&
                        // https://issues.dlang.org/show_bug.cgi?id=19389
                        dve.e1.op != TOK.dotVariable &&
                        e2y.type.implicitConvTo(t1))
                    {
                        /* Look for form of constructor call which is:
                         *    __ctmp.ctor(arguments...)
                         */

                        /* Before calling the constructor, initialize
                         * variable with a bit copy of the default
                         * initializer
                         */
                        Expression einit = getInitExp(sd, exp.loc, sc, t1);
                        if (einit.op == TOK.error)
                        {
                            result = einit;
                            return;
                        }

                        auto ae = new BlitExp(exp.loc, exp.e1, einit);
                        ae.type = e1x.type;

                        /* Replace __ctmp being constructed with e1.
                         * We need to copy constructor call expression,
                         * because it may be used in other place.
                         */
                        auto dvx = cast(DotVarExp)dve.copy();
                        dvx.e1 = e1x;
                        auto cx = cast(CallExp)ce.copy();
                        cx.e1 = dvx;
                        if (checkConstructorEscape(sc, cx, false))
                            return setError();

                        Expression e0;
                        Expression.extractLast(e2x, e0);

                        auto e = Expression.combine(e0, ae, cx);
                        e = e.expressionSemantic(sc);
                        result = e;
                        return;
                    }
                    if (sd.postblit || sd.hasCopyCtor)
                    {
                        /* We have a copy constructor for this
                         */
                        if (e2x.op == TOK.question)
                        {
                            /* Rewrite as:
                             *  a ? e1 = b : e1 = c;
                             */
                            CondExp econd = cast(CondExp)e2x;
                            Expression ea1 = new ConstructExp(econd.e1.loc, e1x, econd.e1);
                            Expression ea2 = new ConstructExp(econd.e1.loc, e1x, econd.e2);
                            Expression e = new CondExp(exp.loc, econd.econd, ea1, ea2);
                            result = e.expressionSemantic(sc);
                            return;
                        }

                        if (e2x.isLvalue())
                        {
                            if (sd.hasCopyCtor)
                            {
                                /* Rewrite as:
                                 * e1 = init, e1.copyCtor(e2);
                                 */
                                Expression einit = new BlitExp(exp.loc, exp.e1, getInitExp(sd, exp.loc, sc, t1));
                                einit.type = e1x.type;

                                Expression e;
                                e = new DotIdExp(exp.loc, e1x, Id.ctor);
                                e = new CallExp(exp.loc, e, e2x);
                                e = new CommaExp(exp.loc, einit, e);

                                //printf("e: %s\n", e.toChars());

                                result = e.expressionSemantic(sc);
                                return;
                            }
                            else
                            {
                                if (!e2x.type.implicitConvTo(e1x.type))
                                {
                                    exp.error("conversion error from `%s` to `%s`",
                                        e2x.type.toChars(), e1x.type.toChars());
                                    return setError();
                                }

                                /* Rewrite as:
                                 *  (e1 = e2).postblit();
                                 *
                                 * Blit assignment e1 = e2 returns a reference to the original e1,
                                 * then call the postblit on it.
                                 */
                                Expression e = e1x.copy();
                                e.type = e.type.mutableOf();
                                if (e.type.isShared && !sd.type.isShared)
                                    e.type = e.type.unSharedOf();
                                e = new BlitExp(exp.loc, e, e2x);
                                e = new DotVarExp(exp.loc, e, sd.postblit, false);
                                e = new CallExp(exp.loc, e);
                                result = e.expressionSemantic(sc);
                                return;
                            }
                        }
                        else
                        {
                            /* The struct value returned from the function is transferred
                             * so should not call the destructor on it.
                             */
                            e2x = valueNoDtor(e2x);
                        }
                    }

                    // https://issues.dlang.org/show_bug.cgi?id=19251
                    // if e2 cannot be converted to e1.type, maybe there is an alias this
                    if (!e2x.implicitConvTo(t1))
                    {
                        AggregateDeclaration ad2 = isAggregate(e2x.type);
                        if (ad2 && ad2.aliasthis && !(exp.att2 && e2x.type == exp.att2))
                        {
                            if (!exp.att2 && exp.e2.type.checkAliasThisRec())
                            exp.att2 = exp.e2.type;
                            /* Rewrite (e1 op e2) as:
                             *      (e1 op e2.aliasthis)
                             */
                            exp.e2 = new DotIdExp(exp.e2.loc, exp.e2, ad2.aliasthis.ident);
                            result = exp.expressionSemantic(sc);
                            return;
                        }
                    }
                }
                else if (!e2x.implicitConvTo(t1))
                {
                    sd.size(exp.loc);
                    if (sd.sizeok != Sizeok.done)
                        return setError();
                    if (!sd.ctor)
                        sd.ctor = sd.searchCtor();

                    if (sd.ctor)
                    {
                        /* Look for implicit constructor call
                         * Rewrite as:
                         *  e1 = init, e1.ctor(e2)
                         */

                        /* Fix Issue 5153 : https://issues.dlang.org/show_bug.cgi?id=5153
                         * Using `new` to initialize a struct object is a common mistake, but
                         * the error message from the compiler is not very helpful in that
                         * case. If exp.e2 is a NewExp and the type of new is the same as
                         * the type as exp.e1 (struct in this case), then we know for sure
                         * that the user wants to instantiate a struct. This is done to avoid
                         * issuing an error when the user actually wants to call a constructor
                         * which receives a class object.
                         *
                         * Foo f = new Foo2(0); is a valid expression if Foo has a constructor
                         * which receives an instance of a Foo2 class
                         */
                        if (exp.e2.op == TOK.new_)
                        {
                            auto newExp = cast(NewExp)(exp.e2);
                            if (newExp.newtype && newExp.newtype == t1)
                            {
                                error(exp.loc, "cannot implicitly convert expression `%s` of type `%s` to `%s`",
                                      newExp.toChars(), newExp.type.toChars(), t1.toChars());
                                errorSupplemental(exp.loc, "Perhaps remove the `new` keyword?");
                                return setError();
                            }
                        }

                        Expression einit = new BlitExp(exp.loc, e1x, getInitExp(sd, exp.loc, sc, t1));
                        einit.type = e1x.type;

                        Expression e;
                        e = new DotIdExp(exp.loc, e1x, Id.ctor);
                        e = new CallExp(exp.loc, e, e2x);
                        e = new CommaExp(exp.loc, einit, e);
                        e = e.expressionSemantic(sc);
                        result = e;
                        return;
                    }
                    if (search_function(sd, Id.call))
                    {
                        /* Look for static opCall
                         * https://issues.dlang.org/show_bug.cgi?id=2702
                         * Rewrite as:
                         *  e1 = typeof(e1).opCall(arguments)
                         */
                        e2x = typeDotIdExp(e2x.loc, e1x.type, Id.call);
                        e2x = new CallExp(exp.loc, e2x, exp.e2);

                        e2x = e2x.expressionSemantic(sc);
                        e2x = resolveProperties(sc, e2x);
                        if (e2x.op == TOK.error)
                        {
                            result = e2x;
                            return;
                        }
                        if (e2x.checkValue() || e2x.checkSharedAccess(sc))
                            return setError();
                    }
                }
                else // https://issues.dlang.org/show_bug.cgi?id=11355
                {
                    AggregateDeclaration ad2 = isAggregate(e2x.type);
                    if (ad2 && ad2.aliasthis && !(exp.att2 && e2x.type == exp.att2))
                    {
                        if (!exp.att2 && exp.e2.type.checkAliasThisRec())
                            exp.att2 = exp.e2.type;
                        /* Rewrite (e1 op e2) as:
                         *      (e1 op e2.aliasthis)
                         */
                        exp.e2 = new DotIdExp(exp.e2.loc, exp.e2, ad2.aliasthis.ident);
                        result = exp.expressionSemantic(sc);
                        return;
                    }
                }
            }
            else if (exp.op == TOK.assign)
            {
                if (e1x.op == TOK.index && (cast(IndexExp)e1x).e1.type.toBasetype().ty == Taarray)
                {
                    /*
                     * Rewrite:
                     *      aa[key] = e2;
                     * as:
                     *      ref __aatmp = aa;
                     *      ref __aakey = key;
                     *      ref __aaval = e2;
                     *      (__aakey in __aatmp
                     *          ? __aatmp[__aakey].opAssign(__aaval)
                     *          : ConstructExp(__aatmp[__aakey], __aaval));
                     */
                    // ensure we keep the expr modifiable
                    Expression esetting = (cast(IndexExp)e1x).markSettingAAElem();
                    if (esetting.op == TOK.error)
                    {
                        result = esetting;
                        return;
                    }
                    assert(esetting.op == TOK.index);
                    IndexExp ie = cast(IndexExp) esetting;
                    Type t2 = e2x.type.toBasetype();

                    Expression e0 = null;
                    Expression ea = extractSideEffect(sc, "__aatmp", e0, ie.e1);
                    Expression ek = extractSideEffect(sc, "__aakey", e0, ie.e2);
                    Expression ev = extractSideEffect(sc, "__aaval", e0, e2x);

                    AssignExp ae = cast(AssignExp)exp.copy();
                    ae.e1 = new IndexExp(exp.loc, ea, ek);
                    ae.e1 = ae.e1.expressionSemantic(sc);
                    ae.e1 = ae.e1.optimize(WANTvalue);
                    ae.e2 = ev;
                    Expression e = ae.op_overload(sc);
                    if (e)
                    {
                        Expression ey = null;
                        if (t2.ty == Tstruct && sd == t2.toDsymbol(sc))
                        {
                            ey = ev;
                        }
                        else if (!ev.implicitConvTo(ie.type) && sd.ctor)
                        {
                            // Look for implicit constructor call
                            // Rewrite as S().ctor(e2)
                            ey = new StructLiteralExp(exp.loc, sd, null);
                            ey = new DotIdExp(exp.loc, ey, Id.ctor);
                            ey = new CallExp(exp.loc, ey, ev);
                            ey = ey.trySemantic(sc);
                        }
                        if (ey)
                        {
                            Expression ex;
                            ex = new IndexExp(exp.loc, ea, ek);
                            ex = ex.expressionSemantic(sc);
                            ex = ex.optimize(WANTvalue);
                            ex = ex.modifiableLvalue(sc, ex); // allocate new slot

                            ey = new ConstructExp(exp.loc, ex, ey);
                            ey = ey.expressionSemantic(sc);
                            if (ey.op == TOK.error)
                            {
                                result = ey;
                                return;
                            }
                            ex = e;

                            // https://issues.dlang.org/show_bug.cgi?id=14144
                            // The whole expression should have the common type
                            // of opAssign() return and assigned AA entry.
                            // Even if there's no common type, expression should be typed as void.
                            Type t = null;
                            if (!typeMerge(sc, TOK.question, &t, &ex, &ey))
                            {
                                ex = new CastExp(ex.loc, ex, Type.tvoid);
                                ey = new CastExp(ey.loc, ey, Type.tvoid);
                            }
                            e = new CondExp(exp.loc, new InExp(exp.loc, ek, ea), ex, ey);
                        }
                        e = Expression.combine(e0, e);
                        e = e.expressionSemantic(sc);
                        result = e;
                        return;
                    }
                }
                else
                {
                    Expression e = exp.op_overload(sc);
                    if (e)
                    {
                        result = e;
                        return;
                    }
                }
            }
            else
                assert(exp.op == TOK.blit);

            exp.e1 = e1x;
            exp.e2 = e2x;
        }
        else if (t1.ty == Tclass)
        {
            // Disallow assignment operator overloads for same type
            if (exp.op == TOK.assign && !exp.e2.implicitConvTo(exp.e1.type))
            {
                Expression e = exp.op_overload(sc);
                if (e)
                {
                    result = e;
                    return;
                }
            }
        }
        else if (t1.ty == Tsarray)
        {
            // SliceExp cannot have static array type without context inference.
            assert(exp.e1.op != TOK.slice);
            Expression e1x = exp.e1;
            Expression e2x = exp.e2;

            if (e2x.implicitConvTo(e1x.type))
            {
                if (exp.op != TOK.blit && (e2x.op == TOK.slice && (cast(UnaExp)e2x).e1.isLvalue() || e2x.op == TOK.cast_ && (cast(UnaExp)e2x).e1.isLvalue() || e2x.op != TOK.slice && e2x.isLvalue()))
                {
                    if (e1x.checkPostblit(sc, t1))
                        return setError();
                }

                // e2 matches to t1 because of the implicit length match, so
                if (isUnaArrayOp(e2x.op) || isBinArrayOp(e2x.op))
                {
                    // convert e1 to e1[]
                    // e.g. e1[] = a[] + b[];
                    auto sle = new SliceExp(e1x.loc, e1x, null, null);
                    sle.arrayop = true;
                    e1x = sle.expressionSemantic(sc);
                }
                else
                {
                    // convert e2 to t1 later
                    // e.g. e1 = [1, 2, 3];
                }
            }
            else
            {
                if (e2x.implicitConvTo(t1.nextOf().arrayOf()) > MATCH.nomatch)
                {
                    uinteger_t dim1 = (cast(TypeSArray)t1).dim.toInteger();
                    uinteger_t dim2 = dim1;
                    if (auto ale = e2x.isArrayLiteralExp())
                    {
                        dim2 = ale.elements ? ale.elements.dim : 0;
                    }
                    else if (auto se = e2x.isSliceExp())
                    {
                        Type tx = toStaticArrayType(se);
                        if (tx)
                            dim2 = (cast(TypeSArray)tx).dim.toInteger();
                    }
                    if (dim1 != dim2)
                    {
                        exp.error("mismatched array lengths, %d and %d", cast(int)dim1, cast(int)dim2);
                        return setError();
                    }
                }

                // May be block or element-wise assignment, so
                // convert e1 to e1[]
                if (exp.op != TOK.assign)
                {
                    // If multidimensional static array, treat as one large array
                    //
                    // Find the appropriate array type depending on the assignment, e.g.
                    // int[3] = int => int[3]
                    // int[3][2] = int => int[6]
                    // int[3][2] = int[] => int[3][2]
                    // int[3][2][4] + int => int[24]
                    // int[3][2][4] + int[] => int[3][8]
                    ulong dim = t1.isTypeSArray().dim.toUInteger();
                    auto type = t1.nextOf();

                    for (TypeSArray tsa; (tsa = type.isTypeSArray()) !is null; )
                    {
                        import core.checkedint : mulu;

                        // Accumulate skipped dimensions
                        bool overflow = false;
                        dim = mulu(dim, tsa.dim.toUInteger(), overflow);
                        if (overflow || dim >= uint.max)
                        {
                            // dym exceeds maximum array size
                            exp.error("static array `%s` size overflowed to %llu",
                                        e1x.type.toChars(), cast(ulong) dim);
                            return setError();
                        }

                        // Move to the element type
                        type = tsa.nextOf().toBasetype();

                        // Rewrite ex1 as a static array if a matching type was found
                        if (e2x.implicitConvTo(type) > MATCH.nomatch)
                        {
                            e1x.type = type.sarrayOf(dim);
                            break;
                        }
                    }
                }
                auto sle = new SliceExp(e1x.loc, e1x, null, null);
                sle.arrayop = true;
                e1x = sle.expressionSemantic(sc);
            }
            if (e1x.op == TOK.error)
                return setResult(e1x);
            if (e2x.op == TOK.error)
                return setResult(e2x);

            exp.e1 = e1x;
            exp.e2 = e2x;
            t1 = e1x.type.toBasetype();
        }
        /* Check the mutability of e1.
         */
        if (auto ale = exp.e1.isArrayLengthExp())
        {
            // e1 is not an lvalue, but we let code generator handle it

            auto ale1x = ale.e1.modifiableLvalue(sc, exp.e1);
            if (ale1x.op == TOK.error)
                return setResult(ale1x);
            ale.e1 = ale1x;

            Type tn = ale.e1.type.toBasetype().nextOf();
            checkDefCtor(ale.loc, tn);

            Identifier hook = global.params.tracegc ? Id._d_arraysetlengthTTrace : Id._d_arraysetlengthT;
            if (!verifyHookExist(exp.loc, *sc, Id._d_arraysetlengthTImpl, "resizing arrays"))
                return setError();

            // Lower to object._d_arraysetlengthTImpl!(typeof(e1))._d_arraysetlengthT{,Trace}(e1, e2)
            Expression id = new IdentifierExp(ale.loc, Id.empty);
            id = new DotIdExp(ale.loc, id, Id.object);
            auto tiargs = new Objects();
            tiargs.push(ale.e1.type);
            id = new DotTemplateInstanceExp(ale.loc, id, Id._d_arraysetlengthTImpl, tiargs);
            id = new DotIdExp(ale.loc, id, hook);
            id = id.expressionSemantic(sc);

            auto arguments = new Expressions();
            arguments.reserve(5);
            if (global.params.tracegc)
            {
                auto funcname = (sc.callsc && sc.callsc.func) ? sc.callsc.func.toPrettyChars() : sc.func.toPrettyChars();
                arguments.push(new StringExp(exp.loc, exp.loc.filename.toDString()));
                arguments.push(new IntegerExp(exp.loc, exp.loc.linnum, Type.tint32));
                arguments.push(new StringExp(exp.loc, funcname.toDString()));
            }
            arguments.push(ale.e1);
            arguments.push(exp.e2);

            Expression ce = new CallExp(ale.loc, id, arguments);
            auto res = ce.expressionSemantic(sc);
            // if (global.params.verbose)
            //     message("lowered   %s =>\n          %s", exp.toChars(), res.toChars());
            return setResult(res);
        }
        else if (auto se = exp.e1.isSliceExp())
        {
            Type tn = se.type.nextOf();
            const fun = sc.func;
            if (exp.op == TOK.assign && !tn.isMutable() &&
                // allow modifiation in module ctor, see
                // https://issues.dlang.org/show_bug.cgi?id=9884
                (!fun || (fun && !fun.isStaticCtorDeclaration())))
            {
                exp.error("slice `%s` is not mutable", se.toChars());
                return setError();
            }

            if (exp.op == TOK.assign && !tn.baseElemOf().isAssignable())
            {
                exp.error("slice `%s` is not mutable, struct `%s` has immutable members",
                    exp.e1.toChars(), tn.baseElemOf().toChars());
                result = new ErrorExp();
                return;
            }

            // For conditional operator, both branches need conversion.
            while (se.e1.op == TOK.slice)
                se = cast(SliceExp)se.e1;
            if (se.e1.op == TOK.question && se.e1.type.toBasetype().ty == Tsarray)
            {
                se.e1 = se.e1.modifiableLvalue(sc, exp.e1);
                if (se.e1.op == TOK.error)
                    return setResult(se.e1);
            }
        }
        else
        {
            if (t1.ty == Tsarray && exp.op == TOK.assign)
            {
                Type tn = exp.e1.type.nextOf();
                if (tn && !tn.baseElemOf().isAssignable())
                {
                    exp.error("array `%s` is not mutable, struct `%s` has immutable members",
                        exp.e1.toChars(), tn.baseElemOf().toChars());
                    result = new ErrorExp();
                    return;
                }
            }

            Expression e1x = exp.e1;

            // Try to do a decent error message with the expression
            // before it got constant folded

            if (e1x.op != TOK.variable)
                e1x = e1x.optimize(WANTvalue);

            if (exp.op == TOK.assign)
                e1x = e1x.modifiableLvalue(sc, e1old);

            if (checkIfIsStructLiteralDotExpr(e1x))
                return setError();

            if (e1x.op == TOK.error)
            {
                result = e1x;
                return;
            }
            exp.e1 = e1x;
        }

        /* Tweak e2 based on the type of e1.
         */
        Expression e2x = exp.e2;
        Type t2 = e2x.type.toBasetype();

        // If it is a array, get the element type. Note that it may be
        // multi-dimensional.
        Type telem = t1;
        while (telem.ty == Tarray)
            telem = telem.nextOf();

        if (exp.e1.op == TOK.slice && t1.nextOf() &&
            (telem.ty != Tvoid || e2x.op == TOK.null_) &&
            e2x.implicitConvTo(t1.nextOf()))
        {
            // Check for block assignment. If it is of type void[], void[][], etc,
            // '= null' is the only allowable block assignment (Bug 7493)
            exp.memset |= MemorySet.blockAssign;    // make it easy for back end to tell what this is
            e2x = e2x.implicitCastTo(sc, t1.nextOf());
            if (exp.op != TOK.blit && e2x.isLvalue() && exp.e1.checkPostblit(sc, t1.nextOf()))
                return setError();
        }
        else if (exp.e1.op == TOK.slice &&
                 (t2.ty == Tarray || t2.ty == Tsarray) &&
                 t2.nextOf().implicitConvTo(t1.nextOf()))
        {
            // Check element-wise assignment.

            /* If assigned elements number is known at compile time,
             * check the mismatch.
             */
            SliceExp se1 = cast(SliceExp)exp.e1;
            TypeSArray tsa1 = cast(TypeSArray)toStaticArrayType(se1);
            TypeSArray tsa2 = null;
            if (auto ale = e2x.isArrayLiteralExp())
                tsa2 = cast(TypeSArray)t2.nextOf().sarrayOf(ale.elements.dim);
            else if (auto se = e2x.isSliceExp())
                tsa2 = cast(TypeSArray)toStaticArrayType(se);
            else
                tsa2 = t2.isTypeSArray();
            if (tsa1 && tsa2)
            {
                uinteger_t dim1 = tsa1.dim.toInteger();
                uinteger_t dim2 = tsa2.dim.toInteger();
                if (dim1 != dim2)
                {
                    exp.error("mismatched array lengths, %d and %d", cast(int)dim1, cast(int)dim2);
                    return setError();
                }
            }

            if (exp.op != TOK.blit &&
                (e2x.op == TOK.slice && (cast(UnaExp)e2x).e1.isLvalue() ||
                 e2x.op == TOK.cast_ && (cast(UnaExp)e2x).e1.isLvalue() ||
                 e2x.op != TOK.slice && e2x.isLvalue()))
            {
                if (exp.e1.checkPostblit(sc, t1.nextOf()))
                    return setError();
            }

            if (0 && global.params.warnings != DiagnosticReporting.off && !global.gag && exp.op == TOK.assign &&
                e2x.op != TOK.slice && e2x.op != TOK.assign &&
                e2x.op != TOK.arrayLiteral && e2x.op != TOK.string_ &&
                !(e2x.op == TOK.add || e2x.op == TOK.min ||
                  e2x.op == TOK.mul || e2x.op == TOK.div ||
                  e2x.op == TOK.mod || e2x.op == TOK.xor ||
                  e2x.op == TOK.and || e2x.op == TOK.or ||
                  e2x.op == TOK.pow ||
                  e2x.op == TOK.tilde || e2x.op == TOK.negate))
            {
                const(char)* e1str = exp.e1.toChars();
                const(char)* e2str = e2x.toChars();
                exp.warning("explicit element-wise assignment `%s = (%s)[]` is better than `%s = %s`", e1str, e2str, e1str, e2str);
            }

            Type t2n = t2.nextOf();
            Type t1n = t1.nextOf();
            int offset;
            if (t2n.equivalent(t1n) ||
                t1n.isBaseOf(t2n, &offset) && offset == 0)
            {
                /* Allow copy of distinct qualifier elements.
                 * eg.
                 *  char[] dst;  const(char)[] src;
                 *  dst[] = src;
                 *
                 *  class C {}   class D : C {}
                 *  C[2] ca;  D[] da;
                 *  ca[] = da;
                 */
                if (isArrayOpValid(e2x))
                {
                    // Don't add CastExp to keep AST for array operations
                    e2x = e2x.copy();
                    e2x.type = exp.e1.type.constOf();
                }
                else
                    e2x = e2x.castTo(sc, exp.e1.type.constOf());
            }
            else
            {
                /* https://issues.dlang.org/show_bug.cgi?id=15778
                 * A string literal has an array type of immutable
                 * elements by default, and normally it cannot be convertible to
                 * array type of mutable elements. But for element-wise assignment,
                 * elements need to be const at best. So we should give a chance
                 * to change code unit size for polysemous string literal.
                 */
                if (e2x.op == TOK.string_)
                    e2x = e2x.implicitCastTo(sc, exp.e1.type.constOf());
                else
                    e2x = e2x.implicitCastTo(sc, exp.e1.type);
            }
            if (t1n.toBasetype.ty == Tvoid && t2n.toBasetype.ty == Tvoid)
            {
                if (!sc.intypeof && sc.func && !(sc.flags & SCOPE.debug_) && sc.func.setUnsafe())
                {
                    exp.error("cannot copy `void[]` to `void[]` in `@safe` code");
                    return setError();
                }
            }
        }
        else
        {
            if (0 && global.params.warnings != DiagnosticReporting.off && !global.gag && exp.op == TOK.assign &&
                t1.ty == Tarray && t2.ty == Tsarray &&
                e2x.op != TOK.slice &&
                t2.implicitConvTo(t1))
            {
                // Disallow ar[] = sa (Converted to ar[] = sa[])
                // Disallow da   = sa (Converted to da   = sa[])
                const(char)* e1str = exp.e1.toChars();
                const(char)* e2str = e2x.toChars();
                const(char)* atypestr = exp.e1.op == TOK.slice ? "element-wise" : "slice";
                exp.warning("explicit %s assignment `%s = (%s)[]` is better than `%s = %s`", atypestr, e1str, e2str, e1str, e2str);
            }
            if (exp.op == TOK.blit)
                e2x = e2x.castTo(sc, exp.e1.type);
            else
            {
                e2x = e2x.implicitCastTo(sc, exp.e1.type);

                // Fix Issue 13435: https://issues.dlang.org/show_bug.cgi?id=13435

                // If the implicit cast has failed and the assign expression is
                // the initialization of a struct member field
                if (e2x.op == TOK.error && exp.op == TOK.construct && t1.ty == Tstruct)
                {
                    scope sd = (cast(TypeStruct)t1).sym;
                    Dsymbol opAssign = search_function(sd, Id.assign);

                    // and the struct defines an opAssign
                    if (opAssign)
                    {
                        // offer more information about the cause of the problem
                        errorSupplemental(exp.loc,
                                          "`%s` is the first assignment of `%s` therefore it represents its initialization",
                                          exp.toChars(), exp.e1.toChars());
                        errorSupplemental(exp.loc,
                                          "`opAssign` methods are not used for initialization, but for subsequent assignments");
                    }
                }
            }
        }
        if (e2x.op == TOK.error)
        {
            result = e2x;
            return;
        }
        exp.e2 = e2x;
        t2 = exp.e2.type.toBasetype();

        /* Look for array operations
         */
        if ((t2.ty == Tarray || t2.ty == Tsarray) && isArrayOpValid(exp.e2))
        {
            // Look for valid array operations
            if (!(exp.memset & MemorySet.blockAssign) &&
                exp.e1.op == TOK.slice &&
                (isUnaArrayOp(exp.e2.op) || isBinArrayOp(exp.e2.op)))
            {
                exp.type = exp.e1.type;
                if (exp.op == TOK.construct) // https://issues.dlang.org/show_bug.cgi?id=10282
                                        // tweak mutability of e1 element
                    exp.e1.type = exp.e1.type.nextOf().mutableOf().arrayOf();
                result = arrayOp(exp, sc);
                return;
            }

            // Drop invalid array operations in e2
            //  d = a[] + b[], d = (a[] + b[])[0..2], etc
            if (checkNonAssignmentArrayOp(exp.e2, !(exp.memset & MemorySet.blockAssign) && exp.op == TOK.assign))
                return setError();

            // Remains valid array assignments
            //  d = d[], d = [1,2,3], etc
        }

        /* Don't allow assignment to classes that were allocated on the stack with:
         *      scope Class c = new Class();
         */
        if (exp.e1.op == TOK.variable && exp.op == TOK.assign)
        {
            VarExp ve = cast(VarExp)exp.e1;
            VarDeclaration vd = ve.var.isVarDeclaration();
            if (vd && (vd.onstack || vd.mynew))
            {
                assert(t1.ty == Tclass);
                exp.error("cannot rebind scope variables");
            }
        }

        if (exp.e1.op == TOK.variable && (cast(VarExp)exp.e1).var.ident == Id.ctfe)
        {
            exp.error("cannot modify compiler-generated variable `__ctfe`");
        }

        exp.type = exp.e1.type;
        assert(exp.type);
        auto res = exp.op == TOK.assign ? exp.reorderSettingAAElem(sc) : exp;
        checkAssignEscape(sc, res, false);
        return setResult(res);
    }

    override void visit(PowAssignExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (exp.e1.checkReadModifyWrite(exp.op, exp.e2))
            return setError();

        assert(exp.e1.type && exp.e2.type);
        if (exp.e1.op == TOK.slice || exp.e1.type.ty == Tarray || exp.e1.type.ty == Tsarray)
        {
            if (checkNonAssignmentArrayOp(exp.e1))
                return setError();

            // T[] ^^= ...
            if (exp.e2.implicitConvTo(exp.e1.type.nextOf()))
            {
                // T[] ^^= T
                exp.e2 = exp.e2.castTo(sc, exp.e1.type.nextOf());
            }
            else if (Expression ex = typeCombine(exp, sc))
            {
                result = ex;
                return;
            }

            // Check element types are arithmetic
            Type tb1 = exp.e1.type.nextOf().toBasetype();
            Type tb2 = exp.e2.type.toBasetype();
            if (tb2.ty == Tarray || tb2.ty == Tsarray)
                tb2 = tb2.nextOf().toBasetype();
            if ((tb1.isintegral() || tb1.isfloating()) && (tb2.isintegral() || tb2.isfloating()))
            {
                exp.type = exp.e1.type;
                result = arrayOp(exp, sc);
                return;
            }
        }
        else
        {
            exp.e1 = exp.e1.modifiableLvalue(sc, exp.e1);
        }

        if ((exp.e1.type.isintegral() || exp.e1.type.isfloating()) && (exp.e2.type.isintegral() || exp.e2.type.isfloating()))
        {
            Expression e0 = null;
            e = exp.reorderSettingAAElem(sc);
            e = Expression.extractLast(e, e0);
            assert(e == exp);

            if (exp.e1.op == TOK.variable)
            {
                // Rewrite: e1 = e1 ^^ e2
                e = new PowExp(exp.loc, exp.e1.syntaxCopy(), exp.e2);
                e = new AssignExp(exp.loc, exp.e1, e);
            }
            else
            {
                // Rewrite: ref tmp = e1; tmp = tmp ^^ e2
                auto v = copyToTemp(STC.ref_, "__powtmp", exp.e1);
                auto de = new DeclarationExp(exp.e1.loc, v);
                auto ve = new VarExp(exp.e1.loc, v);
                e = new PowExp(exp.loc, ve, exp.e2);
                e = new AssignExp(exp.loc, new VarExp(exp.e1.loc, v), e);
                e = new CommaExp(exp.loc, de, e);
            }
            e = Expression.combine(e0, e);
            e = e.expressionSemantic(sc);
            result = e;
            return;
        }
        result = exp.incompatibleTypes();
    }

    override void visit(CatAssignExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        //printf("CatAssignExp::semantic() %s\n", exp.toChars());
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (exp.e1.op == TOK.slice)
        {
            SliceExp se = cast(SliceExp)exp.e1;
            if (se.e1.type.toBasetype().ty == Tsarray)
            {
                exp.error("cannot append to static array `%s`", se.e1.type.toChars());
                return setError();
            }
        }

        if (checkIfIsStructLiteralDotExpr(exp.e1))
            return setError();

        exp.e1 = exp.e1.modifiableLvalue(sc, exp.e1);
        if (exp.e1.op == TOK.error)
        {
            result = exp.e1;
            return;
        }
        if (exp.e2.op == TOK.error)
        {
            result = exp.e2;
            return;
        }

        if (checkNonAssignmentArrayOp(exp.e2))
            return setError();

        Type tb1 = exp.e1.type.toBasetype();
        Type tb1next = tb1.nextOf();
        Type tb2 = exp.e2.type.toBasetype();

        /* Possibilities:
         * TOK.concatenateAssign: appending T[] to T[]
         * TOK.concatenateElemAssign: appending T to T[]
         * TOK.concatenateDcharAssign: appending dchar to T[]
         */
        if ((tb1.ty == Tarray) &&
            (tb2.ty == Tarray || tb2.ty == Tsarray) &&
            (exp.e2.implicitConvTo(exp.e1.type) ||
             (tb2.nextOf().implicitConvTo(tb1next) &&
              (tb2.nextOf().size(Loc.initial) == tb1next.size(Loc.initial)))))
        {
            // TOK.concatenateAssign
            assert(exp.op == TOK.concatenateAssign);
            if (exp.e1.checkPostblit(sc, tb1next))
                return setError();

            exp.e2 = exp.e2.castTo(sc, exp.e1.type);
        }
        else if ((tb1.ty == Tarray) && exp.e2.implicitConvTo(tb1next))
        {
            /* https://issues.dlang.org/show_bug.cgi?id=19782
             *
             * If e2 is implicitly convertible to tb1next, the conversion
             * might be done through alias this, in which case, e2 needs to
             * be modified accordingly (e2 => e2.aliasthis).
             */
            if (tb2.ty == Tstruct && (cast(TypeStruct)tb2).implicitConvToThroughAliasThis(tb1next))
                goto Laliasthis;
            if (tb2.ty == Tclass && (cast(TypeClass)tb2).implicitConvToThroughAliasThis(tb1next))
                goto Laliasthis;
            // Append element
            if (exp.e2.checkPostblit(sc, tb2))
                return setError();

            if (checkNewEscape(sc, exp.e2, false))
                return setError();

            exp = new CatElemAssignExp(exp.loc, exp.type, exp.e1, exp.e2.castTo(sc, tb1next));
            exp.e2 = doCopyOrMove(sc, exp.e2);
        }
        else if (tb1.ty == Tarray &&
                 (tb1next.ty == Tchar || tb1next.ty == Twchar) &&
                 exp.e2.type.ty != tb1next.ty &&
                 exp.e2.implicitConvTo(Type.tdchar))
        {
            // Append dchar to char[] or wchar[]
            exp = new CatDcharAssignExp(exp.loc, exp.type, exp.e1, exp.e2.castTo(sc, Type.tdchar));

            /* Do not allow appending wchar to char[] because if wchar happens
             * to be a surrogate pair, nothing good can result.
             */
        }
        else
        {
            // Try alias this on first operand
            static Expression tryAliasThisForLhs(BinAssignExp exp, Scope* sc)
            {
                AggregateDeclaration ad1 = isAggregate(exp.e1.type);
                if (!ad1 || !ad1.aliasthis)
                    return null;

                /* Rewrite (e1 op e2) as:
                 *      (e1.aliasthis op e2)
                 */
                if (exp.att1 && exp.e1.type == exp.att1)
                    return null;
                //printf("att %s e1 = %s\n", Token::toChars(e.op), e.e1.type.toChars());
                Expression e1 = new DotIdExp(exp.loc, exp.e1, ad1.aliasthis.ident);
                BinExp be = cast(BinExp)exp.copy();
                if (!be.att1 && exp.e1.type.checkAliasThisRec())
                    be.att1 = exp.e1.type;
                be.e1 = e1;
                return be.trySemantic(sc);
            }

            // Try alias this on second operand
            static Expression tryAliasThisForRhs(BinAssignExp exp, Scope* sc)
            {
                AggregateDeclaration ad2 = isAggregate(exp.e2.type);
                if (!ad2 || !ad2.aliasthis)
                    return null;
                /* Rewrite (e1 op e2) as:
                 *      (e1 op e2.aliasthis)
                 */
                if (exp.att2 && exp.e2.type == exp.att2)
                    return null;
                //printf("att %s e2 = %s\n", Token::toChars(e.op), e.e2.type.toChars());
                Expression e2 = new DotIdExp(exp.loc, exp.e2, ad2.aliasthis.ident);
                BinExp be = cast(BinExp)exp.copy();
                if (!be.att2 && exp.e2.type.checkAliasThisRec())
                    be.att2 = exp.e2.type;
                be.e2 = e2;
                return be.trySemantic(sc);
            }

    Laliasthis:
            result = tryAliasThisForLhs(exp, sc);
            if (result)
                return;

            result = tryAliasThisForRhs(exp, sc);
            if (result)
                return;

            exp.error("cannot append type `%s` to type `%s`", tb2.toChars(), tb1.toChars());
            return setError();
        }

        if (exp.e2.checkValue() || exp.e2.checkSharedAccess(sc))
            return setError();

        exp.type = exp.e1.type;
        auto res = exp.reorderSettingAAElem(sc);
        if ((exp.op == TOK.concatenateElemAssign || exp.op == TOK.concatenateDcharAssign) && global.params.vsafe)
            checkAssignEscape(sc, res, false);
        result = res;
    }

    override void visit(AddExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("AddExp::semantic('%s')\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        Type tb1 = exp.e1.type.toBasetype();
        Type tb2 = exp.e2.type.toBasetype();

        bool err = false;
        if (tb1.ty == Tdelegate || tb1.ty == Tpointer && tb1.nextOf().ty == Tfunction)
        {
            err |= exp.e1.checkArithmetic() || exp.e1.checkSharedAccess(sc);
        }
        if (tb2.ty == Tdelegate || tb2.ty == Tpointer && tb2.nextOf().ty == Tfunction)
        {
            err |= exp.e2.checkArithmetic() || exp.e2.checkSharedAccess(sc);
        }
        if (err)
            return setError();

        if (tb1.ty == Tpointer && exp.e2.type.isintegral() || tb2.ty == Tpointer && exp.e1.type.isintegral())
        {
            result = scaleFactor(exp, sc);
            return;
        }

        if (tb1.ty == Tpointer && tb2.ty == Tpointer)
        {
            result = exp.incompatibleTypes();
            return;
        }

        if (Expression ex = typeCombine(exp, sc))
        {
            result = ex;
            return;
        }

        Type tb = exp.type.toBasetype();
        if (tb.ty == Tarray || tb.ty == Tsarray)
        {
            if (!isArrayOpValid(exp))
            {
                result = arrayOpInvalidError(exp);
                return;
            }
            result = exp;
            return;
        }

        tb1 = exp.e1.type.toBasetype();
        if (!target.isVectorOpSupported(tb1, exp.op, tb2))
        {
            result = exp.incompatibleTypes();
            return;
        }
        if ((tb1.isreal() && exp.e2.type.isimaginary()) || (tb1.isimaginary() && exp.e2.type.isreal()))
        {
            switch (exp.type.toBasetype().ty)
            {
            case Tfloat32:
            case Timaginary32:
                exp.type = Type.tcomplex32;
                break;

            case Tfloat64:
            case Timaginary64:
                exp.type = Type.tcomplex64;
                break;

            case Tfloat80:
            case Timaginary80:
                exp.type = Type.tcomplex80;
                break;

            default:
                assert(0);
            }
        }
        result = exp;
    }

    override void visit(MinExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("MinExp::semantic('%s')\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        Type t1 = exp.e1.type.toBasetype();
        Type t2 = exp.e2.type.toBasetype();

        bool err = false;
        if (t1.ty == Tdelegate || t1.ty == Tpointer && t1.nextOf().ty == Tfunction)
        {
            err |= exp.e1.checkArithmetic() || exp.e1.checkSharedAccess(sc);
        }
        if (t2.ty == Tdelegate || t2.ty == Tpointer && t2.nextOf().ty == Tfunction)
        {
            err |= exp.e2.checkArithmetic() || exp.e2.checkSharedAccess(sc);
        }
        if (err)
            return setError();

        if (t1.ty == Tpointer)
        {
            if (t2.ty == Tpointer)
            {
                // https://dlang.org/spec/expression.html#add_expressions
                // "If both operands are pointers, and the operator is -, the pointers are
                // subtracted and the result is divided by the size of the type pointed to
                // by the operands. It is an error if the pointers point to different types."
                Type p1 = t1.nextOf();
                Type p2 = t2.nextOf();

                if (!p1.equivalent(p2))
                {
                    // Deprecation to remain for at least a year, after which this should be
                    // changed to an error
                    // See https://github.com/dlang/dmd/pull/7332
                    deprecation(exp.loc,
                        "cannot subtract pointers to different types: `%s` and `%s`.",
                        t1.toChars(), t2.toChars());
                }

                // Need to divide the result by the stride
                // Replace (ptr - ptr) with (ptr - ptr) / stride
                d_int64 stride;

                // make sure pointer types are compatible
                if (Expression ex = typeCombine(exp, sc))
                {
                    result = ex;
                    return;
                }

                exp.type = Type.tptrdiff_t;
                stride = t2.nextOf().size();
                if (stride == 0)
                {
                    e = new IntegerExp(exp.loc, 0, Type.tptrdiff_t);
                }
                else
                {
                    e = new DivExp(exp.loc, exp, new IntegerExp(Loc.initial, stride, Type.tptrdiff_t));
                    e.type = Type.tptrdiff_t;
                }
            }
            else if (t2.isintegral())
                e = scaleFactor(exp, sc);
            else
            {
                exp.error("can't subtract `%s` from pointer", t2.toChars());
                e = new ErrorExp();
            }
            result = e;
            return;
        }
        if (t2.ty == Tpointer)
        {
            exp.type = exp.e2.type;
            exp.error("can't subtract pointer from `%s`", exp.e1.type.toChars());
            return setError();
        }

        if (Expression ex = typeCombine(exp, sc))
        {
            result = ex;
            return;
        }

        Type tb = exp.type.toBasetype();
        if (tb.ty == Tarray || tb.ty == Tsarray)
        {
            if (!isArrayOpValid(exp))
            {
                result = arrayOpInvalidError(exp);
                return;
            }
            result = exp;
            return;
        }

        t1 = exp.e1.type.toBasetype();
        t2 = exp.e2.type.toBasetype();
        if (!target.isVectorOpSupported(t1, exp.op, t2))
        {
            result = exp.incompatibleTypes();
            return;
        }
        if ((t1.isreal() && t2.isimaginary()) || (t1.isimaginary() && t2.isreal()))
        {
            switch (exp.type.ty)
            {
            case Tfloat32:
            case Timaginary32:
                exp.type = Type.tcomplex32;
                break;

            case Tfloat64:
            case Timaginary64:
                exp.type = Type.tcomplex64;
                break;

            case Tfloat80:
            case Timaginary80:
                exp.type = Type.tcomplex80;
                break;

            default:
                assert(0);
            }
        }
        result = exp;
        return;
    }

    override void visit(CatExp exp)
    {
        // https://dlang.org/spec/expression.html#cat_expressions
        //printf("CatExp.semantic() %s\n", toChars());
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        Type tb1 = exp.e1.type.toBasetype();
        Type tb2 = exp.e2.type.toBasetype();

        auto f1 = checkNonAssignmentArrayOp(exp.e1);
        auto f2 = checkNonAssignmentArrayOp(exp.e2);
        if (f1 || f2)
            return setError();

        /* BUG: Should handle things like:
         *      char c;
         *      c ~ ' '
         *      ' ' ~ c;
         */

        Type tb1next = tb1.nextOf();
        Type tb2next = tb2.nextOf();

        // Check for: array ~ array
        if (tb1next && tb2next && (tb1next.implicitConvTo(tb2next) >= MATCH.constant || tb2next.implicitConvTo(tb1next) >= MATCH.constant || exp.e1.op == TOK.arrayLiteral && exp.e1.implicitConvTo(tb2) || exp.e2.op == TOK.arrayLiteral && exp.e2.implicitConvTo(tb1)))
        {
            /* https://issues.dlang.org/show_bug.cgi?id=9248
             * Here to avoid the case of:
             *    void*[] a = [cast(void*)1];
             *    void*[] b = [cast(void*)2];
             *    a ~ b;
             * becoming:
             *    a ~ [cast(void*)b];
             */

            /* https://issues.dlang.org/show_bug.cgi?id=14682
             * Also to avoid the case of:
             *    int[][] a;
             *    a ~ [];
             * becoming:
             *    a ~ cast(int[])[];
             */
            goto Lpeer;
        }

        // Check for: array ~ element
        if ((tb1.ty == Tsarray || tb1.ty == Tarray) && tb2.ty != Tvoid)
        {
            if (exp.e1.op == TOK.arrayLiteral)
            {
                exp.e2 = doCopyOrMove(sc, exp.e2);
                // https://issues.dlang.org/show_bug.cgi?id=14686
                // Postblit call appears in AST, and this is
                // finally translated  to an ArrayLiteralExp in below optimize().
            }
            else if (exp.e1.op == TOK.string_)
            {
                // No postblit call exists on character (integer) value.
            }
            else
            {
                if (exp.e2.checkPostblit(sc, tb2))
                    return setError();
                // Postblit call will be done in runtime helper function
            }

            if (exp.e1.op == TOK.arrayLiteral && exp.e1.implicitConvTo(tb2.arrayOf()))
            {
                exp.e1 = exp.e1.implicitCastTo(sc, tb2.arrayOf());
                exp.type = tb2.arrayOf();
                goto L2elem;
            }
            if (exp.e2.implicitConvTo(tb1next) >= MATCH.convert)
            {
                exp.e2 = exp.e2.implicitCastTo(sc, tb1next);
                exp.type = tb1next.arrayOf();
            L2elem:
                if (tb2.ty == Tarray || tb2.ty == Tsarray)
                {
                    // Make e2 into [e2]
                    exp.e2 = new ArrayLiteralExp(exp.e2.loc, exp.type, exp.e2);
                }
                else if (checkNewEscape(sc, exp.e2, false))
                    return setError();
                result = exp.optimize(WANTvalue);
                return;
            }
        }
        // Check for: element ~ array
        if ((tb2.ty == Tsarray || tb2.ty == Tarray) && tb1.ty != Tvoid)
        {
            if (exp.e2.op == TOK.arrayLiteral)
            {
                exp.e1 = doCopyOrMove(sc, exp.e1);
            }
            else if (exp.e2.op == TOK.string_)
            {
            }
            else
            {
                if (exp.e1.checkPostblit(sc, tb1))
                    return setError();
            }

            if (exp.e2.op == TOK.arrayLiteral && exp.e2.implicitConvTo(tb1.arrayOf()))
            {
                exp.e2 = exp.e2.implicitCastTo(sc, tb1.arrayOf());
                exp.type = tb1.arrayOf();
                goto L1elem;
            }
            if (exp.e1.implicitConvTo(tb2next) >= MATCH.convert)
            {
                exp.e1 = exp.e1.implicitCastTo(sc, tb2next);
                exp.type = tb2next.arrayOf();
            L1elem:
                if (tb1.ty == Tarray || tb1.ty == Tsarray)
                {
                    // Make e1 into [e1]
                    exp.e1 = new ArrayLiteralExp(exp.e1.loc, exp.type, exp.e1);
                }
                else if (checkNewEscape(sc, exp.e1, false))
                    return setError();
                result = exp.optimize(WANTvalue);
                return;
            }
        }

    Lpeer:
        if ((tb1.ty == Tsarray || tb1.ty == Tarray) && (tb2.ty == Tsarray || tb2.ty == Tarray) && (tb1next.mod || tb2next.mod) && (tb1next.mod != tb2next.mod))
        {
            Type t1 = tb1next.mutableOf().constOf().arrayOf();
            Type t2 = tb2next.mutableOf().constOf().arrayOf();
            if (exp.e1.op == TOK.string_ && !(cast(StringExp)exp.e1).committed)
                exp.e1.type = t1;
            else
                exp.e1 = exp.e1.castTo(sc, t1);
            if (exp.e2.op == TOK.string_ && !(cast(StringExp)exp.e2).committed)
                exp.e2.type = t2;
            else
                exp.e2 = exp.e2.castTo(sc, t2);
        }

        if (Expression ex = typeCombine(exp, sc))
        {
            result = ex;
            return;
        }
        exp.type = exp.type.toHeadMutable();

        Type tb = exp.type.toBasetype();
        if (tb.ty == Tsarray)
            exp.type = tb.nextOf().arrayOf();
        if (exp.type.ty == Tarray && tb1next && tb2next && tb1next.mod != tb2next.mod)
        {
            exp.type = exp.type.nextOf().toHeadMutable().arrayOf();
        }
        if (Type tbn = tb.nextOf())
        {
            if (exp.checkPostblit(sc, tbn))
                return setError();
        }
        Type t1 = exp.e1.type.toBasetype();
        Type t2 = exp.e2.type.toBasetype();
        if ((t1.ty == Tarray || t1.ty == Tsarray) &&
            (t2.ty == Tarray || t2.ty == Tsarray))
        {
            // Normalize to ArrayLiteralExp or StringExp as far as possible
            e = exp.optimize(WANTvalue);
        }
        else
        {
            //printf("(%s) ~ (%s)\n", e1.toChars(), e2.toChars());
            result = exp.incompatibleTypes();
            return;
        }

        result = e;
    }

    override void visit(MulExp exp)
    {
        version (none)
        {
            printf("MulExp::semantic() %s\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (Expression ex = typeCombine(exp, sc))
        {
            result = ex;
            return;
        }

        Type tb = exp.type.toBasetype();
        if (tb.ty == Tarray || tb.ty == Tsarray)
        {
            if (!isArrayOpValid(exp))
            {
                result = arrayOpInvalidError(exp);
                return;
            }
            result = exp;
            return;
        }

        if (exp.checkArithmeticBin() || exp.checkSharedAccessBin(sc))
            return setError();

        if (exp.type.isfloating())
        {
            Type t1 = exp.e1.type;
            Type t2 = exp.e2.type;

            if (t1.isreal())
            {
                exp.type = t2;
            }
            else if (t2.isreal())
            {
                exp.type = t1;
            }
            else if (t1.isimaginary())
            {
                if (t2.isimaginary())
                {
                    switch (t1.toBasetype().ty)
                    {
                    case Timaginary32:
                        exp.type = Type.tfloat32;
                        break;

                    case Timaginary64:
                        exp.type = Type.tfloat64;
                        break;

                    case Timaginary80:
                        exp.type = Type.tfloat80;
                        break;

                    default:
                        assert(0);
                    }

                    // iy * iv = -yv
                    exp.e1.type = exp.type;
                    exp.e2.type = exp.type;
                    e = new NegExp(exp.loc, exp);
                    e = e.expressionSemantic(sc);
                    result = e;
                    return;
                }
                else
                    exp.type = t2; // t2 is complex
            }
            else if (t2.isimaginary())
            {
                exp.type = t1; // t1 is complex
            }
        }
        else if (!target.isVectorOpSupported(tb, exp.op, exp.e2.type.toBasetype()))
        {
            result = exp.incompatibleTypes();
            return;
        }
        result = exp;
    }

    override void visit(DivExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (Expression ex = typeCombine(exp, sc))
        {
            result = ex;
            return;
        }

        Type tb = exp.type.toBasetype();
        if (tb.ty == Tarray || tb.ty == Tsarray)
        {
            if (!isArrayOpValid(exp))
            {
                result = arrayOpInvalidError(exp);
                return;
            }
            result = exp;
            return;
        }

        if (exp.checkArithmeticBin() || exp.checkSharedAccessBin(sc))
            return setError();

        if (exp.type.isfloating())
        {
            Type t1 = exp.e1.type;
            Type t2 = exp.e2.type;

            if (t1.isreal())
            {
                exp.type = t2;
                if (t2.isimaginary())
                {
                    // x/iv = i(-x/v)
                    exp.e2.type = t1;
                    e = new NegExp(exp.loc, exp);
                    e = e.expressionSemantic(sc);
                    result = e;
                    return;
                }
            }
            else if (t2.isreal())
            {
                exp.type = t1;
            }
            else if (t1.isimaginary())
            {
                if (t2.isimaginary())
                {
                    switch (t1.toBasetype().ty)
                    {
                    case Timaginary32:
                        exp.type = Type.tfloat32;
                        break;

                    case Timaginary64:
                        exp.type = Type.tfloat64;
                        break;

                    case Timaginary80:
                        exp.type = Type.tfloat80;
                        break;

                    default:
                        assert(0);
                    }
                }
                else
                    exp.type = t2; // t2 is complex
            }
            else if (t2.isimaginary())
            {
                exp.type = t1; // t1 is complex
            }
        }
        else if (!target.isVectorOpSupported(tb, exp.op, exp.e2.type.toBasetype()))
        {
            result = exp.incompatibleTypes();
            return;
        }
        result = exp;
    }

    override void visit(ModExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (Expression ex = typeCombine(exp, sc))
        {
            result = ex;
            return;
        }

        Type tb = exp.type.toBasetype();
        if (tb.ty == Tarray || tb.ty == Tsarray)
        {
            if (!isArrayOpValid(exp))
            {
                result = arrayOpInvalidError(exp);
                return;
            }
            result = exp;
            return;
        }
        if (!target.isVectorOpSupported(tb, exp.op, exp.e2.type.toBasetype()))
        {
            result = exp.incompatibleTypes();
            return;
        }

        if (exp.checkArithmeticBin() || exp.checkSharedAccessBin(sc))
            return setError();

        if (exp.type.isfloating())
        {
            exp.type = exp.e1.type;
            if (exp.e2.type.iscomplex())
            {
                exp.error("cannot perform modulo complex arithmetic");
                return setError();
            }
        }
        result = exp;
    }

    override void visit(PowExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        //printf("PowExp::semantic() %s\n", toChars());
        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (Expression ex = typeCombine(exp, sc))
        {
            result = ex;
            return;
        }

        Type tb = exp.type.toBasetype();
        if (tb.ty == Tarray || tb.ty == Tsarray)
        {
            if (!isArrayOpValid(exp))
            {
                result = arrayOpInvalidError(exp);
                return;
            }
            result = exp;
            return;
        }

        if (exp.checkArithmeticBin() || exp.checkSharedAccessBin(sc))
            return setError();

        if (!target.isVectorOpSupported(tb, exp.op, exp.e2.type.toBasetype()))
        {
            result = exp.incompatibleTypes();
            return;
        }

        // First, attempt to fold the expression.
        e = exp.optimize(WANTvalue);
        if (e.op != TOK.pow)
        {
            e = e.expressionSemantic(sc);
            result = e;
            return;
        }

        Module mmath = loadStdMath();
        if (!mmath)
        {
            e.error("`%s` requires `std.math` for `^^` operators", e.toChars());
            return setError();
        }
        e = new ScopeExp(exp.loc, mmath);

        if (exp.e2.op == TOK.float64 && exp.e2.toReal() == CTFloat.half)
        {
            // Replace e1 ^^ 0.5 with .std.math.sqrt(e1)
            e = new CallExp(exp.loc, new DotIdExp(exp.loc, e, Id._sqrt), exp.e1);
        }
        else
        {
            // Replace e1 ^^ e2 with .std.math.pow(e1, e2)
            e = new CallExp(exp.loc, new DotIdExp(exp.loc, e, Id._pow), exp.e1, exp.e2);
        }
        e = e.expressionSemantic(sc);
        result = e;
        return;
    }

    override void visit(ShlExp exp)
    {
        //printf("ShlExp::semantic(), type = %p\n", type);
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (exp.checkIntegralBin() || exp.checkSharedAccessBin(sc))
            return setError();

        if (!target.isVectorOpSupported(exp.e1.type.toBasetype(), exp.op, exp.e2.type.toBasetype()))
        {
            result = exp.incompatibleTypes();
            return;
        }
        exp.e1 = integralPromotions(exp.e1, sc);
        if (exp.e2.type.toBasetype().ty != Tvector)
            exp.e2 = exp.e2.castTo(sc, Type.tshiftcnt);

        exp.type = exp.e1.type;
        result = exp;
    }

    override void visit(ShrExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (exp.checkIntegralBin() || exp.checkSharedAccessBin(sc))
            return setError();

        if (!target.isVectorOpSupported(exp.e1.type.toBasetype(), exp.op, exp.e2.type.toBasetype()))
        {
            result = exp.incompatibleTypes();
            return;
        }
        exp.e1 = integralPromotions(exp.e1, sc);
        if (exp.e2.type.toBasetype().ty != Tvector)
            exp.e2 = exp.e2.castTo(sc, Type.tshiftcnt);

        exp.type = exp.e1.type;
        result = exp;
    }

    override void visit(UshrExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (exp.checkIntegralBin() || exp.checkSharedAccessBin(sc))
            return setError();

        if (!target.isVectorOpSupported(exp.e1.type.toBasetype(), exp.op, exp.e2.type.toBasetype()))
        {
            result = exp.incompatibleTypes();
            return;
        }
        exp.e1 = integralPromotions(exp.e1, sc);
        if (exp.e2.type.toBasetype().ty != Tvector)
            exp.e2 = exp.e2.castTo(sc, Type.tshiftcnt);

        exp.type = exp.e1.type;
        result = exp;
    }

    override void visit(AndExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (exp.e1.type.toBasetype().ty == Tbool && exp.e2.type.toBasetype().ty == Tbool)
        {
            exp.type = exp.e1.type;
            result = exp;
            return;
        }

        if (Expression ex = typeCombine(exp, sc))
        {
            result = ex;
            return;
        }

        Type tb = exp.type.toBasetype();
        if (tb.ty == Tarray || tb.ty == Tsarray)
        {
            if (!isArrayOpValid(exp))
            {
                result = arrayOpInvalidError(exp);
                return;
            }
            result = exp;
            return;
        }
        if (!target.isVectorOpSupported(tb, exp.op, exp.e2.type.toBasetype()))
        {
            result = exp.incompatibleTypes();
            return;
        }
        if (exp.checkIntegralBin() || exp.checkSharedAccessBin(sc))
            return setError();

        result = exp;
    }

    override void visit(OrExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (exp.e1.type.toBasetype().ty == Tbool && exp.e2.type.toBasetype().ty == Tbool)
        {
            exp.type = exp.e1.type;
            result = exp;
            return;
        }

        if (Expression ex = typeCombine(exp, sc))
        {
            result = ex;
            return;
        }

        Type tb = exp.type.toBasetype();
        if (tb.ty == Tarray || tb.ty == Tsarray)
        {
            if (!isArrayOpValid(exp))
            {
                result = arrayOpInvalidError(exp);
                return;
            }
            result = exp;
            return;
        }
        if (!target.isVectorOpSupported(tb, exp.op, exp.e2.type.toBasetype()))
        {
            result = exp.incompatibleTypes();
            return;
        }
        if (exp.checkIntegralBin() || exp.checkSharedAccessBin(sc))
            return setError();

        result = exp;
    }

    override void visit(XorExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        if (exp.e1.type.toBasetype().ty == Tbool && exp.e2.type.toBasetype().ty == Tbool)
        {
            exp.type = exp.e1.type;
            result = exp;
            return;
        }

        if (Expression ex = typeCombine(exp, sc))
        {
            result = ex;
            return;
        }

        Type tb = exp.type.toBasetype();
        if (tb.ty == Tarray || tb.ty == Tsarray)
        {
            if (!isArrayOpValid(exp))
            {
                result = arrayOpInvalidError(exp);
                return;
            }
            result = exp;
            return;
        }
        if (!target.isVectorOpSupported(tb, exp.op, exp.e2.type.toBasetype()))
        {
            result = exp.incompatibleTypes();
            return;
        }
        if (exp.checkIntegralBin() || exp.checkSharedAccessBin(sc))
            return setError();

        result = exp;
    }

    override void visit(LogicalExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("LogicalExp::semantic() %s\n", exp.toChars());
        }

        if (exp.type)
        {
            result = exp;
            return;
        }

        exp.setNoderefOperands();

        Expression e1x = exp.e1.expressionSemantic(sc);

        // for static alias this: https://issues.dlang.org/show_bug.cgi?id=17684
        if (e1x.op == TOK.type)
            e1x = resolveAliasThis(sc, e1x);

        e1x = resolveProperties(sc, e1x);
        e1x = e1x.toBoolean(sc);

        if (sc.flags & SCOPE.condition)
        {
            /* If in static if, don't evaluate e2 if we don't have to.
             */
            e1x = e1x.optimize(WANTvalue);
            if (e1x.isBool(exp.op == TOK.orOr))
            {
                result = IntegerExp.createBool(exp.op == TOK.orOr);
                return;
            }
        }

        CtorFlow ctorflow = sc.ctorflow.clone();
        Expression e2x = exp.e2.expressionSemantic(sc);
        sc.merge(exp.loc, ctorflow);
        ctorflow.freeFieldinit();

        // for static alias this: https://issues.dlang.org/show_bug.cgi?id=17684
        if (e2x.op == TOK.type)
            e2x = resolveAliasThis(sc, e2x);

        e2x = resolveProperties(sc, e2x);

        auto f1 = checkNonAssignmentArrayOp(e1x);
        auto f2 = checkNonAssignmentArrayOp(e2x);
        if (f1 || f2)
            return setError();

        // Unless the right operand is 'void', the expression is converted to 'bool'.
        if (e2x.type.ty != Tvoid)
            e2x = e2x.toBoolean(sc);

        if (e2x.op == TOK.type || e2x.op == TOK.scope_)
        {
            exp.error("`%s` is not an expression", exp.e2.toChars());
            return setError();
        }
        if (e1x.op == TOK.error)
        {
            result = e1x;
            return;
        }
        if (e2x.op == TOK.error)
        {
            result = e2x;
            return;
        }

        // The result type is 'bool', unless the right operand has type 'void'.
        if (e2x.type.ty == Tvoid)
            exp.type = Type.tvoid;
        else
            exp.type = Type.tbool;

        exp.e1 = e1x;
        exp.e2 = e2x;
        result = exp;
    }


    override void visit(CmpExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("CmpExp::semantic('%s')\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        exp.setNoderefOperands();

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Type t1 = exp.e1.type.toBasetype();
        Type t2 = exp.e2.type.toBasetype();
        if (t1.ty == Tclass && exp.e2.op == TOK.null_ || t2.ty == Tclass && exp.e1.op == TOK.null_)
        {
            exp.error("do not use `null` when comparing class types");
            return setError();
        }

        TOK cmpop;
        if (auto e = exp.op_overload(sc, &cmpop))
        {
            if (!e.type.isscalar() && e.type.equals(exp.e1.type))
            {
                exp.error("recursive `opCmp` expansion");
                return setError();
            }
            if (e.op == TOK.call)
            {
                e = new CmpExp(cmpop, exp.loc, e, new IntegerExp(exp.loc, 0, Type.tint32));
                e = e.expressionSemantic(sc);
            }
            result = e;
            return;
        }

        if (Expression ex = typeCombine(exp, sc))
        {
            result = ex;
            return;
        }

        auto f1 = checkNonAssignmentArrayOp(exp.e1);
        auto f2 = checkNonAssignmentArrayOp(exp.e2);
        if (f1 || f2)
            return setError();

        exp.type = Type.tbool;

        // Special handling for array comparisons
        Expression arrayLowering = null;
        t1 = exp.e1.type.toBasetype();
        t2 = exp.e2.type.toBasetype();
        if ((t1.ty == Tarray || t1.ty == Tsarray || t1.ty == Tpointer) && (t2.ty == Tarray || t2.ty == Tsarray || t2.ty == Tpointer))
        {
            Type t1next = t1.nextOf();
            Type t2next = t2.nextOf();
            if (t1next.implicitConvTo(t2next) < MATCH.constant && t2next.implicitConvTo(t1next) < MATCH.constant && (t1next.ty != Tvoid && t2next.ty != Tvoid))
            {
                exp.error("array comparison type mismatch, `%s` vs `%s`", t1next.toChars(), t2next.toChars());
                return setError();
            }
            if ((t1.ty == Tarray || t1.ty == Tsarray) && (t2.ty == Tarray || t2.ty == Tsarray))
            {
                if (!verifyHookExist(exp.loc, *sc, Id.__cmp, "comparing arrays"))
                    return setError();

                // Lower to object.__cmp(e1, e2)
                Expression al = new IdentifierExp(exp.loc, Id.empty);
                al = new DotIdExp(exp.loc, al, Id.object);
                al = new DotIdExp(exp.loc, al, Id.__cmp);
                al = al.expressionSemantic(sc);

                auto arguments = new Expressions(2);
                (*arguments)[0] = exp.e1;
                (*arguments)[1] = exp.e2;

                al = new CallExp(exp.loc, al, arguments);
                al = new CmpExp(exp.op, exp.loc, al, IntegerExp.literal!0);

                arrayLowering = al;
            }
        }
        else if (t1.ty == Tstruct || t2.ty == Tstruct || (t1.ty == Tclass && t2.ty == Tclass))
        {
            if (t2.ty == Tstruct)
                exp.error("need member function `opCmp()` for %s `%s` to compare", t2.toDsymbol(sc).kind(), t2.toChars());
            else
                exp.error("need member function `opCmp()` for %s `%s` to compare", t1.toDsymbol(sc).kind(), t1.toChars());
            return setError();
        }
        else if (t1.iscomplex() || t2.iscomplex())
        {
            exp.error("compare not defined for complex operands");
            return setError();
        }
        else if (t1.ty == Taarray || t2.ty == Taarray)
        {
            exp.error("`%s` is not defined for associative arrays", Token.toChars(exp.op));
            return setError();
        }
        else if (!target.isVectorOpSupported(t1, exp.op, t2))
        {
            result = exp.incompatibleTypes();
            return;
        }
        else
        {
            bool r1 = exp.e1.checkValue() || exp.e1.checkSharedAccess(sc);
            bool r2 = exp.e2.checkValue() || exp.e2.checkSharedAccess(sc);
            if (r1 || r2)
                return setError();
        }

        //printf("CmpExp: %s, type = %s\n", e.toChars(), e.type.toChars());
        if (arrayLowering)
        {
            arrayLowering = arrayLowering.expressionSemantic(sc);
            result = arrayLowering;
            return;
        }
        result = exp;
        return;
    }

    override void visit(InExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (Expression ex = binSemanticProp(exp, sc))
        {
            result = ex;
            return;
        }
        Expression e = exp.op_overload(sc);
        if (e)
        {
            result = e;
            return;
        }

        Type t2b = exp.e2.type.toBasetype();
        switch (t2b.ty)
        {
        case Taarray:
            {
                TypeAArray ta = cast(TypeAArray)t2b;

                // Special handling for array keys
                if (!arrayTypeCompatibleWithoutCasting(exp.e1.type, ta.index))
                {
                    // Convert key to type of key
                    exp.e1 = exp.e1.implicitCastTo(sc, ta.index);
                }

                semanticTypeInfo(sc, ta.index);

                // Return type is pointer to value
                exp.type = ta.nextOf().pointerTo();
                break;
            }

        case Terror:
            return setError();

        default:
            result = exp.incompatibleTypes();
            return;
        }
        result = exp;
    }

    override void visit(RemoveExp e)
    {
        if (Expression ex = binSemantic(e, sc))
        {
            result = ex;
            return;
        }
        result = e;
    }

    override void visit(EqualExp exp)
    {
        //printf("EqualExp::semantic('%s')\n", exp.toChars());
        if (exp.type)
        {
            result = exp;
            return;
        }

        exp.setNoderefOperands();

        if (auto e = binSemanticProp(exp, sc))
        {
            result = e;
            return;
        }
        if (exp.e1.op == TOK.type || exp.e2.op == TOK.type)
        {
            /* https://issues.dlang.org/show_bug.cgi?id=12520
             * empty tuples are represented as types so special cases are added
             * so that they can be compared for equality with tuples of values.
             */
            static auto extractTypeTupAndExpTup(Expression e)
            {
                static struct Result { bool ttEmpty; bool te; }
                auto tt = e.op == TOK.type ? e.isTypeExp().type.isTypeTuple() : null;
                return Result(tt && (!tt.arguments || !tt.arguments.dim), e.isTupleExp() !is null);
            }
            auto tups1 = extractTypeTupAndExpTup(exp.e1);
            auto tups2 = extractTypeTupAndExpTup(exp.e2);
            // AliasSeq!() == AliasSeq!(<at least a value>)
            if (tups1.ttEmpty && tups2.te)
            {
                result = IntegerExp.createBool(exp.op != TOK.equal);
                return;
            }
            // AliasSeq!(<at least a value>) == AliasSeq!()
            else if (tups1.te && tups2.ttEmpty)
            {
                result = IntegerExp.createBool(exp.op != TOK.equal);
                return;
            }
            // AliasSeq!() == AliasSeq!()
            else if (tups1.ttEmpty && tups2.ttEmpty)
            {
                result = IntegerExp.createBool(exp.op == TOK.equal);
                return;
            }
            // otherwise, two types are really not comparable
            result = exp.incompatibleTypes();
            return;
        }

        {
            auto t1 = exp.e1.type;
            auto t2 = exp.e2.type;
            if (t1.ty == Tenum && t2.ty == Tenum && !t1.equivalent(t2))
                exp.error("Comparison between different enumeration types `%s` and `%s`; If this behavior is intended consider using `std.conv.asOriginalType`",
                    t1.toChars(), t2.toChars());
        }

        /* Before checking for operator overloading, check to see if we're
         * comparing the addresses of two statics. If so, we can just see
         * if they are the same symbol.
         */
        if (exp.e1.op == TOK.address && exp.e2.op == TOK.address)
        {
            AddrExp ae1 = cast(AddrExp)exp.e1;
            AddrExp ae2 = cast(AddrExp)exp.e2;
            if (ae1.e1.op == TOK.variable && ae2.e1.op == TOK.variable)
            {
                VarExp ve1 = cast(VarExp)ae1.e1;
                VarExp ve2 = cast(VarExp)ae2.e1;
                if (ve1.var == ve2.var)
                {
                    // They are the same, result is 'true' for ==, 'false' for !=
                    result = IntegerExp.createBool(exp.op == TOK.equal);
                    return;
                }
            }
        }

        Type t1 = exp.e1.type.toBasetype();
        Type t2 = exp.e2.type.toBasetype();

        bool needsDirectEq(Type t1, Type t2)
        {
            Type t1n = t1.nextOf().toBasetype();
            Type t2n = t2.nextOf().toBasetype();
            if ((t1n.ty.isSomeChar && t2n.ty.isSomeChar) ||
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

        if (auto e = exp.op_overload(sc))
        {
            result = e;
            return;
        }


        if (!(t1.ty == Tarray && t2.ty == Tarray && needsDirectEq(t1, t2)))
        {
            if (auto e = typeCombine(exp, sc))
            {
                result = e;
                return;
            }
        }

        auto f1 = checkNonAssignmentArrayOp(exp.e1);
        auto f2 = checkNonAssignmentArrayOp(exp.e2);
        if (f1 || f2)
            return setError();

        exp.type = Type.tbool;

        // Special handling for array comparisons
        if (!(t1.ty == Tarray && t2.ty == Tarray && needsDirectEq(t1, t2)))
        {
            if (!arrayTypeCompatible(exp.loc, exp.e1.type, exp.e2.type))
            {
                if (exp.e1.type != exp.e2.type && exp.e1.type.isfloating() && exp.e2.type.isfloating())
                {
                    // Cast both to complex
                    exp.e1 = exp.e1.castTo(sc, Type.tcomplex80);
                    exp.e2 = exp.e2.castTo(sc, Type.tcomplex80);
                }
            }
        }

        if (t1.ty == Tarray && t2.ty == Tarray)
        {
            //printf("Lowering to __equals %s %s\n", e1.toChars(), e2.toChars());

            // For e1 and e2 of struct type, lowers e1 == e2 to object.__equals(e1, e2)
            // and e1 != e2 to !(object.__equals(e1, e2)).

            if (!verifyHookExist(exp.loc, *sc, Id.__equals, "equal checks on arrays"))
                return setError();

            Expression __equals = new IdentifierExp(exp.loc, Id.empty);
            Identifier id = Identifier.idPool("__equals");
            __equals = new DotIdExp(exp.loc, __equals, Id.object);
            __equals = new DotIdExp(exp.loc, __equals, id);

            auto arguments = new Expressions(2);
            (*arguments)[0] = exp.e1;
            (*arguments)[1] = exp.e2;

            __equals = new CallExp(exp.loc, __equals, arguments);
            if (exp.op == TOK.notEqual)
            {
                __equals = new NotExp(exp.loc, __equals);
            }
            __equals = __equals.expressionSemantic(sc);

            result = __equals;
            return;
        }

        if (exp.e1.type.toBasetype().ty == Taarray)
            semanticTypeInfo(sc, exp.e1.type.toBasetype());


        if (!target.isVectorOpSupported(t1, exp.op, t2))
        {
            result = exp.incompatibleTypes();
            return;
        }

        result = exp;
    }

    override void visit(IdentityExp exp)
    {
        if (exp.type)
        {
            result = exp;
            return;
        }

        exp.setNoderefOperands();

        if (auto e = binSemanticProp(exp, sc))
        {
            result = e;
            return;
        }

        if (auto e = typeCombine(exp, sc))
        {
            result = e;
            return;
        }

        auto f1 = checkNonAssignmentArrayOp(exp.e1);
        auto f2 = checkNonAssignmentArrayOp(exp.e2);
        if (f1 || f2)
            return setError();

        if (exp.e1.op == TOK.type || exp.e2.op == TOK.type)
        {
            result = exp.incompatibleTypes();
            return;
        }

        exp.type = Type.tbool;

        if (exp.e1.type != exp.e2.type && exp.e1.type.isfloating() && exp.e2.type.isfloating())
        {
            // Cast both to complex
            exp.e1 = exp.e1.castTo(sc, Type.tcomplex80);
            exp.e2 = exp.e2.castTo(sc, Type.tcomplex80);
        }

        auto tb1 = exp.e1.type.toBasetype();
        auto tb2 = exp.e2.type.toBasetype();
        if (!target.isVectorOpSupported(tb1, exp.op, tb2))
        {
            result = exp.incompatibleTypes();
            return;
        }

        if (exp.e1.op == TOK.call)
            exp.e1 = (cast(CallExp)exp.e1).addDtorHook(sc);
        if (exp.e2.op == TOK.call)
            exp.e2 = (cast(CallExp)exp.e2).addDtorHook(sc);

        if (exp.e1.type.toBasetype().ty == Tsarray ||
            exp.e2.type.toBasetype().ty == Tsarray)
            exp.deprecation("identity comparison of static arrays "
                ~ "implicitly coerces them to slices, "
                ~ "which are compared by reference");

        result = exp;
    }

    override void visit(CondExp exp)
    {
        static if (LOGSEMANTIC)
        {
            printf("CondExp::semantic('%s')\n", exp.toChars());
        }
        if (exp.type)
        {
            result = exp;
            return;
        }

        if (exp.econd.op == TOK.dotIdentifier)
            (cast(DotIdExp)exp.econd).noderef = true;

        Expression ec = exp.econd.expressionSemantic(sc);
        ec = resolveProperties(sc, ec);
        ec = ec.toBoolean(sc);

        CtorFlow ctorflow_root = sc.ctorflow.clone();
        Expression e1x = exp.e1.expressionSemantic(sc);
        e1x = resolveProperties(sc, e1x);

        CtorFlow ctorflow1 = sc.ctorflow;
        sc.ctorflow = ctorflow_root;
        Expression e2x = exp.e2.expressionSemantic(sc);
        e2x = resolveProperties(sc, e2x);

        sc.merge(exp.loc, ctorflow1);
        ctorflow1.freeFieldinit();

        if (ec.op == TOK.error)
        {
            result = ec;
            return;
        }
        if (ec.type == Type.terror)
            return setError();
        exp.econd = ec;

        if (e1x.op == TOK.error)
        {
            result = e1x;
            return;
        }
        if (e1x.type == Type.terror)
            return setError();
        exp.e1 = e1x;

        if (e2x.op == TOK.error)
        {
            result = e2x;
            return;
        }
        if (e2x.type == Type.terror)
            return setError();
        exp.e2 = e2x;

        auto f0 = checkNonAssignmentArrayOp(exp.econd);
        auto f1 = checkNonAssignmentArrayOp(exp.e1);
        auto f2 = checkNonAssignmentArrayOp(exp.e2);
        if (f0 || f1 || f2)
            return setError();

        Type t1 = exp.e1.type;
        Type t2 = exp.e2.type;
        // If either operand is void the result is void, we have to cast both
        // the expression to void so that we explicitly discard the expression
        // value if any
        // https://issues.dlang.org/show_bug.cgi?id=16598
        if (t1.ty == Tvoid || t2.ty == Tvoid)
        {
            exp.type = Type.tvoid;
            exp.e1 = exp.e1.castTo(sc, exp.type);
            exp.e2 = exp.e2.castTo(sc, exp.type);
        }
        else if (t1 == t2)
            exp.type = t1;
        else
        {
            if (Expression ex = typeCombine(exp, sc))
            {
                result = ex;
                return;
            }

            switch (exp.e1.type.toBasetype().ty)
            {
            case Tcomplex32:
            case Tcomplex64:
            case Tcomplex80:
                exp.e2 = exp.e2.castTo(sc, exp.e1.type);
                break;
            default:
                break;
            }
            switch (exp.e2.type.toBasetype().ty)
            {
            case Tcomplex32:
            case Tcomplex64:
            case Tcomplex80:
                exp.e1 = exp.e1.castTo(sc, exp.e2.type);
                break;
            default:
                break;
            }
            if (exp.type.toBasetype().ty == Tarray)
            {
                exp.e1 = exp.e1.castTo(sc, exp.type);
                exp.e2 = exp.e2.castTo(sc, exp.type);
            }
        }
        exp.type = exp.type.merge2();
        version (none)
        {
            printf("res: %s\n", exp.type.toChars());
            printf("e1 : %s\n", exp.e1.type.toChars());
            printf("e2 : %s\n", exp.e2.type.toChars());
        }

        /* https://issues.dlang.org/show_bug.cgi?id=14696
         * If either e1 or e2 contain temporaries which need dtor,
         * make them conditional.
         * Rewrite:
         *      cond ? (__tmp1 = ..., __tmp1) : (__tmp2 = ..., __tmp2)
         * to:
         *      (auto __cond = cond) ? (... __tmp1) : (... __tmp2)
         * and replace edtors of __tmp1 and __tmp2 with:
         *      __tmp1.edtor --> __cond && __tmp1.dtor()
         *      __tmp2.edtor --> __cond || __tmp2.dtor()
         */
        exp.hookDtors(sc);

        result = exp;
    }

    override void visit(FileInitExp e)
    {
        //printf("FileInitExp::semantic()\n");
        e.type = Type.tstring;
        result = e;
    }

    override void visit(LineInitExp e)
    {
        e.type = Type.tint32;
        result = e;
    }

    override void visit(ModuleInitExp e)
    {
        //printf("ModuleInitExp::semantic()\n");
        e.type = Type.tstring;
        result = e;
    }

    override void visit(FuncInitExp e)
    {
        //printf("FuncInitExp::semantic()\n");
        e.type = Type.tstring;
        if (sc.func)
        {
            result = e.resolveLoc(Loc.initial, sc);
            return;
        }
        result = e;
    }

    override void visit(PrettyFuncInitExp e)
    {
        //printf("PrettyFuncInitExp::semantic()\n");
        e.type = Type.tstring;
        if (sc.func)
        {
            result = e.resolveLoc(Loc.initial, sc);
            return;
        }

        result = e;
    }
}

/**********************************
 * Try to run semantic routines.
 * If they fail, return NULL.
 */
Expression trySemantic(Expression exp, Scope* sc)
{
    //printf("+trySemantic(%s)\n", exp.toChars());
    uint errors = global.startGagging();
    Expression e = expressionSemantic(exp, sc);
    if (global.endGagging(errors))
    {
        e = null;
    }
    //printf("-trySemantic(%s)\n", exp.toChars());
    return e;
}

/**************************
 * Helper function for easy error propagation.
 * If error occurs, returns ErrorExp. Otherwise returns NULL.
 */
Expression unaSemantic(UnaExp e, Scope* sc)
{
    static if (LOGSEMANTIC)
    {
        printf("UnaExp::semantic('%s')\n", e.toChars());
    }
    Expression e1x = e.e1.expressionSemantic(sc);
    if (e1x.op == TOK.error)
        return e1x;
    e.e1 = e1x;
    return null;
}

/**************************
 * Helper function for easy error propagation.
 * If error occurs, returns ErrorExp. Otherwise returns NULL.
 */
Expression binSemantic(BinExp e, Scope* sc)
{
    static if (LOGSEMANTIC)
    {
        printf("BinExp::semantic('%s')\n", e.toChars());
    }
    Expression e1x = e.e1.expressionSemantic(sc);
    Expression e2x = e.e2.expressionSemantic(sc);

    // for static alias this: https://issues.dlang.org/show_bug.cgi?id=17684
    if (e1x.op == TOK.type)
        e1x = resolveAliasThis(sc, e1x);
    if (e2x.op == TOK.type)
        e2x = resolveAliasThis(sc, e2x);

    if (e1x.op == TOK.error)
        return e1x;
    if (e2x.op == TOK.error)
        return e2x;
    e.e1 = e1x;
    e.e2 = e2x;
    return null;
}

Expression binSemanticProp(BinExp e, Scope* sc)
{
    if (Expression ex = binSemantic(e, sc))
        return ex;
    Expression e1x = resolveProperties(sc, e.e1);
    Expression e2x = resolveProperties(sc, e.e2);
    if (e1x.op == TOK.error)
        return e1x;
    if (e2x.op == TOK.error)
        return e2x;
    e.e1 = e1x;
    e.e2 = e2x;
    return null;
}

// entrypoint for semantic ExpressionSemanticVisitor
extern (C++) Expression expressionSemantic(Expression e, Scope* sc)
{
    scope v = new ExpressionSemanticVisitor(sc);
    e.accept(v);
    return v.result;
}

Expression semanticX(DotIdExp exp, Scope* sc)
{
    //printf("DotIdExp::semanticX(this = %p, '%s')\n", this, toChars());
    if (Expression ex = unaSemantic(exp, sc))
        return ex;

    if (exp.ident == Id._mangleof)
    {
        // symbol.mangleof
        Dsymbol ds;
        switch (exp.e1.op)
        {
        case TOK.scope_:
            ds = (cast(ScopeExp)exp.e1).sds;
            goto L1;
        case TOK.variable:
            ds = (cast(VarExp)exp.e1).var;
            goto L1;
        case TOK.dotVariable:
            ds = (cast(DotVarExp)exp.e1).var;
            goto L1;
        case TOK.overloadSet:
            ds = (cast(OverExp)exp.e1).vars;
            goto L1;
        case TOK.template_:
            {
                TemplateExp te = cast(TemplateExp)exp.e1;
                ds = te.fd ? cast(Dsymbol)te.fd : te.td;
            }
        L1:
            {
                assert(ds);
                if (auto f = ds.isFuncDeclaration())
                {
                    if (f.checkForwardRef(exp.loc))
                    {
                        return new ErrorExp();
                    }
                    if (f.flags & (FUNCFLAG.purityInprocess | FUNCFLAG.safetyInprocess |
                                   FUNCFLAG.nothrowInprocess | FUNCFLAG.nogcInprocess))
                    {
                        f.error(exp.loc, "cannot retrieve its `.mangleof` while inferring attributes");
                        return new ErrorExp();
                    }
                }
                OutBuffer buf;
                mangleToBuffer(ds, &buf);
                Expression e = new StringExp(exp.loc, buf.extractSlice());
                e = e.expressionSemantic(sc);
                return e;
            }
        default:
            break;
        }
    }

    if (exp.e1.op == TOK.variable && exp.e1.type.toBasetype().ty == Tsarray && exp.ident == Id.length)
    {
        // bypass checkPurity
        return exp.e1.type.dotExp(sc, exp.e1, exp.ident, exp.noderef ? DotExpFlag.noDeref : 0);
    }

    if (exp.e1.op == TOK.dot)
    {
    }
    else
    {
        exp.e1 = resolvePropertiesX(sc, exp.e1);
    }
    if (exp.e1.op == TOK.tuple && exp.ident == Id.offsetof)
    {
        /* 'distribute' the .offsetof to each of the tuple elements.
         */
        TupleExp te = cast(TupleExp)exp.e1;
        auto exps = new Expressions(te.exps.dim);
        for (size_t i = 0; i < exps.dim; i++)
        {
            Expression e = (*te.exps)[i];
            e = e.expressionSemantic(sc);
            e = new DotIdExp(e.loc, e, Id.offsetof);
            (*exps)[i] = e;
        }
        // Don't evaluate te.e0 in runtime
        Expression e = new TupleExp(exp.loc, null, exps);
        e = e.expressionSemantic(sc);
        return e;
    }
    if (exp.e1.op == TOK.tuple && exp.ident == Id.length)
    {
        TupleExp te = cast(TupleExp)exp.e1;
        // Don't evaluate te.e0 in runtime
        Expression e = new IntegerExp(exp.loc, te.exps.dim, Type.tsize_t);
        return e;
    }

    // https://issues.dlang.org/show_bug.cgi?id=14416
    // Template has no built-in properties except for 'stringof'.
    if ((exp.e1.op == TOK.dotTemplateDeclaration || exp.e1.op == TOK.template_) && exp.ident != Id.stringof)
    {
        exp.error("template `%s` does not have property `%s`", exp.e1.toChars(), exp.ident.toChars());
        return new ErrorExp();
    }
    if (!exp.e1.type)
    {
        exp.error("expression `%s` does not have property `%s`", exp.e1.toChars(), exp.ident.toChars());
        return new ErrorExp();
    }

    return exp;
}

// Resolve e1.ident without seeing UFCS.
// If flag == 1, stop "not a property" error and return NULL.
Expression semanticY(DotIdExp exp, Scope* sc, int flag)
{
    //printf("DotIdExp::semanticY(this = %p, '%s')\n", exp, exp.toChars());

    //{ static int z; fflush(stdout); if (++z == 10) *(char*)0=0; }

    /* Special case: rewrite this.id and super.id
     * to be classtype.id and baseclasstype.id
     * if we have no this pointer.
     */
    if ((exp.e1.op == TOK.this_ || exp.e1.op == TOK.super_) && !hasThis(sc))
    {
        if (AggregateDeclaration ad = sc.getStructClassScope())
        {
            if (exp.e1.op == TOK.this_)
            {
                exp.e1 = new TypeExp(exp.e1.loc, ad.type);
            }
            else
            {
                ClassDeclaration cd = ad.isClassDeclaration();
                if (cd && cd.baseClass)
                    exp.e1 = new TypeExp(exp.e1.loc, cd.baseClass.type);
            }
        }
    }

    Expression e = semanticX(exp, sc);
    if (e != exp)
        return e;

    Expression eleft;
    Expression eright;
    if (exp.e1.op == TOK.dot)
    {
        DotExp de = cast(DotExp)exp.e1;
        eleft = de.e1;
        eright = de.e2;
    }
    else
    {
        eleft = null;
        eright = exp.e1;
    }

    Type t1b = exp.e1.type.toBasetype();

    if (eright.op == TOK.scope_) // also used for template alias's
    {
        ScopeExp ie = cast(ScopeExp)eright;

        int flags = SearchLocalsOnly;
        /* Disable access to another module's private imports.
         * The check for 'is sds our current module' is because
         * the current module should have access to its own imports.
         */
        if (ie.sds.isModule() && ie.sds != sc._module)
            flags |= IgnorePrivateImports;
        if (sc.flags & SCOPE.ignoresymbolvisibility)
            flags |= IgnoreSymbolVisibility;
        Dsymbol s = ie.sds.search(exp.loc, exp.ident, flags);
        /* Check for visibility before resolving aliases because public
         * aliases to private symbols are public.
         */
        if (s && !(sc.flags & SCOPE.ignoresymbolvisibility) && !symbolIsVisible(sc._module, s))
        {
            s = null;
        }
        if (s)
        {
            auto p = s.isPackage();
            if (p && checkAccess(sc, p))
            {
                s = null;
            }
        }
        if (s)
        {
            // if 's' is a tuple variable, the tuple is returned.
            s = s.toAlias();

            exp.checkDeprecated(sc, s);
            exp.checkDisabled(sc, s);

            EnumMember em = s.isEnumMember();
            if (em)
            {
                return em.getVarExp(exp.loc, sc);
            }
            VarDeclaration v = s.isVarDeclaration();
            if (v)
            {
                //printf("DotIdExp:: Identifier '%s' is a variable, type '%s'\n", toChars(), v.type.toChars());
                if (!v.type ||
                    !v.type.deco && v.inuse)
                {
                    if (v.inuse)
                        exp.error("circular reference to %s `%s`", v.kind(), v.toPrettyChars());
                    else
                        exp.error("forward reference to %s `%s`", v.kind(), v.toPrettyChars());
                    return new ErrorExp();
                }
                if (v.type.ty == Terror)
                    return new ErrorExp();

                if ((v.storage_class & STC.manifest) && v._init && !exp.wantsym)
                {
                    /* Normally, the replacement of a symbol with its initializer is supposed to be in semantic2().
                     * Introduced by https://github.com/dlang/dmd/pull/5588 which should probably
                     * be reverted. `wantsym` is the hack to work around the problem.
                     */
                    if (v.inuse)
                    {
                        error(exp.loc, "circular initialization of %s `%s`", v.kind(), v.toPrettyChars());
                        return new ErrorExp();
                    }
                    e = v.expandInitializer(exp.loc);
                    v.inuse++;
                    e = e.expressionSemantic(sc);
                    v.inuse--;
                    return e;
                }

                if (v.needThis())
                {
                    if (!eleft)
                        eleft = new ThisExp(exp.loc);
                    e = new DotVarExp(exp.loc, eleft, v);
                    e = e.expressionSemantic(sc);
                }
                else
                {
                    e = new VarExp(exp.loc, v);
                    if (eleft)
                    {
                        e = new CommaExp(exp.loc, eleft, e);
                        e.type = v.type;
                    }
                }
                e = e.deref();
                return e.expressionSemantic(sc);
            }

            FuncDeclaration f = s.isFuncDeclaration();
            if (f)
            {
                //printf("it's a function\n");
                if (!f.functionSemantic())
                    return new ErrorExp();
                if (f.needThis())
                {
                    if (!eleft)
                        eleft = new ThisExp(exp.loc);
                    e = new DotVarExp(exp.loc, eleft, f, true);
                    e = e.expressionSemantic(sc);
                }
                else
                {
                    e = new VarExp(exp.loc, f, true);
                    if (eleft)
                    {
                        e = new CommaExp(exp.loc, eleft, e);
                        e.type = f.type;
                    }
                }
                return e;
            }
            if (auto td = s.isTemplateDeclaration())
            {
                if (eleft)
                    e = new DotTemplateExp(exp.loc, eleft, td);
                else
                    e = new TemplateExp(exp.loc, td);
                e = e.expressionSemantic(sc);
                return e;
            }
            if (OverDeclaration od = s.isOverDeclaration())
            {
                e = new VarExp(exp.loc, od, true);
                if (eleft)
                {
                    e = new CommaExp(exp.loc, eleft, e);
                    e.type = Type.tvoid; // ambiguous type?
                }
                return e;
            }
            OverloadSet o = s.isOverloadSet();
            if (o)
            {
                //printf("'%s' is an overload set\n", o.toChars());
                return new OverExp(exp.loc, o);
            }

            if (auto t = s.getType())
            {
                return (new TypeExp(exp.loc, t)).expressionSemantic(sc);
            }

            TupleDeclaration tup = s.isTupleDeclaration();
            if (tup)
            {
                if (eleft)
                {
                    e = new DotVarExp(exp.loc, eleft, tup);
                    e = e.expressionSemantic(sc);
                    return e;
                }
                e = new TupleExp(exp.loc, tup);
                e = e.expressionSemantic(sc);
                return e;
            }

            ScopeDsymbol sds = s.isScopeDsymbol();
            if (sds)
            {
                //printf("it's a ScopeDsymbol %s\n", ident.toChars());
                e = new ScopeExp(exp.loc, sds);
                e = e.expressionSemantic(sc);
                if (eleft)
                    e = new DotExp(exp.loc, eleft, e);
                return e;
            }

            Import imp = s.isImport();
            if (imp)
            {
                ie = new ScopeExp(exp.loc, imp.pkg);
                return ie.expressionSemantic(sc);
            }
            // BUG: handle other cases like in IdentifierExp::semantic()
            debug
            {
                printf("s = '%s', kind = '%s'\n", s.toChars(), s.kind());
            }
            assert(0);
        }
        else if (exp.ident == Id.stringof)
        {
            e = new StringExp(exp.loc, ie.toString());
            e = e.expressionSemantic(sc);
            return e;
        }
        if (ie.sds.isPackage() || ie.sds.isImport() || ie.sds.isModule())
        {
            flag = 0;
        }
        if (flag)
            return null;
        s = ie.sds.search_correct(exp.ident);
        if (s && symbolIsVisible(sc, s))
        {
            if (s.isPackage())
                exp.error("undefined identifier `%s` in %s `%s`, perhaps add `static import %s;`", exp.ident.toChars(), ie.sds.kind(), ie.sds.toPrettyChars(), s.toPrettyChars());
            else
                exp.error("undefined identifier `%s` in %s `%s`, did you mean %s `%s`?", exp.ident.toChars(), ie.sds.kind(), ie.sds.toPrettyChars(), s.kind(), s.toChars());
        }
        else
            exp.error("undefined identifier `%s` in %s `%s`", exp.ident.toChars(), ie.sds.kind(), ie.sds.toPrettyChars());
        return new ErrorExp();
    }
    else if (t1b.ty == Tpointer && exp.e1.type.ty != Tenum && exp.ident != Id._init && exp.ident != Id.__sizeof && exp.ident != Id.__xalignof && exp.ident != Id.offsetof && exp.ident != Id._mangleof && exp.ident != Id.stringof)
    {
        Type t1bn = t1b.nextOf();
        if (flag)
        {
            AggregateDeclaration ad = isAggregate(t1bn);
            if (ad && !ad.members) // https://issues.dlang.org/show_bug.cgi?id=11312
                return null;
        }

        /* Rewrite:
         *   p.ident
         * as:
         *   (*p).ident
         */
        if (flag && t1bn.ty == Tvoid)
            return null;
        e = new PtrExp(exp.loc, exp.e1);
        e = e.expressionSemantic(sc);
        return e.type.dotExp(sc, e, exp.ident, flag | (exp.noderef ? DotExpFlag.noDeref : 0));
    }
    else
    {
        if (exp.e1.op == TOK.type || exp.e1.op == TOK.template_)
            flag = 0;
        e = exp.e1.type.dotExp(sc, exp.e1, exp.ident, flag | (exp.noderef ? DotExpFlag.noDeref : 0));
        if (e)
            e = e.expressionSemantic(sc);
        return e;
    }
}

// Resolve e1.ident!tiargs without seeing UFCS.
// If flag == 1, stop "not a property" error and return NULL.
Expression semanticY(DotTemplateInstanceExp exp, Scope* sc, int flag)
{
    static if (LOGSEMANTIC)
    {
        printf("DotTemplateInstanceExpY::semantic('%s')\n", exp.toChars());
    }

    static Expression errorExp()
    {
        return new ErrorExp();
    }

    auto die = new DotIdExp(exp.loc, exp.e1, exp.ti.name);

    Expression e = die.semanticX(sc);
    if (e == die)
    {
        exp.e1 = die.e1; // take back
        Type t1b = exp.e1.type.toBasetype();
        if (t1b.ty == Tarray || t1b.ty == Tsarray || t1b.ty == Taarray || t1b.ty == Tnull || (t1b.isTypeBasic() && t1b.ty != Tvoid))
        {
            /* No built-in type has templatized properties, so do shortcut.
             * It is necessary in: 1024.max!"a < b"
             */
            if (flag)
                return null;
        }
        e = die.semanticY(sc, flag);
        if (flag)
        {
            if (!e ||
                isDotOpDispatch(e))
            {
                /* opDispatch!tiargs would be a function template that needs IFTI,
                 * so it's not a template
                 */
                return null;
            }
        }
    }
    assert(e);

    if (e.op == TOK.error)
        return e;
    if (e.op == TOK.dotVariable)
    {
        DotVarExp dve = cast(DotVarExp)e;
        if (FuncDeclaration fd = dve.var.isFuncDeclaration())
        {
            if (TemplateDeclaration td = fd.findTemplateDeclRoot())
            {
                e = new DotTemplateExp(dve.loc, dve.e1, td);
                e = e.expressionSemantic(sc);
            }
        }
        else if (OverDeclaration od = dve.var.isOverDeclaration())
        {
            exp.e1 = dve.e1; // pull semantic() result

            if (!exp.findTempDecl(sc))
                goto Lerr;
            if (exp.ti.needsTypeInference(sc))
                return exp;
            exp.ti.dsymbolSemantic(sc);
            if (!exp.ti.inst || exp.ti.errors) // if template failed to expand
                return errorExp();

            if (Declaration v = exp.ti.toAlias().isDeclaration())
            {
                if (v.type && !v.type.deco)
                    v.type = v.type.typeSemantic(v.loc, sc);
                return new DotVarExp(exp.loc, exp.e1, v)
                       .expressionSemantic(sc);
            }
            return new DotExp(exp.loc, exp.e1, new ScopeExp(exp.loc, exp.ti))
                   .expressionSemantic(sc);
        }
    }
    else if (e.op == TOK.variable)
    {
        VarExp ve = cast(VarExp)e;
        if (FuncDeclaration fd = ve.var.isFuncDeclaration())
        {
            if (TemplateDeclaration td = fd.findTemplateDeclRoot())
            {
                e = new TemplateExp(ve.loc, td)
                    .expressionSemantic(sc);
            }
        }
        else if (OverDeclaration od = ve.var.isOverDeclaration())
        {
            exp.ti.tempdecl = od;
            return new ScopeExp(exp.loc, exp.ti)
                   .expressionSemantic(sc);
        }
    }

    if (e.op == TOK.dotTemplateDeclaration)
    {
        DotTemplateExp dte = cast(DotTemplateExp)e;
        exp.e1 = dte.e1; // pull semantic() result

        exp.ti.tempdecl = dte.td;
        if (!exp.ti.semanticTiargs(sc))
            return errorExp();
        if (exp.ti.needsTypeInference(sc))
            return exp;
        exp.ti.dsymbolSemantic(sc);
        if (!exp.ti.inst || exp.ti.errors) // if template failed to expand
            return errorExp();

        if (Declaration v = exp.ti.toAlias().isDeclaration())
        {
            if (v.isFuncDeclaration() || v.isVarDeclaration())
            {
                return new DotVarExp(exp.loc, exp.e1, v)
                       .expressionSemantic(sc);
            }
        }
        return new DotExp(exp.loc, exp.e1, new ScopeExp(exp.loc, exp.ti))
               .expressionSemantic(sc);
    }
    else if (e.op == TOK.template_)
    {
        exp.ti.tempdecl = (cast(TemplateExp)e).td;
        return new ScopeExp(exp.loc, exp.ti)
               .expressionSemantic(sc);
    }
    else if (e.op == TOK.dot)
    {
        DotExp de = cast(DotExp)e;

        if (de.e2.op == TOK.overloadSet)
        {
            if (!exp.findTempDecl(sc) || !exp.ti.semanticTiargs(sc))
            {
                return errorExp();
            }
            if (exp.ti.needsTypeInference(sc))
                return exp;
            exp.ti.dsymbolSemantic(sc);
            if (!exp.ti.inst || exp.ti.errors) // if template failed to expand
                return errorExp();

            if (Declaration v = exp.ti.toAlias().isDeclaration())
            {
                if (v.type && !v.type.deco)
                    v.type = v.type.typeSemantic(v.loc, sc);
                return new DotVarExp(exp.loc, exp.e1, v)
                       .expressionSemantic(sc);
            }
            return new DotExp(exp.loc, exp.e1, new ScopeExp(exp.loc, exp.ti))
                   .expressionSemantic(sc);
        }
    }
    else if (e.op == TOK.overloadSet)
    {
        OverExp oe = cast(OverExp)e;
        exp.ti.tempdecl = oe.vars;
        return new ScopeExp(exp.loc, exp.ti)
               .expressionSemantic(sc);
    }

Lerr:
    exp.error("`%s` isn't a template", e.toChars());
    return errorExp();
}

/***************************************
 * If expression is shared, check that we can access it.
 * Give error message if not.
 * Params:
 *      e = expression to check
 *      sc = context
 * Returns:
 *      true on error
 */
bool checkSharedAccess(Expression e, Scope* sc)
{
    if (!global.params.noSharedAccess ||
        sc.intypeof ||
        sc.flags & SCOPE.ctfe)
    {
        return false;
    }

    //printf("checkSharedAccess() %s\n", e.toChars());

    static bool check(Expression e)
    {
        static bool sharedError(Expression e)
        {
            // https://dlang.org/phobos/core_atomic.html
            e.error("direct access to shared `%s` is not allowed, see `core.atomic`", e.toChars());
            return true;
        }

        bool visitVar(VarExp ve)
        {
            return ve.var.type.isShared() ? sharedError(ve) : false;
        }

        bool visitPtr(PtrExp pe)
        {
            return pe.e1.type.nextOf().isShared() ? sharedError(pe) : false;
        }

        bool visitDotVar(DotVarExp dve)
        {
            return dve.var.type.isShared() || check(dve.e1) ? sharedError(dve) : false;
        }

        bool visitIndex(IndexExp ie)
        {
            return ie.e1.type.nextOf().isShared() ? sharedError(ie) : false;
        }

        bool visitComma(CommaExp ce)
        {
            return check(ce.e2);
        }

        switch (e.op)
        {
            case TOK.variable:    return visitVar(e.isVarExp());
            case TOK.star:        return visitPtr(e.isPtrExp());
            case TOK.dotVariable: return visitDotVar(e.isDotVarExp());
            case TOK.index:       return visitIndex(e.isIndexExp());
            case TOK.comma:       return visitComma(e.isCommaExp());
            default:
                return false;
        }
    }

    return check(e);
}



/****************************************************
 * Determine if `exp`, which takes the address of `v`, can do so safely.
 * Params:
 *      sc = context
 *      exp = expression that takes the address of `v`
 *      v = the variable getting its address taken
 * Returns:
 *      `true` if ok, `false` for error
 */
private bool checkAddressVar(Scope* sc, UnaExp exp, VarDeclaration v)
{
    //printf("checkAddressVar(exp: %s, v: %s)\n", exp.toChars(), v.toChars());
    if (v)
    {
        if (!v.canTakeAddressOf())
        {
            exp.error("cannot take address of `%s`", exp.e1.toChars());
            return false;
        }
        if (sc.func && !sc.intypeof && !v.isDataseg())
        {
            const(char)* p = v.isParameter() ? "parameter" : "local";
            if (global.params.vsafe)
            {
                // Taking the address of v means it cannot be set to 'scope' later
                v.storage_class &= ~STC.maybescope;
                v.doNotInferScope = true;
                if (exp.e1.type.hasPointers() && v.storage_class & STC.scope_ &&
                    !(sc.flags & SCOPE.debug_) && sc.func.setUnsafe())
                {
                    exp.error("cannot take address of `scope` %s `%s` in `@safe` function `%s`", p, v.toChars(), sc.func.toChars());
                    return false;
                }
            }
            else if (!(sc.flags & SCOPE.debug_) && sc.func.setUnsafe())
            {
                exp.error("cannot take address of %s `%s` in `@safe` function `%s`", p, v.toChars(), sc.func.toChars());
                return false;
            }
        }
    }
    return true;
}

/*******************************
 * Checks the attributes of a function.
 * Purity (`pure`), safety (`@safe`), no GC allocations(`@nogc`)
 * and usage of `deprecated` and `@disabled`-ed symbols are checked.
 *
 * Params:
 *  exp = expression to check attributes for
 *  sc  = scope of the function
 *  f   = function to be checked
 * Returns: `true` if error occur.
 */
private bool checkFunctionAttributes(Expression exp, Scope* sc, FuncDeclaration f)
{
    with(exp)
    {
        bool error = checkDisabled(sc, f);
        error |= checkDeprecated(sc, f);
        error |= checkPurity(sc, f);
        error |= checkSafety(sc, f);
        error |= checkNogc(sc, f);
        return error;
    }
}

/*******************************
 * Helper function for `getRightThis()`.
 * Gets `this` of the next outer aggregate.
 * Params:
 *      loc = location to use for error messages
 *      sc = context
 *      s = the parent symbol of the existing `this`
 *      ad = struct or class we need the correct `this` for
 *      e1 = existing `this`
 *      t = type of the existing `this`
 *      var = the specific member of ad we're accessing
 *      flag = if true, return `null` instead of throwing an error
 * Returns:
 *      Expression representing the `this` for the var
 */
Expression getThisSkipNestedFuncs(const ref Loc loc, Scope* sc, Dsymbol s, AggregateDeclaration ad, Expression e1, Type t, Dsymbol var, bool flag = false)
{
    int n = 0;
    while (s && s.isFuncDeclaration())
    {
        FuncDeclaration f = s.isFuncDeclaration();
        if (f.vthis)
        {
            n++;
            // LDC seems dmd misses it sometimes here :/
            if (IN_LLVM && !flag && f.isMember2())
            {
                f.vthis.nestedrefs.push(sc.parent.isFuncDeclaration());
                f.closureVars.push(f.vthis);
            }
            e1 = new VarExp(loc, f.vthis);
            if (f.isThis2)
            {
                // (*__this)[i]
                if (n > 1)
                    e1 = e1.expressionSemantic(sc);
                e1 = new PtrExp(loc, e1);
                uint i = f.followInstantiationContext(ad);
                e1 = new IndexExp(loc, e1, new IntegerExp(i));
                s = f.toParentP(ad);
                continue;
            }
        }
        else
        {
            if (flag)
                return null;
            e1.error("need `this` of type `%s` to access member `%s` from static function `%s`", ad.toChars(), var.toChars(), f.toChars());
            e1 = new ErrorExp();
            return e1;
        }
        s = s.toParent2();
    }
    if (n > 1 || e1.op == TOK.index)
        e1 = e1.expressionSemantic(sc);
    if (s && e1.type.equivalent(Type.tvoidptr))
    {
        if (auto sad = s.isAggregateDeclaration())
        {
            Type ta = sad.handleType();
            if (ta.ty == Tstruct)
                ta = ta.pointerTo();
            e1.type = ta;
        }
    }
    e1.type = e1.type.addMod(t.mod);
    return e1;
}

/*******************************
 * Make a dual-context container for use as a `this` argument.
 * Params:
 *      loc = location to use for error messages
 *      sc = current scope
 *      fd = target function that will take the `this` argument
 * Returns:
 *      Temporary closure variable.
 * Note:
 *      The function `fd` is added to the nested references of the
 *      newly created variable such that a closure is made for the variable when
 *      the address of `fd` is taken.
 */
VarDeclaration makeThis2Argument(const ref Loc loc, Scope* sc, FuncDeclaration fd)
{
    Type tthis2 = Type.tvoidptr.sarrayOf(2);
    VarDeclaration vthis2 = new VarDeclaration(loc, tthis2, Identifier.generateId("__this"), null);
    vthis2.storage_class |= STC.temp;
    vthis2.dsymbolSemantic(sc);
    vthis2.parent = sc.parent;
    // make it a closure var
    assert(sc.func);
    sc.func.closureVars.push(vthis2);
    // add `fd` to the nested refs
    vthis2.nestedrefs.push(fd);
    return vthis2;
}

/*******************************
 * Make sure that the runtime hook `id` exists.
 * Params:
 *      loc = location to use for error messages
 *      sc = current scope
 *      id = the hook identifier
 *      description = what the hook does
 *      module_ = what module the hook is located in
 * Returns:
 *      a `bool` indicating if the hook is present.
 */
bool verifyHookExist(const ref Loc loc, ref Scope sc, Identifier id, string description, Identifier module_ = Id.object)
{
    auto rootSymbol = sc.search(loc, Id.empty, null);
    if (auto moduleSymbol = rootSymbol.search(loc, module_))
        if (moduleSymbol.search(loc, id))
          return true;
    error(loc, "`%s.%s` not found. The current runtime does not support %.*s, or the runtime is corrupt.", module_.toChars(), id.toChars(), cast(int)description.length, description.ptr);
    return false;
}

/**
 * Check if an expression is an access to a struct member with the struct
 * defined from a literal.
 *
 * This happens with manifest constants since the initializer is reused as is,
 * each time the declaration is part of an expression, which means that the
 * literal used as initializer can become a Lvalue. This Lvalue must not be modifiable.
 *
 * Params:
 *      exp = An expression that's attempted to be written.
 *            Must be the LHS of an `AssignExp`, `BinAssignExp`, `CatAssignExp`,
 *            or the expression passed to a modifiable function parameter.
 * Returns:
 *      `true` if `expr` is a dot var or a dot identifier touching to a struct literal,
 *      in which case an error message is issued, and `false` otherwise.
 */
private bool checkIfIsStructLiteralDotExpr(Expression exp)
{
    // e1.var = ...
    // e1.ident = ...
    Expression e1;
    if (exp.op == TOK.dotVariable)
        e1 = exp.isDotVarExp().e1;
    else if (exp.op == TOK.dotIdentifier)
        e1 = exp.isDotIdExp().e1;
    else
        return false;

    // enum SomeStruct ss = { ... }
    // also true for access from a .init: SomeStruct.init.member = ...
    if (e1.op != TOK.structLiteral)
        return false;

    error(exp.loc, "cannot modify constant expression `%s`", exp.toChars());
    return true;
}
