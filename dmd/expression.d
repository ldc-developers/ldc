/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/expression.d, _expression.d)
 * Documentation:  https://dlang.org/phobos/dmd_expression.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/expression.d
 */

module dmd.expression;

import core.stdc.stdarg;
import core.stdc.stdio;
import core.stdc.string;

import dmd.aggregate;
import dmd.aliasthis;
import dmd.apply;
import dmd.arrayop;
import dmd.arraytypes;
import dmd.gluelayer;
import dmd.canthrow;
import dmd.complex;
import dmd.constfold;
import dmd.ctfeexpr;
import dmd.dcast;
import dmd.dclass;
import dmd.declaration;
import dmd.delegatize;
import dmd.dimport;
import dmd.dinterpret;
import dmd.dmodule;
import dmd.dscope;
import dmd.dstruct;
import dmd.dsymbol;
import dmd.dsymbolsem;
import dmd.dtemplate;
import dmd.errors;
import dmd.escape;
import dmd.expressionsem;
import dmd.func;
import dmd.globals;
import dmd.hdrgen;
import dmd.id;
import dmd.identifier;
import dmd.inline;
import dmd.mtype;
import dmd.nspace;
import dmd.opover;
import dmd.optimize;
import dmd.root.ctfloat;
import dmd.root.filename;
import dmd.root.outbuffer;
import dmd.root.rmem;
import dmd.root.rootobject;
import dmd.safe;
import dmd.sideeffect;
import dmd.target;
import dmd.tokens;
import dmd.typesem;
import dmd.utf;
import dmd.visitor;
version(IN_LLVM) import gen.dpragma;

enum LOGSEMANTIC = false;
void emplaceExp(T : Expression, Args...)(void* p, Args args)
{
    scope tmp = new T(args);
    memcpy(p, cast(void*)tmp, __traits(classInstanceSize, T));
}

void emplaceExp(T : UnionExp)(T* p, Expression e)
{
    memcpy(p, cast(void*)e, e.size);
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
extern (C++) Expression getRightThis(const ref Loc loc, Scope* sc, AggregateDeclaration ad, Expression e1, Declaration var, int flag = 0)
{
    //printf("\ngetRightThis(e1 = %s, ad = %s, var = %s)\n", e1.toChars(), ad.toChars(), var.toChars());
L1:
    Type t = e1.type.toBasetype();
    //printf("e1.type = %s, var.type = %s\n", e1.type.toChars(), var.type.toChars());

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
                e1 = new DotVarExp(loc, e1, tcd.vthis);
                e1.type = tcd.vthis.type;
                e1.type = e1.type.addMod(t.mod);
                // Do not call ensureStaticLinkTo()
                //e1 = e1.semantic(sc);

                // Skip up over nested functions, and get the enclosing
                // class type.
                int n = 0;
                Dsymbol s;
                for (s = tcd.toParent(); s && s.isFuncDeclaration(); s = s.toParent())
                {
                    FuncDeclaration f = s.isFuncDeclaration();
                    if (f.vthis)
                    {
                        //printf("rewriting e1 to %s's this\n", f.toChars());
                        n++;
                        version(IN_LLVM)
                        {
                            // LDC seems dmd misses it sometimes here :/
                            if (f.isMember2()) {
                                f.vthis.nestedrefs.push(sc.parent.isFuncDeclaration());
                                f.closureVars.push(f.vthis);
                            }
                        }
                        e1 = new VarExp(loc, f.vthis);
                    }
                    else
                    {
                        e1.error("need `this` of type `%s` to access member `%s` from static function `%s`", ad.toChars(), var.toChars(), f.toChars());
                        e1 = new ErrorExp();
                        return e1;
                    }
                }
                if (s && s.isClassDeclaration())
                {
                    e1.type = s.isClassDeclaration().type;
                    e1.type = e1.type.addMod(t.mod);
                    if (n > 1)
                        e1 = e1.expressionSemantic(sc);
                }
                else
                    e1 = e1.expressionSemantic(sc);
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

/****************************************
 * Resolve a symbol `s` and wraps it in an expression object.
 *
 * Params:
 *      loc = location of use of `s`
 *      sc = context
 *      s = symbol to resolve
 *      hasOverloads = applies if `s` represents a function.
 *          true means it's overloaded and will be resolved later,
 *          false means it's the exact function symbol.
 * Returns:
 *      `s` turned into an expression, `ErrorExp` if an error occurred
 */
Expression resolve(const ref Loc loc, Scope *sc, Dsymbol s, bool hasOverloads)
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
        Dsymbol p = td.toParent2();
        FuncDeclaration fdthis = hasThis(sc);
        AggregateDeclaration ad = p ? p.isAggregateDeclaration() : null;
        if (fdthis && ad && isAggregate(fdthis.vthis.type) == ad && (td._scope.stc & STC.static_) == 0)
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

/*****************************************
 * Determine if `this` is available by walking up the enclosing
 * scopes until a function is found.
 *
 * Params:
 *      sc = where to start looking for the enclosing function
 * Returns:
 *      Found function if it satisfies `isThis()`, otherwise `null`
 */
extern (C++) FuncDeclaration hasThis(Scope* sc)
{
    //printf("hasThis()\n");
    Dsymbol p = sc.parent;
    while (p && p.isTemplateMixin())
        p = p.parent;
    FuncDeclaration fdthis = p ? p.isFuncDeclaration() : null;
    //printf("fdthis = %p, '%s'\n", fdthis, fdthis ? fdthis.toChars() : "");

    // Go upwards until we find the enclosing member function
    FuncDeclaration fd = fdthis;
    while (1)
    {
        if (!fd)
        {
            goto Lno;
        }
        if (!fd.isNested())
            break;

        Dsymbol parent = fd.parent;
        while (1)
        {
            if (!parent)
                goto Lno;
            TemplateInstance ti = parent.isTemplateInstance();
            if (ti)
                parent = ti.parent;
            else
                break;
        }
        fd = parent.isFuncDeclaration();
    }

    if (!fd.isThis())
    {
        goto Lno;
    }

    assert(fd.vthis);
    return fd;

Lno:
    return null; // don't have 'this' available
}

/***********************************
 * Determine if a `this` is needed to access `d`.
 * Params:
 *      sc = context
 *      d = declaration to check
 * Returns:
 *      true means a `this` is needed
 */
extern (C++) bool isNeedThisScope(Scope* sc, Declaration d)
{
    if (sc.intypeof == 1)
        return false;

    AggregateDeclaration ad = d.isThis();
    if (!ad)
        return false;
    //printf("d = %s, ad = %s\n", d.toChars(), ad.toChars());

    for (Dsymbol s = sc.parent; s; s = s.toParent2())
    {
        //printf("\ts = %s %s, toParent2() = %p\n", s.kind(), s.toChars(), s.toParent2());
        if (AggregateDeclaration ad2 = s.isAggregateDeclaration())
        {
            if (ad2 == ad)
                return false;
            else if (ad2.isNested())
                continue;
            else
                return true;
        }
        if (FuncDeclaration f = s.isFuncDeclaration())
        {
            if (f.isMember2())
                break;
        }
    }
    return true;
}

/******************************
 * Check the tail CallExp is really property function call.
 * Bugs:
 * This doesn't appear to do anything.
 */
private bool checkPropertyCall(Expression e, Expression emsg)
{
    while (e.op == TOK.comma)
        e = (cast(CommaExp)e).e2;

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
 * If e1 is a property function (template), resolve it.
 */
extern (C++) Expression resolvePropertiesOnly(Scope* sc, Expression e1)
{
    //printf("e1 = %s %s\n", Token::toChars(e1.op), e1.toChars());
    OverloadSet os;
    FuncDeclaration fd;
    TemplateDeclaration td;

    if (e1.op == TOK.dot)
    {
        DotExp de = cast(DotExp)e1;
        if (de.e2.op == TOK.overloadSet)
        {
            os = (cast(OverExp)de.e2).vars;
            goto Los;
        }
    }
    else if (e1.op == TOK.overloadSet)
    {
        os = (cast(OverExp)e1).vars;
    Los:
        assert(os);
        for (size_t i = 0; i < os.a.dim; i++)
        {
            Dsymbol s = os.a[i];
            fd = s.isFuncDeclaration();
            td = s.isTemplateDeclaration();
            if (fd)
            {
                if ((cast(TypeFunction)fd.type).isproperty)
                    return resolveProperties(sc, e1);
            }
            else if (td && td.onemember && (fd = td.onemember.isFuncDeclaration()) !is null)
            {
                if ((cast(TypeFunction)fd.type).isproperty || (fd.storage_class2 & STC.property) || (td._scope.stc & STC.property))
                {
                    return resolveProperties(sc, e1);
                }
            }
        }
    }
    else if (e1.op == TOK.dotTemplateInstance)
    {
        DotTemplateInstanceExp dti = cast(DotTemplateInstanceExp)e1;
        if (dti.ti.tempdecl && (td = dti.ti.tempdecl.isTemplateDeclaration()) !is null)
            goto Ltd;
    }
    else if (e1.op == TOK.dotTemplateDeclaration)
    {
        td = (cast(DotTemplateExp)e1).td;
        goto Ltd;
    }
    else if (e1.op == TOK.scope_)
    {
        Dsymbol s = (cast(ScopeExp)e1).sds;
        TemplateInstance ti = s.isTemplateInstance();
        if (ti && !ti.semanticRun && ti.tempdecl)
        {
            if ((td = ti.tempdecl.isTemplateDeclaration()) !is null)
                goto Ltd;
        }
    }
    else if (e1.op == TOK.template_)
    {
        td = (cast(TemplateExp)e1).td;
    Ltd:
        assert(td);
        if (td.onemember && (fd = td.onemember.isFuncDeclaration()) !is null)
        {
            if ((cast(TypeFunction)fd.type).isproperty || (fd.storage_class2 & STC.property) || (td._scope.stc & STC.property))
            {
                return resolveProperties(sc, e1);
            }
        }
    }
    else if (e1.op == TOK.dotVariable && e1.type.ty == Tfunction)
    {
        DotVarExp dve = cast(DotVarExp)e1;
        fd = dve.var.isFuncDeclaration();
        goto Lfd;
    }
    else if (e1.op == TOK.variable && e1.type.ty == Tfunction && (sc.intypeof || !(cast(VarExp)e1).var.needThis()))
    {
        fd = (cast(VarExp)e1).var.isFuncDeclaration();
    Lfd:
        assert(fd);
        if ((cast(TypeFunction)fd.type).isproperty)
            return resolveProperties(sc, e1);
    }
    return e1;
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

    Dsymbol sold = void;
    if (global.params.bug10378 || global.params.check10378)
    {
        sold = searchScopes(flags | IgnoreSymbolVisibility);
        if (!global.params.check10378)
        {
            s = sold;
            goto Lsearchdone;
        }
    }

    // First look in local scopes
    s = searchScopes(flags | SearchLocalsOnly);
    if (!s)
    {
        // Second look in imported modules
        s = searchScopes(flags | SearchImportsOnly);

        /** Still find private symbols, so that symbols that weren't access
         * checked by the compiler remain usable.  Once the deprecation is over,
         * this should be moved to search_correct instead.
         */
        if (!s && !(flags & IgnoreSymbolVisibility))
        {
            s = searchScopes(flags | SearchLocalsOnly | IgnoreSymbolVisibility);
            if (!s)
                s = searchScopes(flags | SearchImportsOnly | IgnoreSymbolVisibility);

            if (s)
                .deprecation(loc, "`%s` is not visible from module `%s`", s.toPrettyChars(), sc._module.toChars());
        }
    }
    if (global.params.check10378)
    {
        alias snew = s;
        if (sold !is snew)
            Scope.deprecation10378(loc, sold, snew);
        if (global.params.bug10378)
            s = sold;
    }
Lsearchdone:

    if (!s)
        return ue.e1.type.Type.getProperty(loc, ident, 0);

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
 * check e is exp.opDispatch!(tiargs) or not
 * It's used to switch to UFCS the semantic analysis path
 */
extern (C++) bool isDotOpDispatch(Expression e)
{
    return e.op == TOK.dotTemplateInstance && (cast(DotTemplateInstanceExp)e).ti.name == Id.opDispatch;
}

/******************************
 * Pull out callable entity with UFCS.
 */
extern (C++) Expression resolveUFCS(Scope* sc, CallExp ce)
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

                if (key.checkValue())
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
                    /* fall down to UFCS */
                }
                else
                    return null;
            }
        }
        e = searchUFCS(sc, die, ident);
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
extern (C++) Expression resolveUFCSProperties(Scope* sc, Expression e1, Expression e2 = null)
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
        auto a1 = new Expressions();
        a1.setDim(1);
        (*a1)[0] = eleft;
        ex = new CallExp(loc, ex, a1);
        ex = ex.trySemantic(sc);

        /* f(e1, e2)
         */
        auto a2 = new Expressions();
        a2.setDim(2);
        (*a2)[0] = eleft;
        (*a2)[1] = e2;
        e = new CallExp(loc, e, a2);
        if (ex)
        {
            // if fallback setter exists, gag errors
            e = e.trySemantic(sc);
            if (!e)
            {
                checkPropertyCall(ex, e1);
                ex = new AssignExp(loc, ex, e2);
                return ex.expressionSemantic(sc);
            }
        }
        else
        {
            // strict setter prints errors if fails
            e = e.expressionSemantic(sc);
        }
        checkPropertyCall(e, e1);
        return e;
    }
    else
    {
        /* f(e1)
         */
        auto arguments = new Expressions();
        arguments.setDim(1);
        (*arguments)[0] = eleft;
        e = new CallExp(loc, e, arguments);
        e = e.expressionSemantic(sc);
        checkPropertyCall(e, e1);
        return e.expressionSemantic(sc);
    }
}

/******************************
 * Perform semantic() on an array of Expressions.
 */
extern (C++) bool arrayExpressionSemantic(Expressions* exps, Scope* sc, bool preserveErrors = false)
{
    bool err = false;
    if (exps)
    {
        for (size_t i = 0; i < exps.dim; i++)
        {
            Expression e = (*exps)[i];
            if (e)
            {
                e = e.expressionSemantic(sc);
                if (e.op == TOK.error)
                    err = true;
                if (preserveErrors || e.op != TOK.error)
                    (*exps)[i] = e;
            }
        }
    }
    return err;
}

/****************************************
 * Expand tuples.
 * Input:
 *      exps    aray of Expressions
 * Output:
 *      exps    rewritten in place
 */
extern (C++) void expandTuples(Expressions* exps)
{
    //printf("expandTuples()\n");
    if (exps)
    {
        for (size_t i = 0; i < exps.dim; i++)
        {
            Expression arg = (*exps)[i];
            if (!arg)
                continue;

            // Look for tuple with 0 members
            if (arg.op == TOK.type)
            {
                TypeExp e = cast(TypeExp)arg;
                if (e.type.toBasetype().ty == Ttuple)
                {
                    TypeTuple tt = cast(TypeTuple)e.type.toBasetype();
                    if (!tt.arguments || tt.arguments.dim == 0)
                    {
                        exps.remove(i);
                        if (i == exps.dim)
                            return;
                        i--;
                        continue;
                    }
                }
            }

            // Inline expand all the tuples
            while (arg.op == TOK.tuple)
            {
                TupleExp te = cast(TupleExp)arg;
                exps.remove(i); // remove arg
                exps.insert(i, te.exps); // replace with tuple contents
                if (i == exps.dim)
                    return; // empty tuple, no more arguments
                (*exps)[i] = Expression.combine(te.e0, (*exps)[i]);
                arg = (*exps)[i];
            }
        }
    }
}

/****************************************
 * Expand alias this tuples.
 */
extern (C++) TupleDeclaration isAliasThisTuple(Expression e)
{
    if (!e.type)
        return null;

    Type t = e.type.toBasetype();
Lagain:
    if (Dsymbol s = t.toDsymbol(null))
    {
        AggregateDeclaration ad = s.isAggregateDeclaration();
        if (ad)
        {
            s = ad.aliasthis;
            if (s && s.isVarDeclaration())
            {
                TupleDeclaration td = s.isVarDeclaration().toAlias().isTupleDeclaration();
                if (td && td.isexp)
                    return td;
            }
            if (Type att = t.aliasthisOf())
            {
                t = att;
                goto Lagain;
            }
        }
    }
    return null;
}

extern (C++) int expandAliasThisTuples(Expressions* exps, size_t starti = 0)
{
    if (!exps || exps.dim == 0)
        return -1;

    for (size_t u = starti; u < exps.dim; u++)
    {
        Expression exp = (*exps)[u];
        TupleDeclaration td = isAliasThisTuple(exp);
        if (td)
        {
            exps.remove(u);
            for (size_t i = 0; i < td.objects.dim; ++i)
            {
                Expression e = isExpression((*td.objects)[i]);
                assert(e);
                assert(e.op == TOK.dSymbol);
                DsymbolExp se = cast(DsymbolExp)e;
                Declaration d = se.s.isDeclaration();
                assert(d);
                e = new DotVarExp(exp.loc, exp, d);
                assert(d.type);
                e.type = d.type;
                exps.insert(u + i, e);
            }
            version (none)
            {
                printf("expansion ->\n");
                for (size_t i = 0; i < exps.dim; ++i)
                {
                    Expression e = (*exps)[i];
                    printf("\texps[%d] e = %s %s\n", i, Token.tochars[e.op], e.toChars());
                }
            }
            return cast(int)u;
        }
    }
    return -1;
}

/****************************************
 * Get TemplateDeclaration enclosing FuncDeclaration.
 */
extern (C++) TemplateDeclaration getFuncTemplateDecl(Dsymbol s)
{
    FuncDeclaration f = s.isFuncDeclaration();
    if (f && f.parent)
    {
        TemplateInstance ti = f.parent.isTemplateInstance();
        if (ti && !ti.isTemplateMixin() && ti.tempdecl && (cast(TemplateDeclaration)ti.tempdecl).onemember && ti.tempdecl.ident == f.ident)
        {
            return cast(TemplateDeclaration)ti.tempdecl;
        }
    }
    return null;
}

/************************************************
 * If we want the value of this expression, but do not want to call
 * the destructor on it.
 */
extern (C++) Expression valueNoDtor(Expression e)
{
    auto ex = e;
    while (ex.op == TOK.comma)
        ex = (cast(CommaExp)ex).e2;

    if (ex.op == TOK.call)
    {
        /* The struct value returned from the function is transferred
         * so do not call the destructor on it.
         * Recognize:
         *       ((S _ctmp = S.init), _ctmp).this(...)
         * and make sure the destructor is not called on _ctmp
         * BUG: if ex is a CommaExp, we should go down the right side.
         */
        CallExp ce = cast(CallExp)ex;
        if (ce.e1.op == TOK.dotVariable)
        {
            DotVarExp dve = cast(DotVarExp)ce.e1;
            if (dve.var.isCtorDeclaration())
            {
                // It's a constructor call
                if (dve.e1.op == TOK.comma)
                {
                    CommaExp comma = cast(CommaExp)dve.e1;
                    if (comma.e2.op == TOK.variable)
                    {
                        VarExp ve = cast(VarExp)comma.e2;
                        VarDeclaration ctmp = ve.var.isVarDeclaration();
                        if (ctmp)
                        {
                            ctmp.storage_class |= STC.nodtor;
                            assert(!ce.isLvalue());
                        }
                    }
                }
            }
        }
    }
    else if (ex.op == TOK.variable)
    {
        auto vtmp = (cast(VarExp)ex).var.isVarDeclaration();
        if (vtmp && (vtmp.storage_class & STC.rvalue))
        {
            vtmp.storage_class |= STC.nodtor;
        }
    }
    return e;
}

/*********************************************
 * If e is an instance of a struct, and that struct has a copy constructor,
 * rewrite e as:
 *    (tmp = e),tmp
 * Input:
 *      sc      just used to specify the scope of created temporary variable
 */
private Expression callCpCtor(Scope* sc, Expression e)
{
    Type tv = e.type.baseElemOf();
    if (tv.ty == Tstruct)
    {
        StructDeclaration sd = (cast(TypeStruct)tv).sym;
        if (sd.postblit)
        {
            /* Create a variable tmp, and replace the argument e with:
             *      (tmp = e),tmp
             * and let AssignExp() handle the construction.
             * This is not the most efficient, ideally tmp would be constructed
             * directly onto the stack.
             */
            auto tmp = copyToTemp(STC.rvalue, "__copytmp", e);
            tmp.storage_class |= STC.nodtor;
            tmp.dsymbolSemantic(sc);
            Expression de = new DeclarationExp(e.loc, tmp);
            Expression ve = new VarExp(e.loc, tmp);
            de.type = Type.tvoid;
            ve.type = e.type;
            e = Expression.combine(de, ve);
        }
    }
    return e;
}

/************************************************
 * Handle the postblit call on lvalue, or the move of rvalue.
 */
extern (C++) Expression doCopyOrMove(Scope *sc, Expression e)
{
    if (e.op == TOK.question)
    {
        auto ce = cast(CondExp)e;
        ce.e1 = doCopyOrMove(sc, ce.e1);
        ce.e2 = doCopyOrMove(sc, ce.e2);
    }
    else
    {
        e = e.isLvalue() ? callCpCtor(sc, e) : valueNoDtor(e);
    }
    return e;
}

/****************************************************************/
/* A type meant as a union of all the Expression types,
 * to serve essentially as a Variant that will sit on the stack
 * during CTFE to reduce memory consumption.
 */
struct UnionExp
{
    // yes, default constructor does nothing
    extern (D) this(Expression e)
    {
        memcpy(&this, cast(void*)e, e.size);
    }

    /* Extract pointer to Expression
     */
    extern (C++) Expression exp()
    {
        return cast(Expression)&u;
    }

    /* Convert to an allocated Expression
     */
    extern (C++) Expression copy()
    {
        Expression e = exp();
        //if (e.size > sizeof(u)) printf("%s\n", Token::toChars(e.op));
        assert(e.size <= u.sizeof);
        if (e.op == TOK.cantExpression)
            return CTFEExp.cantexp;
        if (e.op == TOK.voidExpression)
            return CTFEExp.voidexp;
        if (e.op == TOK.break_)
            return CTFEExp.breakexp;
        if (e.op == TOK.continue_)
            return CTFEExp.continueexp;
        if (e.op == TOK.goto_)
            return CTFEExp.gotoexp;
        return e.copy();
    }

private:
    union __AnonStruct__u
    {
        char[__traits(classInstanceSize, Expression)] exp;
        char[__traits(classInstanceSize, IntegerExp)] integerexp;
        char[__traits(classInstanceSize, ErrorExp)] errorexp;
        char[__traits(classInstanceSize, RealExp)] realexp;
        char[__traits(classInstanceSize, ComplexExp)] complexexp;
        char[__traits(classInstanceSize, SymOffExp)] symoffexp;
        char[__traits(classInstanceSize, StringExp)] stringexp;
        char[__traits(classInstanceSize, ArrayLiteralExp)] arrayliteralexp;
        char[__traits(classInstanceSize, AssocArrayLiteralExp)] assocarrayliteralexp;
        char[__traits(classInstanceSize, StructLiteralExp)] structliteralexp;
        char[__traits(classInstanceSize, NullExp)] nullexp;
        char[__traits(classInstanceSize, DotVarExp)] dotvarexp;
        char[__traits(classInstanceSize, AddrExp)] addrexp;
        char[__traits(classInstanceSize, IndexExp)] indexexp;
        char[__traits(classInstanceSize, SliceExp)] sliceexp;
        // Ensure that the union is suitably aligned.
        real_t for_alignment_only;
    }

    __AnonStruct__u u;
}

/********************************
 * Test to see if two reals are the same.
 * Regard NaN's as equivalent.
 * Regard +0 and -0 as different.
 */
extern (C++) int RealEquals(real_t x1, real_t x2)
{
    return (CTFloat.isNaN(x1) && CTFloat.isNaN(x2)) || CTFloat.isIdentical(x1, x2);
}

/************************ TypeDotIdExp ************************************/
/* Things like:
 *      int.size
 *      foo.size
 *      (foo).size
 *      cast(foo).size
 */
extern (C++) DotIdExp typeDotIdExp(const ref Loc loc, Type type, Identifier ident)
{
    return new DotIdExp(loc, new TypeExp(loc, type), ident);
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

/****************************************************************/

private Expression extractOpDollarSideEffect(Scope* sc, UnaExp ue)
{
    Expression e0;
    Expression e1 = Expression.extractLast(ue.e1, &e0);
    // https://issues.dlang.org/show_bug.cgi?id=12585
    // Extract the side effect part if ue.e1 is comma.

    if (!isTrivialExp(e1))
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
extern (C++) Expression resolveOpDollar(Scope* sc, ArrayExp ae, Expression* pe0)
{
    assert(!ae.lengthVar);
    *pe0 = null;
    AggregateDeclaration ad = isAggregate(ae.e1.type);
    Dsymbol slice = search_function(ad, Id.slice);
    //printf("slice = %s %s\n", slice.kind(), slice.toChars());
    for (size_t i = 0; i < ae.arguments.dim; i++)
    {
        if (i == 0)
            *pe0 = extractOpDollarSideEffect(sc, ae);

        Expression e = (*ae.arguments)[i];
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
        sym.loc = ae.loc;
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

            auto fargs = new Expressions();
            fargs.push(ie.lwr);
            fargs.push(ie.upr);

            uint xerrors = global.startGagging();
            sc = sc.push();
            FuncDeclaration fslice = resolveFuncCall(ae.loc, sc, slice, tiargs, ae.e1.type, fargs, 1);
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
 */
extern (C++) Expression resolveOpDollar(Scope* sc, ArrayExp ae, IntervalExp ie, Expression* pe0)
{
    //assert(!ae.lengthVar);
    if (!ie)
        return ae;

    VarDeclaration lengthVar = ae.lengthVar;

    // create scope for '$'
    auto sym = new ArrayScopeSymbol(sc, ae);
    sym.loc = ae.loc;
    sym.parent = sc.scopesym;
    sc = sc.push(sym);

    for (size_t i = 0; i < 2; ++i)
    {
        Expression e = i == 0 ? ie.lwr : ie.upr;
        e = e.expressionSemantic(sc);
        e = resolveProperties(sc, e);
        if (!e.type)
        {
            ae.error("`%s` has no value", e.toChars());
            return new ErrorExp();
        }
        (i == 0 ? ie.lwr : ie.upr) = e;
    }

    if (lengthVar != ae.lengthVar && sc.func)
    {
        // If $ was used, declare it now
        Expression de = new DeclarationExp(ae.loc, ae.lengthVar);
        de = de.expressionSemantic(sc);
        *pe0 = Expression.combine(*pe0, de);
    }

    sc = sc.pop();

    return ae;
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

enum OwnedBy : int
{
    code,          // normal code expression in AST
    ctfe,          // value expression for CTFE
    cache,         // constant value cached for CTFE
}

enum WANTvalue  = 0;    // default
enum WANTexpand = 1;    // expand const/immutable variables if possible

/***********************************************************
 * http://dlang.org/spec/expression.html#expression
 */
// LDC: Instantiated in gen/asm-x86.h (`Handled = createExpression(...)`).
extern (C++) /* IN_LLVM abstract */ class Expression : RootObject
{
    Loc loc;        // file location
    Type type;      // !=null means that semantic() has been run
    TOK op;         // to minimize use of dynamic_cast
    ubyte size;     // # of bytes in Expression so we can copy() it
    ubyte parens;   // if this is a parenthesized expression

    extern (D) this(const ref Loc loc, TOK op, int size)
    {
        //printf("Expression::Expression(op = %d) this = %p\n", op, this);
        this.loc = loc;
        this.op = op;
        this.size = cast(ubyte)size;
    }

    static void _init()
    {
        CTFEExp.cantexp = new CTFEExp(TOK.cantExpression);
        CTFEExp.voidexp = new CTFEExp(TOK.voidExpression);
        CTFEExp.breakexp = new CTFEExp(TOK.break_);
        CTFEExp.continueexp = new CTFEExp(TOK.continue_);
        CTFEExp.gotoexp = new CTFEExp(TOK.goto_);
    }

    /*********************************
     * Does *not* do a deep copy.
     */
    final Expression copy()
    {
        Expression e;
        if (!size)
        {
            debug
            {
                fprintf(stderr, "No expression copy for: %s\n", toChars());
                printf("op = %d\n", op);
                print();
            }
            assert(0);
        }
        e = cast(Expression)mem.xmalloc(size);
        //printf("Expression::copy(op = %d) e = %p\n", op, e);
        return cast(Expression)memcpy(cast(void*)e, cast(void*)this, size);
    }

    Expression syntaxCopy()
    {
        //printf("Expression::syntaxCopy()\n");
        //print();
        return copy();
    }

    // kludge for template.isExpression()
    override final DYNCAST dyncast() const
    {
        return DYNCAST.expression;
    }

    override final void print()
    {
        fprintf(stderr, "%s\n", toChars());
        fflush(stderr);
    }

    override const(char)* toChars()
    {
        OutBuffer buf;
        HdrGenState hgs;
        toCBuffer(this, &buf, &hgs);
        return buf.extractString();
    }

    final void error(const(char)* format, ...) const
    {
        if (type != Type.terror)
        {
            va_list ap;
            va_start(ap, format);
            .verror(loc, format, ap);
            va_end(ap);
        }
    }

    final void warning(const(char)* format, ...) const
    {
        if (type != Type.terror)
        {
            va_list ap;
            va_start(ap, format);
            .vwarning(loc, format, ap);
            va_end(ap);
        }
    }

    final void deprecation(const(char)* format, ...) const
    {
        if (type != Type.terror)
        {
            va_list ap;
            va_start(ap, format);
            .vdeprecation(loc, format, ap);
            va_end(ap);
        }
    }

    /**********************************
     * Combine e1 and e2 by CommaExp if both are not NULL.
     */
    static Expression combine(Expression e1, Expression e2)
    {
        if (e1)
        {
            if (e2)
            {
                e1 = new CommaExp(e1.loc, e1, e2);
                e1.type = e2.type;
            }
        }
        else
            e1 = e2;
        return e1;
    }

    /**********************************
     * If 'e' is a tree of commas, returns the leftmost expression
     * by stripping off it from the tree. The remained part of the tree
     * is returned via *pe0.
     * Otherwise 'e' is directly returned and *pe0 is set to NULL.
     */
    static Expression extractLast(Expression e, Expression* pe0)
    {
        if (e.op != TOK.comma)
        {
            *pe0 = null;
            return e;
        }

        CommaExp ce = cast(CommaExp)e;
        if (ce.e2.op != TOK.comma)
        {
            *pe0 = ce.e1;
            return ce.e2;
        }
        else
        {
            *pe0 = e;

            Expression* pce = &ce.e2;
            while ((cast(CommaExp)(*pce)).e2.op == TOK.comma)
            {
                pce = &(cast(CommaExp)(*pce)).e2;
            }
            assert((*pce).op == TOK.comma);
            ce = cast(CommaExp)(*pce);
            *pce = ce.e1;

            return ce.e2;
        }
    }

    static Expressions* arraySyntaxCopy(Expressions* exps)
    {
        Expressions* a = null;
        if (exps)
        {
            a = new Expressions();
            a.setDim(exps.dim);
            for (size_t i = 0; i < a.dim; i++)
            {
                Expression e = (*exps)[i];
                (*a)[i] = e ? e.syntaxCopy() : null;
            }
        }
        return a;
    }

    dinteger_t toInteger()
    {
        //printf("Expression %s\n", Token::toChars(op));
        error("integer constant expression expected instead of `%s`", toChars());
        return 0;
    }

    uinteger_t toUInteger()
    {
        //printf("Expression %s\n", Token::toChars(op));
        return cast(uinteger_t)toInteger();
    }

    real_t toReal()
    {
        error("floating point constant expression expected instead of `%s`", toChars());
        return CTFloat.zero;
    }

    real_t toImaginary()
    {
        error("floating point constant expression expected instead of `%s`", toChars());
        return CTFloat.zero;
    }

    complex_t toComplex()
    {
        error("floating point constant expression expected instead of `%s`", toChars());
        return complex_t(CTFloat.zero);
    }

    StringExp toStringExp()
    {
        return null;
    }

    /***************************************
     * Return !=0 if expression is an lvalue.
     */
    bool isLvalue()
    {
        return false;
    }

    /*******************************
     * Give error if we're not an lvalue.
     * If we can, convert expression to be an lvalue.
     */
    Expression toLvalue(Scope* sc, Expression e)
    {
        if (!e)
            e = this;
        else if (!loc.isValid())
            loc = e.loc;

        if (e.op == TOK.type)
            error("%s `%s` is a type, not an lvalue", e.type.kind(), e.type.toChars());
        else
            error("`%s` is not an lvalue", e.toChars());

        return new ErrorExp();
    }

    Expression modifiableLvalue(Scope* sc, Expression e)
    {
        //printf("Expression::modifiableLvalue() %s, type = %s\n", toChars(), type.toChars());
        // See if this expression is a modifiable lvalue (i.e. not const)
        if (checkModifiable(sc) == 1)
        {
            assert(type);
            if (!type.isMutable())
            {
                if (op == TOK.dotVariable)
                {
                    if (isNeedThisScope(sc, (cast(DotVarExp) this).var))
                        for (Dsymbol s = sc.func; s; s = s.toParent2())
                    {
                        FuncDeclaration ff = s.isFuncDeclaration();
                        if (!ff)
                            break;
                        if (!ff.type.isMutable)
                        {
                            error("cannot modify `%s` in `%s` function", toChars(), MODtoChars(type.mod));
                            return new ErrorExp();
                        }
                    }
                }
                error("cannot modify `%s` expression `%s`", MODtoChars(type.mod), toChars());
                return new ErrorExp();
            }
            else if (!type.isAssignable())
            {
                error("cannot modify struct `%s` `%s` with `immutable` members", toChars(), type.toChars());
                return new ErrorExp();
            }
        }
        return toLvalue(sc, e);
    }

    final Expression implicitCastTo(Scope* sc, Type t)
    {
        return .implicitCastTo(this, sc, t);
    }

    final MATCH implicitConvTo(Type t)
    {
        return .implicitConvTo(this, t);
    }

    final Expression castTo(Scope* sc, Type t)
    {
        return .castTo(this, sc, t);
    }

    /****************************************
     * Resolve __FILE__, __LINE__, __MODULE__, __FUNCTION__, __PRETTY_FUNCTION__, __FILE_FULL_PATH__ to loc.
     */
    Expression resolveLoc(const ref Loc loc, Scope* sc)
    {
        this.loc = loc;
        return this;
    }

    /****************************************
     * Check that the expression has a valid type.
     * If not, generates an error "... has no type".
     * Returns:
     *      true if the expression is not valid.
     * Note:
     *      When this function returns true, `checkValue()` should also return true.
     */
    bool checkType()
    {
        return false;
    }

    /****************************************
     * Check that the expression has a valid value.
     * If not, generates an error "... has no value".
     * Returns:
     *      true if the expression is not valid or has void type.
     */
    bool checkValue()
    {
        if (type && type.toBasetype().ty == Tvoid)
        {
            error("expression `%s` is `void` and has no value", toChars());
            //print(); assert(0);
            if (!global.gag)
                type = Type.terror;
            return true;
        }
        return false;
    }

    final bool checkScalar()
    {
        if (op == TOK.error)
            return true;
        if (type.toBasetype().ty == Terror)
            return true;
        if (!type.isscalar())
        {
            error("`%s` is not a scalar, it is a `%s`", toChars(), type.toChars());
            return true;
        }
        return checkValue();
    }

    final bool checkNoBool()
    {
        if (op == TOK.error)
            return true;
        if (type.toBasetype().ty == Terror)
            return true;
        if (type.toBasetype().ty == Tbool)
        {
            error("operation not allowed on `bool` `%s`", toChars());
            return true;
        }
        return false;
    }

    final bool checkIntegral()
    {
        if (op == TOK.error)
            return true;
        if (type.toBasetype().ty == Terror)
            return true;
        if (!type.isintegral())
        {
            error("`%s` is not of integral type, it is a `%s`", toChars(), type.toChars());
            return true;
        }
        return checkValue();
    }

    final bool checkArithmetic()
    {
        if (op == TOK.error)
            return true;
        if (type.toBasetype().ty == Terror)
            return true;
        if (!type.isintegral() && !type.isfloating())
        {
            error("`%s` is not of arithmetic type, it is a `%s`", toChars(), type.toChars());
            return true;
        }
        return checkValue();
    }

    final bool checkDeprecated(Scope* sc, Dsymbol s)
    {
        return s.checkDeprecated(loc, sc);
    }

    final bool checkDisabled(Scope* sc, Dsymbol s)
    {
        if (auto d = s.isDeclaration())
        {
            return d.checkDisabled(loc, sc);
        }

        return false;
    }

    /*********************************************
     * Calling function f.
     * Check the purity, i.e. if we're in a pure function
     * we can only call other pure functions.
     * Returns true if error occurs.
     */
    final bool checkPurity(Scope* sc, FuncDeclaration f)
    {
        if (!sc.func)
            return false;
        if (sc.func == f)
            return false;
        if (sc.intypeof == 1)
            return false;
        if (sc.flags & (SCOPE.ctfe | SCOPE.debug_))
            return false;

        /* Given:
         * void f() {
         *   pure void g() {
         *     /+pure+/ void h() {
         *       /+pure+/ void i() { }
         *     }
         *   }
         * }
         * g() can call h() but not f()
         * i() can call h() and g() but not f()
         */

        // Find the closest pure parent of the calling function
        FuncDeclaration outerfunc = sc.func;
        FuncDeclaration calledparent = f;

        if (outerfunc.isInstantiated())
        {
            // The attributes of outerfunc should be inferred from the call of f.
        }
        else if (f.isInstantiated())
        {
            // The attributes of f are inferred from its body.
        }
        else if (f.isFuncLiteralDeclaration())
        {
            // The attributes of f are always inferred in its declared place.
        }
        else
        {
            /* Today, static local functions are impure by default, but they cannot
             * violate purity of enclosing functions.
             *
             *  auto foo() pure {      // non instantiated function
             *    static auto bar() {  // static, without pure attribute
             *      impureFunc();      // impure call
             *      // Although impureFunc is called inside bar, f(= impureFunc)
             *      // is not callable inside pure outerfunc(= foo <- bar).
             *    }
             *
             *    bar();
             *    // Although bar is called inside foo, f(= bar) is callable
             *    // bacause calledparent(= foo) is same with outerfunc(= foo).
             *  }
             */

            while (outerfunc.toParent2() && outerfunc.isPureBypassingInference() == PURE.impure && outerfunc.toParent2().isFuncDeclaration())
            {
                outerfunc = outerfunc.toParent2().isFuncDeclaration();
                if (outerfunc.type.ty == Terror)
                    return true;
            }
            while (calledparent.toParent2() && calledparent.isPureBypassingInference() == PURE.impure && calledparent.toParent2().isFuncDeclaration())
            {
                calledparent = calledparent.toParent2().isFuncDeclaration();
                if (calledparent.type.ty == Terror)
                    return true;
            }
        }

        // If the caller has a pure parent, then either the called func must be pure,
        // OR, they must have the same pure parent.
        if (!f.isPure() && calledparent != outerfunc)
        {
            FuncDeclaration ff = outerfunc;
            if (sc.flags & SCOPE.compile ? ff.isPureBypassingInference() >= PURE.weak : ff.setImpure())
            {
                error("`pure` %s `%s` cannot call impure %s `%s`",
                    ff.kind(), ff.toPrettyChars(), f.kind(), f.toPrettyChars());
                return true;
            }
        }
        return false;
    }

    /*******************************************
     * Accessing variable v.
     * Check for purity and safety violations.
     * Returns true if error occurs.
     */
    final bool checkPurity(Scope* sc, VarDeclaration v)
    {
        //printf("v = %s %s\n", v.type.toChars(), v.toChars());
        /* Look for purity and safety violations when accessing variable v
         * from current function.
         */
        if (!sc.func)
            return false;
        if (sc.intypeof == 1)
            return false; // allow violations inside typeof(expression)
        if (sc.flags & (SCOPE.ctfe | SCOPE.debug_))
            return false; // allow violations inside compile-time evaluated expressions and debug conditionals
        if (v.ident == Id.ctfe)
            return false; // magic variable never violates pure and safe
        if (v.isImmutable())
            return false; // always safe and pure to access immutables...
        if (v.isConst() && !v.isRef() && (v.isDataseg() || v.isParameter()) && v.type.implicitConvTo(v.type.immutableOf()))
            return false; // or const global/parameter values which have no mutable indirections
        if (v.storage_class & STC.manifest)
            return false; // ...or manifest constants

        bool err = false;
        if (v.isDataseg())
        {
            // https://issues.dlang.org/show_bug.cgi?id=7533
            // Accessing implicit generated __gate is pure.
            if (v.ident == Id.gate)
                return false;

            /* Accessing global mutable state.
             * Therefore, this function and all its immediately enclosing
             * functions must be pure.
             */
            /* Today, static local functions are impure by default, but they cannot
             * violate purity of enclosing functions.
             *
             *  auto foo() pure {      // non instantiated function
             *    static auto bar() {  // static, without pure attribute
             *      globalData++;      // impure access
             *      // Although globalData is accessed inside bar,
             *      // it is not accessible inside pure foo.
             *    }
             *  }
             */
            for (Dsymbol s = sc.func; s; s = s.toParent2())
            {
                FuncDeclaration ff = s.isFuncDeclaration();
                if (!ff)
                    break;
                if (sc.flags & SCOPE.compile ? ff.isPureBypassingInference() >= PURE.weak : ff.setImpure())
                {
                    error("`pure` %s `%s` cannot access mutable static data `%s`",
                        ff.kind(), ff.toPrettyChars(), v.toChars());
                    err = true;
                    break;
                }

                /* If the enclosing is an instantiated function or a lambda, its
                 * attribute inference result is preferred.
                 */
                if (ff.isInstantiated())
                    break;
                if (ff.isFuncLiteralDeclaration())
                    break;
            }
        }
        else
        {
            /* Given:
             * void f() {
             *   int fx;
             *   pure void g() {
             *     int gx;
             *     /+pure+/ void h() {
             *       int hx;
             *       /+pure+/ void i() { }
             *     }
             *   }
             * }
             * i() can modify hx and gx but not fx
             */

            Dsymbol vparent = v.toParent2();
            for (Dsymbol s = sc.func; !err && s; s = s.toParent2())
            {
                if (s == vparent)
                    break;

                if (AggregateDeclaration ad = s.isAggregateDeclaration())
                {
                    if (ad.isNested())
                        continue;
                    break;
                }
                FuncDeclaration ff = s.isFuncDeclaration();
                if (!ff)
                    break;
                if (ff.isNested() || ff.isThis())
                {
                    if (ff.type.isImmutable() ||
                        ff.type.isShared() && !MODimplicitConv(ff.type.mod, v.type.mod))
                    {
                        OutBuffer ffbuf;
                        OutBuffer vbuf;
                        MODMatchToBuffer(&ffbuf, ff.type.mod, v.type.mod);
                        MODMatchToBuffer(&vbuf, v.type.mod, ff.type.mod);
                        error("%s%s `%s` cannot access %sdata `%s`",
                            ffbuf.peekString(), ff.kind(), ff.toPrettyChars(), vbuf.peekString(), v.toChars());
                        err = true;
                        break;
                    }
                    continue;
                }
                break;
            }
        }

        /* Do not allow safe functions to access __gshared data
         */
        if (v.storage_class & STC.gshared)
        {
            if (sc.func.setUnsafe())
            {
                error("`@safe` %s `%s` cannot access `__gshared` data `%s`",
                    sc.func.kind(), sc.func.toChars(), v.toChars());
                err = true;
            }
        }

        return err;
    }

    /*********************************************
     * Calling function f.
     * Check the safety, i.e. if we're in a @safe function
     * we can only call @safe or @trusted functions.
     * Returns true if error occurs.
     */
    final bool checkSafety(Scope* sc, FuncDeclaration f)
    {
        if (!sc.func)
            return false;
        if (sc.func == f)
            return false;
        if (sc.intypeof == 1)
            return false;
        if (sc.flags & SCOPE.ctfe)
            return false;

        if (!f.isSafe() && !f.isTrusted())
        {
            if (sc.flags & SCOPE.compile ? sc.func.isSafeBypassingInference() : sc.func.setUnsafe())
            {
                if (!loc.isValid()) // e.g. implicitly generated dtor
                    loc = sc.func.loc;
                error("`@safe` %s `%s` cannot call `@system` %s `%s`",
                    sc.func.kind(), sc.func.toPrettyChars(), f.kind(), f.toPrettyChars());
                return true;
            }
        }
        return false;
    }

    /*********************************************
     * Calling function f.
     * Check the @nogc-ness, i.e. if we're in a @nogc function
     * we can only call other @nogc functions.
     * Returns true if error occurs.
     */
    final bool checkNogc(Scope* sc, FuncDeclaration f)
    {
        if (!sc.func)
            return false;
        if (sc.func == f)
            return false;
        if (sc.intypeof == 1)
            return false;
        if (sc.flags & SCOPE.ctfe)
            return false;

        if (!f.isNogc())
        {
            if (sc.flags & SCOPE.compile ? sc.func.isNogcBypassingInference() : sc.func.setGC() && !(sc.flags & SCOPE.debug_))
            {
                if (loc.linnum == 0) // e.g. implicitly generated dtor
                    loc = sc.func.loc;
                error("`@nogc` %s `%s` cannot call non-@nogc %s `%s`",
                    sc.func.kind(), sc.func.toPrettyChars(), f.kind(), f.toPrettyChars());
                return true;
            }
        }
        return false;
    }

    /********************************************
     * Check that the postblit is callable if t is an array of structs.
     * Returns true if error happens.
     */
    final bool checkPostblit(Scope* sc, Type t)
    {
        t = t.baseElemOf();
        if (t.ty == Tstruct)
        {
            // https://issues.dlang.org/show_bug.cgi?id=11395
            // Require TypeInfo generation for array concatenation
            semanticTypeInfo(sc, t);

            StructDeclaration sd = (cast(TypeStruct)t).sym;
            if (sd.postblit)
            {
                if (sd.postblit.checkDisabled(loc, sc))
                    return true;

                //checkDeprecated(sc, sd.postblit);        // necessary?
                checkPurity(sc, sd.postblit);
                checkSafety(sc, sd.postblit);
                checkNogc(sc, sd.postblit);
                //checkAccess(sd, loc, sc, sd.postblit);   // necessary?
                return false;
            }
        }
        return false;
    }

    final bool checkRightThis(Scope* sc)
    {
        if (op == TOK.error)
            return true;
        if (op == TOK.variable && type.ty != Terror)
        {
            VarExp ve = cast(VarExp)this;
            if (isNeedThisScope(sc, ve.var))
            {
                //printf("checkRightThis sc.intypeof = %d, ad = %p, func = %p, fdthis = %p\n",
                //        sc.intypeof, sc.getStructClassScope(), func, fdthis);
                error("need `this` for `%s` of type `%s`", ve.var.toChars(), ve.var.type.toChars());
                return true;
            }
        }
        return false;
    }

    /*******************************
     * Check whether the expression allows RMW operations, error with rmw operator diagnostic if not.
     * ex is the RHS expression, or NULL if ++/-- is used (for diagnostics)
     * Returns true if error occurs.
     */
    final bool checkReadModifyWrite(TOK rmwOp, Expression ex = null)
    {
        //printf("Expression::checkReadModifyWrite() %s %s", toChars(), ex ? ex.toChars() : "");
        if (!type || !type.isShared())
            return false;

        // atomicOp uses opAssign (+=/-=) rather than opOp (++/--) for the CT string literal.
        switch (rmwOp)
        {
        case TOK.plusPlus:
        case TOK.prePlusPlus:
            rmwOp = TOK.addAssign;
            break;
        case TOK.minusMinus:
        case TOK.preMinusMinus:
            rmwOp = TOK.minAssign;
            break;
        default:
            break;
        }

        deprecation("read-modify-write operations are not allowed for `shared` variables. Use `core.atomic.atomicOp!\"%s\"(%s, %s)` instead.", Token.toChars(rmwOp), toChars(), ex ? ex.toChars() : "1");
        return false;

        // note: enable when deprecation becomes an error.
        // return true;
    }

    /***************************************
     * Parameters:
     *      sc:     scope
     *      flag:   1: do not issue error message for invalid modification
     * Returns:
     *      0:      is not modifiable
     *      1:      is modifiable in default == being related to type.isMutable()
     *      2:      is modifiable, because this is a part of initializing.
     */
    int checkModifiable(Scope* sc, int flag = 0)
    {
        return type ? 1 : 0; // default modifiable
    }

    /*****************************
     * If expression can be tested for true or false,
     * returns the modified expression.
     * Otherwise returns ErrorExp.
     */
    Expression toBoolean(Scope* sc)
    {
        // Default is 'yes' - do nothing
        debug
        {
            if (!type)
                print();
            assert(type);
        }
        Expression e = this;
        Type t = type;
        Type tb = type.toBasetype();
        Type att = null;
    Lagain:
        // Structs can be converted to bool using opCast(bool)()
        if (tb.ty == Tstruct)
        {
            AggregateDeclaration ad = (cast(TypeStruct)tb).sym;
            /* Don't really need to check for opCast first, but by doing so we
             * get better error messages if it isn't there.
             */
            Dsymbol fd = search_function(ad, Id._cast);
            if (fd)
            {
                e = new CastExp(loc, e, Type.tbool);
                e = e.expressionSemantic(sc);
                return e;
            }

            // Forward to aliasthis.
            if (ad.aliasthis && tb != att)
            {
                if (!att && tb.checkAliasThisRec())
                    att = tb;
                e = resolveAliasThis(sc, e);
                t = e.type;
                tb = e.type.toBasetype();
                goto Lagain;
            }
        }

        if (!t.isBoolean())
        {
            if (tb != Type.terror)
                error("expression `%s` of type `%s` does not have a boolean value", toChars(), t.toChars());
            return new ErrorExp();
        }
        return e;
    }

    /************************************************
     * Destructors are attached to VarDeclarations.
     * Hence, if expression returns a temp that needs a destructor,
     * make sure and create a VarDeclaration for that temp.
     */
    Expression addDtorHook(Scope* sc)
    {
        return this;
    }

    /******************************
     * Take address of expression.
     */
    final Expression addressOf()
    {
        //printf("Expression::addressOf()\n");
        debug
        {
            assert(op == TOK.error || isLvalue());
        }
        Expression e = new AddrExp(loc, this);
        e.type = type.pointerTo();
        return e;
    }

    /******************************
     * If this is a reference, dereference it.
     */
    final Expression deref()
    {
        //printf("Expression::deref()\n");
        // type could be null if forward referencing an 'auto' variable
        if (type && type.ty == Treference)
        {
            Expression e = new PtrExp(loc, this);
            e.type = (cast(TypeReference)type).next;
            return e;
        }
        return this;
    }

    final Expression optimize(int result, bool keepLvalue = false)
    {
        return Expression_optimize(this, result, keepLvalue);
    }

    // Entry point for CTFE.
    // A compile-time result is required. Give an error if not possible
    final Expression ctfeInterpret()
    {
        return .ctfeInterpret(this);
    }

    final int isConst()
    {
        return .isConst(this);
    }

    /********************************
     * Does this expression statically evaluate to a boolean 'result' (true or false)?
     */
    bool isBool(bool result)
    {
        return false;
    }

    final Expression op_overload(Scope* sc)
    {
        return .op_overload(this, sc);
    }

    bool hasCode()
    {
        return true;
    }

    void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class IntegerExp : Expression
{
    dinteger_t value;

    extern (D) this(const ref Loc loc, dinteger_t value, Type type)
    {
        super(loc, TOK.int64, __traits(classInstanceSize, IntegerExp));
        //printf("IntegerExp(value = %lld, type = '%s')\n", value, type ? type.toChars() : "");
        assert(type);
        if (!type.isscalar())
        {
            //printf("%s, loc = %d\n", toChars(), loc.linnum);
            if (type.ty != Terror)
                error("integral constant must be scalar type, not `%s`", type.toChars());
            type = Type.terror;
        }
        this.type = type;
        setInteger(value);
    }

    extern (D) this(dinteger_t value)
    {
        super(Loc.initial, TOK.int64, __traits(classInstanceSize, IntegerExp));
        this.type = Type.tint32;
        this.value = cast(d_int32)value;
    }

    static IntegerExp create(Loc loc, dinteger_t value, Type type)
    {
        return new IntegerExp(loc, value, type);
    }

    override bool equals(RootObject o)
    {
        if (this == o)
            return true;
        if ((cast(Expression)o).op == TOK.int64)
        {
            IntegerExp ne = cast(IntegerExp)o;
            if (type.toHeadMutable().equals(ne.type.toHeadMutable()) && value == ne.value)
            {
                return true;
            }
        }
        return false;
    }

    override dinteger_t toInteger()
    {
        normalize(); // necessary until we fix all the paints of 'type'
        return value;
    }

    override real_t toReal()
    {
        normalize(); // necessary until we fix all the paints of 'type'
        Type t = type.toBasetype();
        if (t.ty == Tuns64)
            return real_t(cast(d_uns64)value);
        else
            return real_t(cast(d_int64)value);
    }

    override real_t toImaginary()
    {
        return CTFloat.zero;
    }

    override complex_t toComplex()
    {
        return complex_t(toReal());
    }

    override bool isBool(bool result)
    {
        bool r = toInteger() != 0;
        return result ? r : !r;
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        if (!e)
            e = this;
        else if (!loc.isValid())
            loc = e.loc;
        e.error("constant `%s` is not an lvalue", e.toChars());
        return new ErrorExp();
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }

    dinteger_t getInteger()
    {
        return value;
    }

    void setInteger(dinteger_t value)
    {
        this.value = value;
        normalize();
    }

    void normalize()
    {
        /* 'Normalize' the value of the integer to be in range of the type
         */
        switch (type.toBasetype().ty)
        {
        case Tbool:
            value = (value != 0);
            break;

        case Tint8:
            value = cast(d_int8)value;
            break;

        case Tchar:
        case Tuns8:
            value = cast(d_uns8)value;
            break;

        case Tint16:
            value = cast(d_int16)value;
            break;

        case Twchar:
        case Tuns16:
            value = cast(d_uns16)value;
            break;

        case Tint32:
            value = cast(d_int32)value;
            break;

        case Tdchar:
        case Tuns32:
            value = cast(d_uns32)value;
            break;

        case Tint64:
            value = cast(d_int64)value;
            break;

        case Tuns64:
            value = cast(d_uns64)value;
            break;

        case Tpointer:
            if (Target.ptrsize == 4)
                value = cast(d_uns32)value;
            else if (Target.ptrsize == 8)
                value = cast(d_uns64)value;
            else
                assert(0);
            break;

        default:
            break;
        }
    }
}

/***********************************************************
 * Use this expression for error recovery.
 * It should behave as a 'sink' to prevent further cascaded error messages.
 */
extern (C++) final class ErrorExp : Expression
{
    extern (D) this()
    {
        super(Loc.initial, TOK.error, __traits(classInstanceSize, ErrorExp));
        type = Type.terror;
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }

    extern (C++) static __gshared ErrorExp errorexp; // handy shared value
}

/***********************************************************
 */
extern (C++) final class RealExp : Expression
{
    real_t value;

    extern (D) this(const ref Loc loc, real_t value, Type type)
    {
        super(loc, TOK.float64, __traits(classInstanceSize, RealExp));
        //printf("RealExp::RealExp(%Lg)\n", value);
        this.value = value;
        this.type = type;
    }

    static RealExp create(Loc loc, real_t value, Type type)
    {
        return new RealExp(loc, value, type);
    }

    override bool equals(RootObject o)
    {
        if (this == o)
            return true;
        if ((cast(Expression)o).op == TOK.float64)
        {
            RealExp ne = cast(RealExp)o;
            if (type.toHeadMutable().equals(ne.type.toHeadMutable()) && RealEquals(value, ne.value))
            {
                return true;
            }
        }
        return false;
    }

    override dinteger_t toInteger()
    {
        return cast(sinteger_t)toReal();
    }

    override uinteger_t toUInteger()
    {
        return cast(uinteger_t)toReal();
    }

    override real_t toReal()
    {
        return type.isreal() ? value : CTFloat.zero;
    }

    override real_t toImaginary()
    {
        return type.isreal() ? CTFloat.zero : value;
    }

    override complex_t toComplex()
    {
        return complex_t(toReal(), toImaginary());
    }

    override bool isBool(bool result)
    {
        return result ? cast(bool)value : !cast(bool)value;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ComplexExp : Expression
{
    complex_t value;

    extern (D) this(const ref Loc loc, complex_t value, Type type)
    {
        super(loc, TOK.complex80, __traits(classInstanceSize, ComplexExp));
        this.value = value;
        this.type = type;
        //printf("ComplexExp::ComplexExp(%s)\n", toChars());
    }

    static ComplexExp create(Loc loc, complex_t value, Type type)
    {
        return new ComplexExp(loc, value, type);
    }

    override bool equals(RootObject o)
    {
        if (this == o)
            return true;
        if ((cast(Expression)o).op == TOK.complex80)
        {
            ComplexExp ne = cast(ComplexExp)o;
            if (type.toHeadMutable().equals(ne.type.toHeadMutable()) && RealEquals(creall(value), creall(ne.value)) && RealEquals(cimagl(value), cimagl(ne.value)))
            {
                return true;
            }
        }
        return false;
    }

    override dinteger_t toInteger()
    {
        return cast(sinteger_t)toReal();
    }

    override uinteger_t toUInteger()
    {
        return cast(uinteger_t)toReal();
    }

    override real_t toReal()
    {
        return creall(value);
    }

    override real_t toImaginary()
    {
        return cimagl(value);
    }

    override complex_t toComplex()
    {
        return value;
    }

    override bool isBool(bool result)
    {
        if (result)
            return cast(bool)value;
        else
            return !value;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) class IdentifierExp : Expression
{
    Identifier ident;

    extern (D) this(const ref Loc loc, Identifier ident)
    {
        super(loc, TOK.identifier, __traits(classInstanceSize, IdentifierExp));
        this.ident = ident;
    }

    static IdentifierExp create(Loc loc, Identifier ident)
    {
        return new IdentifierExp(loc, ident);
    }

    override final bool isLvalue()
    {
        return true;
    }

    override final Expression toLvalue(Scope* sc, Expression e)
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class DollarExp : IdentifierExp
{
    extern (D) this(const ref Loc loc)
    {
        super(loc, Id.dollar);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Won't be generated by parser.
 */
extern (C++) final class DsymbolExp : Expression
{
    Dsymbol s;
    bool hasOverloads;

    extern (D) this(const ref Loc loc, Dsymbol s, bool hasOverloads = true)
    {
        super(loc, TOK.dSymbol, __traits(classInstanceSize, DsymbolExp));
        this.s = s;
        this.hasOverloads = hasOverloads;
    }

    override bool isLvalue()
    {
        return true;
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * http://dlang.org/spec/expression.html#this
 */
extern (C++) class ThisExp : Expression
{
    VarDeclaration var;

    extern (D) this(const ref Loc loc)
    {
        super(loc, TOK.this_, __traits(classInstanceSize, ThisExp));
        //printf("ThisExp::ThisExp() loc = %d\n", loc.linnum);
    }

    override final bool isBool(bool result)
    {
        return result ? true : false;
    }

    override final bool isLvalue()
    {
        // Class `this` should be an rvalue; struct `this` should be an lvalue.
        return type.toBasetype().ty != Tclass;
    }

    override final Expression toLvalue(Scope* sc, Expression e)
    {
        if (type.toBasetype().ty == Tclass)
        {
            // Class `this` is an rvalue; struct `this` is an lvalue.
            return Expression.toLvalue(sc, e);
        }
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * http://dlang.org/spec/expression.html#super
 */
extern (C++) final class SuperExp : ThisExp
{
    extern (D) this(const ref Loc loc)
    {
        super(loc);
        op = TOK.super_;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * http://dlang.org/spec/expression.html#null
 */
extern (C++) final class NullExp : Expression
{
    ubyte committed;    // !=0 if type is committed

    extern (D) this(const ref Loc loc, Type type = null)
    {
        super(loc, TOK.null_, __traits(classInstanceSize, NullExp));
        this.type = type;
    }

    override bool equals(RootObject o)
    {
        if (o && o.dyncast() == DYNCAST.expression)
        {
            Expression e = cast(Expression)o;
            if (e.op == TOK.null_ && type.equals(e.type))
            {
                return true;
            }
        }
        return false;
    }

    override bool isBool(bool result)
    {
        return result ? false : true;
    }

    override StringExp toStringExp()
    {
        if (implicitConvTo(Type.tstring))
        {
            auto se = new StringExp(loc, cast(char*)mem.xcalloc(1, 1), 0);
            se.type = Type.tstring;
            return se;
        }
        return null;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * http://dlang.org/spec/expression.html#string_literals
 */
extern (C++) final class StringExp : Expression
{
    union
    {
        char* string;   // if sz == 1
        wchar* wstring; // if sz == 2
        dchar* dstring; // if sz == 4
    }                   // (const if ownedByCtfe == OwnedBy.code)
    size_t len;         // number of code units
    ubyte sz = 1;       // 1: char, 2: wchar, 4: dchar
    ubyte committed;    // !=0 if type is committed
    char postfix = 0;   // 'c', 'w', 'd'
    OwnedBy ownedByCtfe = OwnedBy.code;

    extern (D) this(const ref Loc loc, char* string)
    {
        super(loc, TOK.string_, __traits(classInstanceSize, StringExp));
        this.string = string;
        this.len = strlen(string);
        this.sz = 1;                    // work around LDC bug #1286
    }

    extern (D) this(const ref Loc loc, void* string, size_t len)
    {
        super(loc, TOK.string_, __traits(classInstanceSize, StringExp));
        this.string = cast(char*)string;
        this.len = len;
        this.sz = 1;                    // work around LDC bug #1286
    }

    extern (D) this(const ref Loc loc, void* string, size_t len, char postfix)
    {
        super(loc, TOK.string_, __traits(classInstanceSize, StringExp));
        this.string = cast(char*)string;
        this.len = len;
        this.postfix = postfix;
        this.sz = 1;                    // work around LDC bug #1286
    }

    static StringExp create(Loc loc, char* s)
    {
        return new StringExp(loc, s);
    }

    static StringExp create(Loc loc, void* string, size_t len)
    {
        return new StringExp(loc, string, len);
    }

    override bool equals(RootObject o)
    {
        //printf("StringExp::equals('%s') %s\n", o.toChars(), toChars());
        if (o && o.dyncast() == DYNCAST.expression)
        {
            Expression e = cast(Expression)o;
            if (e.op == TOK.string_)
            {
                return compare(o) == 0;
            }
        }
        return false;
    }

    /**********************************
     * Return the number of code units the string would be if it were re-encoded
     * as tynto.
     * Params:
     *      tynto = code unit type of the target encoding
     * Returns:
     *      number of code units
     */
    final size_t numberOfCodeUnits(int tynto = 0) const
    {
        int encSize;
        switch (tynto)
        {
            case 0:      return len;
            case Tchar:  encSize = 1; break;
            case Twchar: encSize = 2; break;
            case Tdchar: encSize = 4; break;
            default:
                assert(0);
        }
        if (sz == encSize)
            return len;

        size_t result = 0;
        dchar c;

        switch (sz)
        {
        case 1:
            for (size_t u = 0; u < len;)
            {
                if (const p = utf_decodeChar(string, len, u, c))
                {
                    error("%s", p);
                    return 0;
                }
                result += utf_codeLength(encSize, c);
            }
            break;

        case 2:
            for (size_t u = 0; u < len;)
            {
                if (const p = utf_decodeWchar(wstring, len, u, c))
                {
                    error("%s", p);
                    return 0;
                }
                result += utf_codeLength(encSize, c);
            }
            break;

        case 4:
            foreach (u; 0 .. len)
            {
                result += utf_codeLength(encSize, dstring[u]);
            }
            break;

        default:
            assert(0);
        }
        return result;
    }

    /**********************************************
     * Write the contents of the string to dest.
     * Use numberOfCodeUnits() to determine size of result.
     * Params:
     *  dest = destination
     *  tyto = encoding type of the result
     *  zero = add terminating 0
     */
    void writeTo(void* dest, bool zero, int tyto = 0) const
    {
        int encSize;
        switch (tyto)
        {
            case 0:      encSize = sz; break;
            case Tchar:  encSize = 1; break;
            case Twchar: encSize = 2; break;
            case Tdchar: encSize = 4; break;
            default:
                assert(0);
        }
        if (sz == encSize)
        {
            memcpy(dest, string, len * sz);
            if (zero)
                memset(dest + len * sz, 0, sz);
        }
        else
            assert(0);
    }

    /*********************************************
     * Get the code unit at index i
     * Params:
     *  i = index
     * Returns:
     *  code unit at index i
     */
    final dchar getCodeUnit(size_t i) const pure
    {
        assert(i < len);
        final switch (sz)
        {
        case 1:
            return string[i];
        case 2:
            return wstring[i];
        case 4:
            return dstring[i];
        }
    }

    /*********************************************
     * Set the code unit at index i to c
     * Params:
     *  i = index
     *  c = code unit to set it to
     */
    final void setCodeUnit(size_t i, dchar c)
    {
        assert(i < len);
        final switch (sz)
        {
        case 1:
            string[i] = cast(char)c;
            break;
        case 2:
            wstring[i] = cast(wchar)c;
            break;
        case 4:
            dstring[i] = c;
            break;
        }
    }

    /**************************************************
     * If the string data is UTF-8 and can be accessed directly,
     * return a pointer to it.
     * Do not assume a terminating 0.
     * Returns:
     *  pointer to string data if possible, null if not
     */
    char* toPtr()
    {
        return (sz == 1) ? string : null;
    }

    override StringExp toStringExp()
    {
        return this;
    }

    /****************************************
     * Convert string to char[].
     */
    StringExp toUTF8(Scope* sc)
    {
        if (sz != 1)
        {
            // Convert to UTF-8 string
            committed = 0;
            Expression e = castTo(sc, Type.tchar.arrayOf());
            e = e.optimize(WANTvalue);
            assert(e.op == TOK.string_);
            StringExp se = cast(StringExp)e;
            assert(se.sz == 1);
            return se;
        }
        return this;
    }

    override int compare(RootObject obj)
    {
        //printf("StringExp::compare()\n");
        // Used to sort case statement expressions so we can do an efficient lookup
        StringExp se2 = cast(StringExp)obj;

        // This is a kludge so isExpression() in template.c will return 5
        // for StringExp's.
        if (!se2)
            return 5;

        assert(se2.op == TOK.string_);

        size_t len1 = len;
        size_t len2 = se2.len;

        //printf("sz = %d, len1 = %d, len2 = %d\n", sz, (int)len1, (int)len2);
        if (len1 == len2)
        {
            switch (sz)
            {
            case 1:
                return memcmp(string, se2.string, len1);

            case 2:
                {
                    wchar* s1 = cast(wchar*)string;
                    wchar* s2 = cast(wchar*)se2.string;
                    for (size_t u = 0; u < len; u++)
                    {
                        if (s1[u] != s2[u])
                            return s1[u] - s2[u];
                    }
                }
                break;
            case 4:
                {
                    dchar* s1 = cast(dchar*)string;
                    dchar* s2 = cast(dchar*)se2.string;
                    for (size_t u = 0; u < len; u++)
                    {
                        if (s1[u] != s2[u])
                            return s1[u] - s2[u];
                    }
                }
                break;
            default:
                assert(0);
            }
        }
        return cast(int)(len1 - len2);
    }

    override bool isBool(bool result)
    {
        return result ? true : false;
    }

    override bool isLvalue()
    {
        /* string literal is rvalue in default, but
         * conversion to reference of static array is only allowed.
         */
        return (type && type.toBasetype().ty == Tsarray);
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        //printf("StringExp::toLvalue(%s) type = %s\n", toChars(), type ? type.toChars() : NULL);
        return (type && type.toBasetype().ty == Tsarray) ? this : Expression.toLvalue(sc, e);
    }

    override Expression modifiableLvalue(Scope* sc, Expression e)
    {
        error("cannot modify string literal `%s`", toChars());
        return new ErrorExp();
    }

    uint charAt(uinteger_t i) const
    {
        uint value;
        switch (sz)
        {
        case 1:
            value = (cast(char*)string)[cast(size_t)i];
            break;

        case 2:
            value = (cast(ushort*)string)[cast(size_t)i];
            break;

        case 4:
            value = (cast(uint*)string)[cast(size_t)i];
            break;

        default:
            assert(0);
        }
        return value;
    }

    /********************************
     * Convert string contents to a 0 terminated string,
     * allocated by mem.xmalloc().
     */
    extern (D) final const(char)[] toStringz() const
    {
        auto nbytes = len * sz;
        char* s = cast(char*)mem.xmalloc(nbytes + sz);
        writeTo(s, true);
        return s[0 .. nbytes];
    }

    extern (D) const(char)[] peekSlice() const
    {
        assert(sz == 1);
        return this.string[0 .. len];
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TupleExp : Expression
{
    /* Tuple-field access may need to take out its side effect part.
     * For example:
     *      foo().tupleof
     * is rewritten as:
     *      (ref __tup = foo(); tuple(__tup.field0, __tup.field1, ...))
     * The declaration of temporary variable __tup will be stored in TupleExp.e0.
     */
    Expression e0;

    Expressions* exps;

    extern (D) this(const ref Loc loc, Expression e0, Expressions* exps)
    {
        super(loc, TOK.tuple, __traits(classInstanceSize, TupleExp));
        //printf("TupleExp(this = %p)\n", this);
        this.e0 = e0;
        this.exps = exps;
    }

    extern (D) this(const ref Loc loc, Expressions* exps)
    {
        super(loc, TOK.tuple, __traits(classInstanceSize, TupleExp));
        //printf("TupleExp(this = %p)\n", this);
        this.exps = exps;
    }

    extern (D) this(const ref Loc loc, TupleDeclaration tup)
    {
        super(loc, TOK.tuple, __traits(classInstanceSize, TupleExp));
        this.exps = new Expressions();

        this.exps.reserve(tup.objects.dim);
        for (size_t i = 0; i < tup.objects.dim; i++)
        {
            RootObject o = (*tup.objects)[i];
            if (Dsymbol s = getDsymbol(o))
            {
                /* If tuple element represents a symbol, translate to DsymbolExp
                 * to supply implicit 'this' if needed later.
                 */
                Expression e = new DsymbolExp(loc, s);
                this.exps.push(e);
            }
            else if (o.dyncast() == DYNCAST.expression)
            {
                auto e = (cast(Expression)o).copy();
                e.loc = loc;    // https://issues.dlang.org/show_bug.cgi?id=15669
                this.exps.push(e);
            }
            else if (o.dyncast() == DYNCAST.type)
            {
                Type t = cast(Type)o;
                Expression e = new TypeExp(loc, t);
                this.exps.push(e);
            }
            else
            {
                error("`%s` is not an expression", o.toChars());
            }
        }
    }

    override Expression syntaxCopy()
    {
        return new TupleExp(loc, e0 ? e0.syntaxCopy() : null, arraySyntaxCopy(exps));
    }

    override bool equals(RootObject o)
    {
        if (this == o)
            return true;
        if ((cast(Expression)o).op == TOK.tuple)
        {
            TupleExp te = cast(TupleExp)o;
            if (exps.dim != te.exps.dim)
                return false;
            if (e0 && !e0.equals(te.e0) || !e0 && te.e0)
                return false;
            for (size_t i = 0; i < exps.dim; i++)
            {
                Expression e1 = (*exps)[i];
                Expression e2 = (*te.exps)[i];
                if (!e1.equals(e2))
                    return false;
            }
            return true;
        }
        return false;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * [ e1, e2, e3, ... ]
 *
 * http://dlang.org/spec/expression.html#array_literals
 */
extern (C++) final class ArrayLiteralExp : Expression
{
    /** If !is null, elements[] can be sparse and basis is used for the
     * "default" element value. In other words, non-null elements[i] overrides
     * this 'basis' value.
     */
    Expression basis;

    Expressions* elements;
    OwnedBy ownedByCtfe = OwnedBy.code;

    extern (D) this(const ref Loc loc, Expressions* elements)
    {
        super(loc, TOK.arrayLiteral, __traits(classInstanceSize, ArrayLiteralExp));
        this.elements = elements;
    }

    extern (D) this(const ref Loc loc, Expression e)
    {
        super(loc, TOK.arrayLiteral, __traits(classInstanceSize, ArrayLiteralExp));
        elements = new Expressions();
        elements.push(e);
    }

    extern (D) this(const ref Loc loc, Expression basis, Expressions* elements)
    {
        super(loc, TOK.arrayLiteral, __traits(classInstanceSize, ArrayLiteralExp));
        this.basis = basis;
        this.elements = elements;
    }

    static ArrayLiteralExp create(Loc loc, Expressions* elements)
    {
        return new ArrayLiteralExp(loc, elements);
    }

    override Expression syntaxCopy()
    {
        return new ArrayLiteralExp(loc,
            basis ? basis.syntaxCopy() : null,
            arraySyntaxCopy(elements));
    }

    override bool equals(RootObject o)
    {
        if (this == o)
            return true;
        if (o && o.dyncast() == DYNCAST.expression && (cast(Expression)o).op == TOK.arrayLiteral)
        {
            ArrayLiteralExp ae = cast(ArrayLiteralExp)o;
            if (elements.dim != ae.elements.dim)
                return false;
            if (elements.dim == 0 && !type.equals(ae.type))
            {
                return false;
            }
            for (size_t i = 0; i < elements.dim; i++)
            {
                Expression e1 = (*elements)[i];
                Expression e2 = (*ae.elements)[i];
                if (!e1)
                    e1 = basis;
                if (!e2)
                    e2 = ae.basis;
                if (e1 != e2 && (!e1 || !e2 || !e1.equals(e2)))
                    return false;
            }
            return true;
        }
        return false;
    }

    final Expression getElement(size_t i)
    {
        auto el = (*elements)[i];
        if (!el)
            el = basis;
        return el;
    }

    /** Copy element `Expressions` in the parameters when they're `ArrayLiteralExp`s.
     * Params:
     *      e1  = If it's ArrayLiteralExp, its `elements` will be copied.
     *            Otherwise, `e1` itself will be pushed into the new `Expressions`.
     *      e2  = If it's not `null`, it will be pushed/appended to the new
     *            `Expressions` by the same way with `e1`.
     * Returns:
     *      Newly allocated `Expressions`. Note that it points to the original
     *      `Expression` values in e1 and e2.
     */
    static Expressions* copyElements(Expression e1, Expression e2 = null)
    {
        auto elems = new Expressions();

        void append(ArrayLiteralExp ale)
        {
            if (!ale.elements)
                return;
            auto d = elems.dim;
            elems.append(ale.elements);
            foreach (ref el; (*elems)[][d .. elems.dim])
            {
                if (!el)
                    el = ale.basis;
            }
        }

        if (e1.op == TOK.arrayLiteral)
            append(cast(ArrayLiteralExp)e1);
        else
            elems.push(e1);

        if (e2)
        {
            if (e2.op == TOK.arrayLiteral)
                append(cast(ArrayLiteralExp)e2);
            else
                elems.push(e2);
        }

        return elems;
    }

    override bool isBool(bool result)
    {
        size_t dim = elements ? elements.dim : 0;
        return result ? (dim != 0) : (dim == 0);
    }

    override StringExp toStringExp()
    {
        TY telem = type.nextOf().toBasetype().ty;
        if (telem == Tchar || telem == Twchar || telem == Tdchar || (telem == Tvoid && (!elements || elements.dim == 0)))
        {
            ubyte sz = 1;
            if (telem == Twchar)
                sz = 2;
            else if (telem == Tdchar)
                sz = 4;

            OutBuffer buf;
            if (elements)
            {
                for (size_t i = 0; i < elements.dim; ++i)
                {
                    auto ch = getElement(i);
                    if (ch.op != TOK.int64)
                        return null;
                    if (sz == 1)
                        buf.writeByte(cast(uint)ch.toInteger());
                    else if (sz == 2)
                        buf.writeword(cast(uint)ch.toInteger());
                    else
                        buf.write4(cast(uint)ch.toInteger());
                }
            }
            char prefix;
            if (sz == 1)
            {
                prefix = 'c';
                buf.writeByte(0);
            }
            else if (sz == 2)
            {
                prefix = 'w';
                buf.writeword(0);
            }
            else
            {
                prefix = 'd';
                buf.write4(0);
            }

            const(size_t) len = buf.offset / sz - 1;
            auto se = new StringExp(loc, buf.extractData(), len, prefix);
            se.sz = sz;
            se.type = type;
            return se;
        }
        return null;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * [ key0 : value0, key1 : value1, ... ]
 *
 * http://dlang.org/spec/expression.html#associative_array_literals
 */
extern (C++) final class AssocArrayLiteralExp : Expression
{
    Expressions* keys;
    Expressions* values;

    OwnedBy ownedByCtfe = OwnedBy.code;

    extern (D) this(const ref Loc loc, Expressions* keys, Expressions* values)
    {
        super(loc, TOK.assocArrayLiteral, __traits(classInstanceSize, AssocArrayLiteralExp));
        assert(keys.dim == values.dim);
        this.keys = keys;
        this.values = values;
    }

    override bool equals(RootObject o)
    {
        if (this == o)
            return true;
        if (o && o.dyncast() == DYNCAST.expression && (cast(Expression)o).op == TOK.assocArrayLiteral)
        {
            AssocArrayLiteralExp ae = cast(AssocArrayLiteralExp)o;
            if (keys.dim != ae.keys.dim)
                return false;
            size_t count = 0;
            for (size_t i = 0; i < keys.dim; i++)
            {
                for (size_t j = 0; j < ae.keys.dim; j++)
                {
                    if ((*keys)[i].equals((*ae.keys)[j]))
                    {
                        if (!(*values)[i].equals((*ae.values)[j]))
                            return false;
                        ++count;
                    }
                }
            }
            return count == keys.dim;
        }
        return false;
    }

    override Expression syntaxCopy()
    {
        return new AssocArrayLiteralExp(loc, arraySyntaxCopy(keys), arraySyntaxCopy(values));
    }

    override bool isBool(bool result)
    {
        size_t dim = keys.dim;
        return result ? (dim != 0) : (dim == 0);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

enum stageScrub             = 0x1;  /// scrubReturnValue is running
enum stageSearchPointers    = 0x2;  /// hasNonConstPointers is running
enum stageOptimize          = 0x4;  /// optimize is running
enum stageApply             = 0x8;  /// apply is running
enum stageInlineScan        = 0x10; /// inlineScan is running
enum stageToCBuffer         = 0x20; /// toCBuffer is running

/***********************************************************
 * sd( e1, e2, e3, ... )
 */
extern (C++) final class StructLiteralExp : Expression
{
    StructDeclaration sd;   /// which aggregate this is for
    Expressions* elements;  /// parallels sd.fields[] with null entries for fields to skip
    Type stype;             /// final type of result (can be different from sd's type)

    version(IN_LLVM)
    {
        // With the introduction of pointers returned from CTFE, struct literals can
        // now contain pointers to themselves. While in toElem, contains a pointer
        // to the memory used to build the literal for resolving such references.
        void* inProgressMemory; // llvm::Value*

        // A global variable for taking the address of this struct literal constant,
        // if it already exists. Used to resolve self-references.
        void* globalVar; // llvm::Constant*
    }

    bool useStaticInit;     /// if this is true, use the StructDeclaration's init symbol
    Symbol* sym;            /// back end symbol to initialize with literal

    OwnedBy ownedByCtfe = OwnedBy.code;

    /** pointer to the origin instance of the expression.
     * once a new expression is created, origin is set to 'this'.
     * anytime when an expression copy is created, 'origin' pointer is set to
     * 'origin' pointer value of the original expression.
     */
    StructLiteralExp origin;

    /// those fields need to prevent a infinite recursion when one field of struct initialized with 'this' pointer.
    StructLiteralExp inlinecopy;

    /** anytime when recursive function is calling, 'stageflags' marks with bit flag of
     * current stage and unmarks before return from this function.
     * 'inlinecopy' uses similar 'stageflags' and from multiple evaluation 'doInline'
     * (with infinite recursion) of this expression.
     */
    int stageflags;

    extern (D) this(const ref Loc loc, StructDeclaration sd, Expressions* elements, Type stype = null)
    {
        super(loc, TOK.structLiteral, __traits(classInstanceSize, StructLiteralExp));
        this.sd = sd;
        if (!elements)
            elements = new Expressions();
        this.elements = elements;
        this.stype = stype;
        this.origin = this;
        //printf("StructLiteralExp::StructLiteralExp(%s)\n", toChars());
    }

    static StructLiteralExp create(Loc loc, StructDeclaration sd, void* elements, Type stype = null)
    {
        return new StructLiteralExp(loc, sd, cast(Expressions*)elements, stype);
    }

    override bool equals(RootObject o)
    {
        if (this == o)
            return true;
        if (o && o.dyncast() == DYNCAST.expression && (cast(Expression)o).op == TOK.structLiteral)
        {
            StructLiteralExp se = cast(StructLiteralExp)o;
            if (!type.equals(se.type))
                return false;
            if (elements.dim != se.elements.dim)
                return false;
            for (size_t i = 0; i < elements.dim; i++)
            {
                Expression e1 = (*elements)[i];
                Expression e2 = (*se.elements)[i];
                if (e1 != e2 && (!e1 || !e2 || !e1.equals(e2)))
                    return false;
            }
            return true;
        }
        return false;
    }

    override Expression syntaxCopy()
    {
        auto exp = new StructLiteralExp(loc, sd, arraySyntaxCopy(elements), type ? type : stype);
        exp.origin = this;
        return exp;
    }

    /**************************************
     * Gets expression at offset of type.
     * Returns NULL if not found.
     */
    Expression getField(Type type, uint offset)
    {
        //printf("StructLiteralExp::getField(this = %s, type = %s, offset = %u)\n",
        //  /*toChars()*/"", type.toChars(), offset);
        Expression e = null;
        int i = getFieldIndex(type, offset);

        if (i != -1)
        {
            //printf("\ti = %d\n", i);
            if (i == sd.fields.dim - 1 && sd.isNested())
                return null;

            assert(i < elements.dim);
            e = (*elements)[i];
            if (e)
            {
                //printf("e = %s, e.type = %s\n", e.toChars(), e.type.toChars());

                /* If type is a static array, and e is an initializer for that array,
                 * then the field initializer should be an array literal of e.
                 */
                if (e.type.castMod(0) != type.castMod(0) && type.ty == Tsarray)
                {
                    TypeSArray tsa = cast(TypeSArray)type;
                    size_t length = cast(size_t)tsa.dim.toInteger();
                    auto z = new Expressions();
                    z.setDim(length);
                    for (size_t q = 0; q < length; ++q)
                        (*z)[q] = e.copy();
                    e = new ArrayLiteralExp(loc, z);
                    e.type = type;
                }
                else
                {
                    e = e.copy();
                    e.type = type;
                }
                if (useStaticInit && e.op == TOK.structLiteral && e.type.needsNested())
                {
                    StructLiteralExp se = cast(StructLiteralExp)e;
                    se.useStaticInit = true;
                }
            }
        }
        return e;
    }

    /************************************
     * Get index of field.
     * Returns -1 if not found.
     */
    int getFieldIndex(Type type, uint offset)
    {
        /* Find which field offset is by looking at the field offsets
         */
        if (elements.dim)
        {
            for (size_t i = 0; i < sd.fields.dim; i++)
            {
                VarDeclaration v = sd.fields[i];
                if (offset == v.offset && type.size() == v.type.size())
                {
                    /* context field might not be filled. */
                    if (i == sd.fields.dim - 1 && sd.isNested())
                        return cast(int)i;
                    Expression e = (*elements)[i];
                    if (e)
                    {
                        return cast(int)i;
                    }
                    break;
                }
            }
        }
        return -1;
    }

    override Expression addDtorHook(Scope* sc)
    {
        /* If struct requires a destructor, rewrite as:
         *    (S tmp = S()),tmp
         * so that the destructor can be hung on tmp.
         */
        if (sd.dtor && sc.func)
        {
            /* Make an identifier for the temporary of the form:
             *   __sl%s%d, where %s is the struct name
             */
            const(size_t) len = 10;
            char[len + 1] buf;
            buf[len] = 0;
            strcpy(buf.ptr, "__sl");
            strncat(buf.ptr, sd.ident.toChars(), len - 4 - 1);
            assert(buf[len] == 0);

            auto tmp = copyToTemp(0, buf.ptr, this);
            Expression ae = new DeclarationExp(loc, tmp);
            Expression e = new CommaExp(loc, ae, new VarExp(loc, tmp));
            e = e.expressionSemantic(sc);
            return e;
        }
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Mainly just a placeholder
 */
extern (C++) final class TypeExp : Expression
{
    extern (D) this(const ref Loc loc, Type type)
    {
        super(loc, TOK.type, __traits(classInstanceSize, TypeExp));
        //printf("TypeExp::TypeExp(%s)\n", type.toChars());
        this.type = type;
    }

    override Expression syntaxCopy()
    {
        return new TypeExp(loc, type.syntaxCopy());
    }

    override bool checkType()
    {
        error("type `%s` is not an expression", toChars());
        return true;
    }

    override bool checkValue()
    {
        error("type `%s` has no value", toChars());
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Mainly just a placeholder of
 *  Package, Module, Nspace, and TemplateInstance (including TemplateMixin)
 *
 * A template instance that requires IFTI:
 *      foo!tiargs(fargs)       // foo!tiargs
 * is left until CallExp::semantic() or resolveProperties()
 */
extern (C++) final class ScopeExp : Expression
{
    ScopeDsymbol sds;

    extern (D) this(const ref Loc loc, ScopeDsymbol sds)
    {
        super(loc, TOK.scope_, __traits(classInstanceSize, ScopeExp));
        //printf("ScopeExp::ScopeExp(sds = '%s')\n", sds.toChars());
        //static int count; if (++count == 38) *(char*)0=0;
        this.sds = sds;
        assert(!sds.isTemplateDeclaration());   // instead, you should use TemplateExp
    }

    override Expression syntaxCopy()
    {
        return new ScopeExp(loc, cast(ScopeDsymbol)sds.syntaxCopy(null));
    }

    override bool checkType()
    {
        if (sds.isPackage())
        {
            error("%s `%s` has no type", sds.kind(), sds.toChars());
            return true;
        }
        if (auto ti = sds.isTemplateInstance())
        {
            //assert(ti.needsTypeInference(sc));
            if (ti.tempdecl &&
                ti.semantictiargsdone &&
                ti.semanticRun == PASS.init)
            {
                error("partial %s `%s` has no type", sds.kind(), toChars());
                return true;
            }
        }
        return false;
    }

    override bool checkValue()
    {
        error("%s `%s` has no value", sds.kind(), sds.toChars());
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Mainly just a placeholder
 */
extern (C++) final class TemplateExp : Expression
{
    TemplateDeclaration td;
    FuncDeclaration fd;

    extern (D) this(const ref Loc loc, TemplateDeclaration td, FuncDeclaration fd = null)
    {
        super(loc, TOK.template_, __traits(classInstanceSize, TemplateExp));
        //printf("TemplateExp(): %s\n", td.toChars());
        this.td = td;
        this.fd = fd;
    }

    override bool isLvalue()
    {
        return fd !is null;
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        if (!fd)
            return Expression.toLvalue(sc, e);

        assert(sc);
        return resolve(loc, sc, fd, true);
    }

    override bool checkType()
    {
        error("%s `%s` has no type", td.kind(), toChars());
        return true;
    }

    override bool checkValue()
    {
        error("%s `%s` has no value", td.kind(), toChars());
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * thisexp.new(newargs) newtype(arguments)
 */
extern (C++) final class NewExp : Expression
{
    Expression thisexp;         // if !=null, 'this' for class being allocated
    Expressions* newargs;       // Array of Expression's to call new operator
    Type newtype;
    Expressions* arguments;     // Array of Expression's

    Expression argprefix;       // expression to be evaluated just before arguments[]
    CtorDeclaration member;     // constructor function
    NewDeclaration allocator;   // allocator function
    bool onstack;               // allocate on stack
    bool thrownew;              // this NewExp is the expression of a ThrowStatement

    extern (D) this(const ref Loc loc, Expression thisexp, Expressions* newargs, Type newtype, Expressions* arguments)
    {
        super(loc, TOK.new_, __traits(classInstanceSize, NewExp));
        this.thisexp = thisexp;
        this.newargs = newargs;
        this.newtype = newtype;
        this.arguments = arguments;
    }

    static NewExp create(Loc loc, Expression thisexp, Expressions* newargs, Type newtype, Expressions* arguments)
    {
        return new NewExp(loc, thisexp, newargs, newtype, arguments);
    }

    override Expression syntaxCopy()
    {
        return new NewExp(loc,
            thisexp ? thisexp.syntaxCopy() : null,
            arraySyntaxCopy(newargs),
            newtype.syntaxCopy(),
            arraySyntaxCopy(arguments));
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * thisexp.new(newargs) class baseclasses { } (arguments)
 */
extern (C++) final class NewAnonClassExp : Expression
{
    Expression thisexp;     // if !=null, 'this' for class being allocated
    Expressions* newargs;   // Array of Expression's to call new operator
    ClassDeclaration cd;    // class being instantiated
    Expressions* arguments; // Array of Expression's to call class constructor

    extern (D) this(const ref Loc loc, Expression thisexp, Expressions* newargs, ClassDeclaration cd, Expressions* arguments)
    {
        super(loc, TOK.newAnonymousClass, __traits(classInstanceSize, NewAnonClassExp));
        this.thisexp = thisexp;
        this.newargs = newargs;
        this.cd = cd;
        this.arguments = arguments;
    }

    override Expression syntaxCopy()
    {
        return new NewAnonClassExp(loc, thisexp ? thisexp.syntaxCopy() : null, arraySyntaxCopy(newargs), cast(ClassDeclaration)cd.syntaxCopy(null), arraySyntaxCopy(arguments));
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) class SymbolExp : Expression
{
    Declaration var;
    bool hasOverloads;

    extern (D) this(const ref Loc loc, TOK op, int size, Declaration var, bool hasOverloads)
    {
        super(loc, op, size);
        assert(var);
        this.var = var;
        this.hasOverloads = hasOverloads;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Offset from symbol
 */
extern (C++) final class SymOffExp : SymbolExp
{
    dinteger_t offset;

    extern (D) this(const ref Loc loc, Declaration var, dinteger_t offset, bool hasOverloads = true)
    {
        if (auto v = var.isVarDeclaration())
        {
            // FIXME: This error report will never be handled anyone.
            // It should be done before the SymOffExp construction.
            if (v.needThis())
                .error(loc, "need `this` for address of `%s`", v.toChars());
            hasOverloads = false;
        }
        super(loc, TOK.symbolOffset, __traits(classInstanceSize, SymOffExp), var, hasOverloads);
        this.offset = offset;
    }

    override bool isBool(bool result)
    {
version(IN_LLVM)
{
        // For a weak symbol, we only statically know that it is non-null if the
        // offset is non-zero.
        if (var.llvmInternal == LDCPragma.LLVMextern_weak)
        {
            return result && offset != 0;
        }
}
        return result ? true : false;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Variable
 */
extern (C++) final class VarExp : SymbolExp
{
    extern (D) this(const ref Loc loc, Declaration var, bool hasOverloads = true)
    {
        if (var.isVarDeclaration())
            hasOverloads = false;

        super(loc, TOK.variable, __traits(classInstanceSize, VarExp), var, hasOverloads);
        //printf("VarExp(this = %p, '%s', loc = %s)\n", this, var.toChars(), loc.toChars());
        //if (strcmp(var.ident.toChars(), "func") == 0) assert(0);
        this.type = var.type;
    }

    static VarExp create(Loc loc, Declaration var, bool hasOverloads = true)
    {
        return new VarExp(loc, var, hasOverloads);
    }

    override bool equals(RootObject o)
    {
        if (this == o)
            return true;
        if ((cast(Expression)o).op == TOK.variable)
        {
            VarExp ne = cast(VarExp)o;
            if (type.toHeadMutable().equals(ne.type.toHeadMutable()) && var == ne.var)
            {
                return true;
            }
        }
        return false;
    }

    override int checkModifiable(Scope* sc, int flag)
    {
        //printf("VarExp::checkModifiable %s", toChars());
        assert(type);
        return var.checkModify(loc, sc, type, null, flag);
    }

    bool checkReadModifyWrite();

    override bool isLvalue()
    {
        if (var.storage_class & (STC.lazy_ | STC.rvalue | STC.manifest))
            return false;
        return true;
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        if (var.storage_class & STC.manifest)
        {
            error("manifest constant `%s` is not lvalue", var.toChars());
            return new ErrorExp();
        }
        if (var.storage_class & STC.lazy_)
        {
            error("lazy variables cannot be lvalues");
            return new ErrorExp();
        }
        if (var.ident == Id.ctfe)
        {
            error("compiler-generated variable `__ctfe` is not an lvalue");
            return new ErrorExp();
        }
        if (var.ident == Id.dollar) // https://issues.dlang.org/show_bug.cgi?id=13574
        {
            error("`$` is not an lvalue");
            return new ErrorExp();
        }
        return this;
    }

    override Expression modifiableLvalue(Scope* sc, Expression e)
    {
        //printf("VarExp::modifiableLvalue('%s')\n", var.toChars());
        if (var.storage_class & STC.manifest)
        {
            error("cannot modify manifest constant `%s`", toChars());
            return new ErrorExp();
        }
        // See if this expression is a modifiable lvalue (i.e. not const)
        return Expression.modifiableLvalue(sc, e);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Overload Set
 */
extern (C++) final class OverExp : Expression
{
    OverloadSet vars;

    extern (D) this(const ref Loc loc, OverloadSet s)
    {
        super(loc, TOK.overloadSet, __traits(classInstanceSize, OverExp));
        //printf("OverExp(this = %p, '%s')\n", this, var.toChars());
        vars = s;
        type = Type.tvoid;
    }

    override bool isLvalue()
    {
        return true;
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Function/Delegate literal
 */

extern (C++) final class FuncExp : Expression
{
    FuncLiteralDeclaration fd;
    TemplateDeclaration td;
    TOK tok;

    extern (D) this(const ref Loc loc, Dsymbol s)
    {
        super(loc, TOK.function_, __traits(classInstanceSize, FuncExp));
        this.td = s.isTemplateDeclaration();
        this.fd = s.isFuncLiteralDeclaration();
        if (td)
        {
            assert(td.literal);
            assert(td.members && td.members.dim == 1);
            fd = (*td.members)[0].isFuncLiteralDeclaration();
        }
        tok = fd.tok; // save original kind of function/delegate/(infer)
        assert(fd.fbody);
    }

    override bool equals(RootObject o)
    {
        if (this == o)
            return true;
        if (o.dyncast() != DYNCAST.expression)
            return false;
        if ((cast(Expression)o).op == TOK.function_)
        {
            FuncExp fe = cast(FuncExp)o;
            return fd == fe.fd;
        }
        return false;
    }

    void genIdent(Scope* sc)
    {
        if (fd.ident == Id.empty)
        {
            const(char)* s;
            if (fd.fes)
                s = "__foreachbody";
            else if (fd.tok == TOK.reserved)
            {
                import dmd.lambdacomp;

                s = "__lambda";
                auto serVisitor = new SerializeVisitor(sc);
                fd.accept(serVisitor);
                fd.serialization = serVisitor.buf.offset == 0 ? "uncomparable" : serVisitor.buf.extractString();
                //printf("serialization: %s\n", fd.serialization);
            }
            else if (fd.tok == TOK.delegate_)
                s = "__dgliteral";
            else
                s = "__funcliteral";

            DsymbolTable symtab;
            if (FuncDeclaration func = sc.parent.isFuncDeclaration())
            {
                if (func.localsymtab is null)
                {
                    // Inside template constraint, symtab is not set yet.
                    // Initialize it lazily.
                    func.localsymtab = new DsymbolTable();
                }
                symtab = func.localsymtab;
            }
            else
            {
                ScopeDsymbol sds = sc.parent.isScopeDsymbol();
                if (!sds.symtab)
                {
                    // Inside template constraint, symtab may not be set yet.
                    // Initialize it lazily.
                    assert(sds.isTemplateInstance());
                    sds.symtab = new DsymbolTable();
                }
                symtab = sds.symtab;
            }
            assert(symtab);
            Identifier id = Identifier.generateId(s, symtab.len() + 1);
            fd.ident = id;
            if (td)
                td.ident = id;
            symtab.insert(td ? cast(Dsymbol)td : cast(Dsymbol)fd);
        }
    }

    override Expression syntaxCopy()
    {
        if (td)
            return new FuncExp(loc, td.syntaxCopy(null));
        else if (fd.semanticRun == PASS.init)
            return new FuncExp(loc, fd.syntaxCopy(null));
        else // https://issues.dlang.org/show_bug.cgi?id=13481
             // Prevent multiple semantic analysis of lambda body.
            return new FuncExp(loc, fd);
    }

    MATCH matchType(Type to, Scope* sc, FuncExp* presult, int flag = 0)
    {
        //printf("FuncExp::matchType('%s'), to=%s\n", type ? type.toChars() : "null", to.toChars());
        if (presult)
            *presult = null;

        TypeFunction tof = null;
        if (to.ty == Tdelegate)
        {
            if (tok == TOK.function_)
            {
                if (!flag)
                    error("cannot match function literal to delegate type `%s`", to.toChars());
                return MATCH.nomatch;
            }
            tof = cast(TypeFunction)to.nextOf();
        }
        else if (to.ty == Tpointer && to.nextOf().ty == Tfunction)
        {
            if (tok == TOK.delegate_)
            {
                if (!flag)
                    error("cannot match delegate literal to function pointer type `%s`", to.toChars());
                return MATCH.nomatch;
            }
            tof = cast(TypeFunction)to.nextOf();
        }

        if (td)
        {
            if (!tof)
            {
            L1:
                if (!flag)
                    error("cannot infer parameter types from `%s`", to.toChars());
                return MATCH.nomatch;
            }

            // Parameter types inference from 'tof'
            assert(td._scope);
            TypeFunction tf = cast(TypeFunction)fd.type;
            //printf("\ttof = %s\n", tof.toChars());
            //printf("\ttf  = %s\n", tf.toChars());
            size_t dim = Parameter.dim(tf.parameters);

            if (Parameter.dim(tof.parameters) != dim || tof.varargs != tf.varargs)
                goto L1;

            auto tiargs = new Objects();
            tiargs.reserve(td.parameters.dim);

            for (size_t i = 0; i < td.parameters.dim; i++)
            {
                TemplateParameter tp = (*td.parameters)[i];
                size_t u = 0;
                for (; u < dim; u++)
                {
                    Parameter p = Parameter.getNth(tf.parameters, u);
                    if (p.type.ty == Tident && (cast(TypeIdentifier)p.type).ident == tp.ident)
                    {
                        break;
                    }
                }
                assert(u < dim);
                Parameter pto = Parameter.getNth(tof.parameters, u);
                Type t = pto.type;
                if (t.ty == Terror)
                    goto L1;
                tiargs.push(t);
            }

            // Set target of return type inference
            if (!tf.next && tof.next)
                fd.treq = to;

            auto ti = new TemplateInstance(loc, td, tiargs);
            Expression ex = (new ScopeExp(loc, ti)).expressionSemantic(td._scope);

            // Reset inference target for the later re-semantic
            fd.treq = null;

            if (ex.op == TOK.error)
                return MATCH.nomatch;
            if (ex.op != TOK.function_)
                goto L1;
            return (cast(FuncExp)ex).matchType(to, sc, presult, flag);
        }

        if (!tof || !tof.next)
            return MATCH.nomatch;

        assert(type && type != Type.tvoid);
        TypeFunction tfx = cast(TypeFunction)fd.type;
        bool convertMatch = (type.ty != to.ty);

        if (fd.inferRetType && tfx.next.implicitConvTo(tof.next) == MATCH.convert)
        {
            /* If return type is inferred and covariant return,
             * tweak return statements to required return type.
             *
             * interface I {}
             * class C : Object, I{}
             *
             * I delegate() dg = delegate() { return new class C(); }
             */
            convertMatch = true;

            auto tfy = new TypeFunction(tfx.parameters, tof.next, tfx.varargs, tfx.linkage, STC.undefined_);
            tfy.mod = tfx.mod;
            tfy.isnothrow = tfx.isnothrow;
            tfy.isnogc = tfx.isnogc;
            tfy.purity = tfx.purity;
            tfy.isproperty = tfx.isproperty;
            tfy.isref = tfx.isref;
            tfy.iswild = tfx.iswild;
            tfy.deco = tfy.merge().deco;

            tfx = tfy;
        }
        Type tx;
        if (tok == TOK.delegate_ || tok == TOK.reserved && (type.ty == Tdelegate || type.ty == Tpointer && to.ty == Tdelegate))
        {
            // Allow conversion from implicit function pointer to delegate
            tx = new TypeDelegate(tfx);
            tx.deco = tx.merge().deco;
        }
        else
        {
            assert(tok == TOK.function_ || tok == TOK.reserved && type.ty == Tpointer);
            tx = tfx.pointerTo();
        }
        //printf("\ttx = %s, to = %s\n", tx.toChars(), to.toChars());

        MATCH m = tx.implicitConvTo(to);
        if (m > MATCH.nomatch)
        {
            // MATCH.exact:      exact type match
            // MATCH.constant:      covairiant type match (eg. attributes difference)
            // MATCH.convert:    context conversion
            m = convertMatch ? MATCH.convert : tx.equals(to) ? MATCH.exact : MATCH.constant;

            if (presult)
            {
                (*presult) = cast(FuncExp)copy();
                (*presult).type = to;

                // https://issues.dlang.org/show_bug.cgi?id=12508
                // Tweak function body for covariant returns.
                (*presult).fd.modifyReturns(sc, tof.next);
                version(IN_LLVM)
                    (*presult).fd.type = tof; // Also, update function return type.
            }
        }
        else if (!flag)
        {
            auto ts = toAutoQualChars(tx, to);
            error("cannot implicitly convert expression `%s` of type `%s` to `%s`",
                toChars(), ts[0], ts[1]);
        }
        return m;
    }

    override const(char)* toChars()
    {
        return fd.toChars();
    }

    override bool checkType()
    {
        if (td)
        {
            error("template lambda has no type");
            return true;
        }
        return false;
    }

    override bool checkValue()
    {
        if (td)
        {
            error("template lambda has no value");
            return true;
        }
        return false;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Declaration of a symbol
 *
 * D grammar allows declarations only as statements. However in AST representation
 * it can be part of any expression. This is used, for example, during internal
 * syntax re-writes to inject hidden symbols.
 */
extern (C++) final class DeclarationExp : Expression
{
    Dsymbol declaration;

    extern (D) this(const ref Loc loc, Dsymbol declaration)
    {
        super(loc, TOK.declaration, __traits(classInstanceSize, DeclarationExp));
        this.declaration = declaration;
    }

    override Expression syntaxCopy()
    {
        return new DeclarationExp(loc, declaration.syntaxCopy(null));
    }

    override bool hasCode()
    {
        if (auto vd = declaration.isVarDeclaration())
        {
            return !(vd.storage_class & (STC.manifest | STC.static_));
        }
        return false;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * typeid(int)
 */
extern (C++) final class TypeidExp : Expression
{
    RootObject obj;

    extern (D) this(const ref Loc loc, RootObject o)
    {
        super(loc, TOK.typeid_, __traits(classInstanceSize, TypeidExp));
        this.obj = o;
    }

    override Expression syntaxCopy()
    {
        return new TypeidExp(loc, objectSyntaxCopy(obj));
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * __traits(identifier, args...)
 */
extern (C++) final class TraitsExp : Expression
{
    Identifier ident;
    Objects* args;

    extern (D) this(const ref Loc loc, Identifier ident, Objects* args)
    {
        super(loc, TOK.traits, __traits(classInstanceSize, TraitsExp));
        this.ident = ident;
        this.args = args;
    }

    override Expression syntaxCopy()
    {
        return new TraitsExp(loc, ident, TemplateInstance.arraySyntaxCopy(args));
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class HaltExp : Expression
{
    extern (D) this(const ref Loc loc)
    {
        super(loc, TOK.halt, __traits(classInstanceSize, HaltExp));
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * is(targ id tok tspec)
 * is(targ id == tok2)
 */
extern (C++) final class IsExp : Expression
{
    Type targ;
    Identifier id;      // can be null
    TOK tok;            // ':' or '=='
    Type tspec;         // can be null
    TOK tok2;           // 'struct', 'union', etc.
    TemplateParameters* parameters;

    extern (D) this(const ref Loc loc, Type targ, Identifier id, TOK tok, Type tspec, TOK tok2, TemplateParameters* parameters)
    {
        super(loc, TOK.is_, __traits(classInstanceSize, IsExp));
        this.targ = targ;
        this.id = id;
        this.tok = tok;
        this.tspec = tspec;
        this.tok2 = tok2;
        this.parameters = parameters;
    }

    override Expression syntaxCopy()
    {
        // This section is identical to that in TemplateDeclaration::syntaxCopy()
        TemplateParameters* p = null;
        if (parameters)
        {
            p = new TemplateParameters();
            p.setDim(parameters.dim);
            for (size_t i = 0; i < p.dim; i++)
                (*p)[i] = (*parameters)[i].syntaxCopy();
        }
        return new IsExp(loc, targ.syntaxCopy(), id, tok, tspec ? tspec.syntaxCopy() : null, tok2, p);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) class UnaExp : Expression
{
    Expression e1;
    Type att1;      // Save alias this type to detect recursion

    extern (D) this(const ref Loc loc, TOK op, int size, Expression e1)
    {
        super(loc, op, size);
        this.e1 = e1;
    }

    override Expression syntaxCopy()
    {
        UnaExp e = cast(UnaExp)copy();
        e.type = null;
        e.e1 = e.e1.syntaxCopy();
        return e;
    }

    /********************************
     * The type for a unary expression is incompatible.
     * Print error message.
     * Returns:
     *  ErrorExp
     */
    final Expression incompatibleTypes()
    {
        if (e1.type.toBasetype() == Type.terror)
            return e1;

        if (e1.op == TOK.type)
        {
            error("incompatible type for `%s(%s)`: cannot use `%s` with types", Token.toChars(op), e1.toChars(), Token.toChars(op));
        }
        else
        {
            error("incompatible type for `%s(%s)`: `%s`", Token.toChars(op), e1.toChars(), e1.type.toChars());
        }
        return new ErrorExp();
    }

    /*********************
     * Mark the operand as will never be dereferenced,
     * which is useful info for @safe checks.
     * Do before semantic() on operands rewrites them.
     */
    final void setNoderefOperand()
    {
        if (e1.op == TOK.dotIdentifier)
            (cast(DotIdExp)e1).noderef = true;

    }

    override final Expression resolveLoc(const ref Loc loc, Scope* sc)
    {
        e1 = e1.resolveLoc(loc, sc);
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

extern (C++) alias fp_t = UnionExp function(const ref Loc loc, Type, Expression, Expression);
extern (C++) alias fp2_t = int function(const ref Loc loc, TOK, Expression, Expression);

/***********************************************************
 */
extern (C++) abstract class BinExp : Expression
{
    Expression e1;
    Expression e2;
    Type att1;      // Save alias this type to detect recursion
    Type att2;      // Save alias this type to detect recursion

    extern (D) this(const ref Loc loc, TOK op, int size, Expression e1, Expression e2)
    {
        super(loc, op, size);
        this.e1 = e1;
        this.e2 = e2;
    }

    override Expression syntaxCopy()
    {
        BinExp e = cast(BinExp)copy();
        e.type = null;
        e.e1 = e.e1.syntaxCopy();
        e.e2 = e.e2.syntaxCopy();
        return e;
    }

    /********************************
     * The types for a binary expression are incompatible.
     * Print error message.
     * Returns:
     *  ErrorExp
     */
    final Expression incompatibleTypes()
    {
        if (e1.type.toBasetype() == Type.terror)
            return e1;
        if (e2.type.toBasetype() == Type.terror)
            return e2;

        // CondExp uses 'a ? b : c' but we're comparing 'b : c'
        TOK thisOp = (op == TOK.question) ? TOK.colon : op;
        if (e1.op == TOK.type || e2.op == TOK.type)
        {
            error("incompatible types for `(%s) %s (%s)`: cannot use `%s` with types",
                e1.toChars(), Token.toChars(thisOp), e2.toChars(), Token.toChars(op));
        }
        else if (e1.type.equals(e2.type))
        {
            error("incompatible types for `(%s) %s (%s)`: both operands are of type `%s`",
                e1.toChars(), Token.toChars(thisOp), e2.toChars(), e1.type.toChars());
        }
        else
        {
            auto ts = toAutoQualChars(e1.type, e2.type);
            error("incompatible types for `(%s) %s (%s)`: `%s` and `%s`",
                e1.toChars(), Token.toChars(thisOp), e2.toChars(), ts[0], ts[1]);
        }
        return new ErrorExp();
    }

    final Expression checkOpAssignTypes(Scope* sc)
    {
        // At that point t1 and t2 are the merged types. type is the original type of the lhs.
        Type t1 = e1.type;
        Type t2 = e2.type;

        // T opAssign floating yields a floating. Prevent truncating conversions (float to int).
        // See issue 3841.
        // Should we also prevent double to float (type.isfloating() && type.size() < t2.size()) ?
        if (op == TOK.addAssign || op == TOK.minAssign ||
            op == TOK.mulAssign || op == TOK.divAssign || op == TOK.modAssign ||
            op == TOK.powAssign)
        {
            if ((type.isintegral() && t2.isfloating()))
            {
                warning("`%s %s %s` is performing truncating conversion", type.toChars(), Token.toChars(op), t2.toChars());
            }
        }

        // generate an error if this is a nonsensical *=,/=, or %=, eg real *= imaginary
        if (op == TOK.mulAssign || op == TOK.divAssign || op == TOK.modAssign)
        {
            // Any multiplication by an imaginary or complex number yields a complex result.
            // r *= c, i*=c, r*=i, i*=i are all forbidden operations.
            const(char)* opstr = Token.toChars(op);
            if (t1.isreal() && t2.iscomplex())
            {
                error("`%s %s %s` is undefined. Did you mean `%s %s %s.re`?", t1.toChars(), opstr, t2.toChars(), t1.toChars(), opstr, t2.toChars());
                return new ErrorExp();
            }
            else if (t1.isimaginary() && t2.iscomplex())
            {
                error("`%s %s %s` is undefined. Did you mean `%s %s %s.im`?", t1.toChars(), opstr, t2.toChars(), t1.toChars(), opstr, t2.toChars());
                return new ErrorExp();
            }
            else if ((t1.isreal() || t1.isimaginary()) && t2.isimaginary())
            {
                error("`%s %s %s` is an undefined operation", t1.toChars(), opstr, t2.toChars());
                return new ErrorExp();
            }
        }

        // generate an error if this is a nonsensical += or -=, eg real += imaginary
        if (op == TOK.addAssign || op == TOK.minAssign)
        {
            // Addition or subtraction of a real and an imaginary is a complex result.
            // Thus, r+=i, r+=c, i+=r, i+=c are all forbidden operations.
            if ((t1.isreal() && (t2.isimaginary() || t2.iscomplex())) || (t1.isimaginary() && (t2.isreal() || t2.iscomplex())))
            {
                error("`%s %s %s` is undefined (result is complex)", t1.toChars(), Token.toChars(op), t2.toChars());
                return new ErrorExp();
            }
            if (type.isreal() || type.isimaginary())
            {
                assert(global.errors || t2.isfloating());
                e2 = e2.castTo(sc, t1);
            }
        }
        if (op == TOK.mulAssign)
        {
            if (t2.isfloating())
            {
                if (t1.isreal())
                {
                    if (t2.isimaginary() || t2.iscomplex())
                    {
                        e2 = e2.castTo(sc, t1);
                    }
                }
                else if (t1.isimaginary())
                {
                    if (t2.isimaginary() || t2.iscomplex())
                    {
                        switch (t1.ty)
                        {
                        case Timaginary32:
                            t2 = Type.tfloat32;
                            break;

                        case Timaginary64:
                            t2 = Type.tfloat64;
                            break;

                        case Timaginary80:
                            t2 = Type.tfloat80;
                            break;

                        default:
                            assert(0);
                        }
                        e2 = e2.castTo(sc, t2);
                    }
                }
            }
        }
        else if (op == TOK.divAssign)
        {
            if (t2.isimaginary())
            {
                if (t1.isreal())
                {
                    // x/iv = i(-x/v)
                    // Therefore, the result is 0
                    e2 = new CommaExp(loc, e2, new RealExp(loc, CTFloat.zero, t1));
                    e2.type = t1;
                    Expression e = new AssignExp(loc, e1, e2);
                    e.type = t1;
                    return e;
                }
                else if (t1.isimaginary())
                {
                    Type t3;
                    switch (t1.ty)
                    {
                    case Timaginary32:
                        t3 = Type.tfloat32;
                        break;

                    case Timaginary64:
                        t3 = Type.tfloat64;
                        break;

                    case Timaginary80:
                        t3 = Type.tfloat80;
                        break;

                    default:
                        assert(0);
                    }
                    e2 = e2.castTo(sc, t3);
                    Expression e = new AssignExp(loc, e1, e2);
                    e.type = t1;
                    return e;
                }
            }
        }
        else if (op == TOK.modAssign)
        {
            if (t2.iscomplex())
            {
                error("cannot perform modulo complex arithmetic");
                return new ErrorExp();
            }
        }
        return this;
    }

    final bool checkIntegralBin()
    {
        bool r1 = e1.checkIntegral();
        bool r2 = e2.checkIntegral();
        return (r1 || r2);
    }

    final bool checkArithmeticBin()
    {
        bool r1 = e1.checkArithmetic();
        bool r2 = e2.checkArithmetic();
        return (r1 || r2);
    }

    /*********************
     * Mark the operands as will never be dereferenced,
     * which is useful info for @safe checks.
     * Do before semantic() on operands rewrites them.
     */
    final void setNoderefOperands()
    {
        if (e1.op == TOK.dotIdentifier)
            (cast(DotIdExp)e1).noderef = true;
        if (e2.op == TOK.dotIdentifier)
            (cast(DotIdExp)e2).noderef = true;

    }

    final Expression reorderSettingAAElem(Scope* sc)
    {
        BinExp be = this;

        if (be.e1.op != TOK.index)
            return be;
        auto ie = cast(IndexExp)be.e1;
        if (ie.e1.type.toBasetype().ty != Taarray)
            return be;

        /* Fix evaluation order of setting AA element
         * https://issues.dlang.org/show_bug.cgi?id=3825
         * Rewrite:
         *     aa[k1][k2][k3] op= val;
         * as:
         *     auto ref __aatmp = aa;
         *     auto ref __aakey3 = k1, __aakey2 = k2, __aakey1 = k3;
         *     auto ref __aaval = val;
         *     __aatmp[__aakey3][__aakey2][__aakey1] op= __aaval;  // assignment
         */

        Expression e0;
        while (1)
        {
            Expression de;
            ie.e2 = extractSideEffect(sc, "__aakey", de, ie.e2);
            e0 = Expression.combine(de, e0);

            Expression ie1 = ie.e1;
            if (ie1.op != TOK.index ||
                (cast(IndexExp)ie1).e1.type.toBasetype().ty != Taarray)
            {
                break;
            }
            ie = cast(IndexExp)ie1;
        }
        assert(ie.e1.type.toBasetype().ty == Taarray);

        Expression de;
        ie.e1 = extractSideEffect(sc, "__aatmp", de, ie.e1);
        e0 = Expression.combine(de, e0);

        be.e2 = extractSideEffect(sc, "__aaval", e0, be.e2, true);

        //printf("-e0 = %s, be = %s\n", e0.toChars(), be.toChars());
        return Expression.combine(e0, be);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) class BinAssignExp : BinExp
{
    extern (D) this(const ref Loc loc, TOK op, int size, Expression e1, Expression e2)
    {
        super(loc, op, size, e1, e2);
    }

    override final bool isLvalue()
    {
        return true;
    }

    override final Expression toLvalue(Scope* sc, Expression ex)
    {
        // Lvalue-ness will be handled in glue layer.
        return this;
    }

    override final Expression modifiableLvalue(Scope* sc, Expression e)
    {
        // should check e1.checkModifiable() ?
        return toLvalue(sc, this);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class CompileExp : UnaExp
{
    extern (D) this(const ref Loc loc, Expression e)
    {
        super(loc, TOK.mixin_, __traits(classInstanceSize, CompileExp), e);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ImportExp : UnaExp
{
    extern (D) this(const ref Loc loc, Expression e)
    {
        super(loc, TOK.import_, __traits(classInstanceSize, ImportExp), e);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/expression.html#assert_expressions
 */
extern (C++) final class AssertExp : UnaExp
{
    Expression msg;

    extern (D) this(const ref Loc loc, Expression e, Expression msg = null)
    {
        super(loc, TOK.assert_, __traits(classInstanceSize, AssertExp), e);
        this.msg = msg;
    }

    override Expression syntaxCopy()
    {
        return new AssertExp(loc, e1.syntaxCopy(), msg ? msg.syntaxCopy() : null);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class DotIdExp : UnaExp
{
    Identifier ident;
    bool noderef;       // true if the result of the expression will never be dereferenced
    bool wantsym;       // do not replace Symbol with its initializer during semantic()

    extern (D) this(const ref Loc loc, Expression e, Identifier ident)
    {
        super(loc, TOK.dotIdentifier, __traits(classInstanceSize, DotIdExp), e);
        this.ident = ident;
    }

    static DotIdExp create(Loc loc, Expression e, Identifier ident)
    {
        return new DotIdExp(loc, e, ident);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Mainly just a placeholder
 */
extern (C++) final class DotTemplateExp : UnaExp
{
    TemplateDeclaration td;

    extern (D) this(const ref Loc loc, Expression e, TemplateDeclaration td)
    {
        super(loc, TOK.dotTemplateDeclaration, __traits(classInstanceSize, DotTemplateExp), e);
        this.td = td;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class DotVarExp : UnaExp
{
    Declaration var;
    bool hasOverloads;

    extern (D) this(const ref Loc loc, Expression e, Declaration var, bool hasOverloads = true)
    {
        if (var.isVarDeclaration())
            hasOverloads = false;

        super(loc, TOK.dotVariable, __traits(classInstanceSize, DotVarExp), e);
        //printf("DotVarExp()\n");
        this.var = var;
        this.hasOverloads = hasOverloads;
    }

    override int checkModifiable(Scope* sc, int flag)
    {
        //printf("DotVarExp::checkModifiable %s %s\n", toChars(), type.toChars());
        if (checkUnsafeAccess(sc, this, false, !flag))
            return 2;

        if (e1.op == TOK.this_)
            return var.checkModify(loc, sc, type, e1, flag);

        //printf("\te1 = %s\n", e1.toChars());
        return e1.checkModifiable(sc, flag);
    }

    bool checkReadModifyWrite();

    override bool isLvalue()
    {
        return true;
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        //printf("DotVarExp::toLvalue(%s)\n", toChars());
        return this;
    }

    override Expression modifiableLvalue(Scope* sc, Expression e)
    {
        version (none)
        {
            printf("DotVarExp::modifiableLvalue(%s)\n", toChars());
            printf("e1.type = %s\n", e1.type.toChars());
            printf("var.type = %s\n", var.type.toChars());
        }

        return Expression.modifiableLvalue(sc, e);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * foo.bar!(args)
 */
extern (C++) final class DotTemplateInstanceExp : UnaExp
{
    TemplateInstance ti;

    extern (D) this(const ref Loc loc, Expression e, Identifier name, Objects* tiargs)
    {
        super(loc, TOK.dotTemplateInstance, __traits(classInstanceSize, DotTemplateInstanceExp), e);
        //printf("DotTemplateInstanceExp()\n");
        this.ti = new TemplateInstance(loc, name, tiargs);
    }

    extern (D) this(const ref Loc loc, Expression e, TemplateInstance ti)
    {
        super(loc, TOK.dotTemplateInstance, __traits(classInstanceSize, DotTemplateInstanceExp), e);
        this.ti = ti;
    }

    override Expression syntaxCopy()
    {
        return new DotTemplateInstanceExp(loc, e1.syntaxCopy(), ti.name, TemplateInstance.arraySyntaxCopy(ti.tiargs));
    }

    bool findTempDecl(Scope* sc)
    {
        static if (LOGSEMANTIC)
        {
            printf("DotTemplateInstanceExp::findTempDecl('%s')\n", toChars());
        }
        if (ti.tempdecl)
            return true;

        Expression e = new DotIdExp(loc, e1, ti.name);
        e = e.expressionSemantic(sc);
        if (e.op == TOK.dot)
            e = (cast(DotExp)e).e2;

        Dsymbol s = null;
        switch (e.op)
        {
        case TOK.overloadSet:
            s = (cast(OverExp)e).vars;
            break;

        case TOK.dotTemplateDeclaration:
            s = (cast(DotTemplateExp)e).td;
            break;

        case TOK.scope_:
            s = (cast(ScopeExp)e).sds;
            break;

        case TOK.dotVariable:
            s = (cast(DotVarExp)e).var;
            break;

        case TOK.variable:
            s = (cast(VarExp)e).var;
            break;

        default:
            return false;
        }
        return ti.updateTempDecl(sc, s);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class DelegateExp : UnaExp
{
    FuncDeclaration func;
    bool hasOverloads;

    extern (D) this(const ref Loc loc, Expression e, FuncDeclaration f, bool hasOverloads = true)
    {
        super(loc, TOK.delegate_, __traits(classInstanceSize, DelegateExp), e);
        this.func = f;
        this.hasOverloads = hasOverloads;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class DotTypeExp : UnaExp
{
    Dsymbol sym;        // symbol that represents a type

    extern (D) this(const ref Loc loc, Expression e, Dsymbol s)
    {
        super(loc, TOK.dotType, __traits(classInstanceSize, DotTypeExp), e);
        this.sym = s;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class CallExp : UnaExp
{
    Expressions* arguments; // function arguments
    FuncDeclaration f;      // symbol to call
    bool directcall;        // true if a virtual call is devirtualized

    extern (D) this(const ref Loc loc, Expression e, Expressions* exps)
    {
        super(loc, TOK.call, __traits(classInstanceSize, CallExp), e);
        this.arguments = exps;
    }

    extern (D) this(const ref Loc loc, Expression e)
    {
        super(loc, TOK.call, __traits(classInstanceSize, CallExp), e);
    }

    extern (D) this(const ref Loc loc, Expression e, Expression earg1)
    {
        super(loc, TOK.call, __traits(classInstanceSize, CallExp), e);
        auto arguments = new Expressions();
        if (earg1)
        {
            arguments.setDim(1);
            (*arguments)[0] = earg1;
        }
        this.arguments = arguments;
    }

    extern (D) this(const ref Loc loc, Expression e, Expression earg1, Expression earg2)
    {
        super(loc, TOK.call, __traits(classInstanceSize, CallExp), e);
        auto arguments = new Expressions();
        arguments.setDim(2);
        (*arguments)[0] = earg1;
        (*arguments)[1] = earg2;
        this.arguments = arguments;
    }

    static CallExp create(Loc loc, Expression e, Expressions* exps)
    {
        return new CallExp(loc, e, exps);
    }

    static CallExp create(Loc loc, Expression e)
    {
        return new CallExp(loc, e);
    }

    static CallExp create(Loc loc, Expression e, Expression earg1)
    {
        return new CallExp(loc, e, earg1);
    }

    override Expression syntaxCopy()
    {
        return new CallExp(loc, e1.syntaxCopy(), arraySyntaxCopy(arguments));
    }

    override bool isLvalue()
    {
        Type tb = e1.type.toBasetype();
        if (tb.ty == Tdelegate || tb.ty == Tpointer)
            tb = tb.nextOf();
        if (tb.ty == Tfunction && (cast(TypeFunction)tb).isref)
        {
            if (e1.op == TOK.dotVariable)
                if ((cast(DotVarExp)e1).var.isCtorDeclaration())
                    return false;
            return true; // function returns a reference
        }
        return false;
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        if (isLvalue())
            return this;
        return Expression.toLvalue(sc, e);
    }

    override Expression addDtorHook(Scope* sc)
    {
        /* Only need to add dtor hook if it's a type that needs destruction.
         * Use same logic as VarDeclaration::callScopeDtor()
         */

        if (e1.type && e1.type.ty == Tfunction)
        {
            TypeFunction tf = cast(TypeFunction)e1.type;
            if (tf.isref)
                return this;
        }

        Type tv = type.baseElemOf();
        if (tv.ty == Tstruct)
        {
            TypeStruct ts = cast(TypeStruct)tv;
            StructDeclaration sd = ts.sym;
            if (sd.dtor)
            {
                /* Type needs destruction, so declare a tmp
                 * which the back end will recognize and call dtor on
                 */
                auto tmp = copyToTemp(0, "__tmpfordtor", this);
                auto de = new DeclarationExp(loc, tmp);
                auto ve = new VarExp(loc, tmp);
                Expression e = new CommaExp(loc, de, ve);
                e = e.expressionSemantic(sc);
                return e;
            }
        }
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

FuncDeclaration isFuncAddress(Expression e, bool* hasOverloads = null)
{
    if (e.op == TOK.address)
    {
        auto ae1 = (cast(AddrExp)e).e1;
        if (ae1.op == TOK.variable)
        {
            auto ve = cast(VarExp)ae1;
            if (hasOverloads)
                *hasOverloads = ve.hasOverloads;
            return ve.var.isFuncDeclaration();
        }
        if (ae1.op == TOK.dotVariable)
        {
            auto dve = cast(DotVarExp)ae1;
            if (hasOverloads)
                *hasOverloads = dve.hasOverloads;
            return dve.var.isFuncDeclaration();
        }
    }
    else
    {
        if (e.op == TOK.symbolOffset)
        {
            auto soe = cast(SymOffExp)e;
            if (hasOverloads)
                *hasOverloads = soe.hasOverloads;
            return soe.var.isFuncDeclaration();
        }
        if (e.op == TOK.delegate_)
        {
            auto dge = cast(DelegateExp)e;
            if (hasOverloads)
                *hasOverloads = dge.hasOverloads;
            return dge.func.isFuncDeclaration();
        }
    }
    return null;
}

/***********************************************************
 */
extern (C++) final class AddrExp : UnaExp
{
    extern (D) this(const ref Loc loc, Expression e)
    {
        super(loc, TOK.address, __traits(classInstanceSize, AddrExp), e);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class PtrExp : UnaExp
{
    extern (D) this(const ref Loc loc, Expression e)
    {
        super(loc, TOK.star, __traits(classInstanceSize, PtrExp), e);
        //if (e.type)
        //  type = ((TypePointer *)e.type).next;
    }

    extern (D) this(const ref Loc loc, Expression e, Type t)
    {
        super(loc, TOK.star, __traits(classInstanceSize, PtrExp), e);
        type = t;
    }

    override int checkModifiable(Scope* sc, int flag)
    {
        if (e1.op == TOK.symbolOffset)
        {
            SymOffExp se = cast(SymOffExp)e1;
            return se.var.checkModify(loc, sc, type, null, flag);
        }
        else if (e1.op == TOK.address)
        {
            AddrExp ae = cast(AddrExp)e1;
            return ae.e1.checkModifiable(sc, flag);
        }
        return 1;
    }

    override bool isLvalue()
    {
        return true;
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        return this;
    }

    override Expression modifiableLvalue(Scope* sc, Expression e)
    {
        //printf("PtrExp::modifiableLvalue() %s, type %s\n", toChars(), type.toChars());
        return Expression.modifiableLvalue(sc, e);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class NegExp : UnaExp
{
    extern (D) this(const ref Loc loc, Expression e)
    {
        super(loc, TOK.negate, __traits(classInstanceSize, NegExp), e);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class UAddExp : UnaExp
{
    extern (D) this(const ref Loc loc, Expression e)
    {
        super(loc, TOK.uadd, __traits(classInstanceSize, UAddExp), e);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ComExp : UnaExp
{
    extern (D) this(const ref Loc loc, Expression e)
    {
        super(loc, TOK.tilde, __traits(classInstanceSize, ComExp), e);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class NotExp : UnaExp
{
    extern (D) this(const ref Loc loc, Expression e)
    {
        super(loc, TOK.not, __traits(classInstanceSize, NotExp), e);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class DeleteExp : UnaExp
{
    bool isRAII;        // true if called automatically as a result of scoped destruction

    extern (D) this(const ref Loc loc, Expression e, bool isRAII)
    {
        super(loc, TOK.delete_, __traits(classInstanceSize, DeleteExp), e);
        this.isRAII = isRAII;
    }

    override Expression toBoolean(Scope* sc)
    {
        error("`delete` does not give a boolean result");
        return new ErrorExp();
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Possible to cast to one type while painting to another type
 */
extern (C++) final class CastExp : UnaExp
{
    Type to;                    // type to cast to
    ubyte mod = cast(ubyte)~0;  // MODxxxxx

    extern (D) this(const ref Loc loc, Expression e, Type t)
    {
        super(loc, TOK.cast_, __traits(classInstanceSize, CastExp), e);
        this.to = t;
    }

    /* For cast(const) and cast(immutable)
     */
    extern (D) this(const ref Loc loc, Expression e, ubyte mod)
    {
        super(loc, TOK.cast_, __traits(classInstanceSize, CastExp), e);
        this.mod = mod;
    }

    override Expression syntaxCopy()
    {
        return to ? new CastExp(loc, e1.syntaxCopy(), to.syntaxCopy()) : new CastExp(loc, e1.syntaxCopy(), mod);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class VectorExp : UnaExp
{
    TypeVector to;      // the target vector type before semantic()
    uint dim = ~0;      // number of elements in the vector

    extern (D) this(const ref Loc loc, Expression e, Type t)
    {
        super(loc, TOK.vector, __traits(classInstanceSize, VectorExp), e);
        assert(t.ty == Tvector);
        to = cast(TypeVector)t;
    }

    static VectorExp create(Loc loc, Expression e, Type t)
    {
        return new VectorExp(loc, e, t);
    }

    override Expression syntaxCopy()
    {
        return new VectorExp(loc, e1.syntaxCopy(), to.syntaxCopy());
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * e1 [lwr .. upr]
 *
 * http://dlang.org/spec/expression.html#slice_expressions
 */
extern (C++) final class SliceExp : UnaExp
{
    Expression upr;             // null if implicit 0
    Expression lwr;             // null if implicit [length - 1]

    VarDeclaration lengthVar;
    bool upperIsInBounds;       // true if upr <= e1.length
    bool lowerIsLessThanUpper;  // true if lwr <= upr
    bool arrayop;               // an array operation, rather than a slice

    /************************************************************/
    extern (D) this(const ref Loc loc, Expression e1, IntervalExp ie)
    {
        super(loc, TOK.slice, __traits(classInstanceSize, SliceExp), e1);
        this.upr = ie ? ie.upr : null;
        this.lwr = ie ? ie.lwr : null;
    }

    extern (D) this(const ref Loc loc, Expression e1, Expression lwr, Expression upr)
    {
        super(loc, TOK.slice, __traits(classInstanceSize, SliceExp), e1);
        this.upr = upr;
        this.lwr = lwr;
    }

    override Expression syntaxCopy()
    {
        auto se = new SliceExp(loc, e1.syntaxCopy(), lwr ? lwr.syntaxCopy() : null, upr ? upr.syntaxCopy() : null);
        se.lengthVar = this.lengthVar; // bug7871
        return se;
    }

    override int checkModifiable(Scope* sc, int flag)
    {
        //printf("SliceExp::checkModifiable %s\n", toChars());
        if (e1.type.ty == Tsarray || (e1.op == TOK.index && e1.type.ty != Tarray) || e1.op == TOK.slice)
        {
            return e1.checkModifiable(sc, flag);
        }
        return 1;
    }

    override bool isLvalue()
    {
        /* slice expression is rvalue in default, but
         * conversion to reference of static array is only allowed.
         */
        return (type && type.toBasetype().ty == Tsarray);
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        //printf("SliceExp::toLvalue(%s) type = %s\n", toChars(), type ? type.toChars() : NULL);
        return (type && type.toBasetype().ty == Tsarray) ? this : Expression.toLvalue(sc, e);
    }

    override Expression modifiableLvalue(Scope* sc, Expression e)
    {
        error("slice expression `%s` is not a modifiable lvalue", toChars());
        return this;
    }

    override bool isBool(bool result)
    {
        return e1.isBool(result);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ArrayLengthExp : UnaExp
{
    extern (D) this(const ref Loc loc, Expression e1)
    {
        super(loc, TOK.arrayLength, __traits(classInstanceSize, ArrayLengthExp), e1);
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
    static Expression rewriteOpAssign(BinExp exp)
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

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * e1 [ a0, a1, a2, a3 ,... ]
 *
 * http://dlang.org/spec/expression.html#index_expressions
 */
extern (C++) final class ArrayExp : UnaExp
{
    Expressions* arguments;     // Array of Expression's a0..an

    size_t currentDimension;    // for opDollar
    VarDeclaration lengthVar;

    extern (D) this(const ref Loc loc, Expression e1, Expression index = null)
    {
        super(loc, TOK.array, __traits(classInstanceSize, ArrayExp), e1);
        arguments = new Expressions();
        if (index)
            arguments.push(index);
    }

    extern (D) this(const ref Loc loc, Expression e1, Expressions* args)
    {
        super(loc, TOK.array, __traits(classInstanceSize, ArrayExp), e1);
        arguments = args;
    }

    override Expression syntaxCopy()
    {
        auto ae = new ArrayExp(loc, e1.syntaxCopy(), arraySyntaxCopy(arguments));
        ae.lengthVar = this.lengthVar; // bug7871
        return ae;
    }

    override bool isLvalue()
    {
        if (type && type.toBasetype().ty == Tvoid)
            return false;
        return true;
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        if (type && type.toBasetype().ty == Tvoid)
            error("`void`s have no value");
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class DotExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.dot, __traits(classInstanceSize, DotExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class CommaExp : BinExp
{
    /// This is needed because AssignExp rewrites CommaExp, hence it needs
    /// to trigger the deprecation.
    const bool isGenerated;

    /// Temporary variable to enable / disable deprecation of comma expression
    /// depending on the context.
    /// Since most constructor calls are rewritting, the only place where
    /// false will be passed will be from the parser.
    bool allowCommaExp;


    extern (D) this(const ref Loc loc, Expression e1, Expression e2, bool generated = true)
    {
        super(loc, TOK.comma, __traits(classInstanceSize, CommaExp), e1, e2);
        allowCommaExp = isGenerated = generated;
    }

    override int checkModifiable(Scope* sc, int flag)
    {
        return e2.checkModifiable(sc, flag);
    }

    override bool isLvalue()
    {
        return e2.isLvalue();
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        e2 = e2.toLvalue(sc, null);
        return this;
    }

    override Expression modifiableLvalue(Scope* sc, Expression e)
    {
        e2 = e2.modifiableLvalue(sc, e);
        return this;
    }

    override bool isBool(bool result)
    {
        return e2.isBool(result);
    }

    override Expression toBoolean(Scope* sc)
    {
        auto ex2 = e2.toBoolean(sc);
        if (ex2.op == TOK.error)
            return ex2;
        e2 = ex2;
        type = e2.type;
        return this;
    }

    override Expression addDtorHook(Scope* sc)
    {
        e2 = e2.addDtorHook(sc);
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }

    /**
     * If the argument is a CommaExp, set a flag to prevent deprecation messages
     *
     * It's impossible to know from CommaExp.semantic if the result will
     * be used, hence when there is a result (type != void), a deprecation
     * message is always emitted.
     * However, some construct can produce a result but won't use it
     * (ExpStatement and for loop increment).  Those should call this function
     * to prevent unwanted deprecations to be emitted.
     *
     * Params:
     *   exp = An expression that discards its result.
     *         If the argument is null or not a CommaExp, nothing happens.
     */
    static void allow(Expression exp)
    {
        if (exp && exp.op == TOK.comma)
            (cast(CommaExp)exp).allowCommaExp = true;
    }
}

/***********************************************************
 * Mainly just a placeholder
 */
extern (C++) final class IntervalExp : Expression
{
    Expression lwr;
    Expression upr;

    extern (D) this(const ref Loc loc, Expression lwr, Expression upr)
    {
        super(loc, TOK.interval, __traits(classInstanceSize, IntervalExp));
        this.lwr = lwr;
        this.upr = upr;
    }

    override Expression syntaxCopy()
    {
        return new IntervalExp(loc, lwr.syntaxCopy(), upr.syntaxCopy());
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

extern (C++) final class DelegatePtrExp : UnaExp
{
    extern (D) this(const ref Loc loc, Expression e1)
    {
        super(loc, TOK.delegatePointer, __traits(classInstanceSize, DelegatePtrExp), e1);
    }

    override bool isLvalue()
    {
        return e1.isLvalue();
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        e1 = e1.toLvalue(sc, e);
        return this;
    }

    override Expression modifiableLvalue(Scope* sc, Expression e)
    {
        if (sc.func.setUnsafe())
        {
            error("cannot modify delegate pointer in `@safe` code `%s`", toChars());
            return new ErrorExp();
        }
        return Expression.modifiableLvalue(sc, e);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class DelegateFuncptrExp : UnaExp
{
    extern (D) this(const ref Loc loc, Expression e1)
    {
        super(loc, TOK.delegateFunctionPointer, __traits(classInstanceSize, DelegateFuncptrExp), e1);
    }

    override bool isLvalue()
    {
        return e1.isLvalue();
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        e1 = e1.toLvalue(sc, e);
        return this;
    }

    override Expression modifiableLvalue(Scope* sc, Expression e)
    {
        if (sc.func.setUnsafe())
        {
            error("cannot modify delegate function pointer in `@safe` code `%s`", toChars());
            return new ErrorExp();
        }
        return Expression.modifiableLvalue(sc, e);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * e1 [ e2 ]
 */
extern (C++) final class IndexExp : BinExp
{
    VarDeclaration lengthVar;
    bool modifiable = false;    // assume it is an rvalue
    bool indexIsInBounds;       // true if 0 <= e2 && e2 <= e1.length - 1

    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.index, __traits(classInstanceSize, IndexExp), e1, e2);
        //printf("IndexExp::IndexExp('%s')\n", toChars());
    }

    override Expression syntaxCopy()
    {
        auto ie = new IndexExp(loc, e1.syntaxCopy(), e2.syntaxCopy());
        ie.lengthVar = this.lengthVar; // bug7871
        return ie;
    }

    override int checkModifiable(Scope* sc, int flag)
    {
        if (e1.type.ty == Tsarray || e1.type.ty == Taarray || (e1.op == TOK.index && e1.type.ty != Tarray) || e1.op == TOK.slice)
        {
            return e1.checkModifiable(sc, flag);
        }
        return 1;
    }

    override bool isLvalue()
    {
        return true;
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        return this;
    }

    override Expression modifiableLvalue(Scope* sc, Expression e)
    {
        //printf("IndexExp::modifiableLvalue(%s)\n", toChars());
        Expression ex = markSettingAAElem();
        if (ex.op == TOK.error)
            return ex;

        return Expression.modifiableLvalue(sc, e);
    }

    Expression markSettingAAElem()
    {
        if (e1.type.toBasetype().ty == Taarray)
        {
            Type t2b = e2.type.toBasetype();
            if (t2b.ty == Tarray && t2b.nextOf().isMutable())
            {
                error("associative arrays can only be assigned values with immutable keys, not `%s`", e2.type.toChars());
                return new ErrorExp();
            }
            modifiable = true;

            if (e1.op == TOK.index)
            {
                Expression ex = (cast(IndexExp)e1).markSettingAAElem();
                if (ex.op == TOK.error)
                    return ex;
                assert(ex == e1);
            }
        }
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * For both i++ and i--
 */
extern (C++) final class PostExp : BinExp
{
    extern (D) this(TOK op, const ref Loc loc, Expression e)
    {
        super(loc, op, __traits(classInstanceSize, PostExp), e, new IntegerExp(loc, 1, Type.tint32));
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * For both ++i and --i
 */
extern (C++) final class PreExp : UnaExp
{
    extern (D) this(TOK op, const ref Loc loc, Expression e)
    {
        super(loc, op, __traits(classInstanceSize, PreExp), e);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

enum MemorySet
{
    blockAssign     = 1,    // setting the contents of an array
    referenceInit   = 2,    // setting the reference of STC.ref_ variable
}

/***********************************************************
 */
extern (C++) class AssignExp : BinExp
{
    int memset;         // combination of MemorySet flags

    /************************************************************/
    /* op can be TOK.assign, TOK.construct, or TOK.blit */
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.assign, __traits(classInstanceSize, AssignExp), e1, e2);
    }

    override final bool isLvalue()
    {
        // Array-op 'x[] = y[]' should make an rvalue.
        // Setting array length 'x.length = v' should make an rvalue.
        if (e1.op == TOK.slice || e1.op == TOK.arrayLength)
        {
            return false;
        }
        return true;
    }

    override final Expression toLvalue(Scope* sc, Expression ex)
    {
        if (e1.op == TOK.slice || e1.op == TOK.arrayLength)
        {
            return Expression.toLvalue(sc, ex);
        }

        /* In front-end level, AssignExp should make an lvalue of e1.
         * Taking the address of e1 will be handled in low level layer,
         * so this function does nothing.
         */
        return this;
    }

    override final Expression toBoolean(Scope* sc)
    {
        // Things like:
        //  if (a = b) ...
        // are usually mistakes.

        error("assignment cannot be used as a condition, perhaps `==` was meant?");
        return new ErrorExp();
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ConstructExp : AssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, e1, e2);
        op = TOK.construct;
    }

    // Internal use only. If `v` is a reference variable, the assinment
    // will become a reference initialization automatically.
    extern (D) this(const ref Loc loc, VarDeclaration v, Expression e2)
    {
        auto ve = new VarExp(loc, v);
        assert(v.type && ve.type);

        super(loc, ve, e2);
        op = TOK.construct;

        if (v.storage_class & (STC.ref_ | STC.out_))
            memset |= MemorySet.referenceInit;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class BlitExp : AssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, e1, e2);
        op = TOK.blit;
    }

    // Internal use only. If `v` is a reference variable, the assinment
    // will become a reference rebinding automatically.
    extern (D) this(const ref Loc loc, VarDeclaration v, Expression e2)
    {
        auto ve = new VarExp(loc, v);
        assert(v.type && ve.type);

        super(loc, ve, e2);
        op = TOK.blit;

        if (v.storage_class & (STC.ref_ | STC.out_))
            memset |= MemorySet.referenceInit;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class AddAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.addAssign, __traits(classInstanceSize, AddAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class MinAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.minAssign, __traits(classInstanceSize, MinAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class MulAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.mulAssign, __traits(classInstanceSize, MulAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class DivAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.divAssign, __traits(classInstanceSize, DivAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ModAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.modAssign, __traits(classInstanceSize, ModAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class AndAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.andAssign, __traits(classInstanceSize, AndAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class OrAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.orAssign, __traits(classInstanceSize, OrAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class XorAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.xorAssign, __traits(classInstanceSize, XorAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class PowAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.powAssign, __traits(classInstanceSize, PowAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ShlAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.leftShiftAssign, __traits(classInstanceSize, ShlAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ShrAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.rightShiftAssign, __traits(classInstanceSize, ShrAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class UshrAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.unsignedRightShiftAssign, __traits(classInstanceSize, UshrAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * The ~= operator. It can have one of the following operators:
 *
 * TOK.concatenateAssign      - appending T[] to T[]
 * TOK.concatenateElemAssign  - appending T to T[]
 * TOK.concatenateDcharAssign - appending dchar to T[]
 *
 * The parser initially sets it to TOK.concatenateAssign, and semantic() later decides which
 * of the three it will be set to.
 */
extern (C++) final class CatAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.concatenateAssign, __traits(classInstanceSize, CatAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * http://dlang.org/spec/expression.html#add_expressions
 */
extern (C++) final class AddExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.add, __traits(classInstanceSize, AddExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class MinExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.min, __traits(classInstanceSize, MinExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * http://dlang.org/spec/expression.html#cat_expressions
 */
extern (C++) final class CatExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.concatenate, __traits(classInstanceSize, CatExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * http://dlang.org/spec/expression.html#mul_expressions
 */
extern (C++) final class MulExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.mul, __traits(classInstanceSize, MulExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * http://dlang.org/spec/expression.html#mul_expressions
 */
extern (C++) final class DivExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.div, __traits(classInstanceSize, DivExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * http://dlang.org/spec/expression.html#mul_expressions
 */
extern (C++) final class ModExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.mod, __traits(classInstanceSize, ModExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * http://dlang.org/spec/expression.html#pow_expressions
 */
extern (C++) final class PowExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.pow, __traits(classInstanceSize, PowExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ShlExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.leftShift, __traits(classInstanceSize, ShlExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ShrExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.rightShift, __traits(classInstanceSize, ShrExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class UshrExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.unsignedRightShift, __traits(classInstanceSize, UshrExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class AndExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.and, __traits(classInstanceSize, AndExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class OrExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.or, __traits(classInstanceSize, OrExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class XorExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.xor, __traits(classInstanceSize, XorExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * http://dlang.org/spec/expression.html#andand_expressions
 * http://dlang.org/spec/expression.html#oror_expressions
 */
extern (C++) final class LogicalExp : BinExp
{
    extern (D) this(const ref Loc loc, TOK op, Expression e1, Expression e2)
    {
        super(loc, op, __traits(classInstanceSize, LogicalExp), e1, e2);
    }

    override Expression toBoolean(Scope* sc)
    {
        auto ex2 = e2.toBoolean(sc);
        if (ex2.op == TOK.error)
            return ex2;
        e2 = ex2;
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * `op` is one of:
 *      TOK.lessThan, TOK.lessOrEqual, TOK.greaterThan, TOK.greaterOrEqual,
 *      TOK.unord, TOK.lg, TOK.leg, TOK.ule, TOK.ul, TOK.uge, TOK.ug, TOK.ue
 *
 * http://dlang.org/spec/expression.html#relation_expressions
 */
extern (C++) final class CmpExp : BinExp
{
    extern (D) this(TOK op, const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, op, __traits(classInstanceSize, CmpExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class InExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.in_, __traits(classInstanceSize, InExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * This deletes the key e1 from the associative array e2
 */
extern (C++) final class RemoveExp : BinExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.remove, __traits(classInstanceSize, RemoveExp), e1, e2);
        type = Type.tbool;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * `==` and `!=`
 *
 * TOK.equal and TOK.notEqual
 *
 * http://dlang.org/spec/expression.html#equality_expressions
 */
extern (C++) final class EqualExp : BinExp
{
    extern (D) this(TOK op, const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, op, __traits(classInstanceSize, EqualExp), e1, e2);
        assert(op == TOK.equal || op == TOK.notEqual);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * `is` and `!is`
 *
 * TOK.identity and TOK.notIdentity
 *
 *  http://dlang.org/spec/expression.html#identity_expressions
 */
extern (C++) final class IdentityExp : BinExp
{
    extern (D) this(TOK op, const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, op, __traits(classInstanceSize, IdentityExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * `econd ? e1 : e2`
 *
 * http://dlang.org/spec/expression.html#conditional_expressions
 */
extern (C++) final class CondExp : BinExp
{
    Expression econd;

    extern (D) this(const ref Loc loc, Expression econd, Expression e1, Expression e2)
    {
        super(loc, TOK.question, __traits(classInstanceSize, CondExp), e1, e2);
        this.econd = econd;
    }

    override Expression syntaxCopy()
    {
        return new CondExp(loc, econd.syntaxCopy(), e1.syntaxCopy(), e2.syntaxCopy());
    }

    override int checkModifiable(Scope* sc, int flag)
    {
        return e1.checkModifiable(sc, flag) && e2.checkModifiable(sc, flag);
    }

    override bool isLvalue()
    {
        return e1.isLvalue() && e2.isLvalue();
    }

    override Expression toLvalue(Scope* sc, Expression ex)
    {
        // convert (econd ? e1 : e2) to *(econd ? &e1 : &e2)
        CondExp e = cast(CondExp)copy();
        e.e1 = e1.toLvalue(sc, null).addressOf();
        e.e2 = e2.toLvalue(sc, null).addressOf();
        e.type = type.pointerTo();
        return new PtrExp(loc, e, type);
    }

    override Expression modifiableLvalue(Scope* sc, Expression e)
    {
        //error("conditional expression %s is not a modifiable lvalue", toChars());
        e1 = e1.modifiableLvalue(sc, e1);
        e2 = e2.modifiableLvalue(sc, e2);
        return toLvalue(sc, this);
    }

    override Expression toBoolean(Scope* sc)
    {
        auto ex1 = e1.toBoolean(sc);
        auto ex2 = e2.toBoolean(sc);
        if (ex1.op == TOK.error)
            return ex1;
        if (ex2.op == TOK.error)
            return ex2;
        e1 = ex1;
        e2 = ex2;
        return this;
    }

    void hookDtors(Scope* sc)
    {
        extern (C++) final class DtorVisitor : StoppableVisitor
        {
            alias visit = super.visit;
        public:
            Scope* sc;
            CondExp ce;
            VarDeclaration vcond;
            bool isThen;

            extern (D) this(Scope* sc, CondExp ce)
            {
                this.sc = sc;
                this.ce = ce;
            }

            override void visit(Expression e)
            {
                //printf("(e = %s)\n", e.toChars());
            }

            override void visit(DeclarationExp e)
            {
                auto v = e.declaration.isVarDeclaration();
                if (v && !v.isDataseg())
                {
                    if (v._init)
                    {
                        if (auto ei = v._init.isExpInitializer())
                            ei.exp.accept(this);
                    }

                    if (v.needsScopeDtor())
                    {
                        if (!vcond)
                        {
                            vcond = copyToTemp(STC.volatile_, "__cond", ce.econd);
                            vcond.dsymbolSemantic(sc);

                            Expression de = new DeclarationExp(ce.econd.loc, vcond);
                            de = de.expressionSemantic(sc);

                            Expression ve = new VarExp(ce.econd.loc, vcond);
                            ce.econd = Expression.combine(de, ve);
                        }

                        //printf("\t++v = %s, v.edtor = %s\n", v.toChars(), v.edtor.toChars());
                        Expression ve = new VarExp(vcond.loc, vcond);
                        if (isThen)
                            v.edtor = new LogicalExp(v.edtor.loc, TOK.andAnd, ve, v.edtor);
                        else
                            v.edtor = new LogicalExp(v.edtor.loc, TOK.orOr, ve, v.edtor);
                        v.edtor = v.edtor.expressionSemantic(sc);
                        //printf("\t--v = %s, v.edtor = %s\n", v.toChars(), v.edtor.toChars());
                    }
                }
            }
        }

        scope DtorVisitor v = new DtorVisitor(sc, this);
        //printf("+%s\n", toChars());
        v.isThen = true;
        walkPostorder(e1, v);
        v.isThen = false;
        walkPostorder(e2, v);
        //printf("-%s\n", toChars());
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) class DefaultInitExp : Expression
{
    TOK subop;      // which of the derived classes this is

    extern (D) this(const ref Loc loc, TOK subop, int size)
    {
        super(loc, TOK.default_, size);
        this.subop = subop;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class FileInitExp : DefaultInitExp
{
    extern (D) this(const ref Loc loc, TOK tok)
    {
        super(loc, tok, __traits(classInstanceSize, FileInitExp));
    }

    override Expression resolveLoc(const ref Loc loc, Scope* sc)
    {
        //printf("FileInitExp::resolve() %s\n", toChars());
        const(char)* s;
        if (subop == TOK.fileFullPath)
            s = FileName.toAbsolute(loc.isValid() ? loc.filename : sc._module.srcfile.name.toChars());
        else
            s = loc.isValid() ? loc.filename : sc._module.ident.toChars();

        Expression e = new StringExp(loc, cast(char*)s);
        e = e.expressionSemantic(sc);
        e = e.castTo(sc, type);
        return e;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class LineInitExp : DefaultInitExp
{
    extern (D) this(const ref Loc loc)
    {
        super(loc, TOK.line, __traits(classInstanceSize, LineInitExp));
    }

    override Expression resolveLoc(const ref Loc loc, Scope* sc)
    {
        Expression e = new IntegerExp(loc, loc.linnum, Type.tint32);
        e = e.castTo(sc, type);
        return e;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ModuleInitExp : DefaultInitExp
{
    extern (D) this(const ref Loc loc)
    {
        super(loc, TOK.moduleString, __traits(classInstanceSize, ModuleInitExp));
    }

    override Expression resolveLoc(const ref Loc loc, Scope* sc)
    {
        const(char)* s;
        if (sc.callsc)
            s = sc.callsc._module.toPrettyChars();
        else
            s = sc._module.toPrettyChars();
        Expression e = new StringExp(loc, cast(char*)s);
        e = e.expressionSemantic(sc);
        e = e.castTo(sc, type);
        return e;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class FuncInitExp : DefaultInitExp
{
    extern (D) this(const ref Loc loc)
    {
        super(loc, TOK.functionString, __traits(classInstanceSize, FuncInitExp));
    }

    override Expression resolveLoc(const ref Loc loc, Scope* sc)
    {
        const(char)* s;
        if (sc.callsc && sc.callsc.func)
            s = sc.callsc.func.Dsymbol.toPrettyChars();
        else if (sc.func)
            s = sc.func.Dsymbol.toPrettyChars();
        else
            s = "";
        Expression e = new StringExp(loc, cast(char*)s);
        e = e.expressionSemantic(sc);
        e = e.castTo(sc, type);
        return e;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class PrettyFuncInitExp : DefaultInitExp
{
    extern (D) this(const ref Loc loc)
    {
        super(loc, TOK.prettyFunction, __traits(classInstanceSize, PrettyFuncInitExp));
    }

    override Expression resolveLoc(const ref Loc loc, Scope* sc)
    {
        FuncDeclaration fd;
        if (sc.callsc && sc.callsc.func)
            fd = sc.callsc.func;
        else
            fd = sc.func;

        const(char)* s;
        if (fd)
        {
            const(char)* funcStr = fd.Dsymbol.toPrettyChars();
            OutBuffer buf;
            functionToBufferWithIdent(cast(TypeFunction)fd.type, &buf, funcStr);
            s = buf.extractString();
        }
        else
        {
            s = "";
        }

        Expression e = new StringExp(loc, cast(char*)s);
        e = e.expressionSemantic(sc);
        e = e.castTo(sc, type);
        return e;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}
