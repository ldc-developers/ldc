/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/aliasthis.d, _aliasthis.d)
 * Documentation:  https://dlang.org/phobos/dmd_aliasthis.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/aliasthis.d
 */

module dmd.aliasthis;

import core.stdc.stdio;
import dmd.aggregate;
import dmd.dscope;
import dmd.dsymbol;
import dmd.expression;
import dmd.expressionsem;
import dmd.globals;
import dmd.identifier;
import dmd.mtype;
import dmd.opover;
import dmd.tokens;
import dmd.visitor;

/***********************************************************
 * alias ident this;
 */
extern (C++) final class AliasThis : Dsymbol
{
    Identifier ident;
    /// The symbol this `alias this` resolves to
    Dsymbol sym;
    /// Whether this `alias this` is deprecated or not
    bool isDeprecated_;

    extern (D) this(const ref Loc loc, Identifier ident)
    {
        super(loc, null);    // it's anonymous (no identifier)
        this.ident = ident;
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto at = new AliasThis(loc, ident);
        at.comment = comment;
        return at;
    }

    override const(char)* kind() const
    {
        return "alias this";
    }

    AliasThis isAliasThis()
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }

    override bool isDeprecated() const
    {
        return this.isDeprecated_;
    }
}

Expression resolveAliasThis(Scope* sc, Expression e, bool gag = false)
{
    for (AggregateDeclaration ad = isAggregate(e.type); ad;)
    {
        if (ad.aliasthis)
        {
            uint olderrors = gag ? global.startGagging() : 0;
            Loc loc = e.loc;
            Type tthis = (e.op == TOK.type ? e.type : null);
            e = new DotIdExp(loc, e, ad.aliasthis.ident);
            e = e.expressionSemantic(sc);
            if (tthis && ad.aliasthis.sym.needThis())
            {
                if (e.op == TOK.variable)
                {
                    if (auto fd = (cast(VarExp)e).var.isFuncDeclaration())
                    {
                        // https://issues.dlang.org/show_bug.cgi?id=13009
                        // Support better match for the overloaded alias this.
                        bool hasOverloads;
                        if (auto f = fd.overloadModMatch(loc, tthis, hasOverloads))
                        {
                            if (!hasOverloads)
                                fd = f;     // use exact match
                            e = new VarExp(loc, fd, hasOverloads);
                            e.type = f.type;
                            e = new CallExp(loc, e);
                            goto L1;
                        }
                    }
                }
                /* non-@property function is not called inside typeof(),
                 * so resolve it ahead.
                 */
                {
                    int save = sc.intypeof;
                    sc.intypeof = 1; // bypass "need this" error check
                    e = resolveProperties(sc, e);
                    sc.intypeof = save;
                }
            L1:
                e = new TypeExp(loc, new TypeTypeof(loc, e));
                e = e.expressionSemantic(sc);
            }
            e = resolveProperties(sc, e);
            if (!gag)
                ad.aliasthis.checkDeprecatedAliasThis(loc, sc);
            else if (global.endGagging(olderrors))
                e = null;
        }

        import dmd.dclass : ClassDeclaration;
        auto cd = ad.isClassDeclaration();
        if ((!e || !ad.aliasthis) && cd && cd.baseClass && cd.baseClass != ClassDeclaration.object)
        {
            ad = cd.baseClass;
            continue;
        }
        break;
    }
    return e;
}

/**
 * Check if an `alias this` is deprecated
 *
 * Usually one would use `expression.checkDeprecated(scope, aliasthis)` to
 * check if `expression` uses a deprecated `aliasthis`, but this calls
 * `toPrettyChars` which lead to the following message:
 * "Deprecation: alias this `fullyqualified.aggregate.__anonymous` is deprecated"
 *
 * Params:
 *   at  = The `AliasThis` object to check
 *   loc = `Loc` of the expression triggering the access to `at`
 *   sc  = `Scope` of the expression
 *         (deprecations do not trigger in deprecated scopes)
 *
 * Returns:
 *   Whether the alias this was reported as deprecated.
 */
bool checkDeprecatedAliasThis(AliasThis at, const ref Loc loc, Scope* sc)
{
    import dmd.errors : deprecation;
    import dmd.dsymbolsem : getMessage;

    if (global.params.useDeprecated != DiagnosticReporting.off
        && at.isDeprecated() && !sc.isDeprecated())
    {
            const(char)* message = null;
            for (Dsymbol p = at; p; p = p.parent)
            {
                message = p.depdecl ? p.depdecl.getMessage() : null;
                if (message)
                    break;
            }
            if (message)
                deprecation(loc, "`alias %s this` is deprecated - %s",
                            at.sym.toChars(), message);
            else
                deprecation(loc, "`alias %s this` is deprecated",
                            at.sym.toChars());
        return true;
    }
    return false;
}
