/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/staticassert.d, _staticassert.d)
 * Documentation:  https://dlang.org/phobos/dmd_staticassert.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/staticassert.d
 */

module dmd.staticassert;

import dmd.dscope;
import dmd.dsymbol;
import dmd.expression;
import dmd.globals;
import dmd.id;
import dmd.identifier;
import dmd.mtype;
import dmd.visitor;

/***********************************************************
 */
extern (C++) final class StaticAssert : Dsymbol
{
    Expression exp;
    Expression msg;

    extern (D) this(const ref Loc loc, Expression exp, Expression msg)
    {
        super(Id.empty);
        this.loc = loc;
        this.exp = exp;
        this.msg = msg;
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        return new StaticAssert(loc, exp.syntaxCopy(), msg ? msg.syntaxCopy() : null);
    }

    override void addMember(Scope* sc, ScopeDsymbol sds)
    {
        // we didn't add anything
    }

    override bool oneMember(Dsymbol* ps, Identifier ident)
    {
        //printf("StaticAssert::oneMember())\n");
        *ps = null;
        return true;
    }

    override const(char)* kind() const
    {
        return "static assert";
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}
