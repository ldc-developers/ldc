/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (c) 1999-2017 by Digital Mars, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(DMDSRC _staticassert.d)
 */

module ddmd.staticassert;

import ddmd.dscope;
import ddmd.dsymbol;
import ddmd.errors;
import ddmd.expression;
import ddmd.globals;
import ddmd.id;
import ddmd.identifier;
import ddmd.mtype;
import ddmd.visitor;

/***********************************************************
 */
extern (C++) final class StaticAssert : Dsymbol
{
    Expression exp;
    Expression msg;

    extern (D) this(Loc loc, Expression exp, Expression msg)
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

    override void semantic(Scope* sc)
    {
    }

    override void semantic2(Scope* sc)
    {
        //printf("StaticAssert::semantic2() %s\n", toChars());
        auto sds = new ScopeDsymbol();
        sc = sc.push(sds);
        sc.tinst = null;
        sc.minst = null;
        sc.flags |= SCOPEcondition;
        sc = sc.startCTFE();
        Expression e = exp.semantic(sc);
        e = resolveProperties(sc, e);
        sc = sc.endCTFE();
        sc = sc.pop();
        // Simplify expression, to make error messages nicer if CTFE fails
        e = e.optimize(WANTvalue);
        if (!e.type.isBoolean())
        {
            if (e.type.toBasetype() != Type.terror)
                exp.error("expression %s of type %s does not have a boolean value", exp.toChars(), e.type.toChars());
            return;
        }
        uint olderrs = global.errors;
        e = e.ctfeInterpret();
        if (global.errors != olderrs)
        {
            errorSupplemental(loc, "while evaluating: static assert(%s)", exp.toChars());
        }
        else if (e.isBool(false))
        {
            if (msg)
            {
                sc = sc.startCTFE();
                msg = msg.semantic(sc);
                msg = resolveProperties(sc, msg);
                sc = sc.endCTFE();
                msg = msg.ctfeInterpret();
                if (StringExp se = msg.toStringExp())
                {
                    // same with pragma(msg)
                    se = se.toUTF8(sc);
                    error("\"%.*s\"", cast(int)se.len, se.string);
                }
                else
                    error("%s", msg.toChars());
            }
            else
                error("(%s) is false", exp.toChars());
            if (sc.tinst)
                sc.tinst.printInstantiationTrace();
            if (!global.gag)
                fatal();
        }
        else if (!e.isBool(true))
        {
            error("(%s) is not evaluatable at compile time", exp.toChars());
        }
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
