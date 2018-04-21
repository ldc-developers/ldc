/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/delegatize.d, _delegatize.d)
 * Documentation:  https://dlang.org/phobos/dmd_delegatize.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/delegatize.d
 */

module dmd.delegatize;

import core.stdc.stdio;
import dmd.apply;
import dmd.declaration;
import dmd.dscope;
import dmd.dsymbol;
import dmd.expression;
import dmd.expressionsem;
import dmd.func;
import dmd.globals;
import dmd.initsem;
import dmd.mtype;
import dmd.statement;
import dmd.tokens;
import dmd.visitor;


/*********************************
 * Convert expression into a delegate.
 *
 * Used to convert the argument to a lazy parameter.
 *
 * Params:
 *  e = argument to convert to a delegate
 *  t = the type to be returned by the delegate
 *  sc = context
 * Returns:
 *  A delegate literal
 */
extern (C++) Expression toDelegate(Expression e, Type t, Scope* sc)
{
    //printf("Expression::toDelegate(t = %s) %s\n", t.toChars(), e.toChars());
    Loc loc = e.loc;
    auto tf = new TypeFunction(null, t, 0, LINK.d);
    if (t.hasWild())
        tf.mod = MODFlags.wild;
    auto fld = new FuncLiteralDeclaration(loc, loc, tf, TOK.delegate_, null);
    lambdaSetParent(e, fld);

    sc = sc.push();
    sc.parent = fld; // set current function to be the delegate
    bool r = lambdaCheckForNestedRef(e, sc);
    sc = sc.pop();
    if (r)
        return new ErrorExp();

    Statement s;
    if (t.ty == Tvoid)
        s = new ExpStatement(loc, e);
    else
        s = new ReturnStatement(loc, e);
    fld.fbody = s;
    e = new FuncExp(loc, fld);
    e = e.expressionSemantic(sc);
    return e;
}

/******************************************
 * Patch the parent of declarations to be the new function literal.
 *
 * Since the expression is going to be moved into a function literal,
 * the parent for declarations in the expression needs to be
 * reset to that function literal.
 * Params:
 *   e = expression to check
 *   fd = function literal symbol (the new parent)
 */
private void lambdaSetParent(Expression e, FuncDeclaration fd)
{
    extern (C++) final class LambdaSetParent : StoppableVisitor
    {
        alias visit = super.visit;
        FuncDeclaration fd;

    public:
        extern (D) this(FuncDeclaration fd)
        {
            this.fd = fd;
        }

        override void visit(Expression)
        {
        }

        override void visit(DeclarationExp e)
        {
            e.declaration.parent = fd;
        }

        override void visit(IndexExp e)
        {
            if (e.lengthVar)
            {
                //printf("lengthVar\n");
                e.lengthVar.parent = fd;
            }
        }

        override void visit(SliceExp e)
        {
            if (e.lengthVar)
            {
                //printf("lengthVar\n");
                e.lengthVar.parent = fd;
            }
        }
    }

    scope LambdaSetParent lsp = new LambdaSetParent(fd);
    walkPostorder(e, lsp);
}

/*******************************************
 * Look for references to variables in a scope enclosing the new function literal.
 *
 * Essentially just calls `checkNestedReference() for each variable reference in `e`.
 * Params:
 *      sc = context
 *      e = expression to check
 * Returns:
 *      true if error occurs.
 */
extern (C++) bool lambdaCheckForNestedRef(Expression e, Scope* sc)
{
    extern (C++) final class LambdaCheckForNestedRef : StoppableVisitor
    {
        alias visit = super.visit;
    public:
        Scope* sc;
        bool result;

        extern (D) this(Scope* sc)
        {
            this.sc = sc;
        }

        override void visit(Expression)
        {
        }

        override void visit(SymOffExp e)
        {
            VarDeclaration v = e.var.isVarDeclaration();
            if (v)
                result = v.checkNestedReference(sc, Loc.initial);
        }

        override void visit(VarExp e)
        {
            VarDeclaration v = e.var.isVarDeclaration();
            if (v)
                result = v.checkNestedReference(sc, Loc.initial);
        }

        override void visit(ThisExp e)
        {
            if (e.var)
                result = e.var.checkNestedReference(sc, Loc.initial);
        }

        override void visit(DeclarationExp e)
        {
            VarDeclaration v = e.declaration.isVarDeclaration();
            if (v)
            {
                result = v.checkNestedReference(sc, Loc.initial);
                if (result)
                    return;
                /* Some expressions cause the frontend to create a temporary.
                 * For example, structs with cpctors replace the original
                 * expression e with:
                 *  __cpcttmp = __cpcttmp.cpctor(e);
                 *
                 * In this instance, we need to ensure that the original
                 * expression e does not have any nested references by
                 * checking the declaration initializer too.
                 */
                if (v._init && v._init.isExpInitializer())
                {
                    Expression ie = v._init.initializerToExpression();
                    result = lambdaCheckForNestedRef(ie, sc);
                }
            }
        }
    }

    scope LambdaCheckForNestedRef v = new LambdaCheckForNestedRef(sc);
    walkPostorder(e, v);
    return v.result;
}

/*****************************************
 * See if context `s` is nested within context `p`, meaning
 * it `p` is reachable at runtime by walking the static links.
 * If any of the intervening contexts are function literals,
 * make sure they are delegates.
 * Params:
 *      s = inner context
 *      p = outer context
 * Returns:
 *      true means it is accessible by walking the context pointers at runtime
 * References:
 *      for static links see https://en.wikipedia.org/wiki/Call_stack#Functions_of_the_call_stack
 */
bool ensureStaticLinkTo(Dsymbol s, Dsymbol p)
{
    while (s)
    {
        if (s == p) // hit!
            return true;

        if (auto fd = s.isFuncDeclaration())
        {
            if (!fd.isThis() && !fd.isNested())
                break;

            // https://issues.dlang.org/show_bug.cgi?id=15332
            // change to delegate if fd is actually nested.
            if (auto fld = fd.isFuncLiteralDeclaration())
                fld.tok = TOK.delegate_;
        }
        if (auto ad = s.isAggregateDeclaration())
        {
            if (ad.storage_class & STC.static_)
                break;
        }
        s = s.toParent2();
    }
    return false;
}
