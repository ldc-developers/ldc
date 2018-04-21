/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/printast.d, _printast.d)
 * Documentation:  https://dlang.org/phobos/dmd_printast.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/printast.d
 */

module dmd.printast;

import core.stdc.stdio;

import dmd.expression;
import dmd.tokens;
import dmd.visitor;

/********************
 * Print AST data structure in a nice format.
 * Params:
 *  e = expression AST to print
 *  indent = indentation level
 */
void printAST(Expression e, int indent = 0)
{
    scope PrintASTVisitor pav = new PrintASTVisitor(indent);
    e.accept(pav);
}

private:

extern (C++) final class PrintASTVisitor : Visitor
{
    alias visit = Visitor.visit;

    int indent;

    extern (D) this(int indent)
    {
        this.indent = indent;
    }

    override void visit(Expression e)
    {
        printIndent(indent);
        printf("%s %s\n", Token.toChars(e.op), e.type ? e.type.toChars() : "");
    }

    override void visit(StructLiteralExp e)
    {
        printIndent(indent);
        printf("%s %s, %s\n", Token.toChars(e.op), e.type ? e.type.toChars() : "", e.toChars());
    }

    override void visit(SymbolExp e)
    {
        visit(cast(Expression)e);
        printIndent(indent + 2);
        printf(".var: %s\n", e.var ? e.var.toChars() : "");
    }

    override void visit(DsymbolExp e)
    {
        visit(cast(Expression)e);
        printIndent(indent + 2);
        printf(".s: %s\n", e.s ? e.s.toChars() : "");
    }

    override void visit(DotIdExp e)
    {
        visit(cast(Expression)e);
        printIndent(indent + 2);
        printf(".ident: %s\n", e.ident.toChars());
        printAST(e.e1, indent + 2);
    }

    override void visit(UnaExp e)
    {
        visit(cast(Expression)e);
        printAST(e.e1, indent + 2);
    }

    override void visit(BinExp e)
    {
        visit(cast(Expression)e);
        printAST(e.e1, indent + 2);
        printAST(e.e2, indent + 2);
    }

    override void visit(DelegateExp e)
    {
        visit(cast(Expression)e);
        printIndent(indent + 2);
        printf(".func: %s\n", e.func ? e.func.toChars() : "");
    }

    static void printIndent(int indent)
    {
        foreach (i; 0 .. indent)
            putc(' ', stdout);
    }
}


