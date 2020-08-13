/**
 * Generate $(LINK2 https://dlang.org/dmd-windows.html#interface-files, D interface files).
 *
 * Also used to convert AST nodes to D code in general, e.g. for error messages or `printf` debugging.
 *
 * Copyright:   Copyright (C) 1999-2020 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/hdrgen.d, _hdrgen.d)
 * Documentation:  https://dlang.org/phobos/dmd_hdrgen.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/hdrgen.d
 */

module dmd.hdrgen;

import core.stdc.ctype;
import core.stdc.stdio;
import core.stdc.string;
import dmd.aggregate;
import dmd.aliasthis;
import dmd.arraytypes;
import dmd.attrib;
import dmd.complex;
import dmd.cond;
import dmd.ctfeexpr;
import dmd.dclass;
import dmd.declaration;
import dmd.denum;
import dmd.dimport;
import dmd.dmodule;
import dmd.doc;
import dmd.dstruct;
import dmd.dsymbol;
import dmd.dtemplate;
import dmd.dversion;
import dmd.expression;
import dmd.func;
import dmd.globals;
import dmd.id;
import dmd.identifier;
import dmd.init;
import dmd.mtype;
import dmd.nspace;
import dmd.parse;
import dmd.root.ctfloat;
import dmd.root.outbuffer;
import dmd.root.rootobject;
import dmd.root.string;
import dmd.statement;
import dmd.staticassert;
import dmd.target;
import dmd.tokens;
import dmd.utils;
import dmd.visitor;

struct HdrGenState
{
    bool hdrgen;        /// true if generating header file
    bool ddoc;          /// true if generating Ddoc file
    bool fullDump;      /// true if generating a full AST dump file

    bool fullQual;      /// fully qualify types when printing
    int tpltMember;
    int autoMember;
    int forStmtInit;

    bool declstring; // set while declaring alias for string,wstring or dstring
    EnumDeclaration inEnumDecl;
}

enum TEST_EMIT_ALL = 0;

extern (C++) void genhdrfile(Module m)
{
version (IN_LLVM)
{
    // FIXME: DMD overwrites header files. This should be done only in a DMD mode.
    // m.checkAndAddOutputFile(m.hdrfile);
}
    OutBuffer buf;
    buf.doindent = 1;
    buf.printf("// D import file generated from '%s'", m.srcfile.toChars());
    buf.writenl();
    HdrGenState hgs;
    hgs.hdrgen = true;
    toCBuffer(m, &buf, &hgs);
    writeFile(m.loc, m.hdrfile.toString(), buf[]);
}

/**
 * Dumps the full contents of module `m` to `buf`.
 * Params:
 *   buf = buffer to write to.
 *   m = module to visit all members of.
 */
extern (C++) void moduleToBuffer(OutBuffer* buf, Module m)
{
    HdrGenState hgs;
    hgs.fullDump = true;
    toCBuffer(m, buf, &hgs);
}

void moduleToBuffer2(Module m, OutBuffer* buf, HdrGenState* hgs)
{
    if (m.md)
    {
        if (m.userAttribDecl)
        {
            buf.writestring("@(");
            argsToBuffer(m.userAttribDecl.atts, buf, hgs);
            buf.writeByte(')');
            buf.writenl();
        }
        if (m.md.isdeprecated)
        {
            if (m.md.msg)
            {
                buf.writestring("deprecated(");
                m.md.msg.expressionToBuffer(buf, hgs);
                buf.writestring(") ");
            }
            else
                buf.writestring("deprecated ");
        }
        buf.writestring("module ");
        buf.writestring(m.md.toChars());
        buf.writeByte(';');
        buf.writenl();
    }

    foreach (s; *m.members)
    {
        s.dsymbolToBuffer(buf, hgs);
    }
}

private void statementToBuffer(Statement s, OutBuffer* buf, HdrGenState* hgs)
{
    scope v = new StatementPrettyPrintVisitor(buf, hgs);
    s.accept(v);
}

private extern (C++) final class StatementPrettyPrintVisitor : Visitor
{
    alias visit = Visitor.visit;
public:
    OutBuffer* buf;
    HdrGenState* hgs;

    extern (D) this(OutBuffer* buf, HdrGenState* hgs)
    {
        this.buf = buf;
        this.hgs = hgs;
    }

    override void visit(Statement s)
    {
        buf.writestring("Statement::toCBuffer()");
        buf.writenl();
        assert(0);
    }

    override void visit(ErrorStatement s)
    {
        buf.writestring("__error__");
        buf.writenl();
    }

    override void visit(ExpStatement s)
    {
        if (s.exp && s.exp.op == TOK.declaration &&
            (cast(DeclarationExp)s.exp).declaration)
        {
            // bypass visit(DeclarationExp)
            (cast(DeclarationExp)s.exp).declaration.dsymbolToBuffer(buf, hgs);
            return;
        }
        if (s.exp)
            s.exp.expressionToBuffer(buf, hgs);
        buf.writeByte(';');
        if (!hgs.forStmtInit)
            buf.writenl();
    }

    override void visit(CompileStatement s)
    {
        buf.writestring("mixin(");
        argsToBuffer(s.exps, buf, hgs, null);
        buf.writestring(");");
        if (!hgs.forStmtInit)
            buf.writenl();
    }

    override void visit(CompoundStatement s)
    {
        foreach (sx; *s.statements)
        {
            if (sx)
                sx.accept(this);
        }
    }

    override void visit(CompoundDeclarationStatement s)
    {
        bool anywritten = false;
        foreach (sx; *s.statements)
        {
            auto ds = sx ? sx.isExpStatement() : null;
            if (ds && ds.exp.op == TOK.declaration)
            {
                auto d = (cast(DeclarationExp)ds.exp).declaration;
                assert(d.isDeclaration());
                if (auto v = d.isVarDeclaration())
                {
                    scope ppv = new DsymbolPrettyPrintVisitor(buf, hgs);
                    ppv.visitVarDecl(v, anywritten);
                }
                else
                    d.dsymbolToBuffer(buf, hgs);
                anywritten = true;
            }
        }
        buf.writeByte(';');
        if (!hgs.forStmtInit)
            buf.writenl();
    }

    override void visit(UnrolledLoopStatement s)
    {
        buf.writestring("/*unrolled*/ {");
        buf.writenl();
        buf.level++;
        foreach (sx; *s.statements)
        {
            if (sx)
                sx.accept(this);
        }
        buf.level--;
        buf.writeByte('}');
        buf.writenl();
    }

    override void visit(ScopeStatement s)
    {
        buf.writeByte('{');
        buf.writenl();
        buf.level++;
        if (s.statement)
            s.statement.accept(this);
        buf.level--;
        buf.writeByte('}');
        buf.writenl();
    }

    override void visit(WhileStatement s)
    {
        buf.writestring("while (");
        s.condition.expressionToBuffer(buf, hgs);
        buf.writeByte(')');
        buf.writenl();
        if (s._body)
            s._body.accept(this);
    }

    override void visit(DoStatement s)
    {
        buf.writestring("do");
        buf.writenl();
        if (s._body)
            s._body.accept(this);
        buf.writestring("while (");
        s.condition.expressionToBuffer(buf, hgs);
        buf.writestring(");");
        buf.writenl();
    }

    override void visit(ForStatement s)
    {
        buf.writestring("for (");
        if (s._init)
        {
            hgs.forStmtInit++;
            s._init.accept(this);
            hgs.forStmtInit--;
        }
        else
            buf.writeByte(';');
        if (s.condition)
        {
            buf.writeByte(' ');
            s.condition.expressionToBuffer(buf, hgs);
        }
        buf.writeByte(';');
        if (s.increment)
        {
            buf.writeByte(' ');
            s.increment.expressionToBuffer(buf, hgs);
        }
        buf.writeByte(')');
        buf.writenl();
        buf.writeByte('{');
        buf.writenl();
        buf.level++;
        if (s._body)
            s._body.accept(this);
        buf.level--;
        buf.writeByte('}');
        buf.writenl();
    }

    private void foreachWithoutBody(ForeachStatement s)
    {
        buf.writestring(Token.toString(s.op));
        buf.writestring(" (");
        foreach (i, p; *s.parameters)
        {
            if (i)
                buf.writestring(", ");
            if (stcToBuffer(buf, p.storageClass))
                buf.writeByte(' ');
            if (p.type)
                typeToBuffer(p.type, p.ident, buf, hgs);
            else
                buf.writestring(p.ident.toString());
        }
        buf.writestring("; ");
        s.aggr.expressionToBuffer(buf, hgs);
        buf.writeByte(')');
        buf.writenl();
    }

    override void visit(ForeachStatement s)
    {
        foreachWithoutBody(s);
        buf.writeByte('{');
        buf.writenl();
        buf.level++;
        if (s._body)
            s._body.accept(this);
        buf.level--;
        buf.writeByte('}');
        buf.writenl();
    }

    private void foreachRangeWithoutBody(ForeachRangeStatement s)
    {
        buf.writestring(Token.toString(s.op));
        buf.writestring(" (");
        if (s.prm.type)
            typeToBuffer(s.prm.type, s.prm.ident, buf, hgs);
        else
            buf.writestring(s.prm.ident.toString());
        buf.writestring("; ");
        s.lwr.expressionToBuffer(buf, hgs);
        buf.writestring(" .. ");
        s.upr.expressionToBuffer(buf, hgs);
        buf.writeByte(')');
        buf.writenl();
    }

    override void visit(ForeachRangeStatement s)
    {
        foreachRangeWithoutBody(s);
        buf.writeByte('{');
        buf.writenl();
        buf.level++;
        if (s._body)
            s._body.accept(this);
        buf.level--;
        buf.writeByte('}');
        buf.writenl();
    }

    override void visit(StaticForeachStatement s)
    {
        buf.writestring("static ");
        if (s.sfe.aggrfe)
        {
            visit(s.sfe.aggrfe);
        }
        else
        {
            assert(s.sfe.rangefe);
            visit(s.sfe.rangefe);
        }
    }

    override void visit(ForwardingStatement s)
    {
        s.statement.accept(this);
    }

    override void visit(IfStatement s)
    {
        buf.writestring("if (");
        if (Parameter p = s.prm)
        {
            StorageClass stc = p.storageClass;
            if (!p.type && !stc)
                stc = STC.auto_;
            if (stcToBuffer(buf, stc))
                buf.writeByte(' ');
            if (p.type)
                typeToBuffer(p.type, p.ident, buf, hgs);
            else
                buf.writestring(p.ident.toString());
            buf.writestring(" = ");
        }
        s.condition.expressionToBuffer(buf, hgs);
        buf.writeByte(')');
        buf.writenl();
        if (s.ifbody.isScopeStatement())
        {
            s.ifbody.accept(this);
        }
        else
        {
            buf.level++;
            s.ifbody.accept(this);
            buf.level--;
        }
        if (s.elsebody)
        {
            buf.writestring("else");
            if (!s.elsebody.isIfStatement())
            {
                buf.writenl();
            }
            else
            {
                buf.writeByte(' ');
            }
            if (s.elsebody.isScopeStatement() || s.elsebody.isIfStatement())
            {
                s.elsebody.accept(this);
            }
            else
            {
                buf.level++;
                s.elsebody.accept(this);
                buf.level--;
            }
        }
    }

    override void visit(ConditionalStatement s)
    {
        s.condition.conditionToBuffer(buf, hgs);
        buf.writenl();
        buf.writeByte('{');
        buf.writenl();
        buf.level++;
        if (s.ifbody)
            s.ifbody.accept(this);
        buf.level--;
        buf.writeByte('}');
        buf.writenl();
        if (s.elsebody)
        {
            buf.writestring("else");
            buf.writenl();
            buf.writeByte('{');
            buf.level++;
            buf.writenl();
            s.elsebody.accept(this);
            buf.level--;
            buf.writeByte('}');
        }
        buf.writenl();
    }

    override void visit(PragmaStatement s)
    {
        buf.writestring("pragma (");
        buf.writestring(s.ident.toString());
        if (s.args && s.args.dim)
        {
            buf.writestring(", ");
            argsToBuffer(s.args, buf, hgs);
        }
        buf.writeByte(')');
        if (s._body)
        {
            buf.writenl();
            buf.writeByte('{');
            buf.writenl();
            buf.level++;
            s._body.accept(this);
            buf.level--;
            buf.writeByte('}');
            buf.writenl();
        }
        else
        {
            buf.writeByte(';');
            buf.writenl();
        }
    }

    override void visit(StaticAssertStatement s)
    {
        s.sa.dsymbolToBuffer(buf, hgs);
    }

    override void visit(SwitchStatement s)
    {
        buf.writestring(s.isFinal ? "final switch (" : "switch (");
        s.condition.expressionToBuffer(buf, hgs);
        buf.writeByte(')');
        buf.writenl();
        if (s._body)
        {
            if (!s._body.isScopeStatement())
            {
                buf.writeByte('{');
                buf.writenl();
                buf.level++;
                s._body.accept(this);
                buf.level--;
                buf.writeByte('}');
                buf.writenl();
            }
            else
            {
                s._body.accept(this);
            }
        }
    }

    override void visit(CaseStatement s)
    {
        buf.writestring("case ");
        s.exp.expressionToBuffer(buf, hgs);
        buf.writeByte(':');
        buf.writenl();
        s.statement.accept(this);
    }

    override void visit(CaseRangeStatement s)
    {
        buf.writestring("case ");
        s.first.expressionToBuffer(buf, hgs);
        buf.writestring(": .. case ");
        s.last.expressionToBuffer(buf, hgs);
        buf.writeByte(':');
        buf.writenl();
        s.statement.accept(this);
    }

    override void visit(DefaultStatement s)
    {
        buf.writestring("default:");
        buf.writenl();
        s.statement.accept(this);
    }

    override void visit(GotoDefaultStatement s)
    {
        buf.writestring("goto default;");
        buf.writenl();
    }

    override void visit(GotoCaseStatement s)
    {
        buf.writestring("goto case");
        if (s.exp)
        {
            buf.writeByte(' ');
            s.exp.expressionToBuffer(buf, hgs);
        }
        buf.writeByte(';');
        buf.writenl();
    }

    override void visit(SwitchErrorStatement s)
    {
        buf.writestring("SwitchErrorStatement::toCBuffer()");
        buf.writenl();
    }

    override void visit(ReturnStatement s)
    {
        buf.writestring("return ");
        if (s.exp)
            s.exp.expressionToBuffer(buf, hgs);
        buf.writeByte(';');
        buf.writenl();
    }

    override void visit(BreakStatement s)
    {
        buf.writestring("break");
        if (s.ident)
        {
            buf.writeByte(' ');
            buf.writestring(s.ident.toString());
        }
        buf.writeByte(';');
        buf.writenl();
    }

    override void visit(ContinueStatement s)
    {
        buf.writestring("continue");
        if (s.ident)
        {
            buf.writeByte(' ');
            buf.writestring(s.ident.toString());
        }
        buf.writeByte(';');
        buf.writenl();
    }

    override void visit(SynchronizedStatement s)
    {
        buf.writestring("synchronized");
        if (s.exp)
        {
            buf.writeByte('(');
            s.exp.expressionToBuffer(buf, hgs);
            buf.writeByte(')');
        }
        if (s._body)
        {
            buf.writeByte(' ');
            s._body.accept(this);
        }
    }

    override void visit(WithStatement s)
    {
        buf.writestring("with (");
        s.exp.expressionToBuffer(buf, hgs);
        buf.writestring(")");
        buf.writenl();
        if (s._body)
            s._body.accept(this);
    }

    override void visit(TryCatchStatement s)
    {
        buf.writestring("try");
        buf.writenl();
        if (s._body)
        {
            if (s._body.isScopeStatement())
            {
                s._body.accept(this);
            }
            else
            {
                buf.level++;
                s._body.accept(this);
                buf.level--;
            }
        }
        foreach (c; *s.catches)
        {
            visit(c);
        }
    }

    override void visit(TryFinallyStatement s)
    {
        buf.writestring("try");
        buf.writenl();
        buf.writeByte('{');
        buf.writenl();
        buf.level++;
        s._body.accept(this);
        buf.level--;
        buf.writeByte('}');
        buf.writenl();
        buf.writestring("finally");
        buf.writenl();
        if (s.finalbody.isScopeStatement())
        {
            s.finalbody.accept(this);
        }
        else
        {
            buf.level++;
            s.finalbody.accept(this);
            buf.level--;
        }
    }

    override void visit(ScopeGuardStatement s)
    {
        buf.writestring(Token.toString(s.tok));
        buf.writeByte(' ');
        if (s.statement)
            s.statement.accept(this);
    }

    override void visit(ThrowStatement s)
    {
        buf.writestring("throw ");
        s.exp.expressionToBuffer(buf, hgs);
        buf.writeByte(';');
        buf.writenl();
    }

    override void visit(DebugStatement s)
    {
        if (s.statement)
        {
            s.statement.accept(this);
        }
    }

    override void visit(GotoStatement s)
    {
        buf.writestring("goto ");
        buf.writestring(s.ident.toString());
        buf.writeByte(';');
        buf.writenl();
    }

    override void visit(LabelStatement s)
    {
        buf.writestring(s.ident.toString());
        buf.writeByte(':');
        buf.writenl();
        if (s.statement)
            s.statement.accept(this);
    }

    override void visit(AsmStatement s)
    {
        buf.writestring("asm { ");
        Token* t = s.tokens;
        buf.level++;
        while (t)
        {
            buf.writestring(t.toChars());
            if (t.next &&
                t.value != TOK.min      &&
                t.value != TOK.comma    && t.next.value != TOK.comma    &&
                t.value != TOK.leftBracket && t.next.value != TOK.leftBracket &&
                                          t.next.value != TOK.rightBracket &&
                t.value != TOK.leftParentheses   && t.next.value != TOK.leftParentheses   &&
                                          t.next.value != TOK.rightParentheses   &&
                t.value != TOK.dot      && t.next.value != TOK.dot)
            {
                buf.writeByte(' ');
            }
            t = t.next;
        }
        buf.level--;
        buf.writestring("; }");
        buf.writenl();
    }

    override void visit(ImportStatement s)
    {
        foreach (imp; *s.imports)
        {
            imp.dsymbolToBuffer(buf, hgs);
        }
    }

    void visit(Catch c)
    {
        buf.writestring("catch");
        if (c.type)
        {
            buf.writeByte('(');
            typeToBuffer(c.type, c.ident, buf, hgs);
            buf.writeByte(')');
        }
        buf.writenl();
        buf.writeByte('{');
        buf.writenl();
        buf.level++;
        if (c.handler)
            c.handler.accept(this);
        buf.level--;
        buf.writeByte('}');
        buf.writenl();
    }
}

private void dsymbolToBuffer(Dsymbol s, OutBuffer* buf, HdrGenState* hgs)
{
    scope v = new DsymbolPrettyPrintVisitor(buf, hgs);
    s.accept(v);
}

private extern (C++) final class DsymbolPrettyPrintVisitor : Visitor
{
    alias visit = Visitor.visit;
public:
    OutBuffer* buf;
    HdrGenState* hgs;

    extern (D) this(OutBuffer* buf, HdrGenState* hgs)
    {
        this.buf = buf;
        this.hgs = hgs;
    }

    ////////////////////////////////////////////////////////////////////////////

    override void visit(Dsymbol s)
    {
        buf.writestring(s.toChars());
    }

    override void visit(StaticAssert s)
    {
        buf.writestring(s.kind());
        buf.writeByte('(');
        s.exp.expressionToBuffer(buf, hgs);
        if (s.msg)
        {
            buf.writestring(", ");
            s.msg.expressionToBuffer(buf, hgs);
        }
        buf.writestring(");");
        buf.writenl();
    }

    override void visit(DebugSymbol s)
    {
        buf.writestring("debug = ");
        if (s.ident)
            buf.writestring(s.ident.toString());
        else
            buf.print(s.level);
        buf.writeByte(';');
        buf.writenl();
    }

    override void visit(VersionSymbol s)
    {
        buf.writestring("version = ");
        if (s.ident)
            buf.writestring(s.ident.toString());
        else
            buf.print(s.level);
        buf.writeByte(';');
        buf.writenl();
    }

    override void visit(EnumMember em)
    {
        if (em.type)
            typeToBuffer(em.type, em.ident, buf, hgs);
        else
            buf.writestring(em.ident.toString());
        if (em.value)
        {
            buf.writestring(" = ");
            em.value.expressionToBuffer(buf, hgs);
        }
    }

    override void visit(Import imp)
    {
        if (hgs.hdrgen && imp.id == Id.object)
            return; // object is imported by default
        if (imp.isstatic)
            buf.writestring("static ");
        buf.writestring("import ");
        if (imp.aliasId)
        {
            buf.printf("%s = ", imp.aliasId.toChars());
        }
        if (imp.packages && imp.packages.dim)
        {
            foreach (const pid; *imp.packages)
            {
                buf.printf("%s.", pid.toChars());
            }
        }
        buf.writestring(imp.id.toString());
        if (imp.names.dim)
        {
            buf.writestring(" : ");
            foreach (const i, const name; imp.names)
            {
                if (i)
                    buf.writestring(", ");
                const _alias = imp.aliases[i];
                if (_alias)
                    buf.printf("%s = %s", _alias.toChars(), name.toChars());
                else
                    buf.writestring(name.toChars());
            }
        }
        buf.writeByte(';');
        buf.writenl();
    }

    override void visit(AliasThis d)
    {
        buf.writestring("alias ");
        buf.writestring(d.ident.toString());
        buf.writestring(" this;\n");
    }

    override void visit(AttribDeclaration d)
    {
        if (!d.decl)
        {
            buf.writeByte(';');
            buf.writenl();
            return;
        }
        if (d.decl.dim == 0)
            buf.writestring("{}");
        else if (hgs.hdrgen && d.decl.dim == 1 && (*d.decl)[0].isUnitTestDeclaration())
        {
            // hack for bugzilla 8081
            buf.writestring("{}");
        }
        else if (d.decl.dim == 1)
        {
            (*d.decl)[0].accept(this);
            return;
        }
        else
        {
            buf.writenl();
            buf.writeByte('{');
            buf.writenl();
            buf.level++;
            foreach (de; *d.decl)
                de.accept(this);
            buf.level--;
            buf.writeByte('}');
        }
        buf.writenl();
    }

    override void visit(StorageClassDeclaration d)
    {
        if (stcToBuffer(buf, d.stc))
            buf.writeByte(' ');
        visit(cast(AttribDeclaration)d);
    }

    override void visit(DeprecatedDeclaration d)
    {
        buf.writestring("deprecated(");
        d.msg.expressionToBuffer(buf, hgs);
        buf.writestring(") ");
        visit(cast(AttribDeclaration)d);
    }

    override void visit(LinkDeclaration d)
    {
        buf.writestring("extern (");
        buf.writestring(linkageToString(d.linkage));
        buf.writestring(") ");
        visit(cast(AttribDeclaration)d);
    }

    override void visit(CPPMangleDeclaration d)
    {
        string s;
        final switch (d.cppmangle)
        {
        case CPPMANGLE.asClass:
            s = "class";
            break;
        case CPPMANGLE.asStruct:
            s = "struct";
            break;
        case CPPMANGLE.def:
            break;
        }
        buf.writestring("extern (C++, ");
        buf.writestring(s);
        buf.writestring(") ");
        visit(cast(AttribDeclaration)d);
    }

    override void visit(ProtDeclaration d)
    {
        protectionToBuffer(buf, d.protection);
        buf.writeByte(' ');
        AttribDeclaration ad = cast(AttribDeclaration)d;
        if (ad.decl.dim == 1 && (*ad.decl)[0].isProtDeclaration)
            visit(cast(AttribDeclaration)(*ad.decl)[0]);
        else
            visit(cast(AttribDeclaration)d);
    }

    override void visit(AlignDeclaration d)
    {
        buf.writestring("align ");
        if (d.ealign)
            buf.printf("(%s) ", d.ealign.toChars());
        visit(cast(AttribDeclaration)d);
    }

    override void visit(AnonDeclaration d)
    {
        buf.writestring(d.isunion ? "union" : "struct");
        buf.writenl();
        buf.writestring("{");
        buf.writenl();
        buf.level++;
        if (d.decl)
        {
            foreach (de; *d.decl)
                de.accept(this);
        }
        buf.level--;
        buf.writestring("}");
        buf.writenl();
    }

    override void visit(PragmaDeclaration d)
    {
        buf.writestring("pragma (");
        buf.writestring(d.ident.toString());
        if (d.args && d.args.dim)
        {
            buf.writestring(", ");
            argsToBuffer(d.args, buf, hgs);
        }
        buf.writeByte(')');
        visit(cast(AttribDeclaration)d);
    }

    override void visit(ConditionalDeclaration d)
    {
        d.condition.conditionToBuffer(buf, hgs);
        if (d.decl || d.elsedecl)
        {
            buf.writenl();
            buf.writeByte('{');
            buf.writenl();
            buf.level++;
            if (d.decl)
            {
                foreach (de; *d.decl)
                    de.accept(this);
            }
            buf.level--;
            buf.writeByte('}');
            if (d.elsedecl)
            {
                buf.writenl();
                buf.writestring("else");
                buf.writenl();
                buf.writeByte('{');
                buf.writenl();
                buf.level++;
                foreach (de; *d.elsedecl)
                    de.accept(this);
                buf.level--;
                buf.writeByte('}');
            }
        }
        else
            buf.writeByte(':');
        buf.writenl();
    }

    override void visit(StaticForeachDeclaration s)
    {
        void foreachWithoutBody(ForeachStatement s)
        {
            buf.writestring(Token.toString(s.op));
            buf.writestring(" (");
            foreach (i, p; *s.parameters)
            {
                if (i)
                    buf.writestring(", ");
                if (stcToBuffer(buf, p.storageClass))
                    buf.writeByte(' ');
                if (p.type)
                    typeToBuffer(p.type, p.ident, buf, hgs);
                else
                    buf.writestring(p.ident.toString());
            }
            buf.writestring("; ");
            s.aggr.expressionToBuffer(buf, hgs);
            buf.writeByte(')');
            buf.writenl();
        }

        void foreachRangeWithoutBody(ForeachRangeStatement s)
        {
            /* s.op ( prm ; lwr .. upr )
             */
            buf.writestring(Token.toString(s.op));
            buf.writestring(" (");
            if (s.prm.type)
                typeToBuffer(s.prm.type, s.prm.ident, buf, hgs);
            else
                buf.writestring(s.prm.ident.toString());
            buf.writestring("; ");
            s.lwr.expressionToBuffer(buf, hgs);
            buf.writestring(" .. ");
            s.upr.expressionToBuffer(buf, hgs);
            buf.writeByte(')');
            buf.writenl();
        }

        buf.writestring("static ");
        if (s.sfe.aggrfe)
        {
            foreachWithoutBody(s.sfe.aggrfe);
        }
        else
        {
            assert(s.sfe.rangefe);
            foreachRangeWithoutBody(s.sfe.rangefe);
        }
        buf.writeByte('{');
        buf.writenl();
        buf.level++;
        visit(cast(AttribDeclaration)s);
        buf.level--;
        buf.writeByte('}');
        buf.writenl();

    }

    override void visit(CompileDeclaration d)
    {
        buf.writestring("mixin(");
        argsToBuffer(d.exps, buf, hgs, null);
        buf.writestring(");");
        buf.writenl();
    }

    override void visit(UserAttributeDeclaration d)
    {
        buf.writestring("@(");
        argsToBuffer(d.atts, buf, hgs);
        buf.writeByte(')');
        visit(cast(AttribDeclaration)d);
    }

    override void visit(TemplateDeclaration d)
    {
        version (none)
        {
            // Should handle template functions for doc generation
            if (onemember && onemember.isFuncDeclaration())
                buf.writestring("foo ");
        }
        if ((hgs.hdrgen || hgs.fullDump) && visitEponymousMember(d))
            return;
        if (hgs.ddoc)
            buf.writestring(d.kind());
        else
            buf.writestring("template");
        buf.writeByte(' ');
        buf.writestring(d.ident.toString());
        buf.writeByte('(');
        visitTemplateParameters(hgs.ddoc ? d.origParameters : d.parameters);
        buf.writeByte(')');
        visitTemplateConstraint(d.constraint);
        if (hgs.hdrgen || hgs.fullDump)
        {
            hgs.tpltMember++;
            buf.writenl();
            buf.writeByte('{');
            buf.writenl();
            buf.level++;
            foreach (s; *d.members)
                s.accept(this);
            buf.level--;
            buf.writeByte('}');
            buf.writenl();
            hgs.tpltMember--;
        }
    }

    bool visitEponymousMember(TemplateDeclaration d)
    {
        if (!d.members || d.members.dim != 1)
            return false;
        Dsymbol onemember = (*d.members)[0];
        if (onemember.ident != d.ident)
            return false;
        if (FuncDeclaration fd = onemember.isFuncDeclaration())
        {
            assert(fd.type);
            if (stcToBuffer(buf, fd.storage_class))
                buf.writeByte(' ');
            functionToBufferFull(cast(TypeFunction)fd.type, buf, d.ident, hgs, d);
            visitTemplateConstraint(d.constraint);
            hgs.tpltMember++;
            bodyToBuffer(fd);
            hgs.tpltMember--;
            return true;
        }
        if (AggregateDeclaration ad = onemember.isAggregateDeclaration())
        {
            buf.writestring(ad.kind());
            buf.writeByte(' ');
            buf.writestring(ad.ident.toString());
            buf.writeByte('(');
            visitTemplateParameters(hgs.ddoc ? d.origParameters : d.parameters);
            buf.writeByte(')');
            visitTemplateConstraint(d.constraint);
            visitBaseClasses(ad.isClassDeclaration());
            hgs.tpltMember++;
            if (ad.members)
            {
                buf.writenl();
                buf.writeByte('{');
                buf.writenl();
                buf.level++;
                foreach (s; *ad.members)
                    s.accept(this);
                buf.level--;
                buf.writeByte('}');
            }
            else
                buf.writeByte(';');
            buf.writenl();
            hgs.tpltMember--;
            return true;
        }
        if (VarDeclaration vd = onemember.isVarDeclaration())
        {
            if (d.constraint)
                return false;
            if (stcToBuffer(buf, vd.storage_class))
                buf.writeByte(' ');
            if (vd.type)
                typeToBuffer(vd.type, vd.ident, buf, hgs);
            else
                buf.writestring(vd.ident.toString());
            buf.writeByte('(');
            visitTemplateParameters(hgs.ddoc ? d.origParameters : d.parameters);
            buf.writeByte(')');
            if (vd._init)
            {
                buf.writestring(" = ");
                ExpInitializer ie = vd._init.isExpInitializer();
                if (ie && (ie.exp.op == TOK.construct || ie.exp.op == TOK.blit))
                    (cast(AssignExp)ie.exp).e2.expressionToBuffer(buf, hgs);
                else
                    vd._init.initializerToBuffer(buf, hgs);
            }
            buf.writeByte(';');
            buf.writenl();
            return true;
        }
        return false;
    }

    void visitTemplateParameters(TemplateParameters* parameters)
    {
        if (!parameters || !parameters.dim)
            return;
        foreach (i, p; *parameters)
        {
            if (i)
                buf.writestring(", ");
            p.templateParameterToBuffer(buf, hgs);
        }
    }

    void visitTemplateConstraint(Expression constraint)
    {
        if (!constraint)
            return;
        buf.writestring(" if (");
        constraint.expressionToBuffer(buf, hgs);
        buf.writeByte(')');
    }

    override void visit(TemplateInstance ti)
    {
        buf.writestring(ti.name.toChars());
        tiargsToBuffer(ti, buf, hgs);

        if (hgs.fullDump)
        {
            buf.writenl();
            dumpTemplateInstance(ti, buf, hgs);
        }
    }

    override void visit(TemplateMixin tm)
    {
        buf.writestring("mixin ");
        typeToBuffer(tm.tqual, null, buf, hgs);
        tiargsToBuffer(tm, buf, hgs);
        if (tm.ident && memcmp(tm.ident.toChars(), cast(const(char)*)"__mixin", 7) != 0)
        {
            buf.writeByte(' ');
            buf.writestring(tm.ident.toString());
        }
        buf.writeByte(';');
        buf.writenl();
        if (hgs.fullDump)
            dumpTemplateInstance(tm, buf, hgs);
    }

    override void visit(EnumDeclaration d)
    {
        auto oldInEnumDecl = hgs.inEnumDecl;
        scope(exit) hgs.inEnumDecl = oldInEnumDecl;
        hgs.inEnumDecl = d;
        buf.writestring("enum ");
        if (d.ident)
        {
            buf.writestring(d.ident.toString());
            buf.writeByte(' ');
        }
        if (d.memtype)
        {
            buf.writestring(": ");
            typeToBuffer(d.memtype, null, buf, hgs);
        }
        if (!d.members)
        {
            buf.writeByte(';');
            buf.writenl();
            return;
        }
        buf.writenl();
        buf.writeByte('{');
        buf.writenl();
        buf.level++;
        foreach (em; *d.members)
        {
            if (!em)
                continue;
            em.accept(this);
            buf.writeByte(',');
            buf.writenl();
        }
        buf.level--;
        buf.writeByte('}');
        buf.writenl();
    }

    override void visit(Nspace d)
    {
        buf.writestring("extern (C++, ");
        buf.writestring(d.ident.toString());
        buf.writeByte(')');
        buf.writenl();
        buf.writeByte('{');
        buf.writenl();
        buf.level++;
        foreach (s; *d.members)
            s.accept(this);
        buf.level--;
        buf.writeByte('}');
        buf.writenl();
    }

    override void visit(StructDeclaration d)
    {
        buf.writestring(d.kind());
        buf.writeByte(' ');
        if (!d.isAnonymous())
            buf.writestring(d.toChars());
        if (!d.members)
        {
            buf.writeByte(';');
            buf.writenl();
            return;
        }
        buf.writenl();
        buf.writeByte('{');
        buf.writenl();
        buf.level++;
        foreach (s; *d.members)
            s.accept(this);
        buf.level--;
        buf.writeByte('}');
        buf.writenl();
    }

    override void visit(ClassDeclaration d)
    {
        if (!d.isAnonymous())
        {
            buf.writestring(d.kind());
            buf.writeByte(' ');
            buf.writestring(d.ident.toString());
        }
        visitBaseClasses(d);
        if (d.members)
        {
            buf.writenl();
            buf.writeByte('{');
            buf.writenl();
            buf.level++;
            foreach (s; *d.members)
                s.accept(this);
            buf.level--;
            buf.writeByte('}');
        }
        else
            buf.writeByte(';');
        buf.writenl();
    }

    void visitBaseClasses(ClassDeclaration d)
    {
        if (!d || !d.baseclasses.dim)
            return;
        if (!d.isAnonymous())
            buf.writestring(" : ");
        foreach (i, b; *d.baseclasses)
        {
            if (i)
                buf.writestring(", ");
            typeToBuffer(b.type, null, buf, hgs);
        }
    }

    override void visit(AliasDeclaration d)
    {
        if (d.storage_class & STC.local)
            return;
        buf.writestring("alias ");
        if (d.aliassym)
        {
            buf.writestring(d.ident.toString());
            buf.writestring(" = ");
            if (stcToBuffer(buf, d.storage_class))
                buf.writeByte(' ');
            d.aliassym.accept(this);
        }
        else if (d.type.ty == Tfunction)
        {
            if (stcToBuffer(buf, d.storage_class))
                buf.writeByte(' ');
            typeToBuffer(d.type, d.ident, buf, hgs);
        }
        else if (d.ident)
        {
            hgs.declstring = (d.ident == Id.string || d.ident == Id.wstring || d.ident == Id.dstring);
            buf.writestring(d.ident.toString());
            buf.writestring(" = ");
            if (stcToBuffer(buf, d.storage_class))
                buf.writeByte(' ');
            typeToBuffer(d.type, null, buf, hgs);
            hgs.declstring = false;
        }
        buf.writeByte(';');
        buf.writenl();
    }

    override void visit(VarDeclaration d)
    {
        if (d.storage_class & STC.local)
            return;
        visitVarDecl(d, false);
        buf.writeByte(';');
        buf.writenl();
    }

    void visitVarDecl(VarDeclaration v, bool anywritten)
    {
        if (anywritten)
        {
            buf.writestring(", ");
            buf.writestring(v.ident.toString());
        }
        else
        {
            if (stcToBuffer(buf, v.storage_class))
                buf.writeByte(' ');
            if (v.type)
                typeToBuffer(v.type, v.ident, buf, hgs);
            else
                buf.writestring(v.ident.toString());
        }
        if (v._init)
        {
            buf.writestring(" = ");
            auto ie = v._init.isExpInitializer();
            if (ie && (ie.exp.op == TOK.construct || ie.exp.op == TOK.blit))
                (cast(AssignExp)ie.exp).e2.expressionToBuffer(buf, hgs);
            else
                v._init.initializerToBuffer(buf, hgs);
        }
    }

    override void visit(FuncDeclaration f)
    {
        //printf("FuncDeclaration::toCBuffer() '%s'\n", f.toChars());
        if (stcToBuffer(buf, f.storage_class))
            buf.writeByte(' ');
        auto tf = cast(TypeFunction)f.type;
        typeToBuffer(tf, f.ident, buf, hgs);

        if (hgs.hdrgen)
        {
            // if the return type is missing (e.g. ref functions or auto)
            if (!tf.next || f.storage_class & STC.auto_)
            {
                hgs.autoMember++;
                bodyToBuffer(f);
                hgs.autoMember--;
            }
            else if (hgs.tpltMember == 0 && global.params.hdrStripPlainFunctions)
            {
                buf.writeByte(';');
                buf.writenl();
            }
            else
                bodyToBuffer(f);
        }
        else
            bodyToBuffer(f);
    }

    void bodyToBuffer(FuncDeclaration f)
    {
        if (!f.fbody || (hgs.hdrgen && global.params.hdrStripPlainFunctions && !hgs.autoMember && !hgs.tpltMember))
        {
            buf.writeByte(';');
            buf.writenl();
            return;
        }
        const savetlpt = hgs.tpltMember;
        const saveauto = hgs.autoMember;
        hgs.tpltMember = 0;
        hgs.autoMember = 0;
        buf.writenl();
        bool requireDo = false;
        // in{}
        if (f.frequires)
        {
            foreach (frequire; *f.frequires)
            {
                buf.writestring("in");
                if (auto es = frequire.isExpStatement())
                {
                    assert(es.exp && es.exp.op == TOK.assert_);
                    buf.writestring(" (");
                    (cast(AssertExp)es.exp).e1.expressionToBuffer(buf, hgs);
                    buf.writeByte(')');
                    buf.writenl();
                    requireDo = false;
                }
                else
                {
                    buf.writenl();
                    frequire.statementToBuffer(buf, hgs);
                    requireDo = true;
                }
            }
        }
        // out{}
        if (f.fensures)
        {
            foreach (fensure; *f.fensures)
            {
                buf.writestring("out");
                if (auto es = fensure.ensure.isExpStatement())
                {
                    assert(es.exp && es.exp.op == TOK.assert_);
                    buf.writestring(" (");
                    if (fensure.id)
                    {
                        buf.writestring(fensure.id.toString());
                    }
                    buf.writestring("; ");
                    (cast(AssertExp)es.exp).e1.expressionToBuffer(buf, hgs);
                    buf.writeByte(')');
                    buf.writenl();
                    requireDo = false;
                }
                else
                {
                    if (fensure.id)
                    {
                        buf.writeByte('(');
                        buf.writestring(fensure.id.toString());
                        buf.writeByte(')');
                    }
                    buf.writenl();
                    fensure.ensure.statementToBuffer(buf, hgs);
                    requireDo = true;
                }
            }
        }
        if (requireDo)
        {
            buf.writestring("do");
            buf.writenl();
        }
        buf.writeByte('{');
        buf.writenl();
        buf.level++;
        f.fbody.statementToBuffer(buf, hgs);
        buf.level--;
        buf.writeByte('}');
        buf.writenl();
        hgs.tpltMember = savetlpt;
        hgs.autoMember = saveauto;
    }

    override void visit(FuncLiteralDeclaration f)
    {
        if (f.type.ty == Terror)
        {
            buf.writestring("__error");
            return;
        }
        if (f.tok != TOK.reserved)
        {
            buf.writestring(f.kind());
            buf.writeByte(' ');
        }
        TypeFunction tf = cast(TypeFunction)f.type;

        if (!f.inferRetType && tf.next)
            typeToBuffer(tf.next, null, buf, hgs);
        parametersToBuffer(tf.parameterList, buf, hgs);

        // https://issues.dlang.org/show_bug.cgi?id=20074
        void printAttribute(string str)
        {
            buf.writeByte(' ');
            buf.writestring(str);
        }
        tf.attributesApply(&printAttribute);


        CompoundStatement cs = f.fbody.isCompoundStatement();
        Statement s1;
        if (f.semanticRun >= PASS.semantic3done && cs)
        {
            s1 = (*cs.statements)[cs.statements.dim - 1];
        }
        else
            s1 = !cs ? f.fbody : null;
        ReturnStatement rs = s1 ? s1.endsWithReturnStatement() : null;
        if (rs && rs.exp)
        {
            buf.writestring(" => ");
            rs.exp.expressionToBuffer(buf, hgs);
        }
        else
        {
            hgs.tpltMember++;
            bodyToBuffer(f);
            hgs.tpltMember--;
        }
    }

    override void visit(PostBlitDeclaration d)
    {
        if (stcToBuffer(buf, d.storage_class))
            buf.writeByte(' ');
        buf.writestring("this(this)");
        bodyToBuffer(d);
    }

    override void visit(DtorDeclaration d)
    {
        if (d.storage_class & STC.trusted)
            buf.writestring("@trusted ");
        if (d.storage_class & STC.safe)
            buf.writestring("@safe ");
        if (d.storage_class & STC.nogc)
            buf.writestring("@nogc ");
        if (d.storage_class & STC.disable)
            buf.writestring("@disable ");

        buf.writestring("~this()");
        bodyToBuffer(d);
    }

    override void visit(StaticCtorDeclaration d)
    {
        if (stcToBuffer(buf, d.storage_class & ~STC.static_))
            buf.writeByte(' ');
        if (d.isSharedStaticCtorDeclaration())
            buf.writestring("shared ");
        buf.writestring("static this()");
        if (hgs.hdrgen && !hgs.tpltMember)
        {
            buf.writeByte(';');
            buf.writenl();
        }
        else
            bodyToBuffer(d);
    }

    override void visit(StaticDtorDeclaration d)
    {
        if (stcToBuffer(buf, d.storage_class & ~STC.static_))
            buf.writeByte(' ');
        if (d.isSharedStaticDtorDeclaration())
            buf.writestring("shared ");
        buf.writestring("static ~this()");
        if (hgs.hdrgen && !hgs.tpltMember)
        {
            buf.writeByte(';');
            buf.writenl();
        }
        else
            bodyToBuffer(d);
    }

    override void visit(InvariantDeclaration d)
    {
        if (hgs.hdrgen)
            return;
        if (stcToBuffer(buf, d.storage_class))
            buf.writeByte(' ');
        buf.writestring("invariant");
        if(auto es = d.fbody.isExpStatement())
        {
            assert(es.exp && es.exp.op == TOK.assert_);
            buf.writestring(" (");
            (cast(AssertExp)es.exp).e1.expressionToBuffer(buf, hgs);
            buf.writestring(");");
            buf.writenl();
        }
        else
        {
            bodyToBuffer(d);
        }
    }

    override void visit(UnitTestDeclaration d)
    {
        if (hgs.hdrgen)
            return;
        if (stcToBuffer(buf, d.storage_class))
            buf.writeByte(' ');
        buf.writestring("unittest");
        bodyToBuffer(d);
    }

    override void visit(NewDeclaration d)
    {
        if (stcToBuffer(buf, d.storage_class & ~STC.static_))
            buf.writeByte(' ');
        buf.writestring("new");
        parametersToBuffer(d.parameterList, buf, hgs);
        bodyToBuffer(d);
    }

    override void visit(Module m)
    {
        moduleToBuffer2(m, buf, hgs);
    }
}

private extern (C++) final class ExpressionPrettyPrintVisitor : Visitor
{
    alias visit = Visitor.visit;
public:
    OutBuffer* buf;
    HdrGenState* hgs;

    extern (D) this(OutBuffer* buf, HdrGenState* hgs)
    {
        this.buf = buf;
        this.hgs = hgs;
    }

    ////////////////////////////////////////////////////////////////////////////
    override void visit(Expression e)
    {
        buf.writestring(Token.toString(e.op));
    }

    override void visit(IntegerExp e)
    {
        const dinteger_t v = e.toInteger();
        if (e.type)
        {
            Type t = e.type;
        L1:
            switch (t.ty)
            {
            case Tenum:
                {
                    TypeEnum te = cast(TypeEnum)t;
                    if (hgs.fullDump)
                    {
                        auto sym = te.sym;
                        if (hgs.inEnumDecl && sym && hgs.inEnumDecl != sym)  foreach(i;0 .. sym.members.dim)
                        {
                            EnumMember em = cast(EnumMember) (*sym.members)[i];
                            if (em.value.toInteger == v)
                            {
                                buf.printf("%s.%s", sym.toChars(), em.ident.toChars());
                                return ;
                            }
                        }
                        //assert(0, "We could not find the EmumMember");// for some reason it won't append char* ~ e.toChars() ~ " in " ~ sym.toChars() );
                    }

                    buf.printf("cast(%s)", te.sym.toChars());
                    t = te.sym.memtype;
                    goto L1;
                }
            case Twchar:
                // BUG: need to cast(wchar)
            case Tdchar:
                // BUG: need to cast(dchar)
                if (cast(uinteger_t)v > 0xFF)
                {
                    buf.printf("'\\U%08llx'", cast(long)v);
                    break;
                }
                goto case;
            case Tchar:
                {
                    size_t o = buf.length;
                    if (v == '\'')
                        buf.writestring("'\\''");
                    else if (isprint(cast(int)v) && v != '\\')
                        buf.printf("'%c'", cast(int)v);
                    else
                        buf.printf("'\\x%02x'", cast(int)v);
                    if (hgs.ddoc)
                        escapeDdocString(buf, o);
                    break;
                }
            case Tint8:
                buf.writestring("cast(byte)");
                goto L2;
            case Tint16:
                buf.writestring("cast(short)");
                goto L2;
            case Tint32:
            L2:
                buf.printf("%d", cast(int)v);
                break;
            case Tuns8:
                buf.writestring("cast(ubyte)");
                goto case Tuns32;
            case Tuns16:
                buf.writestring("cast(ushort)");
                goto case Tuns32;
            case Tuns32:
                buf.printf("%uu", cast(uint)v);
                break;
            case Tint64:
                buf.printf("%lldL", v);
                break;
            case Tuns64:
                buf.printf("%lluLU", v);
                break;
            case Tbool:
                buf.writestring(v ? "true" : "false");
                break;
            case Tpointer:
                buf.writestring("cast(");
                buf.writestring(t.toChars());
                buf.writeByte(')');
                if (target.ptrsize == 8)
                    goto case Tuns64;
                else
                    goto case Tuns32;
            default:
                /* This can happen if errors, such as
                 * the type is painted on like in fromConstInitializer().
                 */
                if (!global.errors)
                {
                    assert(0);
                }
                break;
            }
        }
        else if (v & 0x8000000000000000L)
            buf.printf("0x%llx", v);
        else
            buf.print(v);
    }

    override void visit(ErrorExp e)
    {
        buf.writestring("__error");
    }

    override void visit(VoidInitExp e)
    {
        buf.writestring("__void");
    }

    void floatToBuffer(Type type, real_t value)
    {
        /** sizeof(value)*3 is because each byte of mantissa is max
         of 256 (3 characters). The string will be "-M.MMMMe-4932".
         (ie, 8 chars more than mantissa). Plus one for trailing \0.
         Plus one for rounding. */
        const(size_t) BUFFER_LEN = value.sizeof * 3 + 8 + 1 + 1;
        char[BUFFER_LEN] buffer;
        CTFloat.sprint(buffer.ptr, 'g', value);
        assert(strlen(buffer.ptr) < BUFFER_LEN);
        if (hgs.hdrgen)
        {
            real_t r = CTFloat.parse(buffer.ptr);
            if (r != value) // if exact duplication
                CTFloat.sprint(buffer.ptr, 'a', value);
        }
        buf.writestring(buffer.ptr);
        if (buffer.ptr[strlen(buffer.ptr) - 1] == '.')
            buf.remove(buf.length() - 1, 1);

        if (type)
        {
            Type t = type.toBasetype();
            switch (t.ty)
            {
            case Tfloat32:
            case Timaginary32:
            case Tcomplex32:
                buf.writeByte('F');
                break;
            case Tfloat80:
            case Timaginary80:
            case Tcomplex80:
                buf.writeByte('L');
                break;
            default:
                break;
            }
            if (t.isimaginary())
                buf.writeByte('i');
        }
    }

    override void visit(RealExp e)
    {
        floatToBuffer(e.type, e.value);
    }

    override void visit(ComplexExp e)
    {
        /* Print as:
         *  (re+imi)
         */
        buf.writeByte('(');
        floatToBuffer(e.type, creall(e.value));
        buf.writeByte('+');
        floatToBuffer(e.type, cimagl(e.value));
        buf.writestring("i)");
    }

    override void visit(IdentifierExp e)
    {
        if (hgs.hdrgen || hgs.ddoc)
            buf.writestring(e.ident.toHChars2());
        else
            buf.writestring(e.ident.toString());
    }

    override void visit(DsymbolExp e)
    {
        buf.writestring(e.s.toChars());
    }

    override void visit(ThisExp e)
    {
        buf.writestring("this");
    }

    override void visit(SuperExp e)
    {
        buf.writestring("super");
    }

    override void visit(NullExp e)
    {
        buf.writestring("null");
    }

    override void visit(StringExp e)
    {
        buf.writeByte('"');
        const o = buf.length;
        for (size_t i = 0; i < e.len; i++)
        {
            const c = e.charAt(i);
            switch (c)
            {
            case '"':
            case '\\':
                buf.writeByte('\\');
                goto default;
            default:
                if (c <= 0xFF)
                {
                    if (c <= 0x7F && isprint(c))
                        buf.writeByte(c);
                    else
                        buf.printf("\\x%02x", c);
                }
                else if (c <= 0xFFFF)
                    buf.printf("\\x%02x\\x%02x", c & 0xFF, c >> 8);
                else
                    buf.printf("\\x%02x\\x%02x\\x%02x\\x%02x", c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF, c >> 24);
                break;
            }
        }
        if (hgs.ddoc)
            escapeDdocString(buf, o);
        buf.writeByte('"');
        if (e.postfix)
            buf.writeByte(e.postfix);
    }

    override void visit(ArrayLiteralExp e)
    {
        buf.writeByte('[');
        argsToBuffer(e.elements, buf, hgs, e.basis);
        buf.writeByte(']');
    }

    override void visit(AssocArrayLiteralExp e)
    {
        buf.writeByte('[');
        foreach (i, key; *e.keys)
        {
            if (i)
                buf.writestring(", ");
            expToBuffer(key, PREC.assign, buf, hgs);
            buf.writeByte(':');
            auto value = (*e.values)[i];
            expToBuffer(value, PREC.assign, buf, hgs);
        }
        buf.writeByte(']');
    }

    override void visit(StructLiteralExp e)
    {
        buf.writestring(e.sd.toChars());
        buf.writeByte('(');
        // CTFE can generate struct literals that contain an AddrExp pointing
        // to themselves, need to avoid infinite recursion:
        // struct S { this(int){ this.s = &this; } S* s; }
        // const foo = new S(0);
        if (e.stageflags & stageToCBuffer)
            buf.writestring("<recursion>");
        else
        {
            const old = e.stageflags;
            e.stageflags |= stageToCBuffer;
            argsToBuffer(e.elements, buf, hgs);
            e.stageflags = old;
        }
        buf.writeByte(')');
    }

    override void visit(TypeExp e)
    {
        typeToBuffer(e.type, null, buf, hgs);
    }

    override void visit(ScopeExp e)
    {
        if (e.sds.isTemplateInstance())
        {
            e.sds.dsymbolToBuffer(buf, hgs);
        }
        else if (hgs !is null && hgs.ddoc)
        {
            // fixes bug 6491
            if (auto m = e.sds.isModule())
                buf.writestring(m.md.toChars());
            else
                buf.writestring(e.sds.toChars());
        }
        else
        {
            buf.writestring(e.sds.kind());
            buf.writeByte(' ');
            buf.writestring(e.sds.toChars());
        }
    }

    override void visit(TemplateExp e)
    {
        buf.writestring(e.td.toChars());
    }

    override void visit(NewExp e)
    {
        if (e.thisexp)
        {
            expToBuffer(e.thisexp, PREC.primary, buf, hgs);
            buf.writeByte('.');
        }
        buf.writestring("new ");
        if (e.newargs && e.newargs.dim)
        {
            buf.writeByte('(');
            argsToBuffer(e.newargs, buf, hgs);
            buf.writeByte(')');
        }
        typeToBuffer(e.newtype, null, buf, hgs);
        if (e.arguments && e.arguments.dim)
        {
            buf.writeByte('(');
            argsToBuffer(e.arguments, buf, hgs);
            buf.writeByte(')');
        }
    }

    override void visit(NewAnonClassExp e)
    {
        if (e.thisexp)
        {
            expToBuffer(e.thisexp, PREC.primary, buf, hgs);
            buf.writeByte('.');
        }
        buf.writestring("new");
        if (e.newargs && e.newargs.dim)
        {
            buf.writeByte('(');
            argsToBuffer(e.newargs, buf, hgs);
            buf.writeByte(')');
        }
        buf.writestring(" class ");
        if (e.arguments && e.arguments.dim)
        {
            buf.writeByte('(');
            argsToBuffer(e.arguments, buf, hgs);
            buf.writeByte(')');
        }
        if (e.cd)
            e.cd.dsymbolToBuffer(buf, hgs);
    }

    override void visit(SymOffExp e)
    {
        if (e.offset)
            buf.printf("(& %s%+lld)", e.var.toChars(), e.offset);
        else if (e.var.isTypeInfoDeclaration())
            buf.writestring(e.var.toChars());
        else
            buf.printf("& %s", e.var.toChars());
    }

    override void visit(VarExp e)
    {
        buf.writestring(e.var.toChars());
    }

    override void visit(OverExp e)
    {
        buf.writestring(e.vars.ident.toString());
    }

    override void visit(TupleExp e)
    {
        if (e.e0)
        {
            buf.writeByte('(');
            e.e0.accept(this);
            buf.writestring(", tuple(");
            argsToBuffer(e.exps, buf, hgs);
            buf.writestring("))");
        }
        else
        {
            buf.writestring("tuple(");
            argsToBuffer(e.exps, buf, hgs);
            buf.writeByte(')');
        }
    }

    override void visit(FuncExp e)
    {
        e.fd.dsymbolToBuffer(buf, hgs);
        //buf.writestring(e.fd.toChars());
    }

    override void visit(DeclarationExp e)
    {
        /* Normal dmd execution won't reach here - regular variable declarations
         * are handled in visit(ExpStatement), so here would be used only when
         * we'll directly call Expression.toChars() for debugging.
         */
        if (e.declaration)
        {
            if (auto var = e.declaration.isVarDeclaration())
            {
            // For debugging use:
            // - Avoid printing newline.
            // - Intentionally use the format (Type var;)
            //   which isn't correct as regular D code.
                buf.writeByte('(');

                scope v = new DsymbolPrettyPrintVisitor(buf, hgs);
                v.visitVarDecl(var, false);

                buf.writeByte(';');
                buf.writeByte(')');
            }
            else e.declaration.dsymbolToBuffer(buf, hgs);
        }
    }

    override void visit(TypeidExp e)
    {
        buf.writestring("typeid(");
        objectToBuffer(e.obj, buf, hgs);
        buf.writeByte(')');
    }

    override void visit(TraitsExp e)
    {
        buf.writestring("__traits(");
        if (e.ident)
            buf.writestring(e.ident.toString());
        if (e.args)
        {
            foreach (arg; *e.args)
            {
                buf.writestring(", ");
                objectToBuffer(arg, buf, hgs);
            }
        }
        buf.writeByte(')');
    }

    override void visit(HaltExp e)
    {
        buf.writestring("halt");
    }

    override void visit(IsExp e)
    {
        buf.writestring("is(");
        typeToBuffer(e.targ, e.id, buf, hgs);
        if (e.tok2 != TOK.reserved)
        {
            buf.printf(" %s %s", Token.toChars(e.tok), Token.toChars(e.tok2));
        }
        else if (e.tspec)
        {
            if (e.tok == TOK.colon)
                buf.writestring(" : ");
            else
                buf.writestring(" == ");
            typeToBuffer(e.tspec, null, buf, hgs);
        }
        if (e.parameters && e.parameters.dim)
        {
            buf.writestring(", ");
            scope v = new DsymbolPrettyPrintVisitor(buf, hgs);
            v.visitTemplateParameters(e.parameters);
        }
        buf.writeByte(')');
    }

    override void visit(UnaExp e)
    {
        buf.writestring(Token.toString(e.op));
        expToBuffer(e.e1, precedence[e.op], buf, hgs);
    }

    override void visit(BinExp e)
    {
        expToBuffer(e.e1, precedence[e.op], buf, hgs);
        buf.writeByte(' ');
        buf.writestring(Token.toString(e.op));
        buf.writeByte(' ');
        expToBuffer(e.e2, cast(PREC)(precedence[e.op] + 1), buf, hgs);
    }

    override void visit(CommaExp e)
    {
        // CommaExp is generated by the compiler so it shouldn't
        // appear in error messages or header files.
        // For now, this treats the case where the compiler
        // generates CommaExp for temporaries by calling
        // the `sideeffect.copyToTemp` function.
        auto ve = e.e2.isVarExp();

        // not a CommaExp introduced for temporaries, go on
        // the old path
        if (!ve || !(ve.var.storage_class & STC.temp))
        {
            visit(cast(BinExp)e);
            return;
        }

        // CommaExp that contain temporaries inserted via
        // `copyToTemp` are usually of the form
        // ((T __temp = exp), __tmp).
        // Asserts are here to easily spot
        // missing cases where CommaExp
        // are used for other constructs
        auto vd = ve.var.isVarDeclaration();
        assert(vd && vd._init);
        auto exp = vd._init.isExpInitializer.exp;
        assert(exp);
        Expression commaExtract;
        if (auto ce = exp.isConstructExp())
            commaExtract = ce.e2;
        else if (auto se = exp.isStructLiteralExp)
            commaExtract = se;

        // not one of the known cases, go on the old path
        if (!commaExtract)
        {
            visit(cast(BinExp)e);
            return;
        }
        expToBuffer(commaExtract, precedence[exp.op], buf, hgs);
    }

    override void visit(CompileExp e)
    {
        buf.writestring("mixin(");
        argsToBuffer(e.exps, buf, hgs, null);
        buf.writeByte(')');
    }

    override void visit(ImportExp e)
    {
        buf.writestring("import(");
        expToBuffer(e.e1, PREC.assign, buf, hgs);
        buf.writeByte(')');
    }

    override void visit(AssertExp e)
    {
        buf.writestring("assert(");
        expToBuffer(e.e1, PREC.assign, buf, hgs);
        if (e.msg)
        {
            buf.writestring(", ");
            expToBuffer(e.msg, PREC.assign, buf, hgs);
        }
        buf.writeByte(')');
    }

    override void visit(DotIdExp e)
    {
        expToBuffer(e.e1, PREC.primary, buf, hgs);
        buf.writeByte('.');
        buf.writestring(e.ident.toString());
    }

    override void visit(DotTemplateExp e)
    {
        expToBuffer(e.e1, PREC.primary, buf, hgs);
        buf.writeByte('.');
        buf.writestring(e.td.toChars());
    }

    override void visit(DotVarExp e)
    {
        expToBuffer(e.e1, PREC.primary, buf, hgs);
        buf.writeByte('.');
        buf.writestring(e.var.toChars());
    }

    override void visit(DotTemplateInstanceExp e)
    {
        expToBuffer(e.e1, PREC.primary, buf, hgs);
        buf.writeByte('.');
        e.ti.dsymbolToBuffer(buf, hgs);
    }

    override void visit(DelegateExp e)
    {
        buf.writeByte('&');
        if (!e.func.isNested() || e.func.needThis())
        {
            expToBuffer(e.e1, PREC.primary, buf, hgs);
            buf.writeByte('.');
        }
        buf.writestring(e.func.toChars());
    }

    override void visit(DotTypeExp e)
    {
        expToBuffer(e.e1, PREC.primary, buf, hgs);
        buf.writeByte('.');
        buf.writestring(e.sym.toChars());
    }

    override void visit(CallExp e)
    {
        if (e.e1.op == TOK.type)
        {
            /* Avoid parens around type to prevent forbidden cast syntax:
             *   (sometype)(arg1)
             * This is ok since types in constructor calls
             * can never depend on parens anyway
             */
            e.e1.accept(this);
        }
        else
            expToBuffer(e.e1, precedence[e.op], buf, hgs);
        buf.writeByte('(');
        argsToBuffer(e.arguments, buf, hgs);
        buf.writeByte(')');
    }

    override void visit(PtrExp e)
    {
        buf.writeByte('*');
        expToBuffer(e.e1, precedence[e.op], buf, hgs);
    }

    override void visit(DeleteExp e)
    {
        buf.writestring("delete ");
        expToBuffer(e.e1, precedence[e.op], buf, hgs);
    }

    override void visit(CastExp e)
    {
        buf.writestring("cast(");
        if (e.to)
            typeToBuffer(e.to, null, buf, hgs);
        else
        {
            MODtoBuffer(buf, e.mod);
        }
        buf.writeByte(')');
        expToBuffer(e.e1, precedence[e.op], buf, hgs);
    }

    override void visit(VectorExp e)
    {
        buf.writestring("cast(");
        typeToBuffer(e.to, null, buf, hgs);
        buf.writeByte(')');
        expToBuffer(e.e1, precedence[e.op], buf, hgs);
    }

    override void visit(VectorArrayExp e)
    {
        expToBuffer(e.e1, PREC.primary, buf, hgs);
        buf.writestring(".array");
    }

    override void visit(SliceExp e)
    {
        expToBuffer(e.e1, precedence[e.op], buf, hgs);
        buf.writeByte('[');
        if (e.upr || e.lwr)
        {
            if (e.lwr)
                sizeToBuffer(e.lwr, buf, hgs);
            else
                buf.writeByte('0');
            buf.writestring("..");
            if (e.upr)
                sizeToBuffer(e.upr, buf, hgs);
            else
                buf.writeByte('$');
        }
        buf.writeByte(']');
    }

    override void visit(ArrayLengthExp e)
    {
        expToBuffer(e.e1, PREC.primary, buf, hgs);
        buf.writestring(".length");
    }

    override void visit(IntervalExp e)
    {
        expToBuffer(e.lwr, PREC.assign, buf, hgs);
        buf.writestring("..");
        expToBuffer(e.upr, PREC.assign, buf, hgs);
    }

    override void visit(DelegatePtrExp e)
    {
        expToBuffer(e.e1, PREC.primary, buf, hgs);
        buf.writestring(".ptr");
    }

    override void visit(DelegateFuncptrExp e)
    {
        expToBuffer(e.e1, PREC.primary, buf, hgs);
        buf.writestring(".funcptr");
    }

    override void visit(ArrayExp e)
    {
        expToBuffer(e.e1, PREC.primary, buf, hgs);
        buf.writeByte('[');
        argsToBuffer(e.arguments, buf, hgs);
        buf.writeByte(']');
    }

    override void visit(DotExp e)
    {
        expToBuffer(e.e1, PREC.primary, buf, hgs);
        buf.writeByte('.');
        expToBuffer(e.e2, PREC.primary, buf, hgs);
    }

    override void visit(IndexExp e)
    {
        expToBuffer(e.e1, PREC.primary, buf, hgs);
        buf.writeByte('[');
        sizeToBuffer(e.e2, buf, hgs);
        buf.writeByte(']');
    }

    override void visit(PostExp e)
    {
        expToBuffer(e.e1, precedence[e.op], buf, hgs);
        buf.writestring(Token.toString(e.op));
    }

    override void visit(PreExp e)
    {
        buf.writestring(Token.toString(e.op));
        expToBuffer(e.e1, precedence[e.op], buf, hgs);
    }

    override void visit(RemoveExp e)
    {
        expToBuffer(e.e1, PREC.primary, buf, hgs);
        buf.writestring(".remove(");
        expToBuffer(e.e2, PREC.assign, buf, hgs);
        buf.writeByte(')');
    }

    override void visit(CondExp e)
    {
        expToBuffer(e.econd, PREC.oror, buf, hgs);
        buf.writestring(" ? ");
        expToBuffer(e.e1, PREC.expr, buf, hgs);
        buf.writestring(" : ");
        expToBuffer(e.e2, PREC.cond, buf, hgs);
    }

    override void visit(DefaultInitExp e)
    {
        buf.writestring(Token.toString(e.subop));
    }

    override void visit(ClassReferenceExp e)
    {
        buf.writestring(e.value.toChars());
    }
}


private void templateParameterToBuffer(TemplateParameter tp, OutBuffer* buf, HdrGenState* hgs)
{
    scope v = new TemplateParameterPrettyPrintVisitor(buf, hgs);
    tp.accept(v);
}

private extern (C++) final class TemplateParameterPrettyPrintVisitor : Visitor
{
    alias visit = Visitor.visit;
public:
    OutBuffer* buf;
    HdrGenState* hgs;

    extern (D) this(OutBuffer* buf, HdrGenState* hgs)
    {
        this.buf = buf;
        this.hgs = hgs;
    }

    override void visit(TemplateTypeParameter tp)
    {
        buf.writestring(tp.ident.toString());
        if (tp.specType)
        {
            buf.writestring(" : ");
            typeToBuffer(tp.specType, null, buf, hgs);
        }
        if (tp.defaultType)
        {
            buf.writestring(" = ");
            typeToBuffer(tp.defaultType, null, buf, hgs);
        }
    }

    override void visit(TemplateThisParameter tp)
    {
        buf.writestring("this ");
        visit(cast(TemplateTypeParameter)tp);
    }

    override void visit(TemplateAliasParameter tp)
    {
        buf.writestring("alias ");
        if (tp.specType)
            typeToBuffer(tp.specType, tp.ident, buf, hgs);
        else
            buf.writestring(tp.ident.toString());
        if (tp.specAlias)
        {
            buf.writestring(" : ");
            objectToBuffer(tp.specAlias, buf, hgs);
        }
        if (tp.defaultAlias)
        {
            buf.writestring(" = ");
            objectToBuffer(tp.defaultAlias, buf, hgs);
        }
    }

    override void visit(TemplateValueParameter tp)
    {
        typeToBuffer(tp.valType, tp.ident, buf, hgs);
        if (tp.specValue)
        {
            buf.writestring(" : ");
            tp.specValue.expressionToBuffer(buf, hgs);
        }
        if (tp.defaultValue)
        {
            buf.writestring(" = ");
            tp.defaultValue.expressionToBuffer(buf, hgs);
        }
    }

    override void visit(TemplateTupleParameter tp)
    {
        buf.writestring(tp.ident.toString());
        buf.writestring("...");
    }
}

private void conditionToBuffer(Condition c, OutBuffer* buf, HdrGenState* hgs)
{
    scope v = new ConditionPrettyPrintVisitor(buf, hgs);
    c.accept(v);
}

private extern (C++) final class ConditionPrettyPrintVisitor : Visitor
{
    alias visit = Visitor.visit;
public:
    OutBuffer* buf;
    HdrGenState* hgs;

    extern (D) this(OutBuffer* buf, HdrGenState* hgs)
    {
        this.buf = buf;
        this.hgs = hgs;
    }

    override void visit(DebugCondition c)
    {
        buf.writestring("debug (");
        if (c.ident)
            buf.writestring(c.ident.toString());
        else
            buf.print(c.level);
        buf.writeByte(')');
    }

    override void visit(VersionCondition c)
    {
        buf.writestring("version (");
        if (c.ident)
            buf.writestring(c.ident.toString());
        else
            buf.print(c.level);
        buf.writeByte(')');
    }

    override void visit(StaticIfCondition c)
    {
        buf.writestring("static if (");
        c.exp.expressionToBuffer(buf, hgs);
        buf.writeByte(')');
    }
}

void toCBuffer(const Statement s, OutBuffer* buf, HdrGenState* hgs)
{
    scope v = new StatementPrettyPrintVisitor(buf, hgs);
    (cast() s).accept(v);
}

void toCBuffer(const Type t, OutBuffer* buf, const Identifier ident, HdrGenState* hgs)
{
    typeToBuffer(cast() t, ident, buf, hgs);
}

void toCBuffer(Dsymbol s, OutBuffer* buf, HdrGenState* hgs)
{
    scope v = new DsymbolPrettyPrintVisitor(buf, hgs);
    s.accept(v);
}

// used from TemplateInstance::toChars() and TemplateMixin::toChars()
void toCBufferInstance(const TemplateInstance ti, OutBuffer* buf, bool qualifyTypes = false)
{
    HdrGenState hgs;
    hgs.fullQual = qualifyTypes;
    scope v = new DsymbolPrettyPrintVisitor(buf, &hgs);
    v.visit(cast() ti);
}

void toCBuffer(const Initializer iz, OutBuffer* buf, HdrGenState* hgs)
{
    initializerToBuffer(cast() iz, buf, hgs);
}

bool stcToBuffer(OutBuffer* buf, StorageClass stc)
{
    bool result = false;
    if ((stc & (STC.return_ | STC.scope_)) == (STC.return_ | STC.scope_))
        stc &= ~STC.scope_;
    if (stc & STC.scopeinferred)
        stc &= ~(STC.scope_ | STC.scopeinferred);
    while (stc)
    {
        const s = stcToString(stc);
        if (!s.length)
            break;
        if (result)
            buf.writeByte(' ');
        result = true;
        buf.writestring(s);
    }
    return result;
}

/*************************************************
 * Pick off one of the storage classes from stc,
 * and return a string representation of it.
 * stc is reduced by the one picked.
 */
string stcToString(ref StorageClass stc)
{
    struct SCstring
    {
        StorageClass stc;
        TOK tok;
        string id;
    }

    __gshared SCstring* table =
    [
        SCstring(STC.auto_, TOK.auto_),
        SCstring(STC.scope_, TOK.scope_),
        SCstring(STC.static_, TOK.static_),
        SCstring(STC.extern_, TOK.extern_),
        SCstring(STC.const_, TOK.const_),
        SCstring(STC.final_, TOK.final_),
        SCstring(STC.abstract_, TOK.abstract_),
        SCstring(STC.synchronized_, TOK.synchronized_),
        SCstring(STC.deprecated_, TOK.deprecated_),
        SCstring(STC.override_, TOK.override_),
        SCstring(STC.lazy_, TOK.lazy_),
        SCstring(STC.alias_, TOK.alias_),
        SCstring(STC.out_, TOK.out_),
        SCstring(STC.in_, TOK.in_),
        SCstring(STC.manifest, TOK.enum_),
        SCstring(STC.immutable_, TOK.immutable_),
        SCstring(STC.shared_, TOK.shared_),
        SCstring(STC.nothrow_, TOK.nothrow_),
        SCstring(STC.wild, TOK.inout_),
        SCstring(STC.pure_, TOK.pure_),
        SCstring(STC.ref_, TOK.ref_),
        SCstring(STC.return_, TOK.return_),
        SCstring(STC.tls),
        SCstring(STC.gshared, TOK.gshared),
        SCstring(STC.nogc, TOK.at, "@nogc"),
        SCstring(STC.property, TOK.at, "@property"),
        SCstring(STC.safe, TOK.at, "@safe"),
        SCstring(STC.trusted, TOK.at, "@trusted"),
        SCstring(STC.system, TOK.at, "@system"),
        SCstring(STC.disable, TOK.at, "@disable"),
        SCstring(STC.future, TOK.at, "@__future"),
        SCstring(STC.local, TOK.at, "__local"),
        SCstring(0, TOK.reserved)
    ];
    for (int i = 0; table[i].stc; i++)
    {
        StorageClass tbl = table[i].stc;
        assert(tbl & STCStorageClass);
        if (stc & tbl)
        {
            stc &= ~tbl;
            if (tbl == STC.tls) // TOKtls was removed
                return "__thread";
            TOK tok = table[i].tok;
            if (tok != TOK.at && !table[i].id.length)
                table[i].id = Token.toString(tok); // lazilly initialize table
            return table[i].id;
        }
    }
    //printf("stc = %llx\n", stc);
    return null;
}

const(char)* stcToChars(ref StorageClass stc)
{
    const s = stcToString(stc);
    return &s[0];  // assume 0 terminated
}


/// Ditto
extern (D) string trustToString(TRUST trust) pure nothrow
{
    final switch (trust)
    {
    case TRUST.default_:
        return null;
    case TRUST.system:
        return "@system";
    case TRUST.trusted:
        return "@trusted";
    case TRUST.safe:
        return "@safe";
    }
}

private void linkageToBuffer(OutBuffer* buf, LINK linkage)
{
    const s = linkageToString(linkage);
    if (s.length)
    {
        buf.writestring("extern (");
        buf.writestring(s);
        buf.writeByte(')');
    }
}

const(char)* linkageToChars(LINK linkage)
{
    /// Works because we return a literal
    return linkageToString(linkage).ptr;
}

string linkageToString(LINK linkage) pure nothrow
{
    final switch (linkage)
    {
    case LINK.default_:
        return null;
    case LINK.d:
        return "D";
    case LINK.c:
        return "C";
    case LINK.cpp:
        return "C++";
    case LINK.windows:
        return "Windows";
    case LINK.pascal:
        return "Pascal";
    case LINK.objc:
        return "Objective-C";
    case LINK.system:
        return "System";
    }
}

void protectionToBuffer(OutBuffer* buf, Prot prot)
{
    buf.writestring(protectionToString(prot.kind));
    if (prot.kind == Prot.Kind.package_ && prot.pkg)
    {
        buf.writeByte('(');
        buf.writestring(prot.pkg.toPrettyChars(true));
        buf.writeByte(')');
    }
}

/**
 * Returns:
 *   a human readable representation of `kind`
 */
const(char)* protectionToChars(Prot.Kind kind)
{
    // Null terminated because we return a literal
    return protectionToString(kind).ptr;
}

/// Ditto
extern (D) string protectionToString(Prot.Kind kind) nothrow pure
{
    final switch (kind)
    {
    case Prot.Kind.undefined:
        return null;
    case Prot.Kind.none:
        return "none";
    case Prot.Kind.private_:
        return "private";
    case Prot.Kind.package_:
        return "package";
    case Prot.Kind.protected_:
        return "protected";
    case Prot.Kind.public_:
        return "public";
    case Prot.Kind.export_:
        return "export";
    }
}

// Print the full function signature with correct ident, attributes and template args
void functionToBufferFull(TypeFunction tf, OutBuffer* buf, const Identifier ident, HdrGenState* hgs, TemplateDeclaration td)
{
    //printf("TypeFunction::toCBuffer() this = %p\n", this);
    visitFuncIdentWithPrefix(tf, ident, td, buf, hgs);
}

// ident is inserted before the argument list and will be "function" or "delegate" for a type
void functionToBufferWithIdent(TypeFunction tf, OutBuffer* buf, const(char)* ident)
{
    HdrGenState hgs;
    visitFuncIdentWithPostfix(tf, ident.toDString(), buf, &hgs);
}

void toCBuffer(const Expression e, OutBuffer* buf, HdrGenState* hgs)
{
    scope v = new ExpressionPrettyPrintVisitor(buf, hgs);
    (cast() e).accept(v);
}

/**************************************************
 * Write out argument types to buf.
 */
void argExpTypesToCBuffer(OutBuffer* buf, Expressions* arguments)
{
    if (!arguments || !arguments.dim)
        return;
    HdrGenState hgs;
    foreach (i, arg; *arguments)
    {
        if (i)
            buf.writestring(", ");
        typeToBuffer(arg.type, null, buf, &hgs);
    }
}

void toCBuffer(const TemplateParameter tp, OutBuffer* buf, HdrGenState* hgs)
{
    scope v = new TemplateParameterPrettyPrintVisitor(buf, hgs);
    (cast() tp).accept(v);
}

void arrayObjectsToBuffer(OutBuffer* buf, Objects* objects)
{
    if (!objects || !objects.dim)
        return;
    HdrGenState hgs;
    foreach (i, o; *objects)
    {
        if (i)
            buf.writestring(", ");
        objectToBuffer(o, buf, &hgs);
    }
}

/*************************************************************
 * Pretty print function parameters.
 * Params:
 *  pl = parameter list to print
 * Returns: Null-terminated string representing parameters.
 */
extern (C++) const(char)* parametersTypeToChars(ParameterList pl)
{
    OutBuffer buf;
    HdrGenState hgs;
    parametersToBuffer(pl, &buf, &hgs);
    return buf.extractChars();
}

/*************************************************************
 * Pretty print function parameter.
 * Params:
 *  parameter = parameter to print.
 *  tf = TypeFunction which holds parameter.
 *  fullQual = whether to fully qualify types.
 * Returns: Null-terminated string representing parameters.
 */
const(char)* parameterToChars(Parameter parameter, TypeFunction tf, bool fullQual)
{
    OutBuffer buf;
    HdrGenState hgs;
    hgs.fullQual = fullQual;

    parameterToBuffer(parameter, &buf, &hgs);

    if (tf.parameterList.varargs == VarArg.typesafe && parameter == tf.parameterList[tf.parameterList.parameters.dim - 1])
    {
        buf.writestring("...");
    }
    return buf.extractChars();
}


/*************************************************
 * Write ParameterList to buffer.
 * Params:
 *      pl = parameter list to serialize
 *      buf = buffer to write it to
 *      hgs = context
 */

private void parametersToBuffer(ParameterList pl, OutBuffer* buf, HdrGenState* hgs)
{
    buf.writeByte('(');
    foreach (i; 0 .. pl.length)
    {
        if (i)
            buf.writestring(", ");
        pl[i].parameterToBuffer(buf, hgs);
    }
    final switch (pl.varargs)
    {
        case VarArg.none:
            break;

        case VarArg.variadic:
            if (pl.length)
                buf.writestring(", ");

            if (stcToBuffer(buf, pl.stc))
                buf.writeByte(' ');
            goto case VarArg.typesafe;

        case VarArg.typesafe:
            buf.writestring("...");
            break;
    }
    buf.writeByte(')');
}


/***********************************************************
 * Write parameter `p` to buffer `buf`.
 * Params:
 *      p = parameter to serialize
 *      buf = buffer to write it to
 *      hgs = context
 */
private void parameterToBuffer(Parameter p, OutBuffer* buf, HdrGenState* hgs)
{
    if (p.userAttribDecl)
    {
        buf.writeByte('@');

        bool isAnonymous = p.userAttribDecl.atts.dim > 0 && (*p.userAttribDecl.atts)[0].op != TOK.call;
        if (isAnonymous)
            buf.writeByte('(');

        argsToBuffer(p.userAttribDecl.atts, buf, hgs);

        if (isAnonymous)
            buf.writeByte(')');
        buf.writeByte(' ');
    }
    if (p.storageClass & STC.auto_)
        buf.writestring("auto ");
    if (p.storageClass & STC.return_)
        buf.writestring("return ");

    if (p.storageClass & STC.out_)
        buf.writestring("out ");
    else if (p.storageClass & STC.ref_)
        buf.writestring("ref ");
    else if (p.storageClass & STC.in_)
        buf.writestring("in ");
    else if (p.storageClass & STC.lazy_)
        buf.writestring("lazy ");
    else if (p.storageClass & STC.alias_)
        buf.writestring("alias ");

    StorageClass stc = p.storageClass;
    if (p.type && p.type.mod & MODFlags.shared_)
        stc &= ~STC.shared_;

    if (stcToBuffer(buf, stc & (STC.const_ | STC.immutable_ | STC.wild | STC.shared_ | STC.scope_ | STC.scopeinferred)))
        buf.writeByte(' ');

    if (p.storageClass & STC.alias_)
    {
        if (p.ident)
            buf.writestring(p.ident.toString());
    }
    else if (p.type.ty == Tident &&
             (cast(TypeIdentifier)p.type).ident.toString().length > 3 &&
             strncmp((cast(TypeIdentifier)p.type).ident.toChars(), "__T", 3) == 0)
    {
        // print parameter name, instead of undetermined type parameter
        buf.writestring(p.ident.toString());
    }
    else
    {
        typeToBuffer(p.type, p.ident, buf, hgs);
    }

    if (p.defaultArg)
    {
        buf.writestring(" = ");
        p.defaultArg.expToBuffer(PREC.assign, buf, hgs);
    }
}


/**************************************************
 * Write out argument list to buf.
 */
private void argsToBuffer(Expressions* expressions, OutBuffer* buf, HdrGenState* hgs, Expression basis = null)
{
    if (!expressions || !expressions.dim)
        return;
    version (all)
    {
        foreach (i, el; *expressions)
        {
            if (i)
                buf.writestring(", ");
            if (!el)
                el = basis;
            if (el)
                expToBuffer(el, PREC.assign, buf, hgs);
        }
    }
    else
    {
        // Sparse style formatting, for debug use only
        //      [0..dim: basis, 1: e1, 5: e5]
        if (basis)
        {
            buf.writestring("0..");
            buf.print(expressions.dim);
            buf.writestring(": ");
            expToBuffer(basis, PREC.assign, buf, hgs);
        }
        foreach (i, el; *expressions)
        {
            if (el)
            {
                if (basis)
                {
                    buf.writestring(", ");
                    buf.print(i);
                    buf.writestring(": ");
                }
                else if (i)
                    buf.writestring(", ");
                expToBuffer(el, PREC.assign, buf, hgs);
            }
        }
    }
}

private void sizeToBuffer(Expression e, OutBuffer* buf, HdrGenState* hgs)
{
    if (e.type == Type.tsize_t)
    {
        Expression ex = (e.op == TOK.cast_ ? (cast(CastExp)e).e1 : e);
        ex = ex.optimize(WANTvalue);
        const dinteger_t uval = ex.op == TOK.int64 ? ex.toInteger() : cast(dinteger_t)-1;
        if (cast(sinteger_t)uval >= 0)
        {
            dinteger_t sizemax = void;
            if (target.ptrsize == 8)
                sizemax = 0xFFFFFFFFFFFFFFFFUL;
            else if (target.ptrsize == 4)
                sizemax = 0xFFFFFFFFU;
            else if (target.ptrsize == 2)
                sizemax = 0xFFFFU;
            else
                assert(0);
            if (uval <= sizemax && uval <= 0x7FFFFFFFFFFFFFFFUL)
            {
                buf.print(uval);
                return;
            }
        }
    }
    expToBuffer(e, PREC.assign, buf, hgs);
}

private void expressionToBuffer(Expression e, OutBuffer* buf, HdrGenState* hgs)
{
    scope v = new ExpressionPrettyPrintVisitor(buf, hgs);
    e.accept(v);
}

/**************************************************
 * Write expression out to buf, but wrap it
 * in ( ) if its precedence is less than pr.
 */
private void expToBuffer(Expression e, PREC pr, OutBuffer* buf, HdrGenState* hgs)
{
    debug
    {
        if (precedence[e.op] == PREC.zero)
            printf("precedence not defined for token '%s'\n", Token.toChars(e.op));
    }
    if (e.op == 0xFF)
    {
        buf.writestring("<FF>");
        return;
    }
    assert(precedence[e.op] != PREC.zero);
    assert(pr != PREC.zero);
    /* Despite precedence, we don't allow a<b<c expressions.
     * They must be parenthesized.
     */
    if (precedence[e.op] < pr || (pr == PREC.rel && precedence[e.op] == pr)
        || (pr >= PREC.or && pr <= PREC.and && precedence[e.op] == PREC.rel))
    {
        buf.writeByte('(');
        e.expressionToBuffer(buf, hgs);
        buf.writeByte(')');
    }
    else
    {
        e.expressionToBuffer(buf, hgs);
    }
}


/**************************************************
 * An entry point to pretty-print type.
 */
private void typeToBuffer(Type t, const Identifier ident, OutBuffer* buf, HdrGenState* hgs)
{
    if (auto tf = t.isTypeFunction())
    {
        visitFuncIdentWithPrefix(tf, ident, null, buf, hgs);
        return;
    }
    visitWithMask(t, 0, buf, hgs);
    if (ident)
    {
        buf.writeByte(' ');
        buf.writestring(ident.toString());
    }
}

private void visitWithMask(Type t, ubyte modMask, OutBuffer* buf, HdrGenState* hgs)
{
    // Tuples and functions don't use the type constructor syntax
    if (modMask == t.mod || t.ty == Tfunction || t.ty == Ttuple)
    {
        typeToBufferx(t, buf, hgs);
    }
    else
    {
        ubyte m = t.mod & ~(t.mod & modMask);
        if (m & MODFlags.shared_)
        {
            MODtoBuffer(buf, MODFlags.shared_);
            buf.writeByte('(');
        }
        if (m & MODFlags.wild)
        {
            MODtoBuffer(buf, MODFlags.wild);
            buf.writeByte('(');
        }
        if (m & (MODFlags.const_ | MODFlags.immutable_))
        {
            MODtoBuffer(buf, m & (MODFlags.const_ | MODFlags.immutable_));
            buf.writeByte('(');
        }
        typeToBufferx(t, buf, hgs);
        if (m & (MODFlags.const_ | MODFlags.immutable_))
            buf.writeByte(')');
        if (m & MODFlags.wild)
            buf.writeByte(')');
        if (m & MODFlags.shared_)
            buf.writeByte(')');
    }
}


private void dumpTemplateInstance(TemplateInstance ti, OutBuffer* buf, HdrGenState* hgs)
{
    buf.writeByte('{');
    buf.writenl();
    buf.level++;

    if (ti.aliasdecl)
    {
        ti.aliasdecl.dsymbolToBuffer(buf, hgs);
        buf.writenl();
    }
    else if (ti.members)
    {
        foreach(m;*ti.members)
            m.dsymbolToBuffer(buf, hgs);
    }

    buf.level--;
    buf.writeByte('}');
    buf.writenl();

}

private void tiargsToBuffer(TemplateInstance ti, OutBuffer* buf, HdrGenState* hgs)
{
    buf.writeByte('!');
    if (ti.nest)
    {
        buf.writestring("(...)");
        return;
    }
    if (!ti.tiargs)
    {
        buf.writestring("()");
        return;
    }
    if (ti.tiargs.dim == 1)
    {
        RootObject oarg = (*ti.tiargs)[0];
        if (Type t = isType(oarg))
        {
            if (t.equals(Type.tstring) || t.equals(Type.twstring) || t.equals(Type.tdstring) || t.mod == 0 && (t.isTypeBasic() || t.ty == Tident && (cast(TypeIdentifier)t).idents.dim == 0))
            {
                buf.writestring(t.toChars());
                return;
            }
        }
        else if (Expression e = isExpression(oarg))
        {
            if (e.op == TOK.int64 || e.op == TOK.float64 || e.op == TOK.null_ || e.op == TOK.string_ || e.op == TOK.this_)
            {
                buf.writestring(e.toChars());
                return;
            }
        }
    }
    buf.writeByte('(');
    ti.nest++;
    foreach (i, arg; *ti.tiargs)
    {
        if (i)
            buf.writestring(", ");
        objectToBuffer(arg, buf, hgs);
    }
    ti.nest--;
    buf.writeByte(')');
}

/****************************************
 * This makes a 'pretty' version of the template arguments.
 * It's analogous to genIdent() which makes a mangled version.
 */
private void objectToBuffer(RootObject oarg, OutBuffer* buf, HdrGenState* hgs)
{
    //printf("objectToBuffer()\n");
    /* The logic of this should match what genIdent() does. The _dynamic_cast()
     * function relies on all the pretty strings to be unique for different classes
     * See https://issues.dlang.org/show_bug.cgi?id=7375
     * Perhaps it would be better to demangle what genIdent() does.
     */
    if (auto t = isType(oarg))
    {
        //printf("\tt: %s ty = %d\n", t.toChars(), t.ty);
        typeToBuffer(t, null, buf, hgs);
    }
    else if (auto e = isExpression(oarg))
    {
        if (e.op == TOK.variable)
            e = e.optimize(WANTvalue); // added to fix https://issues.dlang.org/show_bug.cgi?id=7375
        expToBuffer(e, PREC.assign, buf, hgs);
    }
    else if (Dsymbol s = isDsymbol(oarg))
    {
        const p = s.ident ? s.ident.toChars() : s.toChars();
        buf.writestring(p);
    }
    else if (auto v = isTuple(oarg))
    {
        auto args = &v.objects;
        foreach (i, arg; *args)
        {
            if (i)
                buf.writestring(", ");
            objectToBuffer(arg, buf, hgs);
        }
    }
    else if (auto p = isParameter(oarg))
    {
        parameterToBuffer(p, buf, hgs);
    }
    else if (!oarg)
    {
        buf.writestring("NULL");
    }
    else
    {
        debug
        {
            printf("bad Object = %p\n", oarg);
        }
        assert(0);
    }
}


private void visitFuncIdentWithPostfix(TypeFunction t, const char[] ident, OutBuffer* buf, HdrGenState* hgs)
{
    if (t.inuse)
    {
        t.inuse = 2; // flag error to caller
        return;
    }
    t.inuse++;
    if (t.linkage > LINK.d && hgs.ddoc != 1 && !hgs.hdrgen)
    {
        linkageToBuffer(buf, t.linkage);
        buf.writeByte(' ');
    }
    if (t.next)
    {
        typeToBuffer(t.next, null, buf, hgs);
        if (ident)
            buf.writeByte(' ');
    }
    else if (hgs.ddoc)
        buf.writestring("auto ");
    if (ident)
        buf.writestring(ident);
    parametersToBuffer(t.parameterList, buf, hgs);
    /* Use postfix style for attributes
     */
    if (t.mod)
    {
        buf.writeByte(' ');
        MODtoBuffer(buf, t.mod);
    }

    void dg(string str)
    {
        buf.writeByte(' ');
        buf.writestring(str);
    }
    t.attributesApply(&dg);

    t.inuse--;
}

private void visitFuncIdentWithPrefix(TypeFunction t, const Identifier ident, TemplateDeclaration td,
    OutBuffer* buf, HdrGenState* hgs)
{
    if (t.inuse)
    {
        t.inuse = 2; // flag error to caller
        return;
    }
    t.inuse++;

    /* Use 'storage class' (prefix) style for attributes
     */
    if (t.mod)
    {
        MODtoBuffer(buf, t.mod);
        buf.writeByte(' ');
    }

    void ignoreReturn(string str)
    {
        if (str != "return")
        {
            // don't write 'ref' for ctors
            if ((ident == Id.ctor) && str == "ref")
                return;
            buf.writestring(str);
            buf.writeByte(' ');
        }
    }
    t.attributesApply(&ignoreReturn);

    if (t.linkage > LINK.d && hgs.ddoc != 1 && !hgs.hdrgen)
    {
        linkageToBuffer(buf, t.linkage);
        buf.writeByte(' ');
    }
    if (ident && ident.toHChars2() != ident.toChars())
    {
        // Don't print return type for ctor, dtor, unittest, etc
    }
    else if (t.next)
    {
        typeToBuffer(t.next, null, buf, hgs);
        if (ident)
            buf.writeByte(' ');
    }
    else if (hgs.ddoc)
        buf.writestring("auto ");
    if (ident)
        buf.writestring(ident.toHChars2());
    if (td)
    {
        buf.writeByte('(');
        foreach (i, p; *td.origParameters)
        {
            if (i)
                buf.writestring(", ");
            p.templateParameterToBuffer(buf, hgs);
        }
        buf.writeByte(')');
    }
    parametersToBuffer(t.parameterList, buf, hgs);
    if (t.isreturn)
    {
        buf.writestring(" return");
    }
    t.inuse--;
}


private void initializerToBuffer(Initializer inx, OutBuffer* buf, HdrGenState* hgs)
{
    void visitError(ErrorInitializer iz)
    {
        buf.writestring("__error__");
    }

    void visitVoid(VoidInitializer iz)
    {
        buf.writestring("void");
    }

    void visitStruct(StructInitializer si)
    {
        //printf("StructInitializer::toCBuffer()\n");
        buf.writeByte('{');
        foreach (i, const id; si.field)
        {
            if (i)
                buf.writestring(", ");
            if (id)
            {
                buf.writestring(id.toString());
                buf.writeByte(':');
            }
            if (auto iz = si.value[i])
                initializerToBuffer(iz, buf, hgs);
        }
        buf.writeByte('}');
    }

    void visitArray(ArrayInitializer ai)
    {
        buf.writeByte('[');
        foreach (i, ex; ai.index)
        {
            if (i)
                buf.writestring(", ");
            if (ex)
            {
                ex.expressionToBuffer(buf, hgs);
                buf.writeByte(':');
            }
            if (auto iz = ai.value[i])
                initializerToBuffer(iz, buf, hgs);
        }
        buf.writeByte(']');
    }

    void visitExp(ExpInitializer ei)
    {
        ei.exp.expressionToBuffer(buf, hgs);
    }

    final switch (inx.kind)
    {
        case InitKind.error:   return visitError (inx.isErrorInitializer ());
        case InitKind.void_:   return visitVoid  (inx.isVoidInitializer  ());
        case InitKind.struct_: return visitStruct(inx.isStructInitializer());
        case InitKind.array:   return visitArray (inx.isArrayInitializer ());
        case InitKind.exp:     return visitExp   (inx.isExpInitializer   ());
    }
}


private void typeToBufferx(Type t, OutBuffer* buf, HdrGenState* hgs)
{
    void visitType(Type t)
    {
        printf("t = %p, ty = %d\n", t, t.ty);
        assert(0);
    }

    void visitError(TypeError t)
    {
        buf.writestring("_error_");
    }

    void visitBasic(TypeBasic t)
    {
        //printf("TypeBasic::toCBuffer2(t.mod = %d)\n", t.mod);
        buf.writestring(t.dstring);
    }

    void visitTraits(TypeTraits t)
    {
        //printf("TypeBasic::toCBuffer2(t.mod = %d)\n", t.mod);
        t.exp.expressionToBuffer(buf, hgs);
    }

    void visitVector(TypeVector t)
    {
        //printf("TypeVector::toCBuffer2(t.mod = %d)\n", t.mod);
        buf.writestring("__vector(");
        visitWithMask(t.basetype, t.mod, buf, hgs);
        buf.writestring(")");
    }

    void visitSArray(TypeSArray t)
    {
        visitWithMask(t.next, t.mod, buf, hgs);
        buf.writeByte('[');
        sizeToBuffer(t.dim, buf, hgs);
        buf.writeByte(']');
    }

    void visitDArray(TypeDArray t)
    {
        Type ut = t.castMod(0);
        if (hgs.declstring)
            goto L1;
        if (ut.equals(Type.tstring))
            buf.writestring("string");
        else if (ut.equals(Type.twstring))
            buf.writestring("wstring");
        else if (ut.equals(Type.tdstring))
            buf.writestring("dstring");
        else
        {
        L1:
            visitWithMask(t.next, t.mod, buf, hgs);
            buf.writestring("[]");
        }
    }

    void visitAArray(TypeAArray t)
    {
        visitWithMask(t.next, t.mod, buf, hgs);
        buf.writeByte('[');
        visitWithMask(t.index, 0, buf, hgs);
        buf.writeByte(']');
    }

    void visitPointer(TypePointer t)
    {
        //printf("TypePointer::toCBuffer2() next = %d\n", t.next.ty);
        if (t.next.ty == Tfunction)
            visitFuncIdentWithPostfix(cast(TypeFunction)t.next, "function", buf, hgs);
        else
        {
            visitWithMask(t.next, t.mod, buf, hgs);
            buf.writeByte('*');
        }
    }

    void visitReference(TypeReference t)
    {
        visitWithMask(t.next, t.mod, buf, hgs);
        buf.writeByte('&');
    }

    void visitFunction(TypeFunction t)
    {
        //printf("TypeFunction::toCBuffer2() t = %p, ref = %d\n", t, t.isref);
        visitFuncIdentWithPostfix(t, null, buf, hgs);
    }

    void visitDelegate(TypeDelegate t)
    {
        visitFuncIdentWithPostfix(cast(TypeFunction)t.next, "delegate", buf, hgs);
    }

    void visitTypeQualifiedHelper(TypeQualified t)
    {
        foreach (id; t.idents)
        {
            if (id.dyncast() == DYNCAST.dsymbol)
            {
                buf.writeByte('.');
                TemplateInstance ti = cast(TemplateInstance)id;
                ti.dsymbolToBuffer(buf, hgs);
            }
            else if (id.dyncast() == DYNCAST.expression)
            {
                buf.writeByte('[');
                (cast(Expression)id).expressionToBuffer(buf, hgs);
                buf.writeByte(']');
            }
            else if (id.dyncast() == DYNCAST.type)
            {
                buf.writeByte('[');
                typeToBufferx(cast(Type)id, buf, hgs);
                buf.writeByte(']');
            }
            else
            {
                buf.writeByte('.');
                buf.writestring(id.toString());
            }
        }
    }

    void visitIdentifier(TypeIdentifier t)
    {
        buf.writestring(t.ident.toString());
        visitTypeQualifiedHelper(t);
    }

    void visitInstance(TypeInstance t)
    {
        t.tempinst.dsymbolToBuffer(buf, hgs);
        visitTypeQualifiedHelper(t);
    }

    void visitTypeof(TypeTypeof t)
    {
        buf.writestring("typeof(");
        t.exp.expressionToBuffer(buf, hgs);
        buf.writeByte(')');
        visitTypeQualifiedHelper(t);
    }

    void visitReturn(TypeReturn t)
    {
        buf.writestring("typeof(return)");
        visitTypeQualifiedHelper(t);
    }

    void visitEnum(TypeEnum t)
    {
        buf.writestring(hgs.fullQual ? t.sym.toPrettyChars() : t.sym.toChars());
    }

    void visitStruct(TypeStruct t)
    {
        // https://issues.dlang.org/show_bug.cgi?id=13776
        // Don't use ti.toAlias() to avoid forward reference error
        // while printing messages.
        TemplateInstance ti = t.sym.parent ? t.sym.parent.isTemplateInstance() : null;
        if (ti && ti.aliasdecl == t.sym)
            buf.writestring(hgs.fullQual ? ti.toPrettyChars() : ti.toChars());
        else
            buf.writestring(hgs.fullQual ? t.sym.toPrettyChars() : t.sym.toChars());
    }

    void visitClass(TypeClass t)
    {
        // https://issues.dlang.org/show_bug.cgi?id=13776
        // Don't use ti.toAlias() to avoid forward reference error
        // while printing messages.
        TemplateInstance ti = t.sym.parent.isTemplateInstance();
        if (ti && ti.aliasdecl == t.sym)
            buf.writestring(hgs.fullQual ? ti.toPrettyChars() : ti.toChars());
        else
            buf.writestring(hgs.fullQual ? t.sym.toPrettyChars() : t.sym.toChars());
    }

    void visitTuple(TypeTuple t)
    {
        parametersToBuffer(ParameterList(t.arguments, VarArg.none), buf, hgs);
    }

    void visitSlice(TypeSlice t)
    {
        visitWithMask(t.next, t.mod, buf, hgs);
        buf.writeByte('[');
        sizeToBuffer(t.lwr, buf, hgs);
        buf.writestring(" .. ");
        sizeToBuffer(t.upr, buf, hgs);
        buf.writeByte(']');
    }

    void visitNull(TypeNull t)
    {
        buf.writestring("typeof(null)");
    }

    void visitMixin(TypeMixin t)
    {
        buf.writestring("mixin(");
        argsToBuffer(t.exps, buf, hgs, null);
        buf.writeByte(')');
    }

    switch (t.ty)
    {
        default:        return t.isTypeBasic() ?
                                visitBasic(cast(TypeBasic)t) :
                                visitType(t);

        case Terror:     return visitError(cast(TypeError)t);
        case Ttraits:    return visitTraits(cast(TypeTraits)t);
        case Tvector:    return visitVector(cast(TypeVector)t);
        case Tsarray:    return visitSArray(cast(TypeSArray)t);
        case Tarray:     return visitDArray(cast(TypeDArray)t);
        case Taarray:    return visitAArray(cast(TypeAArray)t);
        case Tpointer:   return visitPointer(cast(TypePointer)t);
        case Treference: return visitReference(cast(TypeReference)t);
        case Tfunction:  return visitFunction(cast(TypeFunction)t);
        case Tdelegate:  return visitDelegate(cast(TypeDelegate)t);
        case Tident:     return visitIdentifier(cast(TypeIdentifier)t);
        case Tinstance:  return visitInstance(cast(TypeInstance)t);
        case Ttypeof:    return visitTypeof(cast(TypeTypeof)t);
        case Treturn:    return visitReturn(cast(TypeReturn)t);
        case Tenum:      return visitEnum(cast(TypeEnum)t);
        case Tstruct:    return visitStruct(cast(TypeStruct)t);
        case Tclass:     return visitClass(cast(TypeClass)t);
        case Ttuple:     return visitTuple (cast(TypeTuple)t);
        case Tslice:     return visitSlice(cast(TypeSlice)t);
        case Tnull:      return visitNull(cast(TypeNull)t);
        case Tmixin:     return visitMixin(cast(TypeMixin)t);
    }
}
