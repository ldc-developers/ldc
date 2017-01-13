// Compiler implementation of the D programming language
// Copyright (c) 1999-2015 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.statement;

import core.stdc.stdarg;
import core.stdc.stdio;

import ddmd.aggregate;
import ddmd.aliasthis;
import ddmd.arrayop;
import ddmd.arraytypes;
import ddmd.attrib;
import ddmd.gluelayer;
import ddmd.canthrow;
import ddmd.clone;
import ddmd.cond;
import ddmd.ctfeexpr;
import ddmd.dcast;
import ddmd.dclass;
import ddmd.declaration;
import ddmd.denum;
import ddmd.dimport;
import ddmd.dinterpret;
import ddmd.dscope;
import ddmd.dsymbol;
import ddmd.dtemplate;
import ddmd.errors;
import ddmd.escape;
import ddmd.expression;
import ddmd.func;
import ddmd.globals;
import ddmd.hdrgen;
import ddmd.id;
import ddmd.identifier;
import ddmd.init;
import ddmd.inline;
import ddmd.intrange;
import ddmd.mtype;
import ddmd.mtype;
import ddmd.nogc;
import ddmd.opover;
import ddmd.parse;
import ddmd.root.outbuffer;
import ddmd.root.rootobject;
import ddmd.sapply;
import ddmd.sideeffect;
import ddmd.staticassert;
import ddmd.target;
import ddmd.tokens;
import ddmd.visitor;
version(IN_LLVM)
{
    import gen.dpragma;
}

extern (C++) Identifier fixupLabelName(Scope* sc, Identifier ident)
{
    uint flags = (sc.flags & SCOPEcontract);
    if (flags && flags != SCOPEinvariant && !(ident.string[0] == '_' && ident.string[1] == '_'))
    {
        /* CTFE requires FuncDeclaration::labtab for the interpretation.
         * So fixing the label name inside in/out contracts is necessary
         * for the uniqueness in labtab.
         */
        const(char)* prefix = flags == SCOPErequire ? "__in_" : "__out_";
        OutBuffer buf;
        buf.printf("%s%s", prefix, ident.toChars());
        ident = Identifier.idPool(buf.peekSlice());
    }
    return ident;
}

extern (C++) LabelStatement checkLabeledLoop(Scope* sc, Statement statement)
{
    if (sc.slabel && sc.slabel.statement == statement)
    {
        return sc.slabel;
    }
    return null;
}

/***********************************************************
 * Check an assignment is used as a condition.
 * Intended to be use before the `semantic` call on `e`.
 * Params:
 *  e = condition expression which is not yet run semantic analysis.
 * Returns:
 *  `e` or ErrorExp.
 */
Expression checkAssignmentAsCondition(Expression e)
{
    auto ec = e;
    while (ec.op == TOKcomma)
        ec = (cast(CommaExp)ec).e2;
    if (ec.op == TOKassign)
    {
        ec.error("assignment cannot be used as a condition, perhaps == was meant?");
        return new ErrorExp();
    }
    return e;
}

enum BE : int
{
    BEnone = 0,
    BEfallthru = 1,
    BEthrow = 2,
    BEreturn = 4,
    BEgoto = 8,
    BEhalt = 0x10,
    BEbreak = 0x20,
    BEcontinue = 0x40,
    BEerrthrow = 0x80,
    BEany = (BEfallthru | BEthrow | BEreturn | BEgoto | BEhalt),
}

alias BEnone = BE.BEnone;
alias BEfallthru = BE.BEfallthru;
alias BEthrow = BE.BEthrow;
alias BEreturn = BE.BEreturn;
alias BEgoto = BE.BEgoto;
alias BEhalt = BE.BEhalt;
alias BEbreak = BE.BEbreak;
alias BEcontinue = BE.BEcontinue;
alias BEerrthrow = BE.BEerrthrow;
alias BEany = BE.BEany;

/***********************************************************
 */
extern (C++) class Statement : RootObject
{
public:
    Loc loc;

    final extern (D) this(Loc loc)
    {
        this.loc = loc;
        // If this is an in{} contract scope statement (skip for determining
        //  inlineStatus of a function body for header content)
    }

    Statement syntaxCopy()
    {
        assert(0);
    }

    override final void print()
    {
        fprintf(stderr, "%s\n", toChars());
        fflush(stderr);
    }

    override final const(char)* toChars()
    {
        HdrGenState hgs;
        OutBuffer buf;
        .toCBuffer(this, &buf, &hgs);
        return buf.extractString();
    }

    final void error(const(char)* format, ...)
    {
        va_list ap;
        va_start(ap, format);
        .verror(loc, format, ap);
        va_end(ap);
    }

    final void warning(const(char)* format, ...)
    {
        va_list ap;
        va_start(ap, format);
        .vwarning(loc, format, ap);
        va_end(ap);
    }

    final void deprecation(const(char)* format, ...)
    {
        va_list ap;
        va_start(ap, format);
        .vdeprecation(loc, format, ap);
        va_end(ap);
    }

    Statement semantic(Scope* sc)
    {
        return this;
    }

    // Same as semanticNoScope(), but do create a new scope
    final Statement semanticScope(Scope* sc, Statement sbreak, Statement scontinue)
    {
        auto sym = new ScopeDsymbol();
        sym.parent = sc.scopesym;
        Scope* scd = sc.push(sym);
        if (sbreak)
            scd.sbreak = sbreak;
        if (scontinue)
            scd.scontinue = scontinue;
        Statement s = semanticNoScope(scd);
        scd.pop();
        return s;
    }

    final Statement semanticNoScope(Scope* sc)
    {
        //printf("Statement::semanticNoScope() %s\n", toChars());
        Statement s = this;
        if (!s.isCompoundStatement() && !s.isScopeStatement())
        {
            s = new CompoundStatement(loc, this); // so scopeCode() gets called
        }
        s = s.semantic(sc);
        return s;
    }

    Statement getRelatedLabeled()
    {
        return this;
    }

    bool hasBreak()
    {
        //printf("Statement::hasBreak()\n");
        return false;
    }

    bool hasContinue()
    {
        return false;
    }

    /* ============================================== */
    // true if statement uses exception handling
    final bool usesEH()
    {
        extern (C++) final class UsesEH : StoppableVisitor
        {
            alias visit = super.visit;
        public:
            override void visit(Statement s)
            {
            }

            override void visit(TryCatchStatement s)
            {
                stop = true;
            }

            override void visit(TryFinallyStatement s)
            {
                stop = true;
            }

            override void visit(OnScopeStatement s)
            {
                stop = true;
            }

            override void visit(SynchronizedStatement s)
            {
                stop = true;
            }
        }

        scope UsesEH ueh = new UsesEH();
        return walkPostorder(this, ueh);
    }

    /* ============================================== */
    /* Only valid after semantic analysis
     * If 'mustNotThrow' is true, generate an error if it throws
     */
    final int blockExit(FuncDeclaration func, bool mustNotThrow)
    {
        extern (C++) final class BlockExit : Visitor
        {
            alias visit = super.visit;
        public:
            FuncDeclaration func;
            bool mustNotThrow;
            int result;

            extern (D) this(FuncDeclaration func, bool mustNotThrow)
            {
                this.func = func;
                this.mustNotThrow = mustNotThrow;
                result = BEnone;
            }

            override void visit(Statement s)
            {
                printf("Statement::blockExit(%p)\n", s);
                printf("%s\n", s.toChars());
                assert(0);
            }

            override void visit(ErrorStatement s)
            {
                result = BEany;
            }

            override void visit(ExpStatement s)
            {
                result = BEfallthru;
                if (s.exp)
                {
                    if (s.exp.op == TOKhalt)
                    {
                        result = BEhalt;
                        return;
                    }
                    if (s.exp.op == TOKassert)
                    {
                        AssertExp a = cast(AssertExp)s.exp;
                        if (a.e1.isBool(false)) // if it's an assert(0)
                        {
                            result = BEhalt;
                            return;
                        }
                    }
                    if (canThrow(s.exp, func, mustNotThrow))
                        result |= BEthrow;
                }
            }

            override void visit(CompileStatement s)
            {
                assert(global.errors);
                result = BEfallthru;
            }

            override void visit(CompoundStatement cs)
            {
                //printf("CompoundStatement::blockExit(%p) %d\n", cs, cs->statements->dim);
                result = BEfallthru;
                Statement slast = null;
                foreach (s; *cs.statements)
                {
                    if (s)
                    {
                        //printf("result = x%x\n", result);
                        //printf("s: %s\n", s->toChars());
                        if (global.params.warnings && result & BEfallthru && slast)
                        {
                            slast = slast.last();
                            if (slast && (slast.isCaseStatement() || slast.isDefaultStatement()) && (s.isCaseStatement() || s.isDefaultStatement()))
                            {
                                // Allow if last case/default was empty
                                CaseStatement sc = slast.isCaseStatement();
                                DefaultStatement sd = slast.isDefaultStatement();
                                if (sc && (!sc.statement.hasCode() || sc.statement.isCaseStatement() || sc.statement.isErrorStatement()))
                                {
                                }
                                else if (sd && (!sd.statement.hasCode() || sd.statement.isCaseStatement() || sd.statement.isErrorStatement()))
                                {
                                }
                                else
                                {
                                    const(char)* gototype = s.isCaseStatement() ? "case" : "default";
                                    s.warning("switch case fallthrough - use 'goto %s;' if intended", gototype);
                                }
                            }
                        }
                        if (!(result & BEfallthru) && !s.comeFrom())
                        {
                            if (s.blockExit(func, mustNotThrow) != BEhalt && s.hasCode())
                                s.warning("statement is not reachable");
                        }
                        else
                        {
                            result &= ~BEfallthru;
                            result |= s.blockExit(func, mustNotThrow);
                        }
                        slast = s;
                    }
                }
            }

            override void visit(UnrolledLoopStatement uls)
            {
                result = BEfallthru;
                foreach (s; *uls.statements)
                {
                    if (s)
                    {
                        int r = s.blockExit(func, mustNotThrow);
                        result |= r & ~(BEbreak | BEcontinue | BEfallthru);
                        if ((r & (BEfallthru | BEcontinue | BEbreak)) == 0)
                            result &= ~BEfallthru;
                    }
                }
            }

            override void visit(ScopeStatement s)
            {
                //printf("ScopeStatement::blockExit(%p)\n", s->statement);
                result = s.statement ? s.statement.blockExit(func, mustNotThrow) : BEfallthru;
            }

            override void visit(WhileStatement s)
            {
                assert(global.errors);
                result = BEfallthru;
            }

            override void visit(DoStatement s)
            {
                if (s._body)
                {
                    result = s._body.blockExit(func, mustNotThrow);
                    if (result == BEbreak)
                    {
                        result = BEfallthru;
                        return;
                    }
                    if (result & BEcontinue)
                        result |= BEfallthru;
                }
                else
                    result = BEfallthru;
                if (result & BEfallthru)
                {
                    if (canThrow(s.condition, func, mustNotThrow))
                        result |= BEthrow;
                    if (!(result & BEbreak) && s.condition.isBool(true))
                        result &= ~BEfallthru;
                }
                result &= ~(BEbreak | BEcontinue);
            }

            override void visit(ForStatement s)
            {
                result = BEfallthru;
                if (s._init)
                {
                    result = s._init.blockExit(func, mustNotThrow);
                    if (!(result & BEfallthru))
                        return;
                }
                if (s.condition)
                {
                    if (canThrow(s.condition, func, mustNotThrow))
                        result |= BEthrow;
                    if (s.condition.isBool(true))
                        result &= ~BEfallthru;
                    else if (s.condition.isBool(false))
                        return;
                }
                else
                    result &= ~BEfallthru; // the body must do the exiting
                if (s._body)
                {
                    int r = s._body.blockExit(func, mustNotThrow);
                    if (r & (BEbreak | BEgoto))
                        result |= BEfallthru;
                    result |= r & ~(BEfallthru | BEbreak | BEcontinue);
                }
                if (s.increment && canThrow(s.increment, func, mustNotThrow))
                    result |= BEthrow;
            }

            override void visit(ForeachStatement s)
            {
                result = BEfallthru;
                if (canThrow(s.aggr, func, mustNotThrow))
                    result |= BEthrow;
                if (s._body)
                    result |= s._body.blockExit(func, mustNotThrow) & ~(BEbreak | BEcontinue);
            }

            override void visit(ForeachRangeStatement s)
            {
                assert(global.errors);
                result = BEfallthru;
            }

            override void visit(IfStatement s)
            {
                //printf("IfStatement::blockExit(%p)\n", s);
                result = BEnone;
                if (canThrow(s.condition, func, mustNotThrow))
                    result |= BEthrow;
                if (s.condition.isBool(true))
                {
                    if (s.ifbody)
                        result |= s.ifbody.blockExit(func, mustNotThrow);
                    else
                        result |= BEfallthru;
                }
                else if (s.condition.isBool(false))
                {
                    if (s.elsebody)
                        result |= s.elsebody.blockExit(func, mustNotThrow);
                    else
                        result |= BEfallthru;
                }
                else
                {
                    if (s.ifbody)
                        result |= s.ifbody.blockExit(func, mustNotThrow);
                    else
                        result |= BEfallthru;
                    if (s.elsebody)
                        result |= s.elsebody.blockExit(func, mustNotThrow);
                    else
                        result |= BEfallthru;
                }
                //printf("IfStatement::blockExit(%p) = x%x\n", s, result);
            }

            override void visit(ConditionalStatement s)
            {
                result = s.ifbody.blockExit(func, mustNotThrow);
                if (s.elsebody)
                    result |= s.elsebody.blockExit(func, mustNotThrow);
            }

            override void visit(PragmaStatement s)
            {
                result = BEfallthru;
            }

            override void visit(StaticAssertStatement s)
            {
                result = BEfallthru;
            }

            override void visit(SwitchStatement s)
            {
                result = BEnone;
                if (canThrow(s.condition, func, mustNotThrow))
                    result |= BEthrow;
                if (s._body)
                {
                    result |= s._body.blockExit(func, mustNotThrow);
                    if (result & BEbreak)
                    {
                        result |= BEfallthru;
                        result &= ~BEbreak;
                    }
                }
                else
                    result |= BEfallthru;
            }

            override void visit(CaseStatement s)
            {
                result = s.statement.blockExit(func, mustNotThrow);
            }

            override void visit(DefaultStatement s)
            {
                result = s.statement.blockExit(func, mustNotThrow);
            }

            override void visit(GotoDefaultStatement s)
            {
                result = BEgoto;
            }

            override void visit(GotoCaseStatement s)
            {
                result = BEgoto;
            }

            override void visit(SwitchErrorStatement s)
            {
                // Switch errors are non-recoverable
                result = BEhalt;
            }

            override void visit(ReturnStatement s)
            {
                result = BEreturn;
                if (s.exp && canThrow(s.exp, func, mustNotThrow))
                    result |= BEthrow;
            }

            override void visit(BreakStatement s)
            {
                //printf("BreakStatement::blockExit(%p) = x%x\n", s, s->ident ? BEgoto : BEbreak);
                result = s.ident ? BEgoto : BEbreak;
            }

            override void visit(ContinueStatement s)
            {
                result = s.ident ? BEgoto : BEcontinue;
            }

            override void visit(SynchronizedStatement s)
            {
                result = s._body ? s._body.blockExit(func, mustNotThrow) : BEfallthru;
            }

            override void visit(WithStatement s)
            {
                result = BEnone;
                if (canThrow(s.exp, func, mustNotThrow))
                    result = BEthrow;
                if (s._body)
                    result |= s._body.blockExit(func, mustNotThrow);
                else
                    result |= BEfallthru;
            }

            override void visit(TryCatchStatement s)
            {
                assert(s._body);
                result = s._body.blockExit(func, false);
                int catchresult = 0;
                foreach (c; *s.catches)
                {
                    if (c.type == Type.terror)
                        continue;
                    int cresult;
                    if (c.handler)
                        cresult = c.handler.blockExit(func, mustNotThrow);
                    else
                        cresult = BEfallthru;
                    /* If we're catching Object, then there is no throwing
                     */
                    Identifier id = c.type.toBasetype().isClassHandle().ident;
                    if (c.internalCatch && (cresult & BEfallthru))
                    {
                        // Bugzilla 11542: leave blockExit flags of the body
                        cresult &= ~BEfallthru;
                    }
                    else if (id == Id.Object || id == Id.Throwable)
                    {
                        result &= ~(BEthrow | BEerrthrow);
                    }
                    else if (id == Id.Exception)
                    {
                        result &= ~BEthrow;
                    }
                    catchresult |= cresult;
                }
                if (mustNotThrow && (result & BEthrow))
                {
                    // now explain why this is nothrow
                    s._body.blockExit(func, mustNotThrow);
                }
                result |= catchresult;
            }

            override void visit(TryFinallyStatement s)
            {
                result = BEfallthru;
                if (s._body)
                    result = s._body.blockExit(func, false);
                // check finally body as well, it may throw (bug #4082)
                int finalresult = BEfallthru;
                if (s.finalbody)
                    finalresult = s.finalbody.blockExit(func, false);
                // If either body or finalbody halts
                if (result == BEhalt)
                    finalresult = BEnone;
                if (finalresult == BEhalt)
                    result = BEnone;
                if (mustNotThrow)
                {
                    // now explain why this is nothrow
                    if (s._body && (result & BEthrow))
                        s._body.blockExit(func, mustNotThrow);
                    if (s.finalbody && (finalresult & BEthrow))
                        s.finalbody.blockExit(func, mustNotThrow);
                }
                version (none)
                {
                    // Bugzilla 13201: Mask to prevent spurious warnings for
                    // destructor call, exit of synchronized statement, etc.
                    if (result == BEhalt && finalresult != BEhalt && s.finalbody && s.finalbody.hasCode())
                    {
                        s.finalbody.warning("statement is not reachable");
                    }
                }
                if (!(finalresult & BEfallthru))
                    result &= ~BEfallthru;
                result |= finalresult & ~BEfallthru;
            }

            override void visit(OnScopeStatement s)
            {
                // At this point, this statement is just an empty placeholder
                result = BEfallthru;
            }

            override void visit(ThrowStatement s)
            {
                if (s.internalThrow)
                {
                    // Bugzilla 8675: Allow throwing 'Throwable' object even if mustNotThrow.
                    result = BEfallthru;
                    return;
                }
                Type t = s.exp.type.toBasetype();
                ClassDeclaration cd = t.isClassHandle();
                assert(cd);
                if (cd == ClassDeclaration.errorException || ClassDeclaration.errorException.isBaseOf(cd, null))
                {
                    result = BEerrthrow;
                    return;
                }
                if (mustNotThrow)
                    s.error("%s is thrown but not caught", s.exp.type.toChars());
                result = BEthrow;
            }

            override void visit(GotoStatement s)
            {
                //printf("GotoStatement::blockExit(%p)\n", s);
                result = BEgoto;
            }

            override void visit(LabelStatement s)
            {
                //printf("LabelStatement::blockExit(%p)\n", s);
                result = s.statement ? s.statement.blockExit(func, mustNotThrow) : BEfallthru;
                if (s.breaks)
                    result |= BEfallthru;
            }

            override void visit(CompoundAsmStatement s)
            {
                if (mustNotThrow && !(s.stc & STCnothrow))
                    s.deprecation("asm statement is assumed to throw - mark it with 'nothrow' if it does not");
                // Assume the worst
                result = BEfallthru | BEreturn | BEgoto | BEhalt;
                if (!(s.stc & STCnothrow))
                    result |= BEthrow;
            }

            override void visit(ImportStatement s)
            {
                result = BEfallthru;
            }
        }

        scope BlockExit be = new BlockExit(func, mustNotThrow);
        accept(be);
        return be.result;
    }

    /* ============================================== */
    // true if statement 'comes from' somewhere else, like a goto
    final bool comeFrom()
    {
        extern (C++) final class ComeFrom : StoppableVisitor
        {
            alias visit = super.visit;
        public:
            override void visit(Statement s)
            {
            }

            override void visit(CaseStatement s)
            {
                stop = true;
            }

            override void visit(DefaultStatement s)
            {
                stop = true;
            }

            override void visit(LabelStatement s)
            {
                stop = true;
            }

            override void visit(AsmStatement s)
            {
                stop = true;
            }
        }

        scope ComeFrom cf = new ComeFrom();
        return walkPostorder(this, cf);
    }

    /* ============================================== */
    // Return true if statement has executable code.
    final bool hasCode()
    {
        extern (C++) final class HasCode : StoppableVisitor
        {
            alias visit = super.visit;
        public:
            override void visit(Statement s)
            {
                stop = true;
            }

            override void visit(ExpStatement s)
            {
                stop = s.exp !is null;
            }

            override void visit(CompoundStatement s)
            {
            }

            override void visit(ScopeStatement s)
            {
            }

            override void visit(ImportStatement s)
            {
            }
        }

        scope HasCode hc = new HasCode();
        return walkPostorder(this, hc);
    }

    /****************************************
     * If this statement has code that needs to run in a finally clause
     * at the end of the current scope, return that code in the form of
     * a Statement.
     * Output:
     *      *sentry         code executed upon entry to the scope
     *      *sexception     code executed upon exit from the scope via exception
     *      *sfinally       code executed in finally block
     */
    Statement scopeCode(Scope* sc, Statement* sentry, Statement* sexception, Statement* sfinally)
    {
        //printf("Statement::scopeCode()\n");
        //print();
        *sentry = null;
        *sexception = null;
        *sfinally = null;
        return this;
    }

    /*********************************
     * Flatten out the scope by presenting the statement
     * as an array of statements.
     * Returns NULL if no flattening necessary.
     */
    Statements* flatten(Scope* sc)
    {
        return null;
    }

    Statement last()
    {
        return this;
    }

    // Avoid dynamic_cast
    ErrorStatement isErrorStatement()
    {
        return null;
    }

    ScopeStatement isScopeStatement()
    {
        return null;
    }

    ExpStatement isExpStatement()
    {
        return null;
    }

    CompoundStatement isCompoundStatement()
    {
        return null;
    }

    ReturnStatement isReturnStatement()
    {
        return null;
    }

    IfStatement isIfStatement()
    {
        return null;
    }

    CaseStatement isCaseStatement()
    {
        return null;
    }

    DefaultStatement isDefaultStatement()
    {
        return null;
    }

    LabelStatement isLabelStatement()
    {
        return null;
    }

    DtorExpStatement isDtorExpStatement()
    {
        return null;
    }

    void accept(Visitor v)
    {
        v.visit(this);
    }

    version(IN_LLVM)
    {
        CompoundAsmStatement isCompoundAsmBlockStatement()
        {
            return null;
        }

        CompoundAsmStatement endsWithAsm()
        {
            // does not end with inline asm
            return null;
        }
    }
}

/***********************************************************
 * Any Statement that fails semantic() or has a component that is an ErrorExp or
 * a TypeError should return an ErrorStatement from semantic().
 */
extern (C++) final class ErrorStatement : Statement
{
public:
    extern (D) this()
    {
        super(Loc());
        assert(global.gaggedErrors || global.errors);
    }

    override Statement syntaxCopy()
    {
        return this;
    }

    override Statement semantic(Scope* sc)
    {
        return this;
    }

    override ErrorStatement isErrorStatement()
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
extern (C++) final class PeelStatement : Statement
{
public:
    Statement s;

    extern (D) this(Statement s)
    {
        super(s.loc);
        this.s = s;
    }

    override Statement semantic(Scope* sc)
    {
        /* "peel" off this wrapper, and don't run semantic()
         * on the result.
         */
        return s;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Convert TemplateMixin members (== Dsymbols) to Statements.
 */
extern (C++) Statement toStatement(Dsymbol s)
{
    extern (C++) final class ToStmt : Visitor
    {
        alias visit = super.visit;
    public:
        Statement result;

        Statement visitMembers(Loc loc, Dsymbols* a)
        {
            if (!a)
                return null;
            auto statements = new Statements();
            foreach (s; *a)
            {
                statements.push(toStatement(s));
            }
            return new CompoundStatement(loc, statements);
        }

        override void visit(Dsymbol s)
        {
            .error(Loc(), "Internal Compiler Error: cannot mixin %s %s\n", s.kind(), s.toChars());
            result = new ErrorStatement();
        }

        override void visit(TemplateMixin tm)
        {
            auto a = new Statements();
            foreach (m; *tm.members)
            {
                Statement s = toStatement(m);
                if (s)
                    a.push(s);
            }
            result = new CompoundStatement(tm.loc, a);
        }

        /* An actual declaration symbol will be converted to DeclarationExp
         * with ExpStatement.
         */
        Statement declStmt(Dsymbol s)
        {
            auto de = new DeclarationExp(s.loc, s);
            de.type = Type.tvoid; // avoid repeated semantic
            return new ExpStatement(s.loc, de);
        }

        override void visit(VarDeclaration d)
        {
            result = declStmt(d);
        }

        override void visit(AggregateDeclaration d)
        {
            result = declStmt(d);
        }

        override void visit(FuncDeclaration d)
        {
            result = declStmt(d);
        }

        override void visit(EnumDeclaration d)
        {
            result = declStmt(d);
        }

        override void visit(AliasDeclaration d)
        {
            result = declStmt(d);
        }

        override void visit(TemplateDeclaration d)
        {
            result = declStmt(d);
        }

        /* All attributes have been already picked by the semantic analysis of
         * 'bottom' declarations (function, struct, class, etc).
         * So we don't have to copy them.
         */
        override void visit(StorageClassDeclaration d)
        {
            result = visitMembers(d.loc, d.decl);
        }

        override void visit(DeprecatedDeclaration d)
        {
            result = visitMembers(d.loc, d.decl);
        }

        override void visit(LinkDeclaration d)
        {
            result = visitMembers(d.loc, d.decl);
        }

        override void visit(ProtDeclaration d)
        {
            result = visitMembers(d.loc, d.decl);
        }

        override void visit(AlignDeclaration d)
        {
            result = visitMembers(d.loc, d.decl);
        }

        override void visit(UserAttributeDeclaration d)
        {
            result = visitMembers(d.loc, d.decl);
        }

        override void visit(StaticAssert s)
        {
        }

        override void visit(Import s)
        {
        }

        override void visit(PragmaDeclaration d)
        {
        }

        override void visit(ConditionalDeclaration d)
        {
            result = visitMembers(d.loc, d.include(null, null));
        }

        override void visit(CompileDeclaration d)
        {
            result = visitMembers(d.loc, d.include(null, null));
        }
    }

    if (!s)
        return null;
    scope ToStmt v = new ToStmt();
    s.accept(v);
    return v.result;
}

/***********************************************************
 */
extern (C++) class ExpStatement : Statement
{
public:
    Expression exp;

    final extern (D) this(Loc loc, Expression exp)
    {
        super(loc);
        this.exp = exp;
    }

    final extern (D) this(Loc loc, Dsymbol declaration)
    {
        super(loc);
        this.exp = new DeclarationExp(loc, declaration);
    }

    final static ExpStatement create(Loc loc, Expression exp)
    {
        return new ExpStatement(loc, exp);
    }

    override Statement syntaxCopy()
    {
        return new ExpStatement(loc, exp ? exp.syntaxCopy() : null);
    }

    override final Statement semantic(Scope* sc)
    {
        if (exp)
        {
            //printf("ExpStatement::semantic() %s\n", exp.toChars());
            exp = exp.semantic(sc);
            exp = resolveProperties(sc, exp);
            exp = exp.addDtorHook(sc);
            if (checkNonAssignmentArrayOp(exp))
                exp = new ErrorExp();
            if (auto f = isFuncAddress(exp))
            {
                if (f.checkForwardRef(exp.loc))
                    exp = new ErrorExp();
            }
            discardValue(exp);

            exp = exp.optimize(WANTvalue);
            exp = checkGC(sc, exp);
            if (exp.op == TOKerror)
                return new ErrorStatement();
        }
        return this;
    }

    override final Statement scopeCode(Scope* sc, Statement* sentry, Statement* sexception, Statement* sfinally)
    {
        //printf("ExpStatement::scopeCode()\n");
        //print();
        *sentry = null;
        *sexception = null;
        *sfinally = null;
        if (exp && exp.op == TOKdeclaration)
        {
            auto de = cast(DeclarationExp)exp;
            auto v = de.declaration.isVarDeclaration();
            if (v && !v.isDataseg())
            {
                if (v.needsScopeDtor())
                {
                    //printf("dtor is: "); v.edtor.print();
                    *sfinally = new DtorExpStatement(loc, v.edtor, v);
                    v.noscope = true; // don't add in dtor again
                }
            }
        }
        return this;
    }

    override final Statements* flatten(Scope* sc)
    {
        /* Bugzilla 14243: expand template mixin in statement scope
         * to handle variable destructors.
         */
        if (exp && exp.op == TOKdeclaration)
        {
            Dsymbol d = (cast(DeclarationExp)exp).declaration;
            if (TemplateMixin tm = d.isTemplateMixin())
            {
                Expression e = exp.semantic(sc);
                if (e.op == TOKerror || tm.errors)
                {
                    auto a = new Statements();
                    a.push(new ErrorStatement());
                    return a;
                }
                assert(tm.members);
                Statement s = toStatement(tm);
                version (none)
                {
                    OutBuffer buf;
                    buf.doindent = 1;
                    HdrGenState hgs;
                    hgs.hdrgen = true;
                    toCBuffer(s, &buf, &hgs);
                    printf("tm ==> s = %s\n", buf.peekString());
                }
                auto a = new Statements();
                a.push(s);
                return a;
            }
        }
        return null;
    }

    override final ExpStatement isExpStatement()
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
extern (C++) final class DtorExpStatement : ExpStatement
{
public:
    // Wraps an expression that is the destruction of 'var'
    VarDeclaration var;

    extern (D) this(Loc loc, Expression exp, VarDeclaration v)
    {
        super(loc, exp);
        this.var = v;
    }

    override Statement syntaxCopy()
    {
        return new DtorExpStatement(loc, exp ? exp.syntaxCopy() : null, var);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }

    override DtorExpStatement isDtorExpStatement()
    {
        return this;
    }
}

/***********************************************************
 */
extern (C++) final class CompileStatement : Statement
{
public:
    Expression exp;

    extern (D) this(Loc loc, Expression exp)
    {
        super(loc);
        this.exp = exp;
    }

    override Statement syntaxCopy()
    {
        return new CompileStatement(loc, exp.syntaxCopy());
    }

    override Statements* flatten(Scope* sc)
    {
        //printf("CompileStatement::flatten() %s\n", exp->toChars());
        sc = sc.startCTFE();
        exp = exp.semantic(sc);
        exp = resolveProperties(sc, exp);
        sc = sc.endCTFE();
        auto a = new Statements();
        if (exp.op != TOKerror)
        {
            Expression e = exp.ctfeInterpret();
            StringExp se = e.toStringExp();
            if (!se)
                error("argument to mixin must be a string, not (%s) of type %s", exp.toChars(), exp.type.toChars());
            else
            {
                se = se.toUTF8(sc);
                uint errors = global.errors;
                scope Parser p = new Parser(loc, sc._module, se.toStringz(), se.len, 0);
                p.nextToken();
                while (p.token.value != TOKeof)
                {
                    Statement s = p.parseStatement(PSsemi | PScurlyscope);
                    if (!s || p.errors)
                    {
                        assert(!p.errors || global.errors != errors); // make sure we caught all the cases
                        goto Lerror;
                    }
                    a.push(s);
                }
                return a;
            }
        }
    Lerror:
        a.push(new ErrorStatement());
        return a;
    }

    override Statement semantic(Scope* sc)
    {
        //printf("CompileStatement::semantic() %s\n", exp->toChars());
        Statements* a = flatten(sc);
        if (!a)
            return null;
        Statement s = new CompoundStatement(loc, a);
        return s.semantic(sc);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) class CompoundStatement : Statement
{
public:
    Statements* statements;

    final extern (D) this(Loc loc, Statements* s)
    {
        super(loc);
        statements = s;
    }

    final extern (D) this(Loc loc, Statement s1)
    {
        super(loc);
        statements = new Statements();
        statements.push(s1);
    }

    final extern (D) this(Loc loc, Statement s1, Statement s2)
    {
        super(loc);
        statements = new Statements();
        statements.reserve(2);
        statements.push(s1);
        statements.push(s2);
    }

    final static CompoundStatement create(Loc loc, Statement s1, Statement s2)
    {
        return new CompoundStatement(loc, s1, s2);
    }

    override Statement syntaxCopy()
    {
        auto a = new Statements();
        a.setDim(statements.dim);
        foreach (i, s; *statements)
        {
            (*a)[i] = s ? s.syntaxCopy() : null;
        }
        return new CompoundStatement(loc, a);
    }

    override Statement semantic(Scope* sc)
    {
        //printf("CompoundStatement::semantic(this = %p, sc = %p)\n", this, sc);
        version (none)
        {
            foreach (i, s; statements)
            {
                if (s)
                    printf("[%d]: %s", i, s.toChars());
            }
        }
        for (size_t i = 0; i < statements.dim;)
        {
            Statement s = (*statements)[i];
            if (s)
            {
                Statements* flt = s.flatten(sc);
                if (flt)
                {
                    statements.remove(i);
                    statements.insert(i, flt);
                    continue;
                }
                s = s.semantic(sc);
                (*statements)[i] = s;
                if (s)
                {
                    Statement sentry;
                    Statement sexception;
                    Statement sfinally;
                    (*statements)[i] = s.scopeCode(sc, &sentry, &sexception, &sfinally);
                    if (sentry)
                    {
                        sentry = sentry.semantic(sc);
                        statements.insert(i, sentry);
                        i++;
                    }
                    if (sexception)
                        sexception = sexception.semantic(sc);
                    if (sexception)
                    {
                        if (i + 1 == statements.dim && !sfinally)
                        {
                        }
                        else
                        {
                            /* Rewrite:
                             *      s; s1; s2;
                             * As:
                             *      s;
                             *      try { s1; s2; }
                             *      catch (Throwable __o)
                             *      { sexception; throw __o; }
                             */
                            auto a = new Statements();
                            foreach (j; i + 1 .. statements.dim)
                            {
                                a.push((*statements)[j]);
                            }
                            Statement _body = new CompoundStatement(Loc(), a);
                            _body = new ScopeStatement(Loc(), _body);
                            Identifier id = Identifier.generateId("__o");
                            Statement handler = new PeelStatement(sexception);
                            if (sexception.blockExit(sc.func, false) & BEfallthru)
                            {
                                auto ts = new ThrowStatement(Loc(), new IdentifierExp(Loc(), id));
                                ts.internalThrow = true;
                                handler = new CompoundStatement(Loc(), handler, ts);
                            }
                            auto catches = new Catches();
                            auto ctch = new Catch(Loc(), null, id, handler);
                            ctch.internalCatch = true;
                            catches.push(ctch);
                            s = new TryCatchStatement(Loc(), _body, catches);
                            if (sfinally)
                                s = new TryFinallyStatement(Loc(), s, sfinally);
                            s = s.semantic(sc);
                            statements.setDim(i + 1);
                            statements.push(s);
                            break;
                        }
                    }
                    else if (sfinally)
                    {
                        if (0 && i + 1 == statements.dim)
                        {
                            statements.push(sfinally);
                        }
                        else
                        {
                            /* Rewrite:
                             *      s; s1; s2;
                             * As:
                             *      s; try { s1; s2; } finally { sfinally; }
                             */
                            auto a = new Statements();
                            foreach (j; i + 1 .. statements.dim)
                            {
                                a.push((*statements)[j]);
                            }
                            Statement _body = new CompoundStatement(Loc(), a);
                            s = new TryFinallyStatement(Loc(), _body, sfinally);
                            s = s.semantic(sc);
                            statements.setDim(i + 1);
                            statements.push(s);
                            break;
                        }
                    }
                }
                else
                {
                    /* Remove NULL statements from the list.
                     */
                    statements.remove(i);
                    continue;
                }
            }
            i++;
        }
        foreach (i; 0 .. statements.dim)
        {
        Lagain:
            Statement s = (*statements)[i];
            if (!s)
                continue;
            Statement se = s.isErrorStatement();
            if (se)
                return se;
            /* Bugzilla 11653: 'semantic' may return another CompoundStatement
             * (eg. CaseRangeStatement), so flatten it here.
             */
            Statements* flt = s.flatten(sc);
            if (flt)
            {
                statements.remove(i);
                statements.insert(i, flt);
                if (statements.dim <= i)
                    break;
                goto Lagain;
            }
        }
        //IN_LLVM replaced: if (statements.dim == 1)
        if (statements.dim == 1 && !isCompoundAsmBlockStatement())
        {
            return (*statements)[0];
        }
        return this;
    }

    override Statements* flatten(Scope* sc)
    {
        return statements;
    }

    override final ReturnStatement isReturnStatement()
    {
        ReturnStatement rs = null;
        foreach (s; *statements)
        {
            if (s)
            {
                rs = s.isReturnStatement();
                if (rs)
                    break;
            }
        }
        return rs;
    }

    override final Statement last()
    {
        Statement s = null;
        for (size_t i = statements.dim; i; --i)
        {
            s = (*statements)[i - 1];
            if (s)
            {
                s = s.last();
                if (s)
                    break;
            }
        }
        return s;
    }

    // IN_LLVM removed: final
    override CompoundStatement isCompoundStatement()
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }

    version(IN_LLVM)
    {
        override CompoundAsmStatement endsWithAsm()
        {
            // make the last inner statement decide
            if (statements && statements.dim) {
                size_t last = statements.dim - 1;
                Statement s = (*statements)[last];
                if (s) {
                    return s.endsWithAsm();
                }
            }
            return null;
        }
    }
}

/***********************************************************
 */
extern (C++) final class CompoundDeclarationStatement : CompoundStatement
{
public:
    extern (D) this(Loc loc, Statements* s)
    {
        super(loc, s);
        statements = s;
    }

    override Statement syntaxCopy()
    {
        auto a = new Statements();
        a.setDim(statements.dim);
        foreach (i, s; *statements)
        {
            (*a)[i] = s ? s.syntaxCopy() : null;
        }
        return new CompoundDeclarationStatement(loc, a);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * The purpose of this is so that continue will go to the next
 * of the statements, and break will go to the end of the statements.
 */
extern (C++) final class UnrolledLoopStatement : Statement
{
public:
    Statements* statements;

    extern (D) this(Loc loc, Statements* s)
    {
        super(loc);
        statements = s;
    }

    override Statement syntaxCopy()
    {
        auto a = new Statements();
        a.setDim(statements.dim);
        foreach (i, s; *statements)
        {
            (*a)[i] = s ? s.syntaxCopy() : null;
        }
        return new UnrolledLoopStatement(loc, a);
    }

    override Statement semantic(Scope* sc)
    {
        //printf("UnrolledLoopStatement::semantic(this = %p, sc = %p)\n", this, sc);
        Scope* scd = sc.push();
        scd.sbreak = this;
        scd.scontinue = this;
        Statement serror = null;
        foreach (i, ref s; *statements)
        {
            if (s)
            {
                //printf("[%d]: %s\n", i, s->toChars());
                s = s.semantic(scd);
                if (s && !serror)
                    serror = s.isErrorStatement();
            }
        }
        scd.pop();
        return serror ? serror : this;
    }

    override bool hasBreak()
    {
        return true;
    }

    override bool hasContinue()
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ScopeStatement : Statement
{
public:
    Statement statement;

    extern (D) this(Loc loc, Statement s)
    {
        super(loc);
        this.statement = s;
    }

    override Statement syntaxCopy()
    {
        return new ScopeStatement(loc, statement ? statement.syntaxCopy() : null);
    }

    override ScopeStatement isScopeStatement()
    {
        return this;
    }

    override ReturnStatement isReturnStatement()
    {
        if (statement)
            return statement.isReturnStatement();
        return null;
    }

    override Statement semantic(Scope* sc)
    {
        ScopeDsymbol sym;
        //printf("ScopeStatement::semantic(sc = %p)\n", sc);
        if (statement)
        {
            sym = new ScopeDsymbol();
            sym.parent = sc.scopesym;
            sc = sc.push(sym);
            Statements* a = statement.flatten(sc);
            if (a)
            {
                statement = new CompoundStatement(loc, a);
            }
            statement = statement.semantic(sc);
            if (statement)
            {
                if (statement.isErrorStatement())
                {
                    sc.pop();
                    return statement;
                }
                Statement sentry;
                Statement sexception;
                Statement sfinally;
                statement = statement.scopeCode(sc, &sentry, &sexception, &sfinally);
                assert(!sentry);
                assert(!sexception);
                if (sfinally)
                {
                    //printf("adding sfinally\n");
                    sfinally = sfinally.semantic(sc);
                    statement = new CompoundStatement(loc, statement, sfinally);
                }
            }
            sc.pop();
        }
        return this;
    }

    override bool hasBreak()
    {
        //printf("ScopeStatement::hasBreak() %s\n", toChars());
        return statement ? statement.hasBreak() : false;
    }

    override bool hasContinue()
    {
        return statement ? statement.hasContinue() : false;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class WhileStatement : Statement
{
public:
    Expression condition;
    Statement _body;
    Loc endloc;             // location of closing curly bracket

    extern (D) this(Loc loc, Expression c, Statement b, Loc endloc)
    {
        super(loc);
        condition = c;
        _body = b;
        this.endloc = endloc;
    }

    override Statement syntaxCopy()
    {
        return new WhileStatement(loc,
            condition.syntaxCopy(),
            _body ? _body.syntaxCopy() : null,
            endloc);
    }

    override Statement semantic(Scope* sc)
    {
        /* Rewrite as a for(;condition;) loop
         */
        Statement s = new ForStatement(loc, null, condition, null, _body, endloc);
        s = s.semantic(sc);
        return s;
    }

    override bool hasBreak()
    {
        return true;
    }

    override bool hasContinue()
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class DoStatement : Statement
{
public:
    Statement _body;
    Expression condition;

    extern (D) this(Loc loc, Statement b, Expression c)
    {
        super(loc);
        _body = b;
        condition = c;
    }

    override Statement syntaxCopy()
    {
        return new DoStatement(loc,
            _body ? _body.syntaxCopy() : null,
            condition.syntaxCopy());
    }

    override Statement semantic(Scope* sc)
    {
        sc.noctor++;
        if (_body)
            _body = _body.semanticScope(sc, this, this);
        sc.noctor--;

        // check in syntax level
        condition = checkAssignmentAsCondition(condition);

        condition = condition.semantic(sc);
        condition = resolveProperties(sc, condition);
        if (checkNonAssignmentArrayOp(condition))
            condition = new ErrorExp();
        condition = condition.optimize(WANTvalue);
        condition = checkGC(sc, condition);

        condition = condition.toBoolean(sc);

        if (condition.op == TOKerror)
            return new ErrorStatement();
        if (_body && _body.isErrorStatement())
            return _body;

        return this;
    }

    override bool hasBreak()
    {
        return true;
    }

    override bool hasContinue()
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ForStatement : Statement
{
public:
    Statement _init;
    Expression condition;
    Expression increment;
    Statement _body;
    Loc endloc;             // location of closing curly bracket

    // When wrapped in try/finally clauses, this points to the outermost one,
    // which may have an associated label. Internal break/continue statements
    // treat that label as referring to this loop.
    Statement relatedLabeled;

    extern (D) this(Loc loc, Statement _init, Expression condition, Expression increment, Statement _body, Loc endloc)
    {
        super(loc);
        this._init = _init;
        this.condition = condition;
        this.increment = increment;
        this._body = _body;
        this.endloc = endloc;
    }

    override Statement syntaxCopy()
    {
        return new ForStatement(loc,
            _init ? _init.syntaxCopy() : null,
            condition ? condition.syntaxCopy() : null,
            increment ? increment.syntaxCopy() : null,
            _body.syntaxCopy(),
            endloc);
    }

    override Statement semantic(Scope* sc)
    {
        //printf("ForStatement::semantic %s\n", toChars());

        if (_init)
        {
            /* Rewrite:
             *  for (auto v1 = i1, v2 = i2; condition; increment) { ... }
             * to:
             *  { auto v1 = i1, v2 = i2; for (; condition; increment) { ... } }
             * then lowered to:
             *  auto v1 = i1;
             *  try {
             *    auto v2 = i2;
             *    try {
             *      for (; condition; increment) { ... }
             *    } finally { v2.~this(); }
             *  } finally { v1.~this(); }
             */
            auto ainit = new Statements();
            ainit.push(_init), _init = null;
            ainit.push(this);
            Statement s = new CompoundStatement(loc, ainit);
            s = new ScopeStatement(loc, s);
            s = s.semantic(sc);
            if (!s.isErrorStatement())
            {
                if (LabelStatement ls = checkLabeledLoop(sc, this))
                    ls.gotoTarget = this;
                relatedLabeled = s;
            }
            return s;
        }
        assert(_init is null);

        auto sym = new ScopeDsymbol();
        sym.parent = sc.scopesym;
        sc = sc.push(sym);

        sc.noctor++;
        if (condition)
        {
            // check in syntax level
            condition = checkAssignmentAsCondition(condition);

            condition = condition.semantic(sc);
            condition = resolveProperties(sc, condition);
            if (checkNonAssignmentArrayOp(condition))
                condition = new ErrorExp();
            condition = condition.optimize(WANTvalue);
            condition = checkGC(sc, condition);

            condition = condition.toBoolean(sc);
        }
        if (increment)
        {
            increment = increment.semantic(sc);
            increment = resolveProperties(sc, increment);
            if (checkNonAssignmentArrayOp(increment))
                increment = new ErrorExp();
            increment = increment.optimize(WANTvalue);
            increment = checkGC(sc, increment);
        }

        sc.sbreak = this;
        sc.scontinue = this;
        if (_body)
            _body = _body.semanticNoScope(sc);
        sc.noctor--;

        sc.pop();

        if (condition && condition.op == TOKerror ||
            increment && increment.op == TOKerror ||
            _body && _body.isErrorStatement())
        {
            return new ErrorStatement();
        }
        return this;
    }

    override Statement scopeCode(Scope* sc, Statement* sentry, Statement* sexception, Statement* sfinally)
    {
        //printf("ForStatement::scopeCode()\n");
        Statement.scopeCode(sc, sentry, sexception, sfinally);
        return this;
    }

    override Statement getRelatedLabeled()
    {
        return relatedLabeled ? relatedLabeled : this;
    }

    override bool hasBreak()
    {
        //printf("ForStatement::hasBreak()\n");
        return true;
    }

    override bool hasContinue()
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ForeachStatement : Statement
{
public:
    TOK op;                     // TOKforeach or TOKforeach_reverse
    Parameters* parameters;     // array of Parameter*'s
    Expression aggr;
    Statement _body;
    Loc endloc;                 // location of closing curly bracket

    VarDeclaration key;
    VarDeclaration value;

    FuncDeclaration func;       // function we're lexically in

    Statements* cases;          // put breaks, continues, gotos and returns here
    ScopeStatements* gotos;     // forward referenced goto's go here

    extern (D) this(Loc loc, TOK op, Parameters* parameters, Expression aggr, Statement _body, Loc endloc)
    {
        super(loc);
        this.op = op;
        this.parameters = parameters;
        this.aggr = aggr;
        this._body = _body;
        this.endloc = endloc;
    }

    override Statement syntaxCopy()
    {
        return new ForeachStatement(loc, op,
            Parameter.arraySyntaxCopy(parameters),
            aggr.syntaxCopy(),
            _body ? _body.syntaxCopy() : null,
            endloc);
    }

    override Statement semantic(Scope* sc)
    {
        //printf("ForeachStatement::semantic() %p\n", this);
        ScopeDsymbol sym;
        Statement s = this;
        size_t dim = parameters.dim;
        TypeAArray taa = null;
        Dsymbol sapply = null;

        Type tn = null;
        Type tnv = null;

        func = sc.func;
        if (func.fes)
            func = func.fes.func;

        VarDeclaration vinit = null;
        aggr = aggr.semantic(sc);
        aggr = resolveProperties(sc, aggr);
        aggr = aggr.optimize(WANTvalue);
        if (aggr.op == TOKerror)
            return new ErrorStatement();
        Expression oaggr = aggr;
        if (aggr.type && aggr.type.toBasetype().ty == Tstruct &&
            (cast(TypeStruct)(aggr.type.toBasetype())).sym.dtor &&
            aggr.op != TOKtype && !aggr.isLvalue())
        {
            // https://issues.dlang.org/show_bug.cgi?id=14653
            // Extend the life of rvalue aggregate till the end of foreach.
            vinit = new VarDeclaration(loc, aggr.type, Identifier.generateId("__aggr"), new ExpInitializer(loc, aggr));
            vinit.storage_class |= STCtemp;
            vinit.semantic(sc);
            aggr = new VarExp(aggr.loc, vinit);
        }

        if (!inferAggregate(this, sc, sapply))
        {
            const(char)* msg = "";
            if (aggr.type && isAggregate(aggr.type))
            {
                msg = ", define opApply(), range primitives, or use .tupleof";
            }
            error("invalid foreach aggregate %s%s", oaggr.toChars(), msg);
            return new ErrorStatement();
        }

        Dsymbol sapplyOld = sapply; // 'sapply' will be NULL if and after 'inferApplyArgTypes' errors

        /* Check for inference errors
         */
        if (!inferApplyArgTypes(this, sc, sapply))
        {
            /**
             Try and extract the parameter count of the opApply callback function, e.g.:
             int opApply(int delegate(int, float)) => 2 args
             */
            bool foundMismatch = false;
            size_t foreachParamCount = 0;
            if (sapplyOld)
            {
                if (FuncDeclaration fd = sapplyOld.isFuncDeclaration())
                {
                    int fvarargs; // ignored (opApply shouldn't take variadics)
                    Parameters* fparameters = fd.getParameters(&fvarargs);

                    if (Parameter.dim(fparameters) == 1)
                    {
                        // first param should be the callback function
                        Parameter fparam = Parameter.getNth(fparameters, 0);
                        if ((fparam.type.ty == Tpointer ||
                             fparam.type.ty == Tdelegate) &&
                            fparam.type.nextOf().ty == Tfunction)
                        {
                            TypeFunction tf = cast(TypeFunction)fparam.type.nextOf();
                            foreachParamCount = Parameter.dim(tf.parameters);
                            foundMismatch = true;
                        }
                    }
                }
            }

            //printf("dim = %d, parameters->dim = %d\n", dim, parameters->dim);
            if (foundMismatch && dim != foreachParamCount)
            {
                const(char)* plural = foreachParamCount > 1 ? "s" : "";
                error("cannot infer argument types, expected %d argument%s, not %d",
                    foreachParamCount, plural, dim);
            }
            else
                error("cannot uniquely infer foreach argument types");

            return new ErrorStatement();
        }

        Type tab = aggr.type.toBasetype();

        if (tab.ty == Ttuple) // don't generate new scope for tuple loops
        {
            if (dim < 1 || dim > 2)
            {
                error("only one (value) or two (key,value) arguments for tuple foreach");
                return new ErrorStatement();
            }

            Type paramtype = (*parameters)[dim - 1].type;
            if (paramtype)
            {
                paramtype = paramtype.semantic(loc, sc);
                if (paramtype.ty == Terror)
                    return new ErrorStatement();
            }

            TypeTuple tuple = cast(TypeTuple)tab;
            auto statements = new Statements();
            //printf("aggr: op = %d, %s\n", aggr->op, aggr->toChars());
            size_t n;
            TupleExp te = null;
            if (aggr.op == TOKtuple) // expression tuple
            {
                te = cast(TupleExp)aggr;
                n = te.exps.dim;
            }
            else if (aggr.op == TOKtype) // type tuple
            {
                n = Parameter.dim(tuple.arguments);
            }
            else
                assert(0);
            foreach (j; 0 .. n)
            {
                size_t k = (op == TOKforeach) ? j : n - 1 - j;
                Expression e = null;
                Type t = null;
                if (te)
                    e = (*te.exps)[k];
                else
                    t = Parameter.getNth(tuple.arguments, k).type;
                Parameter p = (*parameters)[0];
                auto st = new Statements();

                if (dim == 2)
                {
                    // Declare key
                    if (p.storageClass & (STCout | STCref | STClazy))
                    {
                        error("no storage class for key %s", p.ident.toChars());
                        return new ErrorStatement();
                    }
                    p.type = p.type.semantic(loc, sc);
                    TY keyty = p.type.ty;
                    if (keyty != Tint32 && keyty != Tuns32)
                    {
                        if (global.params.isLP64)
                        {
                            if (keyty != Tint64 && keyty != Tuns64)
                            {
                                error("foreach: key type must be int or uint, long or ulong, not %s", p.type.toChars());
                                return new ErrorStatement();
                            }
                        }
                        else
                        {
                            error("foreach: key type must be int or uint, not %s", p.type.toChars());
                            return new ErrorStatement();
                        }
                    }
                    Initializer ie = new ExpInitializer(Loc(), new IntegerExp(k));
                    auto var = new VarDeclaration(loc, p.type, p.ident, ie);
                    var.storage_class |= STCmanifest;
                    st.push(new ExpStatement(loc, var));
                    p = (*parameters)[1]; // value
                }
                // Declare value
                if (p.storageClass & (STCout | STClazy) ||
                    p.storageClass & STCref && !te)
                {
                    error("no storage class for value %s", p.ident.toChars());
                    return new ErrorStatement();
                }
                Dsymbol var;
                if (te)
                {
                    Type tb = e.type.toBasetype();
                    Dsymbol ds = null;
                    if ((tb.ty == Tfunction || tb.ty == Tsarray) && e.op == TOKvar)
                        ds = (cast(VarExp)e).var;
                    else if (e.op == TOKtemplate)
                        ds = (cast(TemplateExp)e).td;
                    else if (e.op == TOKscope)
                        ds = (cast(ScopeExp)e).sds;
                    else if (e.op == TOKfunction)
                    {
                        auto fe = cast(FuncExp)e;
                        ds = fe.td ? cast(Dsymbol)fe.td : fe.fd;
                    }

                    if (ds)
                    {
                        var = new AliasDeclaration(loc, p.ident, ds);
                        if (p.storageClass & STCref)
                        {
                            error("symbol %s cannot be ref", s.toChars());
                            return new ErrorStatement();
                        }
                        if (paramtype)
                        {
                            error("cannot specify element type for symbol %s", ds.toChars());
                            return new ErrorStatement();
                        }
                    }
                    else if (e.op == TOKtype)
                    {
                        var = new AliasDeclaration(loc, p.ident, e.type);
                        if (paramtype)
                        {
                            error("cannot specify element type for type %s", e.type.toChars());
                            return new ErrorStatement();
                        }
                    }
                    else
                    {
                        p.type = e.type;
                        if (paramtype)
                            p.type = paramtype;
                        Initializer ie = new ExpInitializer(Loc(), e);
                        auto v = new VarDeclaration(loc, p.type, p.ident, ie);
                        if (p.storageClass & STCref)
                            v.storage_class |= STCref | STCforeach;
                        if (e.isConst() ||
                            e.op == TOKstring ||
                            e.op == TOKstructliteral ||
                            e.op == TOKarrayliteral)
                        {
                            if (v.storage_class & STCref)
                            {
                                error("constant value %s cannot be ref", ie.toChars());
                                return new ErrorStatement();
                            }
                            else
                                v.storage_class |= STCmanifest;
                        }
                        var = v;
                    }
                }
                else
                {
                    var = new AliasDeclaration(loc, p.ident, t);
                    if (paramtype)
                    {
                        error("cannot specify element type for symbol %s", s.toChars());
                        return new ErrorStatement();
                    }
                }
                st.push(new ExpStatement(loc, var));

                st.push(_body.syntaxCopy());
                s = new CompoundStatement(loc, st);
                s = new ScopeStatement(loc, s);
                statements.push(s);
            }

            s = new UnrolledLoopStatement(loc, statements);
            if (LabelStatement ls = checkLabeledLoop(sc, this))
                ls.gotoTarget = s;
            if (te && te.e0)
                s = new CompoundStatement(loc, new ExpStatement(te.e0.loc, te.e0), s);
            if (vinit)
                s = new CompoundStatement(loc, new ExpStatement(loc, vinit), s);
            s = s.semantic(sc);
            return s;
        }

        sym = new ScopeDsymbol();
        sym.parent = sc.scopesym;
        auto sc2 = sc.push(sym);
        sc2.noctor++;
        switch (tab.ty)
        {
        case Tarray:
        case Tsarray:
            {
                if (checkForArgTypes())
                    return this;

                if (dim < 1 || dim > 2)
                {
                    error("only one or two arguments for array foreach");
                    goto Lerror2;
                }

                /* Look for special case of parsing char types out of char type
                 * array.
                 */
                tn = tab.nextOf().toBasetype();
                if (tn.ty == Tchar || tn.ty == Twchar || tn.ty == Tdchar)
                {
                    int i = (dim == 1) ? 0 : 1; // index of value
                    Parameter p = (*parameters)[i];
                    p.type = p.type.semantic(loc, sc2);
                    p.type = p.type.addStorageClass(p.storageClass);
                    tnv = p.type.toBasetype();
                    if (tnv.ty != tn.ty &&
                        (tnv.ty == Tchar || tnv.ty == Twchar || tnv.ty == Tdchar))
                    {
                        if (p.storageClass & STCref)
                        {
                            error("foreach: value of UTF conversion cannot be ref");
                            goto Lerror2;
                        }
                        if (dim == 2)
                        {
                            p = (*parameters)[0];
                            if (p.storageClass & STCref)
                            {
                                error("foreach: key cannot be ref");
                                goto Lerror2;
                            }
                        }
                        goto Lapply;
                    }
                }

                foreach (i; 0 .. dim)
                {
                    // Declare parameterss
                    Parameter p = (*parameters)[i];
                    p.type = p.type.semantic(loc, sc2);
                    p.type = p.type.addStorageClass(p.storageClass);
                    VarDeclaration var;

                    if (dim == 2 && i == 0)
                    {
                        var = new VarDeclaration(loc, p.type.mutableOf(), Identifier.generateId("__key"), null);
                        var.storage_class |= STCtemp | STCforeach;
                        if (var.storage_class & (STCref | STCout))
                            var.storage_class |= STCnodtor;

                        key = var;
                        if (p.storageClass & STCref)
                        {
                            if (var.type.constConv(p.type) <= MATCHnomatch)
                            {
                                error("key type mismatch, %s to ref %s",
                                    var.type.toChars(), p.type.toChars());
                                goto Lerror2;
                            }
                        }
                        if (tab.ty == Tsarray)
                        {
                            TypeSArray ta = cast(TypeSArray)tab;
                            IntRange dimrange = getIntRange(ta.dim);
                            if (!IntRange.fromType(var.type).contains(dimrange))
                            {
                                error("index type '%s' cannot cover index range 0..%llu",
                                    p.type.toChars(), ta.dim.toInteger());
                                goto Lerror2;
                            }
                            key.range = new IntRange(SignExtendedNumber(0), dimrange.imax);
                        }
                    }
                    else
                    {
                        var = new VarDeclaration(loc, p.type, p.ident, null);
                        var.storage_class |= STCforeach;
                        var.storage_class |= p.storageClass & (STCin | STCout | STCref | STC_TYPECTOR);
                        if (var.storage_class & (STCref | STCout))
                            var.storage_class |= STCnodtor;

                        value = var;
                        if (var.storage_class & STCref)
                        {
                            if (aggr.checkModifiable(sc2, 1) == 2)
                                var.storage_class |= STCctorinit;

                            Type t = tab.nextOf();
                            if (t.constConv(p.type) <= MATCHnomatch)
                            {
                                error("argument type mismatch, %s to ref %s",
                                    t.toChars(), p.type.toChars());
                                goto Lerror2;
                            }
                        }
                    }
                }

                /* Convert to a ForStatement
                 *   foreach (key, value; a) body =>
                 *   for (T[] tmp = a[], size_t key; key < tmp.length; ++key)
                 *   { T value = tmp[k]; body }
                 *
                 *   foreach_reverse (key, value; a) body =>
                 *   for (T[] tmp = a[], size_t key = tmp.length; key--; )
                 *   { T value = tmp[k]; body }
                 */
                Identifier id = Identifier.generateId("__r");
                auto ie = new ExpInitializer(loc, new SliceExp(loc, aggr, null, null));
                VarDeclaration tmp;
                if (aggr.op == TOKarrayliteral &&
                    !((*parameters)[dim - 1].storageClass & STCref))
                {
                    auto ale = cast(ArrayLiteralExp)aggr;
                    size_t edim = ale.elements ? ale.elements.dim : 0;
                    aggr.type = tab.nextOf().sarrayOf(edim);

                    // for (T[edim] tmp = a, ...)
                    tmp = new VarDeclaration(loc, aggr.type, id, ie);
                }
                else
                    tmp = new VarDeclaration(loc, tab.nextOf().arrayOf(), id, ie);
                tmp.storage_class |= STCtemp;

                Expression tmp_length = new DotIdExp(loc, new VarExp(loc, tmp), Id.length);

                if (!key)
                {
                    Identifier idkey = Identifier.generateId("__key");
                    key = new VarDeclaration(loc, Type.tsize_t, idkey, null);
                    key.storage_class |= STCtemp;
                }
                if (op == TOKforeach_reverse)
                    key._init = new ExpInitializer(loc, tmp_length);
                else
                    key._init = new ExpInitializer(loc, new IntegerExp(loc, 0, key.type));

                auto cs = new Statements();
                if (vinit)
                    cs.push(new ExpStatement(loc, vinit));
                cs.push(new ExpStatement(loc, tmp));
                cs.push(new ExpStatement(loc, key));
                Statement forinit = new CompoundDeclarationStatement(loc, cs);

                Expression cond;
                if (op == TOKforeach_reverse)
                {
                    // key--
                    cond = new PostExp(TOKminusminus, loc, new VarExp(loc, key));
                }
                else
                {
                    // key < tmp.length
                    cond = new CmpExp(TOKlt, loc, new VarExp(loc, key), tmp_length);
                }

                Expression increment = null;
                if (op == TOKforeach)
                {
                    // key += 1
                    increment = new AddAssignExp(loc, new VarExp(loc, key), new IntegerExp(loc, 1, key.type));
                }

                // T value = tmp[key];
                IndexExp indexExp = new IndexExp(loc, new VarExp(loc, tmp), new VarExp(loc, key));
                indexExp.indexIsInBounds = true; // disabling bounds checking in foreach statements.
                value._init = new ExpInitializer(loc, indexExp);
                Statement ds = new ExpStatement(loc, value);

                if (dim == 2)
                {
                    Parameter p = (*parameters)[0];
                    if ((p.storageClass & STCref) && p.type.equals(key.type))
                    {
                        key.range = null;
                        auto v = new AliasDeclaration(loc, p.ident, key);
                        _body = new CompoundStatement(loc, new ExpStatement(loc, v), _body);
                    }
                    else
                    {
                        auto ei = new ExpInitializer(loc, new IdentifierExp(loc, key.ident));
                        auto v = new VarDeclaration(loc, p.type, p.ident, ei);
                        v.storage_class |= STCforeach | (p.storageClass & STCref);
                        _body = new CompoundStatement(loc, new ExpStatement(loc, v), _body);
                        if (key.range && !p.type.isMutable())
                        {
                            /* Limit the range of the key to the specified range
                             */
                            v.range = new IntRange(key.range.imin, key.range.imax - SignExtendedNumber(1));
                        }
                    }
                }
                _body = new CompoundStatement(loc, ds, _body);

                s = new ForStatement(loc, forinit, cond, increment, _body, endloc);
                if (auto ls = checkLabeledLoop(sc, this))   // Bugzilla 15450: don't use sc2
                    ls.gotoTarget = s;
                s = s.semantic(sc2);
                break;
            }
        case Taarray:
            if (op == TOKforeach_reverse)
                warning("cannot use foreach_reverse with an associative array");
            if (checkForArgTypes())
                return this;

            taa = cast(TypeAArray)tab;
            if (dim < 1 || dim > 2)
            {
                error("only one or two arguments for associative array foreach");
                goto Lerror2;
            }
            goto Lapply;

        case Tclass:
        case Tstruct:
            /* Prefer using opApply, if it exists
             */
            if (sapply)
                goto Lapply;
            {
                /* Look for range iteration, i.e. the properties
                 * .empty, .popFront, .popBack, .front and .back
                 *    foreach (e; aggr) { ... }
                 * translates to:
                 *    for (auto __r = aggr[]; !__r.empty; __r.popFront()) {
                 *        auto e = __r.front;
                 *        ...
                 *    }
                 */
                auto ad = (tab.ty == Tclass) ?
                    cast(AggregateDeclaration)(cast(TypeClass)tab).sym :
                    cast(AggregateDeclaration)(cast(TypeStruct)tab).sym;
                Identifier idfront;
                Identifier idpopFront;
                if (op == TOKforeach)
                {
                    idfront = Id.Ffront;
                    idpopFront = Id.FpopFront;
                }
                else
                {
                    idfront = Id.Fback;
                    idpopFront = Id.FpopBack;
                }
                auto sfront = ad.search(Loc(), idfront);
                if (!sfront)
                    goto Lapply;

                /* Generate a temporary __r and initialize it with the aggregate.
                 */
                VarDeclaration r;
                Statement _init;
                if (vinit && aggr.op == TOKvar && (cast(VarExp)aggr).var == vinit)
                {
                    r = vinit;
                    _init = new ExpStatement(loc, vinit);
                }
                else
                {
                    auto rid = Identifier.generateId("__r");
                    r = new VarDeclaration(loc, null, rid, new ExpInitializer(loc, aggr));
                    r.storage_class |= STCtemp;
                    _init = new ExpStatement(loc, r);
                    if (vinit)
                        _init = new CompoundStatement(loc, new ExpStatement(loc, vinit), _init);
                }

                // !__r.empty
                Expression e = new VarExp(loc, r);
                e = new DotIdExp(loc, e, Id.Fempty);
                Expression condition = new NotExp(loc, e);

                // __r.idpopFront()
                e = new VarExp(loc, r);
                Expression increment = new CallExp(loc, new DotIdExp(loc, e, idpopFront));

                /* Declaration statement for e:
                 *    auto e = __r.idfront;
                 */
                e = new VarExp(loc, r);
                Expression einit = new DotIdExp(loc, e, idfront);
                Statement makeargs, forbody;
                if (dim == 1)
                {
                    auto p = (*parameters)[0];
                    auto ve = new VarDeclaration(loc, p.type, p.ident, new ExpInitializer(loc, einit));
                    ve.storage_class |= STCforeach;
                    ve.storage_class |= p.storageClass & (STCin | STCout | STCref | STC_TYPECTOR);

                    makeargs = new ExpStatement(loc, ve);
                }
                else
                {
                    auto id = Identifier.generateId("__front");
                    auto ei = new ExpInitializer(loc, einit);
                    auto vd = new VarDeclaration(loc, null, id, ei);
                    vd.storage_class |= STCtemp | STCctfe | STCref | STCforeach;

                    makeargs = new ExpStatement(loc, vd);

                    Type tfront;
                    if (auto fd = sfront.isFuncDeclaration())
                    {
                        if (!fd.functionSemantic())
                            goto Lrangeerr;
                        tfront = fd.type;
                    }
                    else if (auto td = sfront.isTemplateDeclaration())
                    {
                        Expressions a;
                        if (auto f = resolveFuncCall(loc, sc, td, null, tab, &a, 1))
                            tfront = f.type;
                    }
                    else if (auto d = sfront.isDeclaration())
                    {
                        tfront = d.type;
                    }
                    if (!tfront || tfront.ty == Terror)
                        goto Lrangeerr;
                    if (tfront.toBasetype().ty == Tfunction)
                        tfront = tfront.toBasetype().nextOf();
                    if (tfront.ty == Tvoid)
                    {
                        error("%s.front is void and has no value", oaggr.toChars());
                        goto Lerror2;
                    }
                    // Resolve inout qualifier of front type
                    tfront = tfront.substWildTo(tab.mod);

                    Expression ve = new VarExp(loc, vd);
                    ve.type = tfront;

                    auto exps = new Expressions();
                    exps.push(ve);
                    int pos = 0;
                    while (exps.dim < dim)
                    {
                        pos = expandAliasThisTuples(exps, pos);
                        if (pos == -1)
                            break;
                    }
                    if (exps.dim != dim)
                    {
                        const(char)* plural = exps.dim > 1 ? "s" : "";
                        error("cannot infer argument types, expected %d argument%s, not %d",
                            exps.dim, plural, dim);
                        goto Lerror2;
                    }

                    foreach (i; 0 .. dim)
                    {
                        auto p = (*parameters)[i];
                        auto exp = (*exps)[i];
                        version (none)
                        {
                            printf("[%d] p = %s %s, exp = %s %s\n", i,
                                p.type ? p.type.toChars() : "?", p.ident.toChars(),
                                exp.type.toChars(), exp.toChars());
                        }
                        if (!p.type)
                            p.type = exp.type;
                        p.type = p.type.addStorageClass(p.storageClass).semantic(loc, sc2);
                        if (!exp.implicitConvTo(p.type))
                            goto Lrangeerr;

                        auto var = new VarDeclaration(loc, p.type, p.ident, new ExpInitializer(loc, exp));
                        var.storage_class |= STCctfe | STCref | STCforeach;
                        makeargs = new CompoundStatement(loc, makeargs, new ExpStatement(loc, var));
                    }
                }

                forbody = new CompoundStatement(loc, makeargs, this._body);

                s = new ForStatement(loc, _init, condition, increment, forbody, endloc);
                if (auto ls = checkLabeledLoop(sc, this))
                    ls.gotoTarget = s;

                version (none)
                {
                    printf("init: %s\n", _init.toChars());
                    printf("condition: %s\n", condition.toChars());
                    printf("increment: %s\n", increment.toChars());
                    printf("body: %s\n", forbody.toChars());
                }
                s = s.semantic(sc2);
                break;

            Lrangeerr:
                error("cannot infer argument types");
                goto Lerror2;
            }
        case Tdelegate:
            if (op == TOKforeach_reverse)
                deprecation("cannot use foreach_reverse with a delegate");
        Lapply:
            {
                if (checkForArgTypes())
                {
                    _body = _body.semanticNoScope(sc2);
                    return this;
                }

                TypeFunction tfld = null;
                if (sapply)
                {
                    FuncDeclaration fdapply = sapply.isFuncDeclaration();
                    if (fdapply)
                    {
                        assert(fdapply.type && fdapply.type.ty == Tfunction);
                        tfld = cast(TypeFunction)fdapply.type.semantic(loc, sc2);
                        goto Lget;
                    }
                    else if (tab.ty == Tdelegate)
                    {
                        tfld = cast(TypeFunction)tab.nextOf();
                    Lget:
                        //printf("tfld = %s\n", tfld->toChars());
                        if (tfld.parameters.dim == 1)
                        {
                            Parameter p = Parameter.getNth(tfld.parameters, 0);
                            if (p.type && p.type.ty == Tdelegate)
                            {
                                auto t = p.type.semantic(loc, sc2);
                                assert(t.ty == Tdelegate);
                                tfld = cast(TypeFunction)t.nextOf();
                            }
                        }
                    }
                }

                /* Turn body into the function literal:
                 *  int delegate(ref T param) { body }
                 */
                auto params = new Parameters();
                foreach (i; 0 .. dim)
                {
                    Parameter p = (*parameters)[i];
                    StorageClass stc = STCref;
                    Identifier id;

                    p.type = p.type.semantic(loc, sc2);
                    p.type = p.type.addStorageClass(p.storageClass);
version(IN_LLVM)
{
                    // Type of parameter may be different; see below
                    auto para_type = p.type;
}
                    if (tfld)
                    {
                        Parameter prm = Parameter.getNth(tfld.parameters, i);
                        //printf("\tprm = %s%s\n", (prm->storageClass&STCref?"ref ":""), prm->ident->toChars());
                        stc = prm.storageClass & STCref;
                        id = p.ident; // argument copy is not need.
                        if ((p.storageClass & STCref) != stc)
                        {
                            if (!stc)
                            {
                                error("foreach: cannot make %s ref", p.ident.toChars());
                                goto Lerror2;
                            }
                            goto LcopyArg;
                        }
                    }
                    else if (p.storageClass & STCref)
                    {
                        // default delegate parameters are marked as ref, then
                        // argument copy is not need.
                        id = p.ident;
                    }
                    else
                    {
                        // Make a copy of the ref argument so it isn't
                        // a reference.
                    LcopyArg:
                        id = Identifier.generateId("__applyArg", cast(int)i);
version(IN_LLVM)
{
                        // In case of a foreach loop on an array the index passed
                        // to the delegate is always of type size_t. The type of
                        // the parameter must be changed to size_t and a cast to
                        // the type used must be inserted. Otherwise the index is
                        // always 0 on a big endian architecture. This fixes
                        // issue #326.
                        Initializer ie;
                        if (dim == 2 && i == 0 && (tab.ty == Tarray || tab.ty == Tsarray))
                        {
                            para_type = Type.tsize_t;
                            ie = new ExpInitializer(Loc(),
                                     new CastExp(Loc(),
                                         new IdentifierExp(Loc(), id), p.type));
                        }
                        else
                        {
                            ie = new ExpInitializer(Loc(), new IdentifierExp(Loc(), id));
                        }
}
else
{
                        Initializer ie = new ExpInitializer(Loc(), new IdentifierExp(Loc(), id));
}
                        auto v = new VarDeclaration(Loc(), p.type, p.ident, ie);
                        v.storage_class |= STCtemp;
                        s = new ExpStatement(Loc(), v);
                        _body = new CompoundStatement(loc, s, _body);
                    }
version(IN_LLVM)
                    params.push(new Parameter(stc, para_type, id, null));
else
                    params.push(new Parameter(stc, p.type, id, null));
                }
                // Bugzilla 13840: Throwable nested function inside nothrow function is acceptable.
                StorageClass stc = mergeFuncAttrs(STCsafe | STCpure | STCnogc, func);
                tfld = new TypeFunction(params, Type.tint32, 0, LINKd, stc);
                cases = new Statements();
                gotos = new ScopeStatements();
                auto fld = new FuncLiteralDeclaration(loc, Loc(), tfld, TOKdelegate, this);
                fld.fbody = _body;
                Expression flde = new FuncExp(loc, fld);
                flde = flde.semantic(sc2);
                fld.tookAddressOf = 0;

                // Resolve any forward referenced goto's
                foreach (i; 0 .. gotos.dim)
                {
                    GotoStatement gs = cast(GotoStatement)(*gotos)[i].statement;
                    if (!gs.label.statement)
                    {
                        // 'Promote' it to this scope, and replace with a return
                        cases.push(gs);
                        s = new ReturnStatement(Loc(), new IntegerExp(cases.dim + 1));
                        (*gotos)[i].statement = s;
                    }
                }

                Expression e = null;
                Expression ec;
                if (vinit)
                {
                    e = new DeclarationExp(loc, vinit);
                    e = e.semantic(sc2);
                    if (e.op == TOKerror)
                        goto Lerror2;
                }

                if (taa)
                {
                    // Check types
                    Parameter p = (*parameters)[0];
                    bool isRef = (p.storageClass & STCref) != 0;
                    Type ta = p.type;
                    if (dim == 2)
                    {
                        Type ti = (isRef ? taa.index.addMod(MODconst) : taa.index);
                        if (isRef ? !ti.constConv(ta) : !ti.implicitConvTo(ta))
                        {
                            error("foreach: index must be type %s, not %s",
                                ti.toChars(), ta.toChars());
                            goto Lerror2;
                        }
                        p = (*parameters)[1];
                        isRef = (p.storageClass & STCref) != 0;
                        ta = p.type;
                    }
                    Type taav = taa.nextOf();
                    if (isRef ? !taav.constConv(ta) : !taav.implicitConvTo(ta))
                    {
                        error("foreach: value must be type %s, not %s",
                            taav.toChars(), ta.toChars());
                        goto Lerror2;
                    }

                    /* Call:
                     *  extern(C) int _aaApply(void*, in size_t, int delegate(void*))
                     *      _aaApply(aggr, keysize, flde)
                     *
                     *  extern(C) int _aaApply2(void*, in size_t, int delegate(void*, void*))
                     *      _aaApply2(aggr, keysize, flde)
                     */
                    static __gshared const(char)** name = ["_aaApply", "_aaApply2"];
                    static __gshared FuncDeclaration* fdapply = [null, null];
                    static __gshared TypeDelegate* fldeTy = [null, null];

                    ubyte i = (dim == 2 ? 1 : 0);
                    if (!fdapply[i])
                    {
                        params = new Parameters();
                        params.push(new Parameter(0, Type.tvoid.pointerTo(), null, null));
                        params.push(new Parameter(STCin, Type.tsize_t, null, null));
                        auto dgparams = new Parameters();
                        dgparams.push(new Parameter(0, Type.tvoidptr, null, null));
                        if (dim == 2)
                            dgparams.push(new Parameter(0, Type.tvoidptr, null, null));
                        fldeTy[i] = new TypeDelegate(new TypeFunction(dgparams, Type.tint32, 0, LINKd));
                        params.push(new Parameter(0, fldeTy[i], null, null));
                        fdapply[i] = FuncDeclaration.genCfunc(params, Type.tint32, name[i]);
                    }

                    auto exps = new Expressions();
                    exps.push(aggr);
                    size_t keysize = cast(size_t)taa.index.size();
                    keysize = (keysize + (cast(size_t)Target.ptrsize - 1)) & ~(cast(size_t)Target.ptrsize - 1);
                    // paint delegate argument to the type runtime expects
                    if (!fldeTy[i].equals(flde.type))
                    {
                        flde = new CastExp(loc, flde, flde.type);
                        flde.type = fldeTy[i];
                    }
                    exps.push(new IntegerExp(Loc(), keysize, Type.tsize_t));
                    exps.push(flde);
                    ec = new VarExp(Loc(), fdapply[i], false);
                    ec = new CallExp(loc, ec, exps);
                    ec.type = Type.tint32; // don't run semantic() on ec
                }
                else if (tab.ty == Tarray || tab.ty == Tsarray)
                {
                    /* Call:
                     *      _aApply(aggr, flde)
                     */
                    static __gshared const(char)** fntab =
                    [
                        "cc", "cw", "cd",
                        "wc", "cc", "wd",
                        "dc", "dw", "dd"
                    ];

                    const(size_t) BUFFER_LEN = 7 + 1 + 2 + dim.sizeof * 3 + 1;
                    char[BUFFER_LEN] fdname;
                    int flag;

                    switch (tn.ty)
                    {
                        case Tchar:     flag = 0;   break;
                        case Twchar:    flag = 3;   break;
                        case Tdchar:    flag = 6;   break;
                    default:
                        assert(0);
                    }
                    switch (tnv.ty)
                    {
                        case Tchar:     flag += 0;  break;
                        case Twchar:    flag += 1;  break;
                        case Tdchar:    flag += 2;  break;
                    default:
                        assert(0);
                    }
                    const(char)* r = (op == TOKforeach_reverse) ? "R" : "";
                    int j = sprintf(fdname.ptr, "_aApply%s%.*s%llu", r, 2, fntab[flag], cast(ulong)dim);
                    assert(j < BUFFER_LEN);

                    FuncDeclaration fdapply;
                    TypeDelegate dgty;
                    params = new Parameters();
                    params.push(new Parameter(STCin, tn.arrayOf(), null, null));
                    auto dgparams = new Parameters();
                    dgparams.push(new Parameter(0, Type.tvoidptr, null, null));
                    if (dim == 2)
                        dgparams.push(new Parameter(0, Type.tvoidptr, null, null));
                    dgty = new TypeDelegate(new TypeFunction(dgparams, Type.tint32, 0, LINKd));
                    params.push(new Parameter(0, dgty, null, null));
                    fdapply = FuncDeclaration.genCfunc(params, Type.tint32, fdname.ptr);

                    if (tab.ty == Tsarray)
                        aggr = aggr.castTo(sc2, tn.arrayOf());
                    // paint delegate argument to the type runtime expects
                    if (!dgty.equals(flde.type))
                    {
                        flde = new CastExp(loc, flde, flde.type);
                        flde.type = dgty;
                    }
                    ec = new VarExp(Loc(), fdapply, false);
                    ec = new CallExp(loc, ec, aggr, flde);
                    ec.type = Type.tint32; // don't run semantic() on ec
                }
                else if (tab.ty == Tdelegate)
                {
                    /* Call:
                     *      aggr(flde)
                     */
                    if (aggr.op == TOKdelegate && (cast(DelegateExp)aggr).func.isNested())
                    {
                        // See Bugzilla 3560
                        aggr = (cast(DelegateExp)aggr).e1;
                    }
                    ec = new CallExp(loc, aggr, flde);
                    ec = ec.semantic(sc2);
                    if (ec.op == TOKerror)
                        goto Lerror2;
                    if (ec.type != Type.tint32)
                    {
                        error("opApply() function for %s must return an int", tab.toChars());
                        goto Lerror2;
                    }
                }
                else
                {
                    assert(tab.ty == Tstruct || tab.ty == Tclass);
                    assert(sapply);
                    /* Call:
                     *  aggr.apply(flde)
                     */
                    ec = new DotIdExp(loc, aggr, sapply.ident);
                    ec = new CallExp(loc, ec, flde);
                    ec = ec.semantic(sc2);
                    if (ec.op == TOKerror)
                        goto Lerror2;
                    if (ec.type != Type.tint32)
                    {
                        error("opApply() function for %s must return an int", tab.toChars());
                        goto Lerror2;
                    }
                }
                e = Expression.combine(e, ec);

                if (!cases.dim)
                {
                    // Easy case, a clean exit from the loop
                    e = new CastExp(loc, e, Type.tvoid); // Bugzilla 13899
                    s = new ExpStatement(loc, e);
                }
                else
                {
                    // Construct a switch statement around the return value
                    // of the apply function.
                    auto a = new Statements();

                    // default: break; takes care of cases 0 and 1
                    s = new BreakStatement(Loc(), null);
                    s = new DefaultStatement(Loc(), s);
                    a.push(s);

                    // cases 2...
                    foreach (i, c; *cases)
                    {
                        s = new CaseStatement(Loc(), new IntegerExp(i + 2), c);
                        a.push(s);
                    }

                    s = new CompoundStatement(loc, a);
                    s = new SwitchStatement(loc, e, s, false);
                }
                s = s.semantic(sc2);
                break;
            }
        case Terror:
        Lerror2:
            s = new ErrorStatement();
            break;

        default:
            error("foreach: %s is not an aggregate type", aggr.type.toChars());
            goto Lerror2;
        }
        sc2.noctor--;
        sc2.pop();
        return s;
    }

    bool checkForArgTypes()
    {
        bool result = false;
        foreach (p; *parameters)
        {
            if (!p.type)
            {
                error("cannot infer type for %s", p.ident.toChars());
                p.type = Type.terror;
                result = true;
            }
        }
        return result;
    }

    override bool hasBreak()
    {
        return true;
    }

    override bool hasContinue()
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ForeachRangeStatement : Statement
{
public:
    TOK op;                 // TOKforeach or TOKforeach_reverse
    Parameter prm;          // loop index variable
    Expression lwr;
    Expression upr;
    Statement _body;
    Loc endloc;             // location of closing curly bracket

    VarDeclaration key;

    extern (D) this(Loc loc, TOK op, Parameter prm, Expression lwr, Expression upr, Statement _body, Loc endloc)
    {
        super(loc);
        this.op = op;
        this.prm = prm;
        this.lwr = lwr;
        this.upr = upr;
        this._body = _body;
        this.endloc = endloc;
    }

    override Statement syntaxCopy()
    {
        return new ForeachRangeStatement(loc, op, prm.syntaxCopy(), lwr.syntaxCopy(), upr.syntaxCopy(), _body ? _body.syntaxCopy() : null, endloc);
    }

    override Statement semantic(Scope* sc)
    {
        //printf("ForeachRangeStatement::semantic() %p\n", this);
        lwr = lwr.semantic(sc);
        lwr = resolveProperties(sc, lwr);
        lwr = lwr.optimize(WANTvalue);
        if (!lwr.type)
        {
            error("invalid range lower bound %s", lwr.toChars());
        Lerror:
            return new ErrorStatement();
        }
        upr = upr.semantic(sc);
        upr = resolveProperties(sc, upr);
        upr = upr.optimize(WANTvalue);
        if (!upr.type)
        {
            error("invalid range upper bound %s", upr.toChars());
            goto Lerror;
        }
        if (prm.type)
        {
            prm.type = prm.type.semantic(loc, sc);
            prm.type = prm.type.addStorageClass(prm.storageClass);
            lwr = lwr.implicitCastTo(sc, prm.type);
            if (upr.implicitConvTo(prm.type) || (prm.storageClass & STCref))
            {
                upr = upr.implicitCastTo(sc, prm.type);
            }
            else
            {
                // See if upr-1 fits in prm->type
                Expression limit = new MinExp(loc, upr, new IntegerExp(1));
                limit = limit.semantic(sc);
                limit = limit.optimize(WANTvalue);
                if (!limit.implicitConvTo(prm.type))
                {
                    upr = upr.implicitCastTo(sc, prm.type);
                }
            }
        }
        else
        {
            /* Must infer types from lwr and upr
             */
            Type tlwr = lwr.type.toBasetype();
            if (tlwr.ty == Tstruct || tlwr.ty == Tclass)
            {
                /* Just picking the first really isn't good enough.
                 */
                prm.type = lwr.type;
            }
            else if (lwr.type == upr.type)
            {
                /* Same logic as CondExp ?lwr:upr
                 */
                prm.type = lwr.type;
            }
            else
            {
                scope AddExp ea = new AddExp(loc, lwr, upr);
                if (typeCombine(ea, sc))
                    return new ErrorStatement();
                prm.type = ea.type;
                lwr = ea.e1;
                upr = ea.e2;
            }
            prm.type = prm.type.addStorageClass(prm.storageClass);
        }
        if (prm.type.ty == Terror || lwr.op == TOKerror || upr.op == TOKerror)
        {
            return new ErrorStatement();
        }
        /* Convert to a for loop:
         *  foreach (key; lwr .. upr) =>
         *  for (auto key = lwr, auto tmp = upr; key < tmp; ++key)
         *
         *  foreach_reverse (key; lwr .. upr) =>
         *  for (auto tmp = lwr, auto key = upr; key-- > tmp;)
         */
        auto ie = new ExpInitializer(loc, (op == TOKforeach) ? lwr : upr);
        key = new VarDeclaration(loc, upr.type.mutableOf(), Identifier.generateId("__key"), ie);
        key.storage_class |= STCtemp;
        SignExtendedNumber lower = getIntRange(lwr).imin;
        SignExtendedNumber upper = getIntRange(upr).imax;
        if (lower <= upper)
        {
            key.range = new IntRange(lower, upper);
        }
        Identifier id = Identifier.generateId("__limit");
        ie = new ExpInitializer(loc, (op == TOKforeach) ? upr : lwr);
        auto tmp = new VarDeclaration(loc, upr.type, id, ie);
        tmp.storage_class |= STCtemp;
        auto cs = new Statements();
        // Keep order of evaluation as lwr, then upr
        if (op == TOKforeach)
        {
            cs.push(new ExpStatement(loc, key));
            cs.push(new ExpStatement(loc, tmp));
        }
        else
        {
            cs.push(new ExpStatement(loc, tmp));
            cs.push(new ExpStatement(loc, key));
        }
        Statement forinit = new CompoundDeclarationStatement(loc, cs);
        Expression cond;
        if (op == TOKforeach_reverse)
        {
            cond = new PostExp(TOKminusminus, loc, new VarExp(loc, key));
            if (prm.type.isscalar())
            {
                // key-- > tmp
                cond = new CmpExp(TOKgt, loc, cond, new VarExp(loc, tmp));
            }
            else
            {
                // key-- != tmp
                cond = new EqualExp(TOKnotequal, loc, cond, new VarExp(loc, tmp));
            }
        }
        else
        {
            if (prm.type.isscalar())
            {
                // key < tmp
                cond = new CmpExp(TOKlt, loc, new VarExp(loc, key), new VarExp(loc, tmp));
            }
            else
            {
                // key != tmp
                cond = new EqualExp(TOKnotequal, loc, new VarExp(loc, key), new VarExp(loc, tmp));
            }
        }
        Expression increment = null;
        if (op == TOKforeach)
        {
            // key += 1
            //increment = new AddAssignExp(loc, new VarExp(loc, key), new IntegerExp(1));
            increment = new PreExp(TOKpreplusplus, loc, new VarExp(loc, key));
        }
        if ((prm.storageClass & STCref) && prm.type.equals(key.type))
        {
            key.range = null;
            auto v = new AliasDeclaration(loc, prm.ident, key);
            _body = new CompoundStatement(loc, new ExpStatement(loc, v), _body);
        }
        else
        {
            ie = new ExpInitializer(loc, new CastExp(loc, new VarExp(loc, key), prm.type));
            auto v = new VarDeclaration(loc, prm.type, prm.ident, ie);
            v.storage_class |= STCtemp | STCforeach | (prm.storageClass & STCref);
            _body = new CompoundStatement(loc, new ExpStatement(loc, v), _body);
            if (key.range && !prm.type.isMutable())
            {
                /* Limit the range of the key to the specified range
                 */
                v.range = new IntRange(key.range.imin, key.range.imax - SignExtendedNumber(1));
            }
        }
        if (prm.storageClass & STCref)
        {
            if (key.type.constConv(prm.type) <= MATCHnomatch)
            {
                error("prmument type mismatch, %s to ref %s", key.type.toChars(), prm.type.toChars());
                goto Lerror;
            }
        }
        auto s = new ForStatement(loc, forinit, cond, increment, _body, endloc);
        if (LabelStatement ls = checkLabeledLoop(sc, this))
            ls.gotoTarget = s;
        return s.semantic(sc);
    }

    override bool hasBreak()
    {
        return true;
    }

    override bool hasContinue()
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class IfStatement : Statement
{
public:
    Parameter prm;
    Expression condition;
    Statement ifbody;
    Statement elsebody;
    VarDeclaration match;   // for MatchExpression results

    extern (D) this(Loc loc, Parameter prm, Expression condition, Statement ifbody, Statement elsebody)
    {
        super(loc);
        this.prm = prm;
        this.condition = condition;
        this.ifbody = ifbody;
        this.elsebody = elsebody;
    }

    override Statement syntaxCopy()
    {
        return new IfStatement(loc,
            prm ? prm.syntaxCopy() : null,
            condition.syntaxCopy(),
            ifbody ? ifbody.syntaxCopy() : null,
            elsebody ? elsebody.syntaxCopy() : null);
    }

    override Statement semantic(Scope* sc)
    {
        // Evaluate at runtime
        uint cs0 = sc.callSuper;
        uint cs1;
        uint* fi0 = sc.saveFieldInit();
        uint* fi1 = null;

        // check in syntax level
        condition = checkAssignmentAsCondition(condition);

        auto sym = new ScopeDsymbol();
        sym.parent = sc.scopesym;
        Scope* scd = sc.push(sym);
        if (prm)
        {
            /* Declare prm, which we will set to be the
             * result of condition.
             */
            auto ei = new ExpInitializer(loc, condition);
            match = new VarDeclaration(loc, prm.type, prm.ident, ei);
            match.parent = scd.func;
            match.storage_class |= prm.storageClass;
            match.semantic(scd);

            auto de = new DeclarationExp(loc, match);
            auto ve = new VarExp(loc, match);
            condition = new CommaExp(loc, de, ve);
            condition = condition.semantic(scd);

            if (match.edtor)
            {
                Statement sdtor = new DtorExpStatement(loc, match.edtor, match);
                sdtor = new OnScopeStatement(loc, TOKon_scope_exit, sdtor);
                ifbody = new CompoundStatement(loc, sdtor, ifbody);
                match.noscope = true;
            }
        }
        else
        {
            condition = condition.semantic(scd);
            condition = resolveProperties(scd, condition);
            condition = condition.addDtorHook(scd);
        }
        if (checkNonAssignmentArrayOp(condition))
            condition = new ErrorExp();
        condition = checkGC(scd, condition);

        // Convert to boolean after declaring prm so this works:
        //  if (S prm = S()) {}
        // where S is a struct that defines opCast!bool.
        condition = condition.toBoolean(scd);

        // If we can short-circuit evaluate the if statement, don't do the
        // semantic analysis of the skipped code.
        // This feature allows a limited form of conditional compilation.
        condition = condition.optimize(WANTvalue);

        ifbody = ifbody.semanticNoScope(scd);
        scd.pop();

        cs1 = sc.callSuper;
        fi1 = sc.fieldinit;
        sc.callSuper = cs0;
        sc.fieldinit = fi0;
        if (elsebody)
            elsebody = elsebody.semanticScope(sc, null, null);
        sc.mergeCallSuper(loc, cs1);
        sc.mergeFieldInit(loc, fi1);

        if (condition.op == TOKerror ||
            (ifbody && ifbody.isErrorStatement()) ||
            (elsebody && elsebody.isErrorStatement()))
        {
            return new ErrorStatement();
        }
        return this;
    }

    override IfStatement isIfStatement()
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
extern (C++) final class ConditionalStatement : Statement
{
public:
    Condition condition;
    Statement ifbody;
    Statement elsebody;

    extern (D) this(Loc loc, Condition condition, Statement ifbody, Statement elsebody)
    {
        super(loc);
        this.condition = condition;
        this.ifbody = ifbody;
        this.elsebody = elsebody;
    }

    override Statement syntaxCopy()
    {
        return new ConditionalStatement(loc, condition.syntaxCopy(), ifbody.syntaxCopy(), elsebody ? elsebody.syntaxCopy() : null);
    }

    override Statement semantic(Scope* sc)
    {
        //printf("ConditionalStatement::semantic()\n");
        // If we can short-circuit evaluate the if statement, don't do the
        // semantic analysis of the skipped code.
        // This feature allows a limited form of conditional compilation.
        if (condition.include(sc, null))
        {
            DebugCondition dc = condition.isDebugCondition();
            if (dc)
            {
                sc = sc.push();
                sc.flags |= SCOPEdebug;
                ifbody = ifbody.semantic(sc);
                sc.pop();
            }
            else
                ifbody = ifbody.semantic(sc);
            return ifbody;
        }
        else
        {
            if (elsebody)
                elsebody = elsebody.semantic(sc);
            return elsebody;
        }
    }

    override Statements* flatten(Scope* sc)
    {
        Statement s;
        //printf("ConditionalStatement::flatten()\n");
        if (condition.include(sc, null))
        {
            DebugCondition dc = condition.isDebugCondition();
            if (dc)
                s = new DebugStatement(loc, ifbody);
            else
                s = ifbody;
        }
        else
            s = elsebody;
        auto a = new Statements();
        a.push(s);
        return a;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class PragmaStatement : Statement
{
public:
    Identifier ident;
    Expressions* args;      // array of Expression's
    Statement _body;

    extern (D) this(Loc loc, Identifier ident, Expressions* args, Statement _body)
    {
        super(loc);
        this.ident = ident;
        this.args = args;
        this._body = _body;
    }

    override Statement syntaxCopy()
    {
        return new PragmaStatement(loc, ident, Expression.arraySyntaxCopy(args), _body ? _body.syntaxCopy() : null);
    }

    override Statement semantic(Scope* sc)
    {
        // Should be merged with PragmaDeclaration
        //printf("PragmaStatement::semantic() %s\n", toChars());
        //printf("body = %p\n", body);
        if (ident == Id.msg)
        {
            if (args)
            {
                foreach (arg; *args)
                {
                    sc = sc.startCTFE();
                    auto e = arg.semantic(sc);
                    e = resolveProperties(sc, e);
                    sc = sc.endCTFE();
                    // pragma(msg) is allowed to contain types as well as expressions
                    e = ctfeInterpretForPragmaMsg(e);
                    if (e.op == TOKerror)
                    {
                        errorSupplemental(loc, "while evaluating pragma(msg, %s)", arg.toChars());
                        goto Lerror;
                    }
                    StringExp se = e.toStringExp();
                    if (se)
                    {
                        se = se.toUTF8(sc);
                        fprintf(stderr, "%.*s", cast(int)se.len, se.string);
                    }
                    else
                        fprintf(stderr, "%s", e.toChars());
                }
                fprintf(stderr, "\n");
            }
        }
        else if (ident == Id.lib)
        {
            version (all)
            {
                /* Should this be allowed?
                 */
                error("pragma(lib) not allowed as statement");
                goto Lerror;
            }
            else
            {
                if (!args || args.dim != 1)
                {
                    error("string expected for library name");
                    goto Lerror;
                }
                else
                {
                    Expression e = (*args)[0];
                    sc = sc.startCTFE();
                    e = e.semantic(sc);
                    e = resolveProperties(sc, e);
                    sc = sc.endCTFE();
                    e = e.ctfeInterpret();
                    (*args)[0] = e;
                    StringExp se = e.toStringExp();
                    if (!se)
                    {
                        error("string expected for library name, not '%s'", e.toChars());
                        goto Lerror;
                    }
                    else if (global.params.verbose)
                    {
                        fprintf(global.stdmsg, "library   %.*s\n", cast(int)se.len, se.string);
                    }
                }
            }
        }
        // IN_LLVM. FIXME Move to pragma.cpp
        else if (ident == Id.LDC_allow_inline)
        {
            sc.func.allowInlining = true;
        }
        // IN_LLVM. FIXME Move to pragma.cpp
        else if (ident == Id.LDC_never_inline)
        {
            sc.func.neverInline = true;
        }
        // IN_LLVM. FIXME Move to pragma.cpp
        else if (ident == Id.LDC_profile_instr)
        {
            bool emitInstr = true;
            if (!args || args.dim != 1 || !DtoCheckProfileInstrPragma((*args)[0], emitInstr))
            {
                error("pragma(LDC_profile_instr, true or false) expected");
                goto Lerror;
            }
            else
            {
                FuncDeclaration fd = sc.func;
                if (fd is null)
                {
                    error("pragma(LDC_profile_instr, ...) is not inside a function");
                    goto Lerror;
                }
                fd.emitInstrumentation = emitInstr;
            }
        }
        else if (ident == Id.startaddress)
        {
            if (!args || args.dim != 1)
                error("function name expected for start address");
            else
            {
                Expression e = (*args)[0];
                sc = sc.startCTFE();
                e = e.semantic(sc);
                e = resolveProperties(sc, e);
                sc = sc.endCTFE();
                e = e.ctfeInterpret();
                (*args)[0] = e;
                Dsymbol sa = getDsymbol(e);
                if (!sa || !sa.isFuncDeclaration())
                {
                    error("function name expected for start address, not '%s'", e.toChars());
                    goto Lerror;
                }
                if (_body)
                {
                    _body = _body.semantic(sc);
                    if (_body.isErrorStatement())
                        return _body;
                }
                return this;
            }
        }
        else if (ident == Id.Pinline)
        {
            PINLINE inlining = PINLINEdefault;
            if (!args || args.dim == 0)
                inlining = PINLINEdefault;
            else if (!args || args.dim != 1)
            {
                error("boolean expression expected for pragma(inline)");
                goto Lerror;
            }
            else
            {
                Expression e = (*args)[0];
                if (e.op != TOKint64 || !e.type.equals(Type.tbool))
                {
                    error("pragma(inline, true or false) expected, not %s", e.toChars());
                    goto Lerror;
                }
                if (e.isBool(true))
                    inlining = PINLINEalways;
                else if (e.isBool(false))
                    inlining = PINLINEnever;
                FuncDeclaration fd = sc.func;
                if (!fd)
                {
                    error("pragma(inline) is not inside a function");
                    goto Lerror;
                }
                fd.inlining = inlining;
            }
        }
        else
        {
            error("unrecognized pragma(%s)", ident.toChars());
            goto Lerror;
        }
        if (_body)
        {
            _body = _body.semantic(sc);
        }
        return _body;
    Lerror:
        return new ErrorStatement();
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class StaticAssertStatement : Statement
{
public:
    StaticAssert sa;

    extern (D) this(StaticAssert sa)
    {
        super(sa.loc);
        this.sa = sa;
    }

    override Statement syntaxCopy()
    {
        return new StaticAssertStatement(cast(StaticAssert)sa.syntaxCopy(null));
    }

    override Statement semantic(Scope* sc)
    {
        sa.semantic2(sc);
        return null;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class SwitchStatement : Statement
{
public:
    Expression condition;
    Statement _body;
    bool isFinal;

    DefaultStatement sdefault;
    TryFinallyStatement tf;
    GotoCaseStatements gotoCases;   // array of unresolved GotoCaseStatement's
    CaseStatements* cases;          // array of CaseStatement's
    int hasNoDefault;               // !=0 if no default statement
    int hasVars;                    // !=0 if has variable case values
version(IN_LLVM)
{
    bool hasGotoDefault;            // true iff there is a `goto default` statement for this switch
}

    extern (D) this(Loc loc, Expression c, Statement b, bool isFinal)
    {
        super(loc);
        this.condition = c;
        this._body = b;
        this.isFinal = isFinal;
    }

    override Statement syntaxCopy()
    {
        return new SwitchStatement(loc, condition.syntaxCopy(), _body.syntaxCopy(), isFinal);
    }

    override Statement semantic(Scope* sc)
    {
        //printf("SwitchStatement::semantic(%p)\n", this);
        tf = sc.tf;
        if (cases)
            return this; // already run

        bool conditionError = false;
        condition = condition.semantic(sc);
        condition = resolveProperties(sc, condition);

        Type att = null;
        TypeEnum te = null;
        while (condition.op != TOKerror)
        {
            // preserve enum type for final switches
            if (condition.type.ty == Tenum)
                te = cast(TypeEnum)condition.type;
            if (condition.type.isString())
            {
                // If it's not an array, cast it to one
                if (condition.type.ty != Tarray)
                {
                    condition = condition.implicitCastTo(sc, condition.type.nextOf().arrayOf());
                }
                condition.type = condition.type.constOf();
                break;
            }
            condition = integralPromotions(condition, sc);
            if (condition.op != TOKerror && condition.type.isintegral())
                break;

            auto ad = isAggregate(condition.type);
            if (ad && ad.aliasthis && condition.type != att)
            {
                if (!att && condition.type.checkAliasThisRec())
                    att = condition.type;
                if (auto e = resolveAliasThis(sc, condition, true))
                {
                    condition = e;
                    continue;
                }
            }

            if (condition.op != TOKerror)
            {
                error("'%s' must be of integral or string type, it is a %s",
                    condition.toChars(), condition.type.toChars());
                conditionError = true;
                break;
            }
        }
        if (checkNonAssignmentArrayOp(condition))
            condition = new ErrorExp();
        condition = condition.optimize(WANTvalue);
        condition = checkGC(sc, condition);
        if (condition.op == TOKerror)
            conditionError = true;

        bool needswitcherror = false;
        sc = sc.push();
        sc.sbreak = this;
        sc.sw = this;
        cases = new CaseStatements();
        sc.noctor++; // BUG: should use Scope::mergeCallSuper() for each case instead
        _body = _body.semantic(sc);
        sc.noctor--;
        if (conditionError || _body.isErrorStatement())
            goto Lerror;
        // Resolve any goto case's with exp
        foreach (gcs; gotoCases)
        {
            if (!gcs.exp)
            {
                gcs.error("no case statement following goto case;");
                goto Lerror;
            }
            for (Scope* scx = sc; scx; scx = scx.enclosing)
            {
                if (!scx.sw)
                    continue;
                foreach (cs; *scx.sw.cases)
                {
                    if (cs.exp.equals(gcs.exp))
                    {
                        gcs.cs = cs;
version(IN_LLVM)
{
                        cs.gototarget = true;
}
                        goto Lfoundcase;
                    }
                }
            }
            gcs.error("case %s not found", gcs.exp.toChars());
            goto Lerror;
        Lfoundcase:
        }
        if (isFinal)
        {
            Type t = condition.type;
            Dsymbol ds;
            EnumDeclaration ed = null;
            if (t && ((ds = t.toDsymbol(sc)) !is null))
                ed = ds.isEnumDeclaration(); // typedef'ed enum
            if (!ed && te && ((ds = te.toDsymbol(sc)) !is null))
                ed = ds.isEnumDeclaration();
            if (ed)
            {
                foreach (es; *ed.members)
                {
                    EnumMember em = es.isEnumMember();
                    if (em)
                    {
                        foreach (cs; *cases)
                        {
                            if (cs.exp.equals(em.value) || (!cs.exp.type.isString() && !em.value.type.isString() && cs.exp.toInteger() == em.value.toInteger()))
                                goto L1;
                        }
                        error("enum member %s not represented in final switch", em.toChars());
                        goto Lerror;
                    }
                L1:
                }
            }
            else
                needswitcherror = true;
        }
        if (!sc.sw.sdefault && (!isFinal || needswitcherror || global.params.useAssert))
        {
            hasNoDefault = 1;
            if (!isFinal && !_body.isErrorStatement())
                error("switch statement without a default; use 'final switch' or add 'default: assert(0);' or add 'default: break;'");
            // Generate runtime error if the default is hit
            auto a = new Statements();
            CompoundStatement cs;
            Statement s;
            if (global.params.useSwitchError)
                s = new SwitchErrorStatement(loc);
            else
                s = new ExpStatement(loc, new HaltExp(loc));
            a.reserve(2);
            sc.sw.sdefault = new DefaultStatement(loc, s);
            a.push(_body);
            if (_body.blockExit(sc.func, false) & BEfallthru)
                a.push(new BreakStatement(Loc(), null));
            a.push(sc.sw.sdefault);
            cs = new CompoundStatement(loc, a);
            _body = cs;
        }
version(IN_LLVM)
{
        /+ hasGotoDefault is set by GotoDefaultStatement.semantic
         + at which point sdefault may still be null, therefore
         + set sdefault.gototarget here.
         +/
        if (hasGotoDefault) {
            assert(sdefault);
            sdefault.gototarget = true;
        }
}
        sc.pop();
        return this;
    Lerror:
        sc.pop();
        return new ErrorStatement();
    }

    override bool hasBreak()
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class CaseStatement : Statement
{
public:
    Expression exp;
    Statement statement;
    int index;              // which case it is (since we sort this)

    version(IN_LLVM)
    {
        bool gototarget; // true iff this is the target of a 'goto case'
    }

    extern (D) this(Loc loc, Expression exp, Statement s)
    {
        super(loc);
        this.exp = exp;
        this.statement = s;
    }

    override Statement syntaxCopy()
    {
        return new CaseStatement(loc, exp.syntaxCopy(), statement.syntaxCopy());
    }

    override Statement semantic(Scope* sc)
    {
        SwitchStatement sw = sc.sw;
        bool errors = false;
        //printf("CaseStatement::semantic() %s\n", toChars());
        sc = sc.startCTFE();
        exp = exp.semantic(sc);
        exp = resolveProperties(sc, exp);
        sc = sc.endCTFE();
        if (sw)
        {
            exp = exp.implicitCastTo(sc, sw.condition.type);
            exp = exp.optimize(WANTvalue | WANTexpand);
            /* This is where variables are allowed as case expressions.
             */
            if (exp.op == TOKvar)
            {
                VarExp ve = cast(VarExp)exp;
                VarDeclaration v = ve.var.isVarDeclaration();
                Type t = exp.type.toBasetype();
                if (v && (t.isintegral() || t.ty == Tclass))
                {
                    /* Flag that we need to do special code generation
                     * for this, i.e. generate a sequence of if-then-else
                     */
                    sw.hasVars = 1;
                    if (sw.isFinal)
                    {
                        error("case variables not allowed in final switch statements");
                        errors = true;
                    }
                    goto L1;
                }
            }
            else
                exp = exp.ctfeInterpret();
            if (StringExp se = exp.toStringExp())
                exp = se;
            else if (exp.op != TOKint64 && exp.op != TOKerror)
            {
                error("case must be a string or an integral constant, not %s", exp.toChars());
                errors = true;
            }
        L1:
            foreach (cs; *sw.cases)
            {
                //printf("comparing '%s' with '%s'\n", exp->toChars(), cs->exp->toChars());
                if (cs.exp.equals(exp))
                {
                    error("duplicate case %s in switch statement", exp.toChars());
                    errors = true;
                    break;
                }
            }
            sw.cases.push(this);
            // Resolve any goto case's with no exp to this case statement
            for (size_t i = 0; i < sw.gotoCases.dim;)
            {
                GotoCaseStatement gcs = sw.gotoCases[i];
                if (!gcs.exp)
                {
                    gcs.cs = this;
                    sw.gotoCases.remove(i); // remove from array
                    continue;
                }
                i++;
            }
            if (sc.sw.tf != sc.tf)
            {
                error("switch and case are in different finally blocks");
                errors = true;
            }
        }
        else
        {
            error("case not in switch statement");
            errors = true;
        }
        statement = statement.semantic(sc);
        if (statement.isErrorStatement())
            return statement;
        if (errors || exp.op == TOKerror)
            return new ErrorStatement();
        return this;
    }

    override int compare(RootObject obj)
    {
        // Sort cases so we can do an efficient lookup
        CaseStatement cs2 = cast(CaseStatement)obj;
        return exp.compare(cs2.exp);
    }

    override CaseStatement isCaseStatement()
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
extern (C++) final class CaseRangeStatement : Statement
{
public:
    Expression first;
    Expression last;
    Statement statement;

    extern (D) this(Loc loc, Expression first, Expression last, Statement s)
    {
        super(loc);
        this.first = first;
        this.last = last;
        this.statement = s;
    }

    override Statement syntaxCopy()
    {
        return new CaseRangeStatement(loc, first.syntaxCopy(), last.syntaxCopy(), statement.syntaxCopy());
    }

    override Statement semantic(Scope* sc)
    {
        SwitchStatement sw = sc.sw;
        if (sw is null)
        {
            error("case range not in switch statement");
            return new ErrorStatement();
        }
        //printf("CaseRangeStatement::semantic() %s\n", toChars());
        bool errors = false;
        if (sw.isFinal)
        {
            error("case ranges not allowed in final switch");
            errors = true;
        }
        sc = sc.startCTFE();
        first = first.semantic(sc);
        first = resolveProperties(sc, first);
        sc = sc.endCTFE();
        first = first.implicitCastTo(sc, sw.condition.type);
        first = first.ctfeInterpret();
        sc = sc.startCTFE();
        last = last.semantic(sc);
        last = resolveProperties(sc, last);
        sc = sc.endCTFE();
        last = last.implicitCastTo(sc, sw.condition.type);
        last = last.ctfeInterpret();
        if (first.op == TOKerror || last.op == TOKerror || errors)
        {
            if (statement)
                statement.semantic(sc);
            return new ErrorStatement();
        }
        uinteger_t fval = first.toInteger();
        uinteger_t lval = last.toInteger();
        if ((first.type.isunsigned() && fval > lval) || (!first.type.isunsigned() && cast(sinteger_t)fval > cast(sinteger_t)lval))
        {
            error("first case %s is greater than last case %s", first.toChars(), last.toChars());
            errors = true;
            lval = fval;
        }
        if (lval - fval > 256)
        {
            error("had %llu cases which is more than 256 cases in case range", lval - fval);
            errors = true;
            lval = fval + 256;
        }
        if (errors)
            return new ErrorStatement();
        /* This works by replacing the CaseRange with an array of Case's.
         *
         * case a: .. case b: s;
         *    =>
         * case a:
         *   [...]
         * case b:
         *   s;
         */
        auto statements = new Statements();
        for (uinteger_t i = fval; i != lval + 1; i++)
        {
            Statement s = statement;
            if (i != lval) // if not last case
                s = new ExpStatement(loc, cast(Expression)null);
            Expression e = new IntegerExp(loc, i, first.type);
            Statement cs = new CaseStatement(loc, e, s);
            statements.push(cs);
        }
        Statement s = new CompoundStatement(loc, statements);
        s = s.semantic(sc);
        return s;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class DefaultStatement : Statement
{
public:
    Statement statement;

    version(IN_LLVM)
    {
        bool gototarget; // true iff this is the target of a 'goto default'
    }

    extern (D) this(Loc loc, Statement s)
    {
        super(loc);
        this.statement = s;
    }

    override Statement syntaxCopy()
    {
        return new DefaultStatement(loc, statement.syntaxCopy());
    }

    override Statement semantic(Scope* sc)
    {
        //printf("DefaultStatement::semantic()\n");
        bool errors = false;
        if (sc.sw)
        {
            if (sc.sw.sdefault)
            {
                error("switch statement already has a default");
                errors = true;
            }
            sc.sw.sdefault = this;
            if (sc.sw.tf != sc.tf)
            {
                error("switch and default are in different finally blocks");
                errors = true;
            }
            if (sc.sw.isFinal)
            {
                error("default statement not allowed in final switch statement");
                errors = true;
            }
        }
        else
        {
            error("default not in switch statement");
            errors = true;
        }
        statement = statement.semantic(sc);
        if (errors || statement.isErrorStatement())
            return new ErrorStatement();
        return this;
    }

    override DefaultStatement isDefaultStatement()
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
extern (C++) final class GotoDefaultStatement : Statement
{
public:
    SwitchStatement sw;

    extern (D) this(Loc loc)
    {
        super(loc);
    }

    override Statement syntaxCopy()
    {
        return new GotoDefaultStatement(loc);
    }

    override Statement semantic(Scope* sc)
    {
        sw = sc.sw;
        if (!sw)
        {
            error("goto default not in switch statement");
            return new ErrorStatement();
        }
        if (sw.isFinal)
        {
            error("goto default not allowed in final switch statement");
            return new ErrorStatement();
        }

version(IN_LLVM)
{
        sw.hasGotoDefault = true;
}

        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class GotoCaseStatement : Statement
{
public:
    Expression exp;     // null, or which case to goto
    CaseStatement cs;   // case statement it resolves to

    version(IN_LLVM)
    {
        SwitchStatement sw;
    }

    extern (D) this(Loc loc, Expression exp)
    {
        super(loc);
        this.exp = exp;
    }

    override Statement syntaxCopy()
    {
        return new GotoCaseStatement(loc, exp ? exp.syntaxCopy() : null);
    }

    override Statement semantic(Scope* sc)
    {
        if (!sc.sw)
        {
            error("goto case not in switch statement");
            return new ErrorStatement();
        }
        version(IN_LLVM)
        {
            sw = sc.sw;
        }
        if (exp)
        {
            exp = exp.semantic(sc);
            exp = exp.implicitCastTo(sc, sc.sw.condition.type);
            exp = exp.optimize(WANTvalue);
            if (exp.op == TOKerror)
                return new ErrorStatement();
        }
        sc.sw.gotoCases.push(this);
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class SwitchErrorStatement : Statement
{
public:
    extern (D) this(Loc loc)
    {
        super(loc);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ReturnStatement : Statement
{
public:
    Expression exp;
    size_t caseDim;

    extern (D) this(Loc loc, Expression exp)
    {
        super(loc);
        this.exp = exp;
    }

    override Statement syntaxCopy()
    {
        return new ReturnStatement(loc, exp ? exp.syntaxCopy() : null);
    }

    override Statement semantic(Scope* sc)
    {
        //printf("ReturnStatement::semantic() %s\n", toChars());
        FuncDeclaration fd = sc.parent.isFuncDeclaration();
        if (fd.fes)
            fd = fd.fes.func; // fd is now function enclosing foreach
        TypeFunction tf = cast(TypeFunction)fd.type;
        assert(tf.ty == Tfunction);
        if (exp && exp.op == TOKvar && (cast(VarExp)exp).var == fd.vresult)
        {
            // return vresult;
            if (sc.fes)
            {
                assert(caseDim == 0);
                sc.fes.cases.push(this);
                return new ReturnStatement(Loc(), new IntegerExp(sc.fes.cases.dim + 1));
            }
            if (fd.returnLabel)
            {
                auto gs = new GotoStatement(loc, Id.returnLabel);
                gs.label = fd.returnLabel;
                return gs;
            }
            if (!fd.returns)
                fd.returns = new ReturnStatements();
            fd.returns.push(this);
            return this;
        }
        Type tret = tf.next;
        Type tbret = tret ? tret.toBasetype() : null;
        bool inferRef = (tf.isref && (fd.storage_class & STCauto));
        Expression e0 = null;
        bool errors = false;
        if (sc.flags & SCOPEcontract)
        {
            error("return statements cannot be in contracts");
            errors = true;
        }
        if (sc.os && sc.os.tok != TOKon_scope_failure)
        {
            error("return statements cannot be in %s bodies", Token.toChars(sc.os.tok));
            errors = true;
        }
        if (sc.tf)
        {
            error("return statements cannot be in finally bodies");
            errors = true;
        }
        if (fd.isCtorDeclaration())
        {
            if (exp)
            {
                error("cannot return expression from constructor");
                errors = true;
            }
            // Constructors implicitly do:
            //      return this;
            exp = new ThisExp(Loc());
            exp.type = tret;
        }
        else if (exp)
        {
            fd.hasReturnExp |= 1;
            FuncLiteralDeclaration fld = fd.isFuncLiteralDeclaration();
            if (tret)
                exp = inferType(exp, tret);
            else if (fld && fld.treq)
                exp = inferType(exp, fld.treq.nextOf().nextOf());

            exp = exp.semantic(sc);
            exp = resolveProperties(sc, exp);
            if (exp.checkType())
                exp = new ErrorExp();
            if (auto f = isFuncAddress(exp))
            {
                if (fd.inferRetType && f.checkForwardRef(exp.loc))
                    exp = new ErrorExp();
            }
            if (checkNonAssignmentArrayOp(exp))
                exp = new ErrorExp();

            // Extract side-effect part
            exp = Expression.extractLast(exp, &e0);
            if (exp.op == TOKcall)
                exp = valueNoDtor(exp);

            /* Void-return function can have void typed expression
             * on return statement.
             */
            if (tbret && tbret.ty == Tvoid || exp.type.ty == Tvoid)
            {
                if (exp.type.ty != Tvoid)
                {
                    error("cannot return non-void from void function");
                    errors = true;
                    exp = new CastExp(loc, exp, Type.tvoid);
                    exp = exp.semantic(sc);
                }

                /* Replace:
                 *      return exp;
                 * with:
                 *      exp; return;
                 */
                e0 = Expression.combine(e0, exp);
                exp = null;
            }
            if (e0)
                e0 = checkGC(sc, e0);
        }
        if (exp)
        {
            if (fd.inferRetType) // infer return type
            {
                if (!tret)
                {
                    tf.next = exp.type;
                }
                else if (tret.ty != Terror && !exp.type.equals(tret))
                {
                    int m1 = exp.type.implicitConvTo(tret);
                    int m2 = tret.implicitConvTo(exp.type);
                    //printf("exp->type = %s m2<-->m1 tret %s\n", exp->type->toChars(), tret->toChars());
                    //printf("m1 = %d, m2 = %d\n", m1, m2);
                    if (m1 && m2)
                    {
                    }
                    else if (!m1 && m2)
                        tf.next = exp.type;
                    else if (m1 && !m2)
                    {
                    }
                    else if (exp.op != TOKerror)
                    {
                        error("mismatched function return type inference of %s and %s", exp.type.toChars(), tret.toChars());
                        errors = true;
                        tf.next = Type.terror;
                    }
                }
                tret = tf.next;
                tbret = tret.toBasetype();
            }
            if (inferRef) // deduce 'auto ref'
            {
                /* Determine "refness" of function return:
                 * if it's an lvalue, return by ref, else return by value
                 */
                if (exp.isLvalue())
                {
                    /* May return by ref
                     */
                    if (checkEscapeRef(sc, exp, true))
                        tf.isref = false; // return by value
                }
                else
                    tf.isref = false; // return by value
                /* The "refness" is determined by all of return statements.
                 * This means:
                 *    return 3; return x;  // ok, x can be a value
                 *    return x; return 3;  // ok, x can be a value
                 */
            }
            // handle NRVO
            if (fd.nrvo_can && exp.op == TOKvar)
            {
                VarExp ve = cast(VarExp)exp;
                VarDeclaration v = ve.var.isVarDeclaration();
                if (tf.isref)
                {
                    // Function returns a reference
                    if (!inferRef)
                        fd.nrvo_can = 0;
                }
                else if (!v || v.isOut() || v.isRef())
                    fd.nrvo_can = 0;
                else if (fd.nrvo_var is null)
                {
                    if (!v.isDataseg() && !v.isParameter() && v.toParent2() == fd)
                    {
                        //printf("Setting nrvo to %s\n", v->toChars());
                        fd.nrvo_var = v;
                    }
                    else
                        fd.nrvo_can = 0;
                }
                else if (fd.nrvo_var != v)
                    fd.nrvo_can = 0;
            }
            else //if (!exp->isLvalue())    // keep NRVO-ability
                fd.nrvo_can = 0;
        }
        else
        {
            // handle NRVO
            fd.nrvo_can = 0;
            // infer return type
            if (fd.inferRetType)
            {
                if (tf.next && tf.next.ty != Tvoid)
                {
                    if (tf.next.ty != Terror)
                    {
                        error("mismatched function return type inference of void and %s", tf.next.toChars());
                    }
                    errors = true;
                    tf.next = Type.terror;
                }
                else
                    tf.next = Type.tvoid;
                tret = tf.next;
                tbret = tret.toBasetype();
            }
            if (inferRef) // deduce 'auto ref'
                tf.isref = false;
            if (tbret.ty != Tvoid) // if non-void return
            {
                if (tbret.ty != Terror)
                    error("return expression expected");
                errors = true;
            }
            else if (fd.isMain())
            {
                // main() returns 0, even if it returns void
                exp = new IntegerExp(0);
            }
        }
        // If any branches have called a ctor, but this branch hasn't, it's an error
        if (sc.callSuper & CSXany_ctor && !(sc.callSuper & (CSXthis_ctor | CSXsuper_ctor)))
        {
            error("return without calling constructor");
            errors = true;
        }
        sc.callSuper |= CSXreturn;
        if (sc.fieldinit)
        {
            AggregateDeclaration ad = fd.isAggregateMember2();
            assert(ad);
            size_t dim = sc.fieldinit_dim;
            foreach (i; 0 .. dim)
            {
                VarDeclaration v = ad.fields[i];
                bool mustInit = (v.storage_class & STCnodefaultctor || v.type.needsNested());
                if (mustInit && !(sc.fieldinit[i] & CSXthis_ctor))
                {
                    error("an earlier return statement skips field %s initialization", v.toChars());
                    errors = true;
                }
                sc.fieldinit[i] |= CSXreturn;
            }
        }
        if (errors)
            return new ErrorStatement();
        if (sc.fes)
        {
            if (!exp)
            {
                // Send out "case receiver" statement to the foreach.
                //  return exp;
                Statement s = new ReturnStatement(Loc(), exp);
                sc.fes.cases.push(s);
                // Immediately rewrite "this" return statement as:
                //  return cases->dim+1;
                this.exp = new IntegerExp(sc.fes.cases.dim + 1);
                if (e0)
                    return new CompoundStatement(loc, new ExpStatement(loc, e0), this);
                return this;
            }
            else
            {
                fd.buildResultVar(null, exp.type);
                bool r = fd.vresult.checkNestedReference(sc, Loc());
                assert(!r); // vresult should be always accessible
                // Send out "case receiver" statement to the foreach.
                //  return vresult;
                Statement s = new ReturnStatement(Loc(), new VarExp(Loc(), fd.vresult));
                sc.fes.cases.push(s);
                // Save receiver index for the later rewriting from:
                //  return exp;
                // to:
                //  vresult = exp; retrun caseDim;
                caseDim = sc.fes.cases.dim + 1;
            }
        }
        if (exp)
        {
            if (!fd.returns)
                fd.returns = new ReturnStatements();
            fd.returns.push(this);
        }
        if (e0)
            return new CompoundStatement(loc, new ExpStatement(loc, e0), this);
        return this;
    }

    override ReturnStatement isReturnStatement()
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
extern (C++) final class BreakStatement : Statement
{
public:
    Identifier ident;

    version(IN_LLVM)
    {
        // LDC: only set if ident is set: label statement to jump to
        LabelStatement target;
    }

    extern (D) this(Loc loc, Identifier ident)
    {
        super(loc);
        this.ident = ident;
    }

    override Statement syntaxCopy()
    {
        return new BreakStatement(loc, ident);
    }

    override Statement semantic(Scope* sc)
    {
        //printf("BreakStatement::semantic()\n");
        // If:
        //  break Identifier;
        if (ident)
        {
            ident = fixupLabelName(sc, ident);
            FuncDeclaration thisfunc = sc.func;
            for (Scope* scx = sc; scx; scx = scx.enclosing)
            {
                if (scx.func != thisfunc) // if in enclosing function
                {
                    if (sc.fes) // if this is the body of a foreach
                    {
                        /* Post this statement to the fes, and replace
                         * it with a return value that caller will put into
                         * a switch. Caller will figure out where the break
                         * label actually is.
                         * Case numbers start with 2, not 0, as 0 is continue
                         * and 1 is break.
                         */
                        sc.fes.cases.push(this);
                        Statement s = new ReturnStatement(Loc(), new IntegerExp(sc.fes.cases.dim + 1));
                        return s;
                    }
                    break;
                    // can't break to it
                }
                LabelStatement ls = scx.slabel;
                if (ls && ls.ident == ident)
                {
                    Statement s = ls.statement;
                    if (!s || !s.hasBreak())
                        error("label '%s' has no break", ident.toChars());
                    else if (ls.tf != sc.tf)
                        error("cannot break out of finally block");
                    else
                    {
                        version(IN_LLVM)
                            target = ls;

                        ls.breaks = true;
                        return this;
                    }
                    return new ErrorStatement();
                }
            }
            error("enclosing label '%s' for break not found", ident.toChars());
            return new ErrorStatement();
        }
        else if (!sc.sbreak)
        {
            if (sc.os && sc.os.tok != TOKon_scope_failure)
            {
                error("break is not inside %s bodies", Token.toChars(sc.os.tok));
            }
            else if (sc.fes)
            {
                // Replace break; with return 1;
                Statement s = new ReturnStatement(Loc(), new IntegerExp(1));
                return s;
            }
            else
                error("break is not inside a loop or switch");
            return new ErrorStatement();
        }
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ContinueStatement : Statement
{
public:
    Identifier ident;

    version(IN_LLVM)
    {
        // LDC: only set if ident is set: label statement to jump to
        LabelStatement target;
    }

    extern (D) this(Loc loc, Identifier ident)
    {
        super(loc);
        this.ident = ident;
    }

    override Statement syntaxCopy()
    {
        return new ContinueStatement(loc, ident);
    }

    override Statement semantic(Scope* sc)
    {
        //printf("ContinueStatement::semantic() %p\n", this);
        if (ident)
        {
            ident = fixupLabelName(sc, ident);
            Scope* scx;
            FuncDeclaration thisfunc = sc.func;
            for (scx = sc; scx; scx = scx.enclosing)
            {
                LabelStatement ls;
                if (scx.func != thisfunc) // if in enclosing function
                {
                    if (sc.fes) // if this is the body of a foreach
                    {
                        for (; scx; scx = scx.enclosing)
                        {
                            ls = scx.slabel;
                            if (ls && ls.ident == ident && ls.statement == sc.fes)
                            {
                                // Replace continue ident; with return 0;
                                return new ReturnStatement(Loc(), new IntegerExp(0));
                            }
                        }
                        /* Post this statement to the fes, and replace
                         * it with a return value that caller will put into
                         * a switch. Caller will figure out where the break
                         * label actually is.
                         * Case numbers start with 2, not 0, as 0 is continue
                         * and 1 is break.
                         */
                        sc.fes.cases.push(this);
                        Statement s = new ReturnStatement(Loc(), new IntegerExp(sc.fes.cases.dim + 1));
                        return s;
                    }
                    break;
                    // can't continue to it
                }
                ls = scx.slabel;
                if (ls && ls.ident == ident)
                {
                    Statement s = ls.statement;
                    if (!s || !s.hasContinue())
                        error("label '%s' has no continue", ident.toChars());
                    else if (ls.tf != sc.tf)
                        error("cannot continue out of finally block");
                    else
                    {
                        version(IN_LLVM)
                            target = ls;

                        return this;
                    }
                    return new ErrorStatement();
                }
            }
            error("enclosing label '%s' for continue not found", ident.toChars());
            return new ErrorStatement();
        }
        else if (!sc.scontinue)
        {
            if (sc.os && sc.os.tok != TOKon_scope_failure)
            {
                error("continue is not inside %s bodies", Token.toChars(sc.os.tok));
            }
            else if (sc.fes)
            {
                // Replace continue; with return 0;
                Statement s = new ReturnStatement(Loc(), new IntegerExp(0));
                return s;
            }
            else
                error("continue is not inside a loop");
            return new ErrorStatement();
        }
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class SynchronizedStatement : Statement
{
public:
    Expression exp;
    Statement _body;

    extern (D) this(Loc loc, Expression exp, Statement _body)
    {
        super(loc);
        this.exp = exp;
        this._body = _body;
    }

    override Statement syntaxCopy()
    {
        return new SynchronizedStatement(loc, exp ? exp.syntaxCopy() : null, _body ? _body.syntaxCopy() : null);
    }

    override Statement semantic(Scope* sc)
    {
        if (exp)
        {
            exp = exp.semantic(sc);
            exp = resolveProperties(sc, exp);
            exp = exp.optimize(WANTvalue);
            exp = checkGC(sc, exp);
            if (exp.op == TOKerror)
                goto Lbody;

            ClassDeclaration cd = exp.type.isClassHandle();
            if (!cd)
            {
                error("can only synchronize on class objects, not '%s'", exp.type.toChars());
                return new ErrorStatement();
            }
            else if (cd.isInterfaceDeclaration())
            {
                /* Cast the interface to an object, as the object has the monitor,
                 * not the interface.
                 */
                if (!ClassDeclaration.object)
                {
                    error("missing or corrupt object.d");
                    fatal();
                }

                Type t = ClassDeclaration.object.type;
                t = t.semantic(Loc(), sc).toBasetype();
                assert(t.ty == Tclass);

                exp = new CastExp(loc, exp, t);
                exp = exp.semantic(sc);
            }
            version (all)
            {
                /* Rewrite as:
                 *  auto tmp = exp;
                 *  _d_monitorenter(tmp);
                 *  try { body } finally { _d_monitorexit(tmp); }
                 */
                Identifier id = Identifier.generateId("__sync");
                auto ie = new ExpInitializer(loc, exp);
                auto tmp = new VarDeclaration(loc, exp.type, id, ie);
                tmp.storage_class |= STCtemp;

                auto cs = new Statements();
                cs.push(new ExpStatement(loc, tmp));

                auto args = new Parameters();
                args.push(new Parameter(0, ClassDeclaration.object.type, null, null));

                FuncDeclaration fdenter = FuncDeclaration.genCfunc(args, Type.tvoid, Id.monitorenter);
                Expression e = new CallExp(loc, new VarExp(loc, fdenter, false), new VarExp(loc, tmp));
                e.type = Type.tvoid; // do not run semantic on e

                cs.push(new ExpStatement(loc, e));
                FuncDeclaration fdexit = FuncDeclaration.genCfunc(args, Type.tvoid, Id.monitorexit);
                e = new CallExp(loc, new VarExp(loc, fdexit, false), new VarExp(loc, tmp));
                e.type = Type.tvoid; // do not run semantic on e
                Statement s = new ExpStatement(loc, e);
                s = new TryFinallyStatement(loc, _body, s);
                cs.push(s);

                s = new CompoundStatement(loc, cs);
                return s.semantic(sc);
            }
        }
        else
        {
            /* Generate our own critical section, then rewrite as:
             *  __gshared byte[CriticalSection.sizeof] critsec;
             *  _d_criticalenter(critsec.ptr);
             *  try { body } finally { _d_criticalexit(critsec.ptr); }
             */
            Identifier id = Identifier.generateId("__critsec");
            Type t = new TypeSArray(Type.tint8, new IntegerExp(Target.ptrsize + Target.critsecsize()));
            auto tmp = new VarDeclaration(loc, t, id, null);
            tmp.storage_class |= STCtemp | STCgshared | STCstatic;

            auto cs = new Statements();
            cs.push(new ExpStatement(loc, tmp));

            /* This is just a dummy variable for "goto skips declaration" error.
             * Backend optimizer could remove this unused variable.
             */
            auto v = new VarDeclaration(loc, Type.tvoidptr, Identifier.generateId("__sync"), null);
            v.semantic(sc);
            cs.push(new ExpStatement(loc, v));

            auto args = new Parameters();
            args.push(new Parameter(0, t.pointerTo(), null, null));

            FuncDeclaration fdenter = FuncDeclaration.genCfunc(args, Type.tvoid, Id.criticalenter, STCnothrow);
            Expression e = new DotIdExp(loc, new VarExp(loc, tmp), Id.ptr);
            e = e.semantic(sc);
            e = new CallExp(loc, new VarExp(loc, fdenter, false), e);
            e.type = Type.tvoid; // do not run semantic on e
            cs.push(new ExpStatement(loc, e));

            FuncDeclaration fdexit = FuncDeclaration.genCfunc(args, Type.tvoid, Id.criticalexit, STCnothrow);
            e = new DotIdExp(loc, new VarExp(loc, tmp), Id.ptr);
            e = e.semantic(sc);
            e = new CallExp(loc, new VarExp(loc, fdexit, false), e);
            e.type = Type.tvoid; // do not run semantic on e
            Statement s = new ExpStatement(loc, e);
            s = new TryFinallyStatement(loc, _body, s);
            cs.push(s);

            s = new CompoundStatement(loc, cs);
          version(IN_LLVM) // backport alignment fix for issue #1955
          {
            s = s.semantic(sc);
            tmp.alignment = Target.ptrsize; // must be set after semantic()
            return s;
          }
          else
          {
            return s.semantic(sc);
          }
        }
    Lbody:
        if (_body)
            _body = _body.semantic(sc);
        if (_body && _body.isErrorStatement())
            return _body;
        return this;
    }

    override bool hasBreak()
    {
        return false; //true;
    }

    override bool hasContinue()
    {
        return false; //true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class WithStatement : Statement
{
public:
    Expression exp;
    Statement _body;
    VarDeclaration wthis;

    extern (D) this(Loc loc, Expression exp, Statement _body)
    {
        super(loc);
        this.exp = exp;
        this._body = _body;
    }

    override Statement syntaxCopy()
    {
        return new WithStatement(loc, exp.syntaxCopy(), _body ? _body.syntaxCopy() : null);
    }

    override Statement semantic(Scope* sc)
    {
        ScopeDsymbol sym;
        Initializer _init;
        //printf("WithStatement::semantic()\n");
        exp = exp.semantic(sc);
        exp = resolveProperties(sc, exp);
        exp = exp.optimize(WANTvalue);
        exp = checkGC(sc, exp);
        if (exp.op == TOKerror)
            return new ErrorStatement();
        if (exp.op == TOKscope)
        {
            sym = new WithScopeSymbol(this);
            sym.parent = sc.scopesym;
        }
        else if (exp.op == TOKtype)
        {
            Dsymbol s = (cast(TypeExp)exp).type.toDsymbol(sc);
            if (!s || !s.isScopeDsymbol())
            {
                error("with type %s has no members", exp.toChars());
                return new ErrorStatement();
            }
            sym = new WithScopeSymbol(this);
            sym.parent = sc.scopesym;
        }
        else
        {
            Type t = exp.type.toBasetype();
            Expression olde = exp;
            if (t.ty == Tpointer)
            {
                exp = new PtrExp(loc, exp);
                exp = exp.semantic(sc);
                t = exp.type.toBasetype();
            }
            assert(t);
            t = t.toBasetype();
            if (t.isClassHandle())
            {
                _init = new ExpInitializer(loc, exp);
                wthis = new VarDeclaration(loc, exp.type, Id.withSym, _init);
                wthis.semantic(sc);
                sym = new WithScopeSymbol(this);
                sym.parent = sc.scopesym;
            }
            else if (t.ty == Tstruct)
            {
                if (!exp.isLvalue())
                {
                    /* Re-write to
                     * {
                     *   auto __withtmp = exp
                     *   with(__withtmp)
                     *   {
                     *     ...
                     *   }
                     * }
                     */
                    _init = new ExpInitializer(loc, exp);
                    wthis = new VarDeclaration(loc, exp.type, Identifier.generateId("__withtmp"), _init);
                    wthis.storage_class |= STCtemp;
                    auto es = new ExpStatement(loc, wthis);
                    exp = new VarExp(loc, wthis);
                    Statement ss = new ScopeStatement(loc, new CompoundStatement(loc, es, this));
                    return ss.semantic(sc);
                }
                Expression e = exp.addressOf();
                _init = new ExpInitializer(loc, e);
                wthis = new VarDeclaration(loc, e.type, Id.withSym, _init);
                wthis.semantic(sc);
                sym = new WithScopeSymbol(this);
                // Need to set the scope to make use of resolveAliasThis
                sym.setScope(sc);
                sym.parent = sc.scopesym;
            }
            else
            {
                error("with expressions must be aggregate types or pointers to them, not '%s'", olde.type.toChars());
                return new ErrorStatement();
            }
        }
        if (_body)
        {
            sym._scope = sc;
            sc = sc.push(sym);
            sc.insert(sym);
            _body = _body.semantic(sc);
            sc.pop();
            if (_body && _body.isErrorStatement())
                return _body;
        }
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class TryCatchStatement : Statement
{
public:
    Statement _body;
    Catches* catches;

    extern (D) this(Loc loc, Statement _body, Catches* catches)
    {
        super(loc);
        this._body = _body;
        this.catches = catches;
    }

    override Statement syntaxCopy()
    {
        auto a = new Catches();
        a.setDim(catches.dim);
        foreach (i, c; *catches)
        {
            (*a)[i] = c.syntaxCopy();
        }
        return new TryCatchStatement(loc, _body.syntaxCopy(), a);
    }

    override Statement semantic(Scope* sc)
    {
        uint flags;
        enum FLAGcpp = 1;
        enum FLAGd = 2;

        _body = _body.semanticScope(sc, null, null);
        assert(_body);
        /* Even if body is empty, still do semantic analysis on catches
         */
        bool catchErrors = false;
        foreach (i, c; *catches)
        {
            c.semantic(sc);
            if (c.errors)
            {
                catchErrors = true;
                continue;
            }
            auto cd = c.type.toBasetype().isClassHandle();
            flags |= cd.isCPPclass() ? FLAGcpp : FLAGd;

            // Determine if current catch 'hides' any previous catches
            foreach (j; 0 .. i)
            {
                Catch cj = (*catches)[j];
                const si = c.loc.toChars();
                const sj = cj.loc.toChars();
                if (c.type.toBasetype().implicitConvTo(cj.type.toBasetype()))
                {
                    error("catch at %s hides catch at %s", sj, si);
                    catchErrors = true;
                }
            }
        }

        if (sc.func)
        {
            if (flags == (FLAGcpp | FLAGd))
            {
                error("cannot mix catching D and C++ exceptions in the same try-catch");
                catchErrors = true;
            }
        }

        if (catchErrors)
            return new ErrorStatement();
        if (_body.isErrorStatement())
            return _body;
        /* If the try body never throws, we can eliminate any catches
         * of recoverable exceptions.
         */
        if (!(_body.blockExit(sc.func, false) & BEthrow) && ClassDeclaration.exception)
        {
            foreach_reverse (i; 0 .. catches.dim)
            {
                Catch c = (*catches)[i];
                /* If catch exception type is derived from Exception
                 */
                if (c.type.toBasetype().implicitConvTo(ClassDeclaration.exception.type) && (!c.handler || !c.handler.comeFrom()))
                {
                    // Remove c from the array of catches
                    catches.remove(i);
                }
            }
        }
        if (catches.dim == 0)
            return _body.hasCode() ? _body : null;
        return this;
    }

    override bool hasBreak()
    {
        return false;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class Catch : RootObject
{
public:
    Loc loc;
    Type type;
    Identifier ident;
    VarDeclaration var;
    Statement handler;

    bool errors;                // set if semantic processing errors

    // was generated by the compiler, wasn't present in source code
    bool internalCatch;

    extern (D) this(Loc loc, Type t, Identifier id, Statement handler)
    {
        //printf("Catch(%s, loc = %s)\n", id->toChars(), loc.toChars());
        this.loc = loc;
        this.type = t;
        this.ident = id;
        this.handler = handler;
    }

    Catch syntaxCopy()
    {
        auto c = new Catch(loc, type ? type.syntaxCopy() : null, ident, (handler ? handler.syntaxCopy() : null));
        c.internalCatch = internalCatch;
        return c;
    }

    void semantic(Scope* sc)
    {
        //printf("Catch::semantic(%s)\n", ident->toChars());
        static if (!IN_GCC)
        {
            if (sc.os && sc.os.tok != TOKon_scope_failure)
            {
                // If enclosing is scope(success) or scope(exit), this will be placed in finally block.
                error(loc, "cannot put catch statement inside %s", Token.toChars(sc.os.tok));
                errors = true;
            }
            if (sc.tf)
            {
                /* This is because the _d_local_unwind() gets the stack munged
                 * up on this. The workaround is to place any try-catches into
                 * a separate function, and call that.
                 * To fix, have the compiler automatically convert the finally
                 * body into a nested function.
                 */
                error(loc, "cannot put catch statement inside finally block");
                errors = true;
            }
        }
        auto sym = new ScopeDsymbol();
        sym.parent = sc.scopesym;
        sc = sc.push(sym);
        if (!type)
        {
            // reference .object.Throwable
            auto tid = new TypeIdentifier(Loc(), Id.empty);
            tid.addIdent(Id.object);
            tid.addIdent(Id.Throwable);
            type = tid;
        }
        type = type.semantic(loc, sc);
        if (type == Type.terror)
            errors = true;
        else
        {
            auto cd = type.toBasetype().isClassHandle();
            if (!cd)
            {
                error(loc, "can only catch class objects, not '%s'", type.toChars());
                errors = true;
            }
            else if (cd.isCPPclass())
            {
                if (!Target.cppExceptions)
                {
                    error(loc, "catching C++ class objects not supported for this target");
                    errors = true;
                }
                if (sc.func && !sc.intypeof && !internalCatch && sc.func.setUnsafe())
                {
                    error(loc, "cannot catch C++ class objects in @safe code");
                    errors = true;
                }
            }
            else if (cd != ClassDeclaration.throwable && !ClassDeclaration.throwable.isBaseOf(cd, null))
            {
                error(loc, "can only catch class objects derived from Throwable, not '%s'", type.toChars());
                errors = true;
            }
            else if (sc.func && !sc.intypeof && !internalCatch &&
                     cd != ClassDeclaration.exception && !ClassDeclaration.exception.isBaseOf(cd, null) &&
                     sc.func.setUnsafe())
            {
                error(loc, "can only catch class objects derived from Exception in @safe code, not '%s'", type.toChars());
                errors = true;
            }

            if (ident)
            {
                var = new VarDeclaration(loc, type, ident, null);
                var.semantic(sc);
                sc.insert(var);
            }
            handler = handler.semantic(sc);
            if (handler && handler.isErrorStatement())
                errors = true;
        }
        sc.pop();
    }
}

/***********************************************************
 */
extern (C++) final class TryFinallyStatement : Statement
{
public:
    Statement _body;
    Statement finalbody;

    extern (D) this(Loc loc, Statement _body, Statement finalbody)
    {
        super(loc);
        this._body = _body;
        this.finalbody = finalbody;
    }

    static TryFinallyStatement create(Loc loc, Statement _body, Statement finalbody)
    {
        return new TryFinallyStatement(loc, _body, finalbody);
    }

    override Statement syntaxCopy()
    {
        return new TryFinallyStatement(loc, _body.syntaxCopy(), finalbody.syntaxCopy());
    }

    override Statement semantic(Scope* sc)
    {
        //printf("TryFinallyStatement::semantic()\n");
        _body = _body.semantic(sc);
        sc = sc.push();
        sc.tf = this;
        sc.sbreak = null;
        sc.scontinue = null; // no break or continue out of finally block
        finalbody = finalbody.semanticNoScope(sc);
        sc.pop();
        if (!_body)
            return finalbody;
        if (!finalbody)
            return _body;
        if (_body.blockExit(sc.func, false) == BEfallthru)
        {
            Statement s = new CompoundStatement(loc, _body, finalbody);
            return s;
        }
        return this;
    }

    override bool hasBreak()
    {
        return false; //true;
    }

    override bool hasContinue()
    {
        return false; //true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class OnScopeStatement : Statement
{
public:
    TOK tok;
    Statement statement;

    extern (D) this(Loc loc, TOK tok, Statement statement)
    {
        super(loc);
        this.tok = tok;
        this.statement = statement;
    }

    override Statement syntaxCopy()
    {
        return new OnScopeStatement(loc, tok, statement.syntaxCopy());
    }

    override Statement semantic(Scope* sc)
    {
        static if (!IN_GCC)
        {
            if (tok != TOKon_scope_exit)
            {
                // scope(success) and scope(failure) are rewritten to try-catch(-finally) statement,
                // so the generated catch block cannot be placed in finally block.
                // See also Catch::semantic.
                if (sc.os && sc.os.tok != TOKon_scope_failure)
                {
                    // If enclosing is scope(success) or scope(exit), this will be placed in finally block.
                    error("cannot put %s statement inside %s", Token.toChars(tok), Token.toChars(sc.os.tok));
                    return new ErrorStatement();
                }
                if (sc.tf)
                {
                    error("cannot put %s statement inside finally block", Token.toChars(tok));
                    return new ErrorStatement();
                }
            }
        }
        sc = sc.push();
        sc.tf = null;
        sc.os = this;
        if (tok != TOKon_scope_failure)
        {
            // Jump out from scope(failure) block is allowed.
            sc.sbreak = null;
            sc.scontinue = null;
        }
        statement = statement.semanticNoScope(sc);
        sc.pop();
        if (!statement || statement.isErrorStatement())
            return statement;
        return this;
    }

    override Statement scopeCode(Scope* sc, Statement* sentry, Statement* sexception, Statement* sfinally)
    {
        //printf("OnScopeStatement::scopeCode()\n");
        //print();
        *sentry = null;
        *sexception = null;
        *sfinally = null;
        Statement s = new PeelStatement(statement);
        switch (tok)
        {
        case TOKon_scope_exit:
            *sfinally = s;
            break;
        case TOKon_scope_failure:
            *sexception = s;
            break;
        case TOKon_scope_success:
            {
                /* Create:
                 *  sentry:   bool x = false;
                 *  sexception:    x = true;
                 *  sfinally: if (!x) statement;
                 */
                Identifier id = Identifier.generateId("__os");
                auto ie = new ExpInitializer(loc, new IntegerExp(Loc(), 0, Type.tbool));
                auto v = new VarDeclaration(loc, Type.tbool, id, ie);
                v.storage_class |= STCtemp;
                *sentry = new ExpStatement(loc, v);
                Expression e = new IntegerExp(Loc(), 1, Type.tbool);
                e = new AssignExp(Loc(), new VarExp(Loc(), v), e);
                *sexception = new ExpStatement(Loc(), e);
                e = new VarExp(Loc(), v);
                e = new NotExp(Loc(), e);
                *sfinally = new IfStatement(Loc(), null, e, s, null);
                break;
            }
        default:
            assert(0);
        }
        return null;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class ThrowStatement : Statement
{
public:
    Expression exp;

    // was generated by the compiler, wasn't present in source code
    bool internalThrow;

    extern (D) this(Loc loc, Expression exp)
    {
        super(loc);
        this.exp = exp;
    }

    override Statement syntaxCopy()
    {
        auto s = new ThrowStatement(loc, exp.syntaxCopy());
        s.internalThrow = internalThrow;
        return s;
    }

    override Statement semantic(Scope* sc)
    {
        //printf("ThrowStatement::semantic()\n");
        FuncDeclaration fd = sc.parent.isFuncDeclaration();
        fd.hasReturnExp |= 2;
        exp = exp.semantic(sc);
        exp = resolveProperties(sc, exp);
        exp = checkGC(sc, exp);
        if (exp.op == TOKerror)
            return new ErrorStatement();
        ClassDeclaration cd = exp.type.toBasetype().isClassHandle();
        if (!cd || ((cd != ClassDeclaration.throwable) && !ClassDeclaration.throwable.isBaseOf(cd, null)))
        {
            error("can only throw class objects derived from Throwable, not type %s", exp.type.toChars());
            return new ErrorStatement();
        }
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class DebugStatement : Statement
{
public:
    Statement statement;

    extern (D) this(Loc loc, Statement statement)
    {
        super(loc);
        this.statement = statement;
    }

    override Statement syntaxCopy()
    {
        return new DebugStatement(loc, statement ? statement.syntaxCopy() : null);
    }

    override Statement semantic(Scope* sc)
    {
        if (statement)
        {
            sc = sc.push();
            sc.flags |= SCOPEdebug;
            statement = statement.semantic(sc);
            sc.pop();
        }
        return statement;
    }

    override Statements* flatten(Scope* sc)
    {
        Statements* a = statement ? statement.flatten(sc) : null;
        if (a)
        {
            foreach (ref s; *a)
            {
                s = new DebugStatement(loc, s);
            }
        }
        return a;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class GotoStatement : Statement
{
public:
    Identifier ident;
    LabelDsymbol label;
    TryFinallyStatement tf;
    OnScopeStatement os;
    VarDeclaration lastVar;

    extern (D) this(Loc loc, Identifier ident)
    {
        super(loc);
        this.ident = ident;
    }

    override Statement syntaxCopy()
    {
        return new GotoStatement(loc, ident);
    }

    override Statement semantic(Scope* sc)
    {
        //printf("GotoStatement::semantic()\n");
        FuncDeclaration fd = sc.func;
        ident = fixupLabelName(sc, ident);
        label = fd.searchLabel(ident);
        tf = sc.tf;
        os = sc.os;
        lastVar = sc.lastVar;
        if (!label.statement && sc.fes)
        {
            /* Either the goto label is forward referenced or it
             * is in the function that the enclosing foreach is in.
             * Can't know yet, so wrap the goto in a scope statement
             * so we can patch it later, and add it to a 'look at this later'
             * list.
             */
            auto ss = new ScopeStatement(loc, this);
            sc.fes.gotos.push(ss); // 'look at this later' list
            return ss;
        }
        // Add to fwdref list to check later
        if (!label.statement)
        {
            if (!fd.gotos)
                fd.gotos = new GotoStatements();
            fd.gotos.push(this);
        }
        else if (checkLabel())
            return new ErrorStatement();
        return this;
    }

    bool checkLabel()
    {
        if (!label.statement)
        {
            error("label '%s' is undefined", label.toChars());
            return true;
        }
        if (label.statement.os != os)
        {
            if (os && os.tok == TOKon_scope_failure && !label.statement.os)
            {
                // Jump out from scope(failure) block is allowed.
            }
            else
            {
                if (label.statement.os)
                    error("cannot goto in to %s block", Token.toChars(label.statement.os.tok));
                else
                    error("cannot goto out of %s block", Token.toChars(os.tok));
                return true;
            }
        }
        // IN_LLVM replaced: if (label.statement.tf != tf)
        if ( (label.statement !is null) && label.statement.tf != tf)
        {
            error("cannot goto in or out of finally block");
            return true;
        }
        VarDeclaration vd = label.statement.lastVar;
        if (!vd || vd.isDataseg() || (vd.storage_class & STCmanifest))
            return false;
        VarDeclaration last = lastVar;
        while (last && last != vd)
            last = last.lastVar;
        if (last == vd)
        {
            // All good, the label's scope has no variables
        }
        else if (vd.ident == Id.withSym)
        {
            error("goto skips declaration of with temporary at %s", vd.loc.toChars());
            return true;
        }
        else
        {
            error("goto skips declaration of variable %s at %s", vd.toPrettyChars(), vd.loc.toChars());
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
 */
extern (C++) final class LabelStatement : Statement
{
public:
    Identifier ident;
    Statement statement;
    TryFinallyStatement tf;
    OnScopeStatement os;
    VarDeclaration lastVar;
    Statement gotoTarget;       // interpret
    bool breaks;                // someone did a 'break ident'

    extern (D) this(Loc loc, Identifier ident, Statement statement)
    {
        super(loc);
        this.ident = ident;
        this.statement = statement;
    }

    override Statement syntaxCopy()
    {
        return new LabelStatement(loc, ident, statement ? statement.syntaxCopy() : null);
    }

    override Statement semantic(Scope* sc)
    {
        //printf("LabelStatement::semantic()\n");
        FuncDeclaration fd = sc.parent.isFuncDeclaration();
        ident = fixupLabelName(sc, ident);
        tf = sc.tf;
        os = sc.os;
        lastVar = sc.lastVar;
        LabelDsymbol ls = fd.searchLabel(ident);
        if (ls.statement)
        {
            error("label '%s' already defined", ls.toChars());
            return new ErrorStatement();
        }
        else
            ls.statement = this;
        sc = sc.push();
        sc.scopesym = sc.enclosing.scopesym;
        sc.callSuper |= CSXlabel;
        if (sc.fieldinit)
        {
            size_t dim = sc.fieldinit_dim;
            foreach (i; 0 .. dim)
                sc.fieldinit[i] |= CSXlabel;
        }
        sc.slabel = this;
        if (statement)
            statement = statement.semantic(sc);
        sc.pop();
        return this;
    }

    override Statements* flatten(Scope* sc)
    {
        Statements* a = null;
        if (statement)
        {
            a = statement.flatten(sc);
            if (a)
            {
                if (!a.dim)
                {
                    a.push(new ExpStatement(loc, cast(Expression)null));
                }
                // reuse 'this' LabelStatement
                this.statement = (*a)[0];
                (*a)[0] = this;
            }
        }
        return a;
    }

    override Statement scopeCode(Scope* sc, Statement* sentry, Statement* sexit, Statement* sfinally)
    {
        //printf("LabelStatement::scopeCode()\n");
        if (statement)
            statement = statement.scopeCode(sc, sentry, sexit, sfinally);
        else
        {
            *sentry = null;
            *sexit = null;
            *sfinally = null;
        }
        return this;
    }

    override LabelStatement isLabelStatement()
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
extern (C++) final class LabelDsymbol : Dsymbol
{
public:
    LabelStatement statement;

    extern (D) this(Identifier ident)
    {
        super(ident);
    }

    static LabelDsymbol create(Identifier ident)
    {
        return new LabelDsymbol(ident);
    }

    // is this a LabelDsymbol()?
    override LabelDsymbol isLabel()
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
extern (C++) final class AsmStatement : Statement
{
public:
    Token* tokens;
    code* asmcode;
    uint asmalign;  // alignment of this statement
    uint regs;      // mask of registers modified (must match regm_t in back end)
    bool refparam;  // true if function parameter is referenced
    bool naked;     // true if function is to be naked

    version(IN_LLVM)
    {
        // non-zero if this is a branch, contains the target label
        LabelDsymbol isBranchToLabel;
    }

    extern (D) this(Loc loc, Token* tokens)
    {
        super(loc);
        this.tokens = tokens;
    }

    override Statement syntaxCopy()
    {
version(IN_LLVM)
{
        auto a_s = new AsmStatement(loc, tokens);
        a_s.refparam = refparam;
        a_s.naked = naked;
        return a_s;
}
else
{
        return new AsmStatement(loc, tokens);
}
    }

    override Statement semantic(Scope* sc)
    {
        return asmSemantic(this, sc);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * a complete asm {} block
 */
extern (C++) final class CompoundAsmStatement : CompoundStatement
{
public:
    StorageClass stc; // postfix attributes like nothrow/pure/@trusted

    version(IN_LLVM)
    {
        void* abiret; // llvm::Value*
    }

    extern (D) this(Loc loc, Statements* s, StorageClass stc)
    {
        super(loc, s);
        this.stc = stc;
    }

    override CompoundAsmStatement syntaxCopy()
    {
        auto a = new Statements();
        a.setDim(statements.dim);
        foreach (i, s; *statements)
        {
            (*a)[i] = s ? s.syntaxCopy() : null;
        }
        return new CompoundAsmStatement(loc, a, stc);
    }

    override CompoundAsmStatement semantic(Scope* sc)
    {
        foreach (ref s; *statements)
        {
            s = s ? s.semantic(sc) : null;
        }
        assert(sc.func);
        // use setImpure/setGC when the deprecation cycle is over
        PURE purity;
        if (!(stc & STCpure) && (purity = sc.func.isPureBypassingInference()) != PUREimpure && purity != PUREfwdref)
            deprecation("asm statement is assumed to be impure - mark it with 'pure' if it is not");
        if (!(stc & STCnogc) && sc.func.isNogcBypassingInference())
            deprecation("asm statement is assumed to use the GC - mark it with '@nogc' if it does not");
        if (!(stc & (STCtrusted | STCsafe)) && sc.func.setUnsafe())
            error("asm statement is assumed to be @system - mark it with '@trusted' if it is not");
        return this;
    }

    override Statements* flatten(Scope* sc)
    {
        return null;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }

    version(IN_LLVM)
    {
        override final CompoundStatement isCompoundStatement()
        {
            return null;
        }
        override final CompoundAsmStatement isCompoundAsmBlockStatement()
        {
            return this;
        }

        override final CompoundAsmStatement endsWithAsm()
        {
            // yes this is inline asm
            return this;
        }
    }
}

/***********************************************************
 */
extern (C++) final class ImportStatement : Statement
{
public:
    Dsymbols* imports;      // Array of Import's

    extern (D) this(Loc loc, Dsymbols* imports)
    {
        super(loc);
        this.imports = imports;
    }

    override Statement syntaxCopy()
    {
        auto m = new Dsymbols();
        m.setDim(imports.dim);
        foreach (i, s; *imports)
        {
            (*m)[i] = s.syntaxCopy(null);
        }
        return new ImportStatement(loc, m);
    }

    override Statement semantic(Scope* sc)
    {
        foreach (i; 0 .. imports.dim)
        {
            Import s = (*imports)[i].isImport();
            assert(!s.aliasdecls.dim);
            foreach (j, name; s.names)
            {
                Identifier _alias = s.aliases[j];
                if (!_alias)
                    _alias = name;
                auto tname = new TypeIdentifier(s.loc, name);
                auto ad = new AliasDeclaration(s.loc, _alias, tname);
                ad._import = s;
                s.aliasdecls.push(ad);
            }
            s.semantic(sc);
            //s->semantic2(sc);     // Bugzilla 14666
            sc.insert(s);
            foreach (aliasdecl; s.aliasdecls)
            {
                sc.insert(aliasdecl);
            }
        }
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}
