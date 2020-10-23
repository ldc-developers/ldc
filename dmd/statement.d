/**
 * Defines AST nodes for statements.
 *
 * Specification: $(LINK2 https://dlang.org/spec/statement.html, Statements)
 *
 * Copyright:   Copyright (C) 1999-2020 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/statement.d, _statement.d)
 * Documentation:  https://dlang.org/phobos/dmd_statement.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/statement.d
 */

module dmd.statement;

import core.stdc.stdarg;
import core.stdc.stdio;

import dmd.aggregate;
import dmd.arraytypes;
import dmd.attrib;
import dmd.astcodegen;
import dmd.ast_node;
import dmd.gluelayer;
import dmd.canthrow;
import dmd.cond;
import dmd.dclass;
import dmd.declaration;
import dmd.denum;
import dmd.dimport;
import dmd.dscope;
import dmd.dsymbol;
import dmd.dsymbolsem;
import dmd.dtemplate;
import dmd.errors;
import dmd.expression;
import dmd.expressionsem;
import dmd.func;
import dmd.globals;
import dmd.hdrgen;
import dmd.id;
import dmd.identifier;
import dmd.dinterpret;
import dmd.mtype;
import dmd.parse;
import dmd.root.outbuffer;
import dmd.root.rootobject;
import dmd.sapply;
import dmd.sideeffect;
import dmd.staticassert;
import dmd.tokens;
import dmd.visitor;

version (IN_LLVM) import gen.dpragma;

/**
 * Returns:
 *     `TypeIdentifier` corresponding to `object.Throwable`
 */
TypeIdentifier getThrowable()
{
    auto tid = new TypeIdentifier(Loc.initial, Id.empty);
    tid.addIdent(Id.object);
    tid.addIdent(Id.Throwable);
    return tid;
}

/**
 * Returns:
 *      TypeIdentifier corresponding to `object.Exception`
 */
TypeIdentifier getException()
{
    auto tid = new TypeIdentifier(Loc.initial, Id.empty);
    tid.addIdent(Id.object);
    tid.addIdent(Id.Exception);
    return tid;
}

/********************************
 * Identify Statement types with this enum rather than
 * virtual functions.
 */

enum STMT : ubyte
{
    Error,
    Peel,
    Exp, DtorExp,
    Compile,
    Compound, CompoundDeclaration, CompoundAsm,
    UnrolledLoop,
    Scope,
    Forwarding,
    While,
    Do,
    For,
    Foreach,
    ForeachRange,
    If,
    Conditional,
    StaticForeach,
    Pragma,
    StaticAssert,
    Switch,
    Case,
    CaseRange,
    Default,
    GotoDefault,
    GotoCase,
    SwitchError,
    Return,
    Break,
    Continue,
    Synchronized,
    With,
    TryCatch,
    TryFinally,
    ScopeGuard,
    Throw,
    Debug,
    Goto,
    Label,
    Asm, InlineAsm, GccAsm,
    Import,
}


/***********************************************************
 * Specification: http://dlang.org/spec/statement.html
 */
extern (C++) abstract class Statement : ASTNode
{
    const Loc loc;
    const STMT stmt;

    override final DYNCAST dyncast() const
    {
        return DYNCAST.statement;
    }

    final extern (D) this(const ref Loc loc, STMT stmt)
    {
        this.loc = loc;
        this.stmt = stmt;
        // If this is an in{} contract scope statement (skip for determining
        //  inlineStatus of a function body for header content)
    }

    Statement syntaxCopy()
    {
        assert(0);
    }

    /*************************************
     * Do syntax copy of an array of Statement's.
     */
    static Statements* arraySyntaxCopy(Statements* a)
    {
        Statements* b = null;
        if (a)
        {
            b = a.copy();
            foreach (i, s; *a)
            {
                (*b)[i] = s ? s.syntaxCopy() : null;
            }
        }
        return b;
    }

    override final const(char)* toChars() const
    {
        HdrGenState hgs;
        OutBuffer buf;
        .toCBuffer(this, &buf, &hgs);
        buf.writeByte(0);
        return buf.extractSlice().ptr;
    }

    static if (__VERSION__ < 2092)
    {
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
    }
    else
    {
        pragma(printf) final void error(const(char)* format, ...)
        {
            va_list ap;
            va_start(ap, format);
            .verror(loc, format, ap);
            va_end(ap);
        }

        pragma(printf) final void warning(const(char)* format, ...)
        {
            va_list ap;
            va_start(ap, format);
            .vwarning(loc, format, ap);
            va_end(ap);
        }

        pragma(printf) final void deprecation(const(char)* format, ...)
        {
            va_list ap;
            va_start(ap, format);
            .vdeprecation(loc, format, ap);
            va_end(ap);
        }
    }

    Statement getRelatedLabeled()
    {
        return this;
    }

    /****************************
     * Determine if an enclosed `break` would apply to this
     * statement, such as if it is a loop or switch statement.
     * Returns:
     *     `true` if it does
     */
    bool hasBreak() const pure nothrow
    {
        //printf("Statement::hasBreak()\n");
        return false;
    }

    /****************************
     * Determine if an enclosed `continue` would apply to this
     * statement, such as if it is a loop statement.
     * Returns:
     *     `true` if it does
     */
    bool hasContinue() const pure nothrow
    {
        return false;
    }

    /**********************************
     * Returns:
     *     `true` if statement uses exception handling
     */
    final bool usesEH()
    {
        extern (C++) final class UsesEH : StoppableVisitor
        {
            alias visit = typeof(super).visit;
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

            override void visit(ScopeGuardStatement s)
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

    /**********************************
     * Returns:
     *   `true` if statement 'comes from' somewhere else, like a goto
     */
    final bool comeFrom()
    {
        extern (C++) final class ComeFrom : StoppableVisitor
        {
            alias visit = typeof(super).visit;
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

    /**********************************
     * Returns:
     *   `true` if statement has executable code.
     */
    final bool hasCode()
    {
        extern (C++) final class HasCode : StoppableVisitor
        {
            alias visit = typeof(super).visit;
        public:
            override void visit(Statement s)
            {
                stop = true;
            }

            override void visit(ExpStatement s)
            {
                if (s.exp !is null)
                {
                    stop = s.exp.hasCode();
                }
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
     * Params:
     *     sc = context
     *     sentry     = set to code executed upon entry to the scope
     *     sexception = set to code executed upon exit from the scope via exception
     *     sfinally   = set to code executed in finally block
     * Returns:
     *    code to be run in the finally clause
     */
    Statement scopeCode(Scope* sc, Statement* sentry, Statement* sexception, Statement* sfinally)
    {
        //printf("Statement::scopeCode()\n");
        *sentry = null;
        *sexception = null;
        *sfinally = null;
        return this;
    }

    /*********************************
     * Flatten out the scope by presenting the statement
     * as an array of statements.
     * Params:
     *     sc = context
     * Returns:
     *     The array of `Statements`, or `null` if no flattening necessary
     */
    Statements* flatten(Scope* sc)
    {
        return null;
    }

    /*******************************
     * Find last statement in a sequence of statements.
     * Returns:
     *  the last statement, or `null` if there isn't one
     */
    inout(Statement) last() inout nothrow pure
    {
        return this;
    }

    /**************************
     * Support Visitor Pattern
     * Params:
     *  v = visitor
     */
    override void accept(Visitor v)
    {
        v.visit(this);
    }

    /************************************
     * Does this statement end with a return statement?
     *
     * I.e. is it a single return statement or some compound statement
     * that unconditionally hits a return statement.
     * Returns:
     *  return statement it ends with, otherwise null
     */
    pure nothrow @nogc
    inout(ReturnStatement) endsWithReturnStatement() inout { return null; }

version (IN_LLVM)
{
    pure nothrow @nogc
    inout(CompoundAsmStatement) endsWithAsm() inout { return null; }
}

  final pure inout nothrow @nogc:

    /********************
     * A cheaper method of doing downcasting of Statements.
     * Returns:
     *    the downcast statement if it can be downcasted, otherwise `null`
     */
    inout(ErrorStatement)       isErrorStatement()       { return stmt == STMT.Error       ? cast(typeof(return))this : null; }
    inout(ScopeStatement)       isScopeStatement()       { return stmt == STMT.Scope       ? cast(typeof(return))this : null; }
    inout(ExpStatement)         isExpStatement()         { return stmt == STMT.Exp         ? cast(typeof(return))this : null; }
    inout(CompoundStatement)    isCompoundStatement()    { return stmt == STMT.Compound    ? cast(typeof(return))this : null; }
    version (IN_LLVM)
    inout(CompoundAsmStatement) isCompoundAsmStatement() { return stmt == STMT.CompoundAsm ? cast(typeof(return))this : null; }
    inout(ReturnStatement)      isReturnStatement()      { return stmt == STMT.Return      ? cast(typeof(return))this : null; }
    inout(IfStatement)          isIfStatement()          { return stmt == STMT.If          ? cast(typeof(return))this : null; }
    inout(CaseStatement)        isCaseStatement()        { return stmt == STMT.Case        ? cast(typeof(return))this : null; }
    inout(DefaultStatement)     isDefaultStatement()     { return stmt == STMT.Default     ? cast(typeof(return))this : null; }
    inout(LabelStatement)       isLabelStatement()       { return stmt == STMT.Label       ? cast(typeof(return))this : null; }
    inout(GotoStatement)        isGotoStatement()        { return stmt == STMT.Goto        ? cast(typeof(return))this : null; }
    inout(GotoDefaultStatement) isGotoDefaultStatement() { return stmt == STMT.GotoDefault ? cast(typeof(return))this : null; }
    inout(GotoCaseStatement)    isGotoCaseStatement()    { return stmt == STMT.GotoCase    ? cast(typeof(return))this : null; }
    inout(BreakStatement)       isBreakStatement()       { return stmt == STMT.Break       ? cast(typeof(return))this : null; }
    inout(DtorExpStatement)     isDtorExpStatement()     { return stmt == STMT.DtorExp     ? cast(typeof(return))this : null; }
    inout(ForwardingStatement)  isForwardingStatement()  { return stmt == STMT.Forwarding  ? cast(typeof(return))this : null; }
    inout(DoStatement)          isDoStatement()          { return stmt == STMT.Do          ? cast(typeof(return))this : null; }
    inout(WhileStatement)       isWhileStatement()       { return stmt == STMT.While       ? cast(typeof(return))this : null; }
    inout(ForStatement)         isForStatement()         { return stmt == STMT.For         ? cast(typeof(return))this : null; }
    inout(ForeachStatement)     isForeachStatement()     { return stmt == STMT.Foreach     ? cast(typeof(return))this : null; }
    inout(SwitchStatement)      isSwitchStatement()      { return stmt == STMT.Switch      ? cast(typeof(return))this : null; }
    inout(ContinueStatement)    isContinueStatement()    { return stmt == STMT.Continue    ? cast(typeof(return))this : null; }
    inout(WithStatement)        isWithStatement()        { return stmt == STMT.With        ? cast(typeof(return))this : null; }
    inout(TryCatchStatement)    isTryCatchStatement()    { return stmt == STMT.TryCatch    ? cast(typeof(return))this : null; }
    inout(ThrowStatement)       isThrowStatement()       { return stmt == STMT.Throw       ? cast(typeof(return))this : null; }
    inout(TryFinallyStatement)  isTryFinallyStatement()  { return stmt == STMT.TryFinally  ? cast(typeof(return))this : null; }
    inout(SwitchErrorStatement)  isSwitchErrorStatement()  { return stmt == STMT.SwitchError  ? cast(typeof(return))this : null; }
    inout(UnrolledLoopStatement) isUnrolledLoopStatement() { return stmt == STMT.UnrolledLoop ? cast(typeof(return))this : null; }
    inout(ForeachRangeStatement) isForeachRangeStatement() { return stmt == STMT.ForeachRange ? cast(typeof(return))this : null; }
    inout(CompoundDeclarationStatement) isCompoundDeclarationStatement() { return stmt == STMT.CompoundDeclaration ? cast(typeof(return))this : null; }
}

/***********************************************************
 * Any Statement that fails semantic() or has a component that is an ErrorExp or
 * a TypeError should return an ErrorStatement from semantic().
 */
extern (C++) final class ErrorStatement : Statement
{
    extern (D) this()
    {
        super(Loc.initial, STMT.Error);
        assert(global.gaggedErrors || global.errors);
    }

    override Statement syntaxCopy()
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
    Statement s;

    extern (D) this(Statement s)
    {
        super(s.loc, STMT.Peel);
        this.s = s;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Convert TemplateMixin members (== Dsymbols) to Statements.
 */
private Statement toStatement(Dsymbol s)
{
    extern (C++) final class ToStmt : Visitor
    {
        alias visit = Visitor.visit;
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
            .error(Loc.initial, "Internal Compiler Error: cannot mixin %s `%s`\n", s.kind(), s.toChars());
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

        override void visit(ForwardingAttribDeclaration d)
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
            result = visitMembers(d.loc, d.include(null));
        }

        override void visit(StaticForeachDeclaration d)
        {
            assert(d.sfe && !!d.sfe.aggrfe ^ !!d.sfe.rangefe);
            result = visitMembers(d.loc, d.include(null));
        }

        override void visit(CompileDeclaration d)
        {
            result = visitMembers(d.loc, d.include(null));
        }
    }

    if (!s)
        return null;

    scope ToStmt v = new ToStmt();
    s.accept(v);
    return v.result;
}

/***********************************************************
 * https://dlang.org/spec/statement.html#ExpressionStatement
 */
extern (C++) class ExpStatement : Statement
{
    Expression exp;

    final extern (D) this(const ref Loc loc, Expression exp)
    {
        super(loc, STMT.Exp);
        this.exp = exp;
    }

    final extern (D) this(const ref Loc loc, Expression exp, STMT stmt)
    {
        super(loc, stmt);
        this.exp = exp;
    }

    final extern (D) this(const ref Loc loc, Dsymbol declaration)
    {
        super(loc, STMT.Exp);
        this.exp = new DeclarationExp(loc, declaration);
    }

    static ExpStatement create(Loc loc, Expression exp)
    {
        return new ExpStatement(loc, exp);
    }

    override Statement syntaxCopy()
    {
        return new ExpStatement(loc, exp ? exp.syntaxCopy() : null);
    }

    override final Statement scopeCode(Scope* sc, Statement* sentry, Statement* sexception, Statement* sfinally)
    {
        //printf("ExpStatement::scopeCode()\n");

        *sentry = null;
        *sexception = null;
        *sfinally = null;

        if (exp && exp.op == TOK.declaration)
        {
            auto de = cast(DeclarationExp)exp;
            auto v = de.declaration.isVarDeclaration();
            if (v && !v.isDataseg())
            {
                if (v.needsScopeDtor())
                {
                    *sfinally = new DtorExpStatement(loc, v.edtor, v);
                    v.storage_class |= STC.nodtor; // don't add in dtor again
                }
            }
        }
        return this;
    }

    override final Statements* flatten(Scope* sc)
    {
        /* https://issues.dlang.org/show_bug.cgi?id=14243
         * expand template mixin in statement scope
         * to handle variable destructors.
         */
        if (exp && exp.op == TOK.declaration)
        {
            Dsymbol d = (cast(DeclarationExp)exp).declaration;
            if (TemplateMixin tm = d.isTemplateMixin())
            {
                Expression e = exp.expressionSemantic(sc);
                if (e.op == TOK.error || tm.errors)
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
                    printf("tm ==> s = %s\n", buf.peekChars());
                }
                auto a = new Statements();
                a.push(s);
                return a;
            }
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
extern (C++) final class DtorExpStatement : ExpStatement
{
    // Wraps an expression that is the destruction of 'var'
    VarDeclaration var;

    extern (D) this(const ref Loc loc, Expression exp, VarDeclaration var)
    {
        super(loc, exp, STMT.DtorExp);
        this.var = var;
    }

    override Statement syntaxCopy()
    {
        return new DtorExpStatement(loc, exp ? exp.syntaxCopy() : null, var);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#mixin-statement
 */
extern (C++) final class CompileStatement : Statement
{
    Expressions* exps;

    extern (D) this(const ref Loc loc, Expression exp)
    {
        Expressions* exps = new Expressions();
        exps.push(exp);
        this(loc, exps);
    }

    extern (D) this(const ref Loc loc, Expressions* exps)
    {
        super(loc, STMT.Compile);
        this.exps = exps;
    }

    override Statement syntaxCopy()
    {
        return new CompileStatement(loc, Expression.arraySyntaxCopy(exps));
    }

    private Statements* compileIt(Scope* sc)
    {
        //printf("CompileStatement::compileIt() %s\n", exp.toChars());

        auto errorStatements()
        {
            auto a = new Statements();
            a.push(new ErrorStatement());
            return a;
        }


        OutBuffer buf;
        if (expressionsToString(buf, sc, exps))
            return errorStatements();

        const errors = global.errors;
        const len = buf.length;
        buf.writeByte(0);
        const str = buf.extractSlice()[0 .. len];
        scope p = new Parser!ASTCodegen(loc, sc._module, str, false);
        p.nextToken();

        auto a = new Statements();
        while (p.token.value != TOK.endOfFile)
        {
            Statement s = p.parseStatement(ParseStatementFlags.semi | ParseStatementFlags.curlyScope);
            if (!s || global.errors != errors)
                return errorStatements();
            a.push(s);
        }
        return a;
    }

    override Statements* flatten(Scope* sc)
    {
        //printf("CompileStatement::flatten() %s\n", exp.toChars());
        return compileIt(sc);
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
    Statements* statements;

    /**
     * Construct a `CompoundStatement` using an already existing
     * array of `Statement`s
     *
     * Params:
     *   loc = Instantiation information
     *   statements   = An array of `Statement`s, that will referenced by this class
     */
    final extern (D) this(const ref Loc loc, Statements* statements)
    {
        super(loc, STMT.Compound);
        this.statements = statements;
    }

    final extern (D) this(const ref Loc loc, Statements* statements, STMT stmt)
    {
        super(loc, stmt);
        this.statements = statements;
    }

    /**
     * Construct a `CompoundStatement` from an array of `Statement`s
     *
     * Params:
     *   loc = Instantiation information
     *   sts   = A variadic array of `Statement`s, that will copied in this class
     *         The entries themselves will not be copied.
     */
    final extern (D) this(const ref Loc loc, Statement[] sts...)
    {
        super(loc, STMT.Compound);
        statements = new Statements();
        statements.reserve(sts.length);
        foreach (s; sts)
            statements.push(s);
    }

    static CompoundStatement create(Loc loc, Statement s1, Statement s2)
    {
        return new CompoundStatement(loc, s1, s2);
    }

    override Statement syntaxCopy()
    {
        return new CompoundStatement(loc, Statement.arraySyntaxCopy(statements));
    }

    override Statements* flatten(Scope* sc)
    {
        return statements;
    }

    override final inout(ReturnStatement) endsWithReturnStatement() inout nothrow pure
    {
        foreach (s; *statements)
        {
            if (s)
            {
                if (inout rs = s.endsWithReturnStatement())
                    return rs;
            }
        }
        return null;
    }

    override final inout(Statement) last() inout nothrow pure
    {
        Statement s = null;
        for (size_t i = statements.dim; i; --i)
        {
            s = cast(Statement)(*statements)[i - 1];
            if (s)
            {
                s = cast(Statement)s.last();
                if (s)
                    break;
            }
        }
        return cast(inout)s;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }

version (IN_LLVM)
{
    override inout(CompoundAsmStatement) endsWithAsm() inout pure nothrow @nogc
    {
        // make the last inner statement decide
        if (statements && statements.dim)
        {
            size_t last = statements.dim - 1;
            if (auto s = (*statements)[last])
                return s.endsWithAsm();
        }
        return null;
    }
}
}

/***********************************************************
 */
extern (C++) final class CompoundDeclarationStatement : CompoundStatement
{
    extern (D) this(const ref Loc loc, Statements* statements)
    {
        super(loc, statements, STMT.CompoundDeclaration);
    }

    override Statement syntaxCopy()
    {
        auto a = new Statements(statements.dim);
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
    Statements* statements;

    extern (D) this(const ref Loc loc, Statements* statements)
    {
        super(loc, STMT.UnrolledLoop);
        this.statements = statements;
    }

    override Statement syntaxCopy()
    {
        auto a = new Statements(statements.dim);
        foreach (i, s; *statements)
        {
            (*a)[i] = s ? s.syntaxCopy() : null;
        }
        return new UnrolledLoopStatement(loc, a);
    }

    override bool hasBreak() const pure nothrow
    {
        return true;
    }

    override bool hasContinue() const pure nothrow
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
extern (C++) class ScopeStatement : Statement
{
    Statement statement;
    Loc endloc;                 // location of closing curly bracket

    extern (D) this(const ref Loc loc, Statement statement, Loc endloc)
    {
        super(loc, STMT.Scope);
        this.statement = statement;
        this.endloc = endloc;
    }
    override Statement syntaxCopy()
    {
        return new ScopeStatement(loc, statement ? statement.syntaxCopy() : null, endloc);
    }

    override inout(ReturnStatement) endsWithReturnStatement() inout nothrow pure
    {
        if (statement)
            return statement.endsWithReturnStatement();
        return null;
    }

    override bool hasBreak() const pure nothrow
    {
        //printf("ScopeStatement::hasBreak() %s\n", toChars());
        return statement ? statement.hasBreak() : false;
    }

    override bool hasContinue() const pure nothrow
    {
        return statement ? statement.hasContinue() : false;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Statement whose symbol table contains foreach index variables in a
 * local scope and forwards other members to the parent scope.  This
 * wraps a statement.
 *
 * Also see: `dmd.attrib.ForwardingAttribDeclaration`
 */
extern (C++) final class ForwardingStatement : Statement
{
    /// The symbol containing the `static foreach` variables.
    ForwardingScopeDsymbol sym = null;
    /// The wrapped statement.
    Statement statement;

    extern (D) this(const ref Loc loc, ForwardingScopeDsymbol sym, Statement statement)
    {
        super(loc, STMT.Forwarding);
        this.sym = sym;
        assert(statement);
        this.statement = statement;
    }

    extern (D) this(const ref Loc loc, Statement statement)
    {
        auto sym = new ForwardingScopeDsymbol(null);
        sym.symtab = new DsymbolTable();
        this(loc, sym, statement);
    }

    override Statement syntaxCopy()
    {
        return new ForwardingStatement(loc, statement.syntaxCopy());
    }

    /***********************
     * ForwardingStatements are distributed over the flattened
     * sequence of statements. This prevents flattening to be
     * "blocked" by a ForwardingStatement and is necessary, for
     * example, to support generating scope guards with `static
     * foreach`:
     *
     *     static foreach(i; 0 .. 10) scope(exit) writeln(i);
     *     writeln("this is printed first");
     *     // then, it prints 10, 9, 8, 7, ...
     */

    override Statements* flatten(Scope* sc)
    {
        if (!statement)
        {
            return null;
        }
        sc = sc.push(sym);
        auto a = statement.flatten(sc);
        sc = sc.pop();
        if (!a)
        {
            return a;
        }
        auto b = new Statements(a.dim);
        foreach (i, s; *a)
        {
            (*b)[i] = s ? new ForwardingStatement(s.loc, sym, s) : null;
        }
        return b;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}


/***********************************************************
 * https://dlang.org/spec/statement.html#while-statement
 */
extern (C++) final class WhileStatement : Statement
{
    Expression condition;
    Statement _body;
    Loc endloc;             // location of closing curly bracket

    extern (D) this(const ref Loc loc, Expression condition, Statement _body, Loc endloc)
    {
        super(loc, STMT.While);
        this.condition = condition;
        this._body = _body;
        this.endloc = endloc;
    }

    override Statement syntaxCopy()
    {
        return new WhileStatement(loc,
            condition.syntaxCopy(),
            _body ? _body.syntaxCopy() : null,
            endloc);
    }

    override bool hasBreak() const pure nothrow
    {
        return true;
    }

    override bool hasContinue() const pure nothrow
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#do-statement
 */
extern (C++) final class DoStatement : Statement
{
    Statement _body;
    Expression condition;
    Loc endloc;                 // location of ';' after while

    extern (D) this(const ref Loc loc, Statement _body, Expression condition, Loc endloc)
    {
        super(loc, STMT.Do);
        this._body = _body;
        this.condition = condition;
        this.endloc = endloc;
    }

    override Statement syntaxCopy()
    {
        return new DoStatement(loc,
            _body ? _body.syntaxCopy() : null,
            condition.syntaxCopy(),
            endloc);
    }

    override bool hasBreak() const pure nothrow
    {
        return true;
    }

    override bool hasContinue() const pure nothrow
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#for-statement
 */
extern (C++) final class ForStatement : Statement
{
    Statement _init;
    Expression condition;
    Expression increment;
    Statement _body;
    Loc endloc;             // location of closing curly bracket

    // When wrapped in try/finally clauses, this points to the outermost one,
    // which may have an associated label. Internal break/continue statements
    // treat that label as referring to this loop.
    Statement relatedLabeled;

    extern (D) this(const ref Loc loc, Statement _init, Expression condition, Expression increment, Statement _body, Loc endloc)
    {
        super(loc, STMT.For);
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

    override bool hasBreak() const pure nothrow
    {
        //printf("ForStatement::hasBreak()\n");
        return true;
    }

    override bool hasContinue() const pure nothrow
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#foreach-statement
 */
extern (C++) final class ForeachStatement : Statement
{
    TOK op;                     // TOK.foreach_ or TOK.foreach_reverse_
    Parameters* parameters;     // array of Parameters, one for each ForeachType
    Expression aggr;            // ForeachAggregate
    Statement _body;            // NoScopeNonEmptyStatement
    Loc endloc;                 // location of closing curly bracket

    VarDeclaration key;
    VarDeclaration value;

    FuncDeclaration func;       // function we're lexically in

    Statements* cases;          // put breaks, continues, gotos and returns here
    ScopeStatements* gotos;     // forward referenced goto's go here

    extern (D) this(const ref Loc loc, TOK op, Parameters* parameters, Expression aggr, Statement _body, Loc endloc)
    {
        super(loc, STMT.Foreach);
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

    override bool hasBreak() const pure nothrow
    {
        return true;
    }

    override bool hasContinue() const pure nothrow
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#foreach-range-statement
 */
extern (C++) final class ForeachRangeStatement : Statement
{
    TOK op;                 // TOK.foreach_ or TOK.foreach_reverse_
    Parameter prm;          // loop index variable
    Expression lwr;
    Expression upr;
    Statement _body;
    Loc endloc;             // location of closing curly bracket

    VarDeclaration key;

    extern (D) this(const ref Loc loc, TOK op, Parameter prm, Expression lwr, Expression upr, Statement _body, Loc endloc)
    {
        super(loc, STMT.ForeachRange);
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

    override bool hasBreak() const pure nothrow
    {
        return true;
    }

    override bool hasContinue() const pure nothrow
    {
        return true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#if-statement
 */
extern (C++) final class IfStatement : Statement
{
    Parameter prm;
    Expression condition;
    Statement ifbody;
    Statement elsebody;
    VarDeclaration match;   // for MatchExpression results
    Loc endloc;                 // location of closing curly bracket

    extern (D) this(const ref Loc loc, Parameter prm, Expression condition, Statement ifbody, Statement elsebody, Loc endloc)
    {
        super(loc, STMT.If);
        this.prm = prm;
        this.condition = condition;
        this.ifbody = ifbody;
        this.elsebody = elsebody;
        this.endloc = endloc;
    }

    override Statement syntaxCopy()
    {
        return new IfStatement(loc,
            prm ? prm.syntaxCopy() : null,
            condition.syntaxCopy(),
            ifbody ? ifbody.syntaxCopy() : null,
            elsebody ? elsebody.syntaxCopy() : null,
            endloc);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/version.html#ConditionalStatement
 */
extern (C++) final class ConditionalStatement : Statement
{
    Condition condition;
    Statement ifbody;
    Statement elsebody;

    extern (D) this(const ref Loc loc, Condition condition, Statement ifbody, Statement elsebody)
    {
        super(loc, STMT.Conditional);
        this.condition = condition;
        this.ifbody = ifbody;
        this.elsebody = elsebody;
    }

    override Statement syntaxCopy()
    {
        return new ConditionalStatement(loc, condition.syntaxCopy(), ifbody.syntaxCopy(), elsebody ? elsebody.syntaxCopy() : null);
    }

    override Statements* flatten(Scope* sc)
    {
        Statement s;

        //printf("ConditionalStatement::flatten()\n");
        if (condition.include(sc))
        {
            DebugCondition dc = condition.isDebugCondition();
            if (dc)
            {
                s = new DebugStatement(loc, ifbody);
                debugThrowWalker(ifbody);
            }
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

/**
Marks all occurring ThrowStatements as internalThrows.
This is intended to be called from a DebugStatement as it allows
to mark all its nodes as nothrow.

Params:
    s = AST Node to traverse
*/
private void debugThrowWalker(Statement s)
{

    extern(C++) final class DebugWalker : SemanticTimeTransitiveVisitor
    {
        alias visit = SemanticTimeTransitiveVisitor.visit;
    public:

        override void visit(ThrowStatement s)
        {
            s.internalThrow = true;
        }

        override void visit(CallExp s)
        {
            s.inDebugStatement = true;
        }
    }

    scope walker = new DebugWalker();
    s.accept(walker);
}

/***********************************************************
 * https://dlang.org/spec/version.html#StaticForeachStatement
 * Static foreach statements, like:
 *      void main()
 *      {
 *           static foreach(i; 0 .. 10)
 *           {
 *               pragma(msg, i);
 *           }
 *      }
 */
extern (C++) final class StaticForeachStatement : Statement
{
    StaticForeach sfe;

    extern (D) this(const ref Loc loc, StaticForeach sfe)
    {
        super(loc, STMT.StaticForeach);
        this.sfe = sfe;
    }

    override Statement syntaxCopy()
    {
        return new StaticForeachStatement(loc, sfe.syntaxCopy());
    }

    override Statements* flatten(Scope* sc)
    {
        sfe.prepare(sc);
        if (sfe.ready())
        {
            import dmd.statementsem;
            auto s = makeTupleForeach!(true, false)(sc, sfe.aggrfe, sfe.needExpansion);
            auto result = s.flatten(sc);
            if (result)
            {
                return result;
            }
            result = new Statements();
            result.push(s);
            return result;
        }
        else
        {
            auto result = new Statements();
            result.push(new ErrorStatement());
            return result;
        }
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#pragma-statement
 */
extern (C++) final class PragmaStatement : Statement
{
    const Identifier ident;
    Expressions* args;      // array of Expression's
    Statement _body;

    extern (D) this(const ref Loc loc, const Identifier ident, Expressions* args, Statement _body)
    {
        super(loc, STMT.Pragma);
        this.ident = ident;
        this.args = args;
        this._body = _body;
    }

    override Statement syntaxCopy()
    {
        return new PragmaStatement(loc, ident, Expression.arraySyntaxCopy(args), _body ? _body.syntaxCopy() : null);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/version.html#StaticAssert
 */
extern (C++) final class StaticAssertStatement : Statement
{
    StaticAssert sa;

    extern (D) this(StaticAssert sa)
    {
        super(sa.loc, STMT.StaticAssert);
        this.sa = sa;
    }

    override Statement syntaxCopy()
    {
        return new StaticAssertStatement(cast(StaticAssert)sa.syntaxCopy(null));
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#switch-statement
 */
extern (C++) final class SwitchStatement : Statement
{
    Expression condition;           /// switch(condition)
    Statement _body;                ///
    bool isFinal;                   /// https://dlang.org/spec/statement.html#final-switch-statement

    DefaultStatement sdefault;      /// default:
    Statement tryBody;              /// set to TryCatchStatement or TryFinallyStatement if in _body portion
    TryFinallyStatement tf;         /// set if in the 'finally' block of a TryFinallyStatement
    GotoCaseStatements gotoCases;   /// array of unresolved GotoCaseStatement's
    CaseStatements* cases;          /// array of CaseStatement's
    int hasNoDefault;               /// !=0 if no default statement
    int hasVars;                    /// !=0 if has variable case values
    VarDeclaration lastVar;         /// last observed variable declaration in this statement
version (IN_LLVM)
{
    bool hasGotoDefault;            // true iff there is a `goto default` statement for this switch
}

    extern (D) this(const ref Loc loc, Expression condition, Statement _body, bool isFinal)
    {
        super(loc, STMT.Switch);
        this.condition = condition;
        this._body = _body;
        this.isFinal = isFinal;
    }

    override Statement syntaxCopy()
    {
        return new SwitchStatement(loc, condition.syntaxCopy(), _body.syntaxCopy(), isFinal);
    }

    override bool hasBreak() const pure nothrow
    {
        return true;
    }

    /************************************
     * Returns:
     *  true if error
     */
    extern (D) bool checkLabel()
    {
        /*
         * Checks the scope of a label for existing variable declaration.
         * Params:
         *   vd = last variable declared before this case/default label
         * Returns: `true` if the variables declared in this label would be skipped.
         */
        bool checkVar(VarDeclaration vd)
        {
            for (auto v = vd; v && v != lastVar; v = v.lastVar)
            {
                if (v.isDataseg() || (v.storage_class & (STC.manifest | STC.temp)) || v._init.isVoidInitializer())
                    continue;
                if (vd.ident == Id.withSym)
                    error("`switch` skips declaration of `with` temporary at %s", v.loc.toChars());
                else
                    error("`switch` skips declaration of variable `%s` at %s", v.toPrettyChars(), v.loc.toChars());
                return true;
            }
            return false;
        }

        enum error = true;

        if (sdefault && checkVar(sdefault.lastVar))
            return !error; // return error once fully deprecated

        foreach (scase; *cases)
        {
            if (scase && checkVar(scase.lastVar))
                return !error; // return error once fully deprecated
        }
        return !error;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#CaseStatement
 */
extern (C++) final class CaseStatement : Statement
{
    Expression exp;
    Statement statement;

    int index;              // which case it is (since we sort this)
    VarDeclaration lastVar;
    void* extra;            // for use by Statement_toIR()

version (IN_LLVM)
{
    bool gototarget; // true iff this is the target of a 'goto case'
}

    extern (D) this(const ref Loc loc, Expression exp, Statement statement)
    {
        super(loc, STMT.Case);
        this.exp = exp;
        this.statement = statement;
    }

    override Statement syntaxCopy()
    {
        return new CaseStatement(loc, exp.syntaxCopy(), statement.syntaxCopy());
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#CaseRangeStatement
 */
extern (C++) final class CaseRangeStatement : Statement
{
    Expression first;
    Expression last;
    Statement statement;

    extern (D) this(const ref Loc loc, Expression first, Expression last, Statement statement)
    {
        super(loc, STMT.CaseRange);
        this.first = first;
        this.last = last;
        this.statement = statement;
    }

    override Statement syntaxCopy()
    {
        return new CaseRangeStatement(loc, first.syntaxCopy(), last.syntaxCopy(), statement.syntaxCopy());
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#DefaultStatement
 */
extern (C++) final class DefaultStatement : Statement
{
    Statement statement;

    VarDeclaration lastVar;

version (IN_LLVM)
{
    bool gototarget; // true iff this is the target of a 'goto default'
}

    extern (D) this(const ref Loc loc, Statement statement)
    {
        super(loc, STMT.Default);
        this.statement = statement;
    }

    override Statement syntaxCopy()
    {
        return new DefaultStatement(loc, statement.syntaxCopy());
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#GotoStatement
 */
extern (C++) final class GotoDefaultStatement : Statement
{
    SwitchStatement sw;

    extern (D) this(const ref Loc loc)
    {
        super(loc, STMT.GotoDefault);
    }

    override Statement syntaxCopy()
    {
        return new GotoDefaultStatement(loc);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#GotoStatement
 */
extern (C++) final class GotoCaseStatement : Statement
{
    Expression exp;     // null, or which case to goto

    CaseStatement cs;   // case statement it resolves to

version (IN_LLVM)
{
    SwitchStatement sw;
}

    extern (D) this(const ref Loc loc, Expression exp)
    {
        super(loc, STMT.GotoCase);
        this.exp = exp;
    }

    override Statement syntaxCopy()
    {
        return new GotoCaseStatement(loc, exp ? exp.syntaxCopy() : null);
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
    Expression exp;

    extern (D) this(const ref Loc loc)
    {
        super(loc, STMT.SwitchError);
    }

    final extern (D) this(const ref Loc loc, Expression exp)
    {
        super(loc, STMT.SwitchError);
        this.exp = exp;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#return-statement
 */
extern (C++) final class ReturnStatement : Statement
{
    Expression exp;
    size_t caseDim;

    extern (D) this(const ref Loc loc, Expression exp)
    {
        super(loc, STMT.Return);
        this.exp = exp;
    }

    override Statement syntaxCopy()
    {
        return new ReturnStatement(loc, exp ? exp.syntaxCopy() : null);
    }

    override inout(ReturnStatement) endsWithReturnStatement() inout nothrow pure
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#break-statement
 */
extern (C++) final class BreakStatement : Statement
{
    Identifier ident;

version (IN_LLVM)
{
    // LDC: only set if ident is set: label statement to jump to
    LabelStatement target;
}

    extern (D) this(const ref Loc loc, Identifier ident)
    {
        super(loc, STMT.Break);
        this.ident = ident;
    }

    override Statement syntaxCopy()
    {
        return new BreakStatement(loc, ident);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#continue-statement
 */
extern (C++) final class ContinueStatement : Statement
{
    Identifier ident;

version (IN_LLVM)
{
    // LDC: only set if ident is set: label statement to jump to
    LabelStatement target;
}

    extern (D) this(const ref Loc loc, Identifier ident)
    {
        super(loc, STMT.Continue);
        this.ident = ident;
    }

    override Statement syntaxCopy()
    {
        return new ContinueStatement(loc, ident);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#SynchronizedStatement
 */
extern (C++) final class SynchronizedStatement : Statement
{
    Expression exp;
    Statement _body;

    extern (D) this(const ref Loc loc, Expression exp, Statement _body)
    {
        super(loc, STMT.Synchronized);
        this.exp = exp;
        this._body = _body;
    }

    override Statement syntaxCopy()
    {
        return new SynchronizedStatement(loc, exp ? exp.syntaxCopy() : null, _body ? _body.syntaxCopy() : null);
    }

    override bool hasBreak() const pure nothrow
    {
        return false; //true;
    }

    override bool hasContinue() const pure nothrow
    {
        return false; //true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#with-statement
 */
extern (C++) final class WithStatement : Statement
{
    Expression exp;
    Statement _body;
    VarDeclaration wthis;
    Loc endloc;

    extern (D) this(const ref Loc loc, Expression exp, Statement _body, Loc endloc)
    {
        super(loc, STMT.With);
        this.exp = exp;
        this._body = _body;
        this.endloc = endloc;
    }

    override Statement syntaxCopy()
    {
        return new WithStatement(loc, exp.syntaxCopy(), _body ? _body.syntaxCopy() : null, endloc);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#try-statement
 */
extern (C++) final class TryCatchStatement : Statement
{
    Statement _body;
    Catches* catches;

    Statement tryBody;   /// set to enclosing TryCatchStatement or TryFinallyStatement if in _body portion

    extern (D) this(const ref Loc loc, Statement _body, Catches* catches)
    {
        super(loc, STMT.TryCatch);
        this._body = _body;
        this.catches = catches;
    }

    override Statement syntaxCopy()
    {
        auto a = new Catches(catches.dim);
        foreach (i, c; *catches)
        {
            (*a)[i] = c.syntaxCopy();
        }
        return new TryCatchStatement(loc, _body.syntaxCopy(), a);
    }

    override bool hasBreak() const pure nothrow
    {
        return false;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#Catch
 */
extern (C++) final class Catch : RootObject
{
    const Loc loc;
    Type type;
    Identifier ident;
    Statement handler;

    VarDeclaration var;
    bool errors;                // set if semantic processing errors

    // was generated by the compiler, wasn't present in source code
    bool internalCatch;

    extern (D) this(const ref Loc loc, Type type, Identifier ident, Statement handler)
    {
        //printf("Catch(%s, loc = %s)\n", id.toChars(), loc.toChars());
        this.loc = loc;
        this.type = type;
        this.ident = ident;
        this.handler = handler;
    }

    Catch syntaxCopy()
    {
        auto c = new Catch(loc, type ? type.syntaxCopy() : getThrowable(), ident, (handler ? handler.syntaxCopy() : null));
        c.internalCatch = internalCatch;
        return c;
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#try-statement
 */
extern (C++) final class TryFinallyStatement : Statement
{
    Statement _body;
    Statement finalbody;

    Statement tryBody;   /// set to enclosing TryCatchStatement or TryFinallyStatement if in _body portion
    bool bodyFallsThru;  /// true if _body falls through to finally

    extern (D) this(const ref Loc loc, Statement _body, Statement finalbody)
    {
        super(loc, STMT.TryFinally);
        this._body = _body;
        this.finalbody = finalbody;
        this.bodyFallsThru = true;      // assume true until statementSemantic()
    }

    static TryFinallyStatement create(Loc loc, Statement _body, Statement finalbody)
    {
        return new TryFinallyStatement(loc, _body, finalbody);
    }

    override Statement syntaxCopy()
    {
        return new TryFinallyStatement(loc, _body.syntaxCopy(), finalbody.syntaxCopy());
    }

    override bool hasBreak() const pure nothrow
    {
        return false; //true;
    }

    override bool hasContinue() const pure nothrow
    {
        return false; //true;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/statement.html#scope-guard-statement
 */
extern (C++) final class ScopeGuardStatement : Statement
{
    TOK tok;
    Statement statement;

    extern (D) this(const ref Loc loc, TOK tok, Statement statement)
    {
        super(loc, STMT.ScopeGuard);
        this.tok = tok;
        this.statement = statement;
    }

    override Statement syntaxCopy()
    {
        return new ScopeGuardStatement(loc, tok, statement.syntaxCopy());
    }

    override Statement scopeCode(Scope* sc, Statement* sentry, Statement* sexception, Statement* sfinally)
    {
        //printf("ScopeGuardStatement::scopeCode()\n");
        *sentry = null;
        *sexception = null;
        *sfinally = null;

        Statement s = new PeelStatement(statement);

        switch (tok)
        {
        case TOK.onScopeExit:
            *sfinally = s;
            break;

        case TOK.onScopeFailure:
            *sexception = s;
            break;

        case TOK.onScopeSuccess:
            {
                /* Create:
                 *  sentry:   bool x = false;
                 *  sexception:    x = true;
                 *  sfinally: if (!x) statement;
                 */
                auto v = copyToTemp(0, "__os", IntegerExp.createBool(false));
                v.dsymbolSemantic(sc);
                *sentry = new ExpStatement(loc, v);

                Expression e = IntegerExp.createBool(true);
                e = new AssignExp(Loc.initial, new VarExp(Loc.initial, v), e);
                *sexception = new ExpStatement(Loc.initial, e);

                e = new VarExp(Loc.initial, v);
                e = new NotExp(Loc.initial, e);
                *sfinally = new IfStatement(Loc.initial, null, e, s, null, Loc.initial);

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
 * https://dlang.org/spec/statement.html#throw-statement
 */
extern (C++) final class ThrowStatement : Statement
{
    Expression exp;

    // was generated by the compiler, wasn't present in source code
    bool internalThrow;

    extern (D) this(const ref Loc loc, Expression exp)
    {
        super(loc, STMT.Throw);
        this.exp = exp;
    }

    override Statement syntaxCopy()
    {
        auto s = new ThrowStatement(loc, exp.syntaxCopy());
        s.internalThrow = internalThrow;
        return s;
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
    Statement statement;

    extern (D) this(const ref Loc loc, Statement statement)
    {
        super(loc, STMT.Debug);
        this.statement = statement;
    }

    override Statement syntaxCopy()
    {
        return new DebugStatement(loc, statement ? statement.syntaxCopy() : null);
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
 * https://dlang.org/spec/statement.html#goto-statement
 */
extern (C++) final class GotoStatement : Statement
{
    Identifier ident;
    LabelDsymbol label;
    Statement tryBody;              /// set to TryCatchStatement or TryFinallyStatement if in _body portion
    TryFinallyStatement tf;
    ScopeGuardStatement os;
    VarDeclaration lastVar;

    extern (D) this(const ref Loc loc, Identifier ident)
    {
        super(loc, STMT.Goto);
        this.ident = ident;
    }

    override Statement syntaxCopy()
    {
        return new GotoStatement(loc, ident);
    }

    extern (D) bool checkLabel()
    {
        if (!label.statement)
            return true;        // error should have been issued for this already

        if (label.statement.os != os)
        {
            if (os && os.tok == TOK.onScopeFailure && !label.statement.os)
            {
                // Jump out from scope(failure) block is allowed.
            }
            else
            {
                if (label.statement.os)
                    error("cannot `goto` in to `%s` block", Token.toChars(label.statement.os.tok));
                else
                    error("cannot `goto` out of `%s` block", Token.toChars(os.tok));
                return true;
            }
        }

        if (label.statement.tf != tf)
        {
            error("cannot `goto` in or out of `finally` block");
            return true;
        }

        Statement stbnext;
        for (auto stb = tryBody; stb != label.statement.tryBody; stb = stbnext)
        {
            if (!stb)
            {
                error("cannot `goto` into `try` block");
                return true;
            }
            if (auto stf = stb.isTryFinallyStatement())
                stbnext = stf.tryBody;
            else if (auto stc = stb.isTryCatchStatement())
                stbnext = stc.tryBody;
            else
                assert(0);
        }

        VarDeclaration vd = label.statement.lastVar;
        if (!vd || vd.isDataseg() || (vd.storage_class & STC.manifest))
            return false;

        VarDeclaration last = lastVar;
        while (last && last != vd)
            last = last.lastVar;
        if (last == vd)
        {
            // All good, the label's scope has no variables
        }
        else if (vd.storage_class & STC.exptemp)
        {
            // Lifetime ends at end of expression, so no issue with skipping the statement
        }
        else if (vd.ident == Id.withSym)
        {
            error("`goto` skips declaration of `with` temporary at %s", vd.loc.toChars());
            return true;
        }
        else
        {
            error("`goto` skips declaration of variable `%s` at %s", vd.toPrettyChars(), vd.loc.toChars());
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
 * https://dlang.org/spec/statement.html#LabeledStatement
 */
extern (C++) final class LabelStatement : Statement
{
    Identifier ident;
    Statement statement;

    Statement tryBody;              /// set to TryCatchStatement or TryFinallyStatement if in _body portion
    TryFinallyStatement tf;
    ScopeGuardStatement os;
    VarDeclaration lastVar;
    Statement gotoTarget;       // interpret
    void* extra;                // used by Statement_toIR()
    bool breaks;                // someone did a 'break ident'

    extern (D) this(const ref Loc loc, Identifier ident, Statement statement)
    {
        super(loc, STMT.Label);
        this.ident = ident;
        this.statement = statement;
    }

    override Statement syntaxCopy()
    {
        return new LabelStatement(loc, ident, statement ? statement.syntaxCopy() : null);
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

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class LabelDsymbol : Dsymbol
{
    LabelStatement statement;

    bool deleted;           // set if rewritten to return in foreach delegate
    bool iasm;              // set if used by inline assembler

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
 * https://dlang.org/spec/statement.html#asm
 */
extern (C++) class AsmStatement : Statement
{
    Token* tokens;

    extern (D) this(const ref Loc loc, Token* tokens)
    {
        super(loc, STMT.Asm);
        this.tokens = tokens;
    }

    extern (D) this(const ref Loc loc, Token* tokens, STMT stmt)
    {
        super(loc, stmt);
        this.tokens = tokens;
    }

    override Statement syntaxCopy()
    {
        return new AsmStatement(loc, tokens);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://dlang.org/spec/iasm.html
 */
extern (C++) final class InlineAsmStatement : AsmStatement
{
    code* asmcode;
    uint asmalign;  // alignment of this statement
    uint regs;      // mask of registers modified (must match regm_t in back end)
    bool refparam;  // true if function parameter is referenced
    bool naked;     // true if function is to be naked

version (IN_LLVM)
{
    // non-zero if this is a branch, contains the target label
    LabelDsymbol isBranchToLabel;
}

    extern (D) this(const ref Loc loc, Token* tokens)
    {
        super(loc, tokens, STMT.InlineAsm);
    }

    override Statement syntaxCopy()
    {
version (IN_LLVM)
{
        auto a_s = new InlineAsmStatement(loc, tokens);
        a_s.refparam = refparam;
        a_s.naked = naked;
        return a_s;
}
else
{
        return new InlineAsmStatement(loc, tokens);
}
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html
 * Assembler instructions with D expression operands.
 */
extern (C++) final class GccAsmStatement : AsmStatement
{
    StorageClass stc;           // attributes of the asm {} block
    Expression insn;            // string expression that is the template for assembler code
    Expressions* args;          // input and output operands of the statement
    uint outputargs;            // of the operands in 'args', the number of output operands
    Identifiers* names;         // list of symbolic names for the operands
    Expressions* constraints;   // list of string constants specifying constraints on operands
    Expressions* clobbers;      // list of string constants specifying clobbers and scratch registers
    Identifiers* labels;        // list of goto labels
    GotoStatements* gotos;      // of the goto labels, the equivalent statements they represent

    extern (D) this(const ref Loc loc, Token* tokens)
    {
        super(loc, tokens, STMT.GccAsm);
    }

    override Statement syntaxCopy()
    {
        return new GccAsmStatement(loc, tokens);
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
    StorageClass stc; // postfix attributes like nothrow/pure/@trusted

version (IN_LLVM)
{
    void* abiret; // llvm::Value*
}

    extern (D) this(const ref Loc loc, Statements* statements, StorageClass stc)
    {
        super(loc, statements, STMT.CompoundAsm);
        this.stc = stc;
    }

    override CompoundAsmStatement syntaxCopy()
    {
        auto a = new Statements(statements.dim);
        foreach (i, s; *statements)
        {
            (*a)[i] = s ? s.syntaxCopy() : null;
        }
        return new CompoundAsmStatement(loc, a, stc);
    }

    override Statements* flatten(Scope* sc)
    {
        return null;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }

version (IN_LLVM)
{
    override final inout(CompoundAsmStatement) endsWithAsm() inout pure nothrow @nogc
    {
        // yes this is inline asm
        return this;
    }
}
}

/***********************************************************
 * https://dlang.org/spec/module.html#ImportDeclaration
 */
extern (C++) final class ImportStatement : Statement
{
    Dsymbols* imports;      // Array of Import's

    extern (D) this(const ref Loc loc, Dsymbols* imports)
    {
        super(loc, STMT.Import);
        this.imports = imports;
    }

    override Statement syntaxCopy()
    {
        auto m = new Dsymbols(imports.dim);
        foreach (i, s; *imports)
        {
            (*m)[i] = s.syntaxCopy(null);
        }
        return new ImportStatement(loc, m);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}
