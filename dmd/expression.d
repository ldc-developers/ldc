/**
 * Defines the bulk of the classes which represent the AST at the expression level.
 *
 * Specification: ($LINK2 https://dlang.org/spec/expression.html, Expressions)
 *
 * Copyright:   Copyright (C) 1999-2020 by The D Language Foundation, All Rights Reserved
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
import dmd.ast_node;
import dmd.gluelayer;
import dmd.canthrow;
import dmd.complex;
import dmd.constfold;
import dmd.ctfeexpr;
import dmd.ctorflow;
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
import dmd.objc;
import dmd.opover;
import dmd.optimize;
import dmd.root.ctfloat;
import dmd.root.filename;
import dmd.root.outbuffer;
import dmd.root.rmem;
import dmd.root.rootobject;
import dmd.root.string;
import dmd.safe;
import dmd.sideeffect;
import dmd.target;
import dmd.tokens;
import dmd.typesem;
import dmd.utf;
import dmd.visitor;

version (IN_LLVM) import gen.dpragma;

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

// Return value for `checkModifiable`
enum Modifiable
{
    /// Not modifiable
    no,
    /// Modifiable (the type is mutable)
    yes,
    /// Modifiable because it is initialization
    initialization,
}

/****************************************
 * Find the first non-comma expression.
 * Params:
 *      e = Expressions connected by commas
 * Returns:
 *      left-most non-comma expression
 */
inout(Expression) firstComma(inout Expression e)
{
    Expression ex = cast()e;
    while (ex.op == TOK.comma)
        ex = (cast(CommaExp)ex).e1;
    return cast(inout)ex;

}

/****************************************
 * Find the last non-comma expression.
 * Params:
 *      e = Expressions connected by commas
 * Returns:
 *      right-most non-comma expression
 */

inout(Expression) lastComma(inout Expression e)
{
    Expression ex = cast()e;
    while (ex.op == TOK.comma)
        ex = (cast(CommaExp)ex).e2;
    return cast(inout)ex;

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
FuncDeclaration hasThis(Scope* sc)
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
            return null;
        }
        if (!fd.isNested() || fd.isThis() || (fd.isThis2 && fd.isMember2()))
            break;

        Dsymbol parent = fd.parent;
        while (1)
        {
            if (!parent)
                return null;
            TemplateInstance ti = parent.isTemplateInstance();
            if (ti)
                parent = ti.parent;
            else
                break;
        }
        fd = parent.isFuncDeclaration();
    }

    if (!fd.isThis() && !(fd.isThis2 && fd.isMember2()))
    {
        return null;
    }

    assert(fd.vthis);
    return fd;

}

/***********************************
 * Determine if a `this` is needed to access `d`.
 * Params:
 *      sc = context
 *      d = declaration to check
 * Returns:
 *      true means a `this` is needed
 */
bool isNeedThisScope(Scope* sc, Declaration d)
{
    if (sc.intypeof == 1)
        return false;

    AggregateDeclaration ad = d.isThis();
    if (!ad)
        return false;
    //printf("d = %s, ad = %s\n", d.toChars(), ad.toChars());

    for (Dsymbol s = sc.parent; s; s = s.toParentLocal())
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
            if (f.isMemberLocal())
                break;
        }
    }
    return true;
}

/******************************
 * check e is exp.opDispatch!(tiargs) or not
 * It's used to switch to UFCS the semantic analysis path
 */
bool isDotOpDispatch(Expression e)
{
    if (auto dtie = e.isDotTemplateInstanceExp())
        return dtie.ti.name == Id.opDispatch;
    return false;
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
    if (exps is null)
        return;

    for (size_t i = 0; i < exps.dim; i++)
    {
        Expression arg = (*exps)[i];
        if (!arg)
            continue;

        // Look for tuple with 0 members
        if (auto e = arg.isTypeExp())
        {
            if (auto tt = e.type.toBasetype().isTypeTuple())
            {
                if (!tt.arguments || tt.arguments.dim == 0)
                {
                    exps.remove(i);
                    if (i == exps.dim)
                        return;
                }
                else // Expand a TypeTuple
                {
                    exps.remove(i);
                    auto texps = new Expressions(tt.arguments.length);
                    foreach (j, a; *tt.arguments)
                        (*texps)[j] = new TypeExp(e.loc, a.type);
                    exps.insert(i, texps);
                }
                i--;
                continue;
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

/****************************************
 * Expand alias this tuples.
 */
TupleDeclaration isAliasThisTuple(Expression e)
{
    if (!e.type)
        return null;

    Type t = e.type.toBasetype();
    while (true)
    {
        if (Dsymbol s = t.toDsymbol(null))
        {
            if (auto ad = s.isAggregateDeclaration())
            {
                s = ad.aliasthis ? ad.aliasthis.sym : null;
                if (s && s.isVarDeclaration())
                {
                    TupleDeclaration td = s.isVarDeclaration().toAlias().isTupleDeclaration();
                    if (td && td.isexp)
                        return td;
                }
                if (Type att = t.aliasthisOf())
                {
                    t = att;
                    continue;
                }
            }
        }
        return null;
    }
}

int expandAliasThisTuples(Expressions* exps, size_t starti = 0)
{
    if (!exps || exps.dim == 0)
        return -1;

    for (size_t u = starti; u < exps.dim; u++)
    {
        Expression exp = (*exps)[u];
        if (TupleDeclaration td = exp.isAliasThisTuple)
        {
            exps.remove(u);
            foreach (i, o; *td.objects)
            {
                auto d = o.isExpression().isDsymbolExp().s.isDeclaration();
                auto e = new DotVarExp(exp.loc, exp, d);
                assert(d.type);
                e.type = d.type;
                exps.insert(u + i, e);
            }
            version (none)
            {
                printf("expansion ->\n");
                foreach (e; exps)
                {
                    printf("\texps[%d] e = %s %s\n", i, Token.tochars[e.op], e.toChars());
                }
            }
            return cast(int)u;
        }
    }
    return -1;
}

/****************************************
 * If `s` is a function template, i.e. the only member of a template
 * and that member is a function, return that template.
 * Params:
 *      s = symbol that might be a function template
 * Returns:
 *      template for that function, otherwise null
 */
TemplateDeclaration getFuncTemplateDecl(Dsymbol s)
{
    FuncDeclaration f = s.isFuncDeclaration();
    if (f && f.parent)
    {
        if (auto ti = f.parent.isTemplateInstance())
        {
            if (!ti.isTemplateMixin() && ti.tempdecl)
            {
                auto td = ti.tempdecl.isTemplateDeclaration();
                if (td.onemember && td.ident == f.ident)
                {
                    return td;
                }
            }
        }
    }
    return null;
}

/************************************************
 * If we want the value of this expression, but do not want to call
 * the destructor on it.
 */
Expression valueNoDtor(Expression e)
{
    auto ex = lastComma(e);

    if (auto ce = ex.isCallExp())
    {
        /* The struct value returned from the function is transferred
         * so do not call the destructor on it.
         * Recognize:
         *       ((S _ctmp = S.init), _ctmp).this(...)
         * and make sure the destructor is not called on _ctmp
         * BUG: if ex is a CommaExp, we should go down the right side.
         */
        if (auto dve = ce.e1.isDotVarExp())
        {
            if (dve.var.isCtorDeclaration())
            {
                // It's a constructor call
                if (auto comma = dve.e1.isCommaExp())
                {
                    if (auto ve = comma.e2.isVarExp())
                    {
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
    else if (auto ve = ex.isVarExp())
    {
        auto vtmp = ve.var.isVarDeclaration();
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
 *      sc = just used to specify the scope of created temporary variable
 *      destinationType = the type of the object on which the copy constructor is called;
 *                        may be null if the struct defines a postblit
 */
private Expression callCpCtor(Scope* sc, Expression e, Type destinationType)
{
    if (auto ts = e.type.baseElemOf().isTypeStruct())
    {
        StructDeclaration sd = ts.sym;
        if (sd.postblit || sd.hasCopyCtor)
        {
            /* Create a variable tmp, and replace the argument e with:
             *      (tmp = e),tmp
             * and let AssignExp() handle the construction.
             * This is not the most efficient, ideally tmp would be constructed
             * directly onto the stack.
             */
            auto tmp = copyToTemp(STC.rvalue, "__copytmp", e);
            if (sd.hasCopyCtor && destinationType)
                tmp.type = destinationType;
            tmp.storage_class |= STC.nodtor;
            tmp.dsymbolSemantic(sc);
            Expression de = new DeclarationExp(e.loc, tmp);
            Expression ve = new VarExp(e.loc, tmp);
            de.type = Type.tvoid;
            ve.type = e.type;
            return Expression.combine(de, ve);
        }
    }
    return e;
}

/************************************************
 * Handle the postblit call on lvalue, or the move of rvalue.
 *
 * Params:
 *   sc = the scope where the expression is encountered
 *   e = the expression the needs to be moved or copied (source)
 *   t = if the struct defines a copy constructor, the type of the destination
 *
 * Returns:
 *  The expression that copy constructs or moves the value.
 */
extern (D) Expression doCopyOrMove(Scope *sc, Expression e, Type t = null)
{
    if (auto ce = e.isCondExp())
    {
        ce.e1 = doCopyOrMove(sc, ce.e1);
        ce.e2 = doCopyOrMove(sc, ce.e2);
    }
    else
    {
        e = e.isLvalue() ? callCpCtor(sc, e, t) : valueNoDtor(e);
    }
    return e;
}

/****************************************************************/
/* A type meant as a union of all the Expression types,
 * to serve essentially as a Variant that will sit on the stack
 * during CTFE to reduce memory consumption.
 */
extern (C++) struct UnionExp
{
    // yes, default constructor does nothing
    extern (D) this(Expression e)
    {
        memcpy(&this, cast(void*)e, e.size);
    }

    /* Extract pointer to Expression
     */
    extern (C++) Expression exp() return
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
        switch (e.op)
        {
            case TOK.cantExpression:    return CTFEExp.cantexp;
            case TOK.voidExpression:    return CTFEExp.voidexp;
            case TOK.break_:            return CTFEExp.breakexp;
            case TOK.continue_:         return CTFEExp.continueexp;
            case TOK.goto_:             return CTFEExp.gotoexp;
            default:                    return e.copy();
        }
    }

private:
    // Ensure that the union is suitably aligned.
    align(8) union __AnonStruct__u
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
        char[__traits(classInstanceSize, VectorExp)] vectorexp;
    }

    __AnonStruct__u u;
}

/********************************
 * Test to see if two reals are the same.
 * Regard NaN's as equivalent.
 * Regard +0 and -0 as different.
 * Params:
 *      x1 = first operand
 *      x2 = second operand
 * Returns:
 *      true if x1 is x2
 *      else false
 */
bool RealIdentical(real_t x1, real_t x2)
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
DotIdExp typeDotIdExp(const ref Loc loc, Type type, Identifier ident)
{
    return new DotIdExp(loc, new TypeExp(loc, type), ident);
}

/***************************************************
 * Given an Expression, find the variable it really is.
 *
 * For example, `a[index]` is really `a`, and `s.f` is really `s`.
 * Params:
 *      e = Expression to look at
 * Returns:
 *      variable if there is one, null if not
 */
VarDeclaration expToVariable(Expression e)
{
    while (1)
    {
        switch (e.op)
        {
            case TOK.variable:
                return (cast(VarExp)e).var.isVarDeclaration();

            case TOK.dotVariable:
                e = (cast(DotVarExp)e).e1;
                continue;

            case TOK.index:
            {
                IndexExp ei = cast(IndexExp)e;
                e = ei.e1;
                Type ti = e.type.toBasetype();
                if (ti.ty == Tsarray)
                    continue;
                return null;
            }

            case TOK.slice:
            {
                SliceExp ei = cast(SliceExp)e;
                e = ei.e1;
                Type ti = e.type.toBasetype();
                if (ti.ty == Tsarray)
                    continue;
                return null;
            }

            case TOK.this_:
            case TOK.super_:
                return (cast(ThisExp)e).var.isVarDeclaration();

            default:
                return null;
        }
    }
}

enum OwnedBy : ubyte
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
extern (C++) /* IN_LLVM abstract */ class Expression : ASTNode
{
    const TOK op;   // to minimize use of dynamic_cast
    ubyte size;     // # of bytes in Expression so we can copy() it
    ubyte parens;   // if this is a parenthesized expression
    Type type;      // !=null means that semantic() has been run
    Loc loc;        // file location

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
        CTFEExp.showcontext = new CTFEExp(TOK.showCtfeContext);
    }

    /**
     * Deinitializes the global state of the compiler.
     *
     * This can be used to restore the state set by `_init` to its original
     * state.
     */
    static void deinitialize()
    {
        CTFEExp.cantexp = CTFEExp.cantexp.init;
        CTFEExp.voidexp = CTFEExp.voidexp.init;
        CTFEExp.breakexp = CTFEExp.breakexp.init;
        CTFEExp.continueexp = CTFEExp.continueexp.init;
        CTFEExp.gotoexp = CTFEExp.gotoexp.init;
        CTFEExp.showcontext = CTFEExp.showcontext.init;
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
            }
            assert(0);
        }

        // memory never freed, so can use the faster bump-pointer-allocation
        e = cast(Expression)_d_allocmemory(size);
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

    override const(char)* toChars() const
    {
        OutBuffer buf;
        HdrGenState hgs;
        toCBuffer(this, &buf, &hgs);
        return buf.extractChars();
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

    final void errorSupplemental(const(char)* format, ...)
    {
        if (type == Type.terror)
            return;

        va_list ap;
        va_start(ap, format);
        .verrorSupplemental(loc, format, ap);
        va_end(ap);
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
    extern (D) static Expression combine(Expression e1, Expression e2)
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

    extern (D) static Expression combine(Expression e1, Expression e2, Expression e3)
    {
        return combine(combine(e1, e2), e3);
    }

    extern (D) static Expression combine(Expression e1, Expression e2, Expression e3, Expression e4)
    {
        return combine(combine(e1, e2), combine(e3, e4));
    }

    /**********************************
     * If 'e' is a tree of commas, returns the rightmost expression
     * by stripping off it from the tree. The remained part of the tree
     * is returned via e0.
     * Otherwise 'e' is directly returned and e0 is set to NULL.
     */
    extern (D) static Expression extractLast(Expression e, out Expression e0)
    {
        if (e.op != TOK.comma)
        {
            return e;
        }

        CommaExp ce = cast(CommaExp)e;
        if (ce.e2.op != TOK.comma)
        {
            e0 = ce.e1;
            return ce.e2;
        }
        else
        {
            e0 = e;

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

    extern (D) static Expressions* arraySyntaxCopy(Expressions* exps)
    {
        Expressions* a = null;
        if (exps)
        {
            a = new Expressions(exps.dim);
            foreach (i, e; *exps)
            {
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

    TupleExp toTupleExp()
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
            error("`%s` is a `%s` definition and cannot be modified", e.type.toChars(), e.type.kind());
        else
            error("`%s` is not an lvalue and cannot be modified", e.toChars());

        return new ErrorExp();
    }

    Expression modifiableLvalue(Scope* sc, Expression e)
    {
        //printf("Expression::modifiableLvalue() %s, type = %s\n", toChars(), type.toChars());
        // See if this expression is a modifiable lvalue (i.e. not const)
        if (checkModifiable(sc) == Modifiable.yes)
        {
            assert(type);
            if (!type.isMutable())
            {
                if (auto dve = this.isDotVarExp())
                {
                    if (isNeedThisScope(sc, dve.var))
                        for (Dsymbol s = sc.func; s; s = s.toParentLocal())
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
                error("cannot modify struct instance `%s` of type `%s` because it contains `const` or `immutable` members",
                    toChars(), type.toChars());
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

    extern (D) final bool checkScalar()
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

    extern (D) final bool checkNoBool()
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

    extern (D) final bool checkIntegral()
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

    extern (D) final bool checkArithmetic()
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

    extern (D) final bool checkDisabled(Scope* sc, Dsymbol s)
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
    extern (D) final bool checkPurity(Scope* sc, FuncDeclaration f)
    {
        if (!sc.func)
            return false;
        if (sc.func == f)
            return false;
        if (sc.intypeof == 1)
            return false;
        if (sc.flags & (SCOPE.ctfe | SCOPE.debug_))
            return false;

        // If the call has a pure parent, then the called func must be pure.
        if (!f.isPure() && checkImpure(sc))
        {
            error("`pure` %s `%s` cannot call impure %s `%s`",
                sc.func.kind(), sc.func.toPrettyChars(), f.kind(),
                f.toPrettyChars());
            return true;
        }
        return false;
    }

    /*******************************************
     * Accessing variable v.
     * Check for purity and safety violations.
     * Returns true if error occurs.
     */
    extern (D) final bool checkPurity(Scope* sc, VarDeclaration v)
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

        if (v.type.ty == Tstruct)
        {
            StructDeclaration sd = (cast(TypeStruct)v.type).sym;
            if (sd.hasNoFields)
                return false;
        }

        bool err = false;
        if (v.isDataseg())
        {
            // https://issues.dlang.org/show_bug.cgi?id=7533
            // Accessing implicit generated __gate is pure.
            if (v.ident == Id.gate)
                return false;

            if (checkImpure(sc))
            {
                error("`pure` %s `%s` cannot access mutable static data `%s`",
                    sc.func.kind(), sc.func.toPrettyChars(), v.toChars());
                err = true;
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
            for (Dsymbol s = sc.func; !err && s; s = s.toParentP(vparent))
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
                            ffbuf.peekChars(), ff.kind(), ff.toPrettyChars(), vbuf.peekChars(), v.toChars());
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

    /*
    Check if sc.func is impure or can be made impure.
    Returns true on error, i.e. if sc.func is pure and cannot be made impure.
    */
    private static bool checkImpure(Scope* sc)
    {
        return sc.func && (sc.flags & SCOPE.compile
                ? sc.func.isPureBypassingInference() >= PURE.weak
                : sc.func.setImpure());
    }

    /*********************************************
     * Calling function f.
     * Check the safety, i.e. if we're in a @safe function
     * we can only call @safe or @trusted functions.
     * Returns true if error occurs.
     */
    extern (D) final bool checkSafety(Scope* sc, FuncDeclaration f)
    {
        if (!sc.func)
            return false;
        if (sc.func == f)
            return false;
        if (sc.intypeof == 1)
            return false;
        if (sc.flags & (SCOPE.ctfe | SCOPE.debug_))
            return false;

        if (!f.isSafe() && !f.isTrusted())
        {
            if (sc.flags & SCOPE.compile ? sc.func.isSafeBypassingInference() : sc.func.setUnsafe())
            {
                if (!loc.isValid()) // e.g. implicitly generated dtor
                    loc = sc.func.loc;

                const prettyChars = f.toPrettyChars();
                error("`@safe` %s `%s` cannot call `@system` %s `%s`",
                    sc.func.kind(), sc.func.toPrettyChars(), f.kind(),
                    prettyChars);
                .errorSupplemental(f.loc, "`%s` is declared here", prettyChars);
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
    extern (D) final bool checkNogc(Scope* sc, FuncDeclaration f)
    {
        if (!sc.func)
            return false;
        if (sc.func == f)
            return false;
        if (sc.intypeof == 1)
            return false;
        if (sc.flags & (SCOPE.ctfe | SCOPE.debug_))
            return false;

        if (!f.isNogc())
        {
            if (sc.flags & SCOPE.compile ? sc.func.isNogcBypassingInference() : sc.func.setGC())
            {
                if (loc.linnum == 0) // e.g. implicitly generated dtor
                    loc = sc.func.loc;

                // Lowered non-@nogc'd hooks will print their own error message inside of nogc.d (NOGCVisitor.visit(CallExp e)),
                // so don't print anything to avoid double error messages.
                if (!(f.ident == Id._d_HookTraceImpl || f.ident == Id._d_arraysetlengthT))
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
    extern (D) final bool checkPostblit(Scope* sc, Type t)
    {
        if (auto ts = t.baseElemOf().isTypeStruct())
        {
            if (global.params.useTypeInfo)
            {
                // https://issues.dlang.org/show_bug.cgi?id=11395
                // Require TypeInfo generation for array concatenation
                semanticTypeInfo(sc, t);
            }

            StructDeclaration sd = ts.sym;
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

    extern (D) final bool checkRightThis(Scope* sc)
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
    extern (D) final bool checkReadModifyWrite(TOK rmwOp, Expression ex = null)
    {
        //printf("Expression::checkReadModifyWrite() %s %s", toChars(), ex ? ex.toChars() : "");
        if (!type || !type.isShared() || type.isTypeStruct() || type.isTypeClass())
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

        error("read-modify-write operations are not allowed for `shared` variables. Use `core.atomic.atomicOp!\"%s\"(%s, %s)` instead.", Token.toChars(rmwOp), toChars(), ex ? ex.toChars() : "1");

         return true;
    }

    /***************************************
     * Parameters:
     *      sc:     scope
     *      flag:   1: do not issue error message for invalid modification
     * Returns:
     *      Whether the type is modifiable
     */
    Modifiable checkModifiable(Scope* sc, int flag = 0)
    {
        return type ? Modifiable.yes : Modifiable.no; // default modifiable
    }

    /*****************************
     * If expression can be tested for true or false,
     * returns the modified expression.
     * Otherwise returns ErrorExp.
     */
    Expression toBoolean(Scope* sc)
    {
        // Default is 'yes' - do nothing
        Expression e = this;
        Type t = type;
        Type tb = type.toBasetype();
        Type att = null;

        while (1)
        {
            // Structs can be converted to bool using opCast(bool)()
            if (auto ts = tb.isTypeStruct())
            {
                AggregateDeclaration ad = ts.sym;
                /* Don't really need to check for opCast first, but by doing so we
                 * get better error messages if it isn't there.
                 */
                if (Dsymbol fd = search_function(ad, Id._cast))
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
                    continue;
                }
            }
            break;
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
        Expression e = new AddrExp(loc, this, type.pointerTo());
        return e;
    }

    /******************************
     * If this is a reference, dereference it.
     */
    final Expression deref()
    {
        //printf("Expression::deref()\n");
        // type could be null if forward referencing an 'auto' variable
        if (type)
            if (auto tr = type.isTypeReference())
            {
                Expression e = new PtrExp(loc, this, tr.next);
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

    bool hasCode()
    {
        return true;
    }

    final pure inout nothrow @nogc
    {
        inout(IntegerExp)   isIntegerExp() { return op == TOK.int64 ? cast(typeof(return))this : null; }
        inout(ErrorExp)     isErrorExp() { return op == TOK.error ? cast(typeof(return))this : null; }
        inout(VoidInitExp)  isVoidInitExp() { return op == TOK.void_ ? cast(typeof(return))this : null; }
        inout(RealExp)      isRealExp() { return op == TOK.float64 ? cast(typeof(return))this : null; }
        inout(ComplexExp)   isComplexExp() { return op == TOK.complex80 ? cast(typeof(return))this : null; }
        inout(IdentifierExp) isIdentifierExp() { return op == TOK.identifier ? cast(typeof(return))this : null; }
        inout(DollarExp)    isDollarExp() { return op == TOK.dollar ? cast(typeof(return))this : null; }
        inout(DsymbolExp)   isDsymbolExp() { return op == TOK.dSymbol ? cast(typeof(return))this : null; }
        inout(ThisExp)      isThisExp() { return op == TOK.this_ ? cast(typeof(return))this : null; }
        inout(SuperExp)     isSuperExp() { return op == TOK.super_ ? cast(typeof(return))this : null; }
        inout(NullExp)      isNullExp() { return op == TOK.null_ ? cast(typeof(return))this : null; }
        inout(StringExp)    isStringExp() { return op == TOK.string_ ? cast(typeof(return))this : null; }
        inout(TupleExp)     isTupleExp() { return op == TOK.tuple ? cast(typeof(return))this : null; }
        inout(ArrayLiteralExp) isArrayLiteralExp() { return op == TOK.arrayLiteral ? cast(typeof(return))this : null; }
        inout(AssocArrayLiteralExp) isAssocArrayLiteralExp() { return op == TOK.assocArrayLiteral ? cast(typeof(return))this : null; }
        inout(StructLiteralExp) isStructLiteralExp() { return op == TOK.structLiteral ? cast(typeof(return))this : null; }
        inout(TypeExp)      isTypeExp() { return op == TOK.type ? cast(typeof(return))this : null; }
        inout(ScopeExp)     isScopeExp() { return op == TOK.scope_ ? cast(typeof(return))this : null; }
        inout(TemplateExp)  isTemplateExp() { return op == TOK.template_ ? cast(typeof(return))this : null; }
        inout(NewExp) isNewExp() { return op == TOK.new_ ? cast(typeof(return))this : null; }
        inout(NewAnonClassExp) isNewAnonClassExp() { return op == TOK.newAnonymousClass ? cast(typeof(return))this : null; }
        inout(SymOffExp)    isSymOffExp() { return op == TOK.symbolOffset ? cast(typeof(return))this : null; }
        inout(VarExp)       isVarExp() { return op == TOK.variable ? cast(typeof(return))this : null; }
        inout(OverExp)      isOverExp() { return op == TOK.overloadSet ? cast(typeof(return))this : null; }
        inout(FuncExp)      isFuncExp() { return op == TOK.function_ ? cast(typeof(return))this : null; }
        inout(DeclarationExp) isDeclarationExp() { return op == TOK.declaration ? cast(typeof(return))this : null; }
        inout(TypeidExp)    isTypeidExp() { return op == TOK.typeid_ ? cast(typeof(return))this : null; }
        inout(TraitsExp)    isTraitsExp() { return op == TOK.traits ? cast(typeof(return))this : null; }
        inout(HaltExp)      isHaltExp() { return op == TOK.halt ? cast(typeof(return))this : null; }
        inout(IsExp)        isExp() { return op == TOK.is_ ? cast(typeof(return))this : null; }
        inout(CompileExp)   isCompileExp() { return op == TOK.mixin_ ? cast(typeof(return))this : null; }
        inout(ImportExp)    isImportExp() { return op == TOK.import_ ? cast(typeof(return))this : null; }
        inout(AssertExp)    isAssertExp() { return op == TOK.assert_ ? cast(typeof(return))this : null; }
        inout(DotIdExp)     isDotIdExp() { return op == TOK.dotIdentifier ? cast(typeof(return))this : null; }
        inout(DotTemplateExp) isDotTemplateExp() { return op == TOK.dotTemplateDeclaration ? cast(typeof(return))this : null; }
        inout(DotVarExp)    isDotVarExp() { return op == TOK.dotVariable ? cast(typeof(return))this : null; }
        inout(DotTemplateInstanceExp) isDotTemplateInstanceExp() { return op == TOK.dotTemplateInstance ? cast(typeof(return))this : null; }
        inout(DelegateExp)  isDelegateExp() { return op == TOK.delegate_ ? cast(typeof(return))this : null; }
        inout(DotTypeExp)   isDotTypeExp() { return op == TOK.dotType ? cast(typeof(return))this : null; }
        inout(CallExp)      isCallExp() { return op == TOK.call ? cast(typeof(return))this : null; }
        inout(AddrExp)      isAddrExp() { return op == TOK.address ? cast(typeof(return))this : null; }
        inout(PtrExp)       isPtrExp() { return op == TOK.star ? cast(typeof(return))this : null; }
        inout(NegExp)       isNegExp() { return op == TOK.negate ? cast(typeof(return))this : null; }
        inout(UAddExp)      isUAddExp() { return op == TOK.uadd ? cast(typeof(return))this : null; }
        inout(ComExp)       isComExp() { return op == TOK.tilde ? cast(typeof(return))this : null; }
        inout(NotExp)       isNotExp() { return op == TOK.not ? cast(typeof(return))this : null; }
        inout(DeleteExp)    isDeleteExp() { return op == TOK.delete_ ? cast(typeof(return))this : null; }
        inout(CastExp)      isCastExp() { return op == TOK.cast_ ? cast(typeof(return))this : null; }
        inout(VectorExp)    isVectorExp() { return op == TOK.vector ? cast(typeof(return))this : null; }
        inout(VectorArrayExp) isVectorArrayExp() { return op == TOK.vectorArray ? cast(typeof(return))this : null; }
        inout(SliceExp)     isSliceExp() { return op == TOK.slice ? cast(typeof(return))this : null; }
        inout(ArrayLengthExp) isArrayLengthExp() { return op == TOK.arrayLength ? cast(typeof(return))this : null; }
        inout(ArrayExp)     isArrayExp() { return op == TOK.array ? cast(typeof(return))this : null; }
        inout(DotExp)       isDotExp() { return op == TOK.dot ? cast(typeof(return))this : null; }
        inout(CommaExp)     isCommaExp() { return op == TOK.comma ? cast(typeof(return))this : null; }
        inout(IntervalExp)  isIntervalExp() { return op == TOK.interval ? cast(typeof(return))this : null; }
        inout(DelegatePtrExp)     isDelegatePtrExp() { return op == TOK.delegatePointer ? cast(typeof(return))this : null; }
        inout(DelegateFuncptrExp) isDelegateFuncptrExp() { return op == TOK.delegateFunctionPointer ? cast(typeof(return))this : null; }
        inout(IndexExp)     isIndexExp() { return op == TOK.index ? cast(typeof(return))this : null; }
        inout(PostExp)      isPostExp()  { return (op == TOK.plusPlus || op == TOK.minusMinus) ? cast(typeof(return))this : null; }
        inout(PreExp)       isPreExp()   { return (op == TOK.prePlusPlus || op == TOK.preMinusMinus) ? cast(typeof(return))this : null; }
        inout(AssignExp)    isAssignExp()    { return op == TOK.assign ? cast(typeof(return))this : null; }
        inout(ConstructExp) isConstructExp() { return op == TOK.construct ? cast(typeof(return))this : null; }
        inout(BlitExp)      isBlitExp()      { return op == TOK.blit ? cast(typeof(return))this : null; }
        inout(AddAssignExp) isAddAssignExp() { return op == TOK.addAssign ? cast(typeof(return))this : null; }
        inout(MinAssignExp) isMinAssignExp() { return op == TOK.minAssign ? cast(typeof(return))this : null; }
        inout(MulAssignExp) isMulAssignExp() { return op == TOK.mulAssign ? cast(typeof(return))this : null; }

        inout(DivAssignExp) isDivAssignExp() { return op == TOK.divAssign ? cast(typeof(return))this : null; }
        inout(ModAssignExp) isModAssignExp() { return op == TOK.modAssign ? cast(typeof(return))this : null; }
        inout(AndAssignExp) isAndAssignExp() { return op == TOK.andAssign ? cast(typeof(return))this : null; }
        inout(OrAssignExp)  isOrAssignExp()  { return op == TOK.orAssign ? cast(typeof(return))this : null; }
        inout(XorAssignExp) isXorAssignExp() { return op == TOK.xorAssign ? cast(typeof(return))this : null; }
        inout(PowAssignExp) isPowAssignExp() { return op == TOK.powAssign ? cast(typeof(return))this : null; }

        inout(ShlAssignExp)  isShlAssignExp()  { return op == TOK.leftShiftAssign ? cast(typeof(return))this : null; }
        inout(ShrAssignExp)  isShrAssignExp()  { return op == TOK.rightShiftAssign ? cast(typeof(return))this : null; }
        inout(UshrAssignExp) isUshrAssignExp() { return op == TOK.unsignedRightShiftAssign ? cast(typeof(return))this : null; }

        inout(CatAssignExp) isCatAssignExp() { return op == TOK.concatenateAssign
                                                ? cast(typeof(return))this
                                                : null; }

        inout(CatElemAssignExp) isCatElemAssignExp() { return op == TOK.concatenateElemAssign
                                                ? cast(typeof(return))this
                                                : null; }

        inout(CatDcharAssignExp) isCatDcharAssignExp() { return op == TOK.concatenateDcharAssign
                                                ? cast(typeof(return))this
                                                : null; }

        inout(AddExp)      isAddExp() { return op == TOK.add ? cast(typeof(return))this : null; }
        inout(MinExp)      isMinExp() { return op == TOK.min ? cast(typeof(return))this : null; }
        inout(CatExp)      isCatExp() { return op == TOK.concatenate ? cast(typeof(return))this : null; }
        inout(MulExp)      isMulExp() { return op == TOK.mul ? cast(typeof(return))this : null; }
        inout(DivExp)      isDivExp() { return op == TOK.div ? cast(typeof(return))this : null; }
        inout(ModExp)      isModExp() { return op == TOK.mod ? cast(typeof(return))this : null; }
        inout(PowExp)      isPowExp() { return op == TOK.pow ? cast(typeof(return))this : null; }
        inout(ShlExp)      isShlExp() { return op == TOK.leftShift ? cast(typeof(return))this : null; }
        inout(ShrExp)      isShrExp() { return op == TOK.rightShift ? cast(typeof(return))this : null; }
        inout(UshrExp)     isUshrExp() { return op == TOK.unsignedRightShift ? cast(typeof(return))this : null; }
        inout(AndExp)      isAndExp() { return op == TOK.and ? cast(typeof(return))this : null; }
        inout(OrExp)       isOrExp() { return op == TOK.or ? cast(typeof(return))this : null; }
        inout(XorExp)      isXorExp() { return op == TOK.xor ? cast(typeof(return))this : null; }
        inout(LogicalExp)  isLogicalExp() { return (op == TOK.andAnd || op == TOK.orOr) ? cast(typeof(return))this : null; }
        //inout(CmpExp)    isCmpExp() { return op == TOK. ? cast(typeof(return))this : null; }
        inout(InExp)       isInExp() { return op == TOK.in_ ? cast(typeof(return))this : null; }
        inout(RemoveExp)   isRemoveExp() { return op == TOK.remove ? cast(typeof(return))this : null; }
        inout(EqualExp)    isEqualExp() { return (op == TOK.equal || op == TOK.notEqual) ? cast(typeof(return))this : null; }
        inout(IdentityExp) isIdentityExp() { return (op == TOK.identity || op == TOK.notIdentity) ? cast(typeof(return))this : null; }
        inout(CondExp)     isCondExp() { return op == TOK.question ? cast(typeof(return))this : null; }

        inout(DefaultInitExp)    isDefaultInitExp() { return op == TOK.default_ ? cast(typeof(return))this : null; }
        inout(FileInitExp)       isFileInitExp() { return (op == TOK.file || op == TOK.fileFullPath) ? cast(typeof(return))this : null; }
        inout(LineInitExp)       isLineInitExp() { return op == TOK.line ? cast(typeof(return))this : null; }
        inout(ModuleInitExp)     isModuleInitExp() { return op == TOK.moduleString ? cast(typeof(return))this : null; }
        inout(FuncInitExp)       isFuncInitExp() { return op == TOK.functionString ? cast(typeof(return))this : null; }
        inout(PrettyFuncInitExp) isPrettyFuncInitExp() { return op == TOK.prettyFunction ? cast(typeof(return))this : null; }
        inout(ClassReferenceExp) isClassReferenceExp() { return op == TOK.classReference ? cast(typeof(return))this : null; }
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class IntegerExp : Expression
{
    private dinteger_t value;

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
        this.value = normalize(type.toBasetype().ty, value);
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

    // Same as create, but doesn't allocate memory.
    static void emplace(UnionExp* pue, Loc loc, dinteger_t value, Type type)
    {
        emplaceExp!(IntegerExp)(pue, loc, value, type);
    }

    override bool equals(const RootObject o) const
    {
        if (this == o)
            return true;
        if (auto ne = (cast(Expression)o).isIntegerExp())
        {
            if (type.toHeadMutable().equals(ne.type.toHeadMutable()) && value == ne.value)
            {
                return true;
            }
        }
        return false;
    }

    override dinteger_t toInteger()
    {
        // normalize() is necessary until we fix all the paints of 'type'
        return value = normalize(type.toBasetype().ty, value);
    }

    override real_t toReal()
    {
        // normalize() is necessary until we fix all the paints of 'type'
        const ty = type.toBasetype().ty;
        const val = normalize(ty, value);
        value = val;
        return (ty == Tuns64)
            ? real_t(cast(d_uns64)val)
            : real_t(cast(d_int64)val);
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
        e.error("cannot modify constant `%s`", e.toChars());
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
        this.value = normalize(type.toBasetype().ty, value);
    }

    extern (D) static dinteger_t normalize(TY ty, dinteger_t value)
    {
        /* 'Normalize' the value of the integer to be in range of the type
         */
        dinteger_t result;
        switch (ty)
        {
        case Tbool:
            result = (value != 0);
            break;

        case Tint8:
            result = cast(d_int8)value;
            break;

        case Tchar:
        case Tuns8:
            result = cast(d_uns8)value;
            break;

        case Tint16:
            result = cast(d_int16)value;
            break;

        case Twchar:
        case Tuns16:
            result = cast(d_uns16)value;
            break;

        case Tint32:
            result = cast(d_int32)value;
            break;

        case Tdchar:
        case Tuns32:
            result = cast(d_uns32)value;
            break;

        case Tint64:
            result = cast(d_int64)value;
            break;

        case Tuns64:
            result = cast(d_uns64)value;
            break;

        case Tpointer:
            if (target.ptrsize == 8)
                goto case Tuns64;
            if (target.ptrsize == 4)
                goto case Tuns32;
            if (target.ptrsize == 2)
                goto case Tuns16;
            assert(0);

        default:
            break;
        }
        return result;
    }

    override Expression syntaxCopy()
    {
        return this;
    }

    /**
     * Use this instead of creating new instances for commonly used literals
     * such as 0 or 1.
     *
     * Parameters:
     *      v = The value of the expression
     * Returns:
     *      A static instance of the expression, typed as `Tint32`.
     */
    static IntegerExp literal(int v)()
    {
        __gshared IntegerExp theConstant;
        if (!theConstant)
            theConstant = new IntegerExp(v);
        return theConstant;
    }

    /**
     * Use this instead of creating new instances for commonly used bools.
     *
     * Parameters:
     *      b = The value of the expression
     * Returns:
     *      A static instance of the expression, typed as `Type.tbool`.
     */
    static IntegerExp createBool(bool b)
    {
        __gshared IntegerExp trueExp, falseExp;
        if (!trueExp)
        {
            trueExp = new IntegerExp(Loc.initial, 1, Type.tbool);
            falseExp = new IntegerExp(Loc.initial, 0, Type.tbool);
        }
        return b ? trueExp : falseExp;
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
        if (global.errors == 0 && global.gaggedErrors == 0)
        {
             /* Unfortunately, errors can still leak out of gagged errors,
              * and we need to set the error count to prevent bogus code
              * generation. At least give a message.
              */
             error("unknown, please file report on issues.dlang.org");
        }

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

    extern (C++) __gshared ErrorExp errorexp; // handy shared value
}


/***********************************************************
 * An uninitialized value,
 * generated from void initializers.
 */
extern (C++) final class VoidInitExp : Expression
{
    VarDeclaration var; /// the variable from where the void value came from, null if not known
                        /// Useful for error messages

    extern (D) this(VarDeclaration var)
    {
        super(var.loc, TOK.void_, __traits(classInstanceSize, VoidInitExp));
        this.var = var;
        this.type = var.type;
    }

    override const(char)* toChars() const
    {
        return "void";
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
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

    // Same as create, but doesn't allocate memory.
    static void emplace(UnionExp* pue, Loc loc, real_t value, Type type)
    {
        emplaceExp!(RealExp)(pue, loc, value, type);
    }

    override bool equals(const RootObject o) const
    {
        if (this == o)
            return true;
        if (auto ne = (cast(Expression)o).isRealExp())
        {
            if (type.toHeadMutable().equals(ne.type.toHeadMutable()) && RealIdentical(value, ne.value))
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

    // Same as create, but doesn't allocate memory.
    static void emplace(UnionExp* pue, Loc loc, complex_t value, Type type)
    {
        emplaceExp!(ComplexExp)(pue, loc, value, type);
    }

    override bool equals(const RootObject o) const
    {
        if (this == o)
            return true;
        if (auto ne = (cast(Expression)o).isComplexExp())
        {
            if (type.toHeadMutable().equals(ne.type.toHeadMutable()) && RealIdentical(creall(value), creall(ne.value)) && RealIdentical(cimagl(value), cimagl(ne.value)))
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

    this(const ref Loc loc, const TOK tok)
    {
        super(loc, tok, __traits(classInstanceSize, ThisExp));
        //printf("ThisExp::ThisExp() loc = %d\n", loc.linnum);
    }

    override Expression syntaxCopy()
    {
        auto r = cast(ThisExp) super.syntaxCopy();
        // require new semantic (possibly new `var` etc.)
        r.type = null;
        r.var = null;
        return r;
    }

    override final bool isBool(bool result)
    {
        return result;
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
        super(loc, TOK.super_);
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

    override bool equals(const RootObject o) const
    {
        if (auto e = o.isExpression())
        {
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
            auto se = new StringExp(loc, (cast(char*)mem.xcalloc(1, 1))[0 .. 0]);
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
    private union
    {
        char* string;   // if sz == 1
        wchar* wstring; // if sz == 2
        dchar* dstring; // if sz == 4
    }                   // (const if ownedByCtfe == OwnedBy.code)
    size_t len;         // number of code units
    ubyte sz = 1;       // 1: char, 2: wchar, 4: dchar
    ubyte committed;    // !=0 if type is committed
    enum char NoPostfix = 0;
    char postfix = NoPostfix;   // 'c', 'w', 'd'
    OwnedBy ownedByCtfe = OwnedBy.code;

    extern (D) this(const ref Loc loc, const(void)[] string)
    {
        super(loc, TOK.string_, __traits(classInstanceSize, StringExp));
        this.string = cast(char*)string.ptr; // note that this.string should be const
        this.len = string.length;
        this.sz = 1;                    // work around LDC bug #1286
    }

    extern (D) this(const ref Loc loc, const(void)[] string, size_t len, ubyte sz, char postfix = NoPostfix)
    {
        super(loc, TOK.string_, __traits(classInstanceSize, StringExp));
        this.string = cast(char*)string.ptr; // note that this.string should be const
        this.len = len;
        this.sz = sz;
        this.postfix = postfix;
    }

    static StringExp create(Loc loc, char* s)
    {
        return new StringExp(loc, s.toDString());
    }

    static StringExp create(Loc loc, void* string, size_t len)
    {
        return new StringExp(loc, string[0 .. len]);
    }

    // Same as create, but doesn't allocate memory.
    static void emplace(UnionExp* pue, Loc loc, char* s)
    {
        emplaceExp!(StringExp)(pue, loc, s.toDString());
    }

    extern (D) static void emplace(UnionExp* pue, Loc loc, const(void)[] string)
    {
        emplaceExp!(StringExp)(pue, loc, string);
    }

    extern (D) static void emplace(UnionExp* pue, Loc loc, const(void)[] string, size_t len, ubyte sz, char postfix)
    {
        emplaceExp!(StringExp)(pue, loc, string, len, sz, postfix);
    }

    override bool equals(const RootObject o) const
    {
        //printf("StringExp::equals('%s') %s\n", o.toChars(), toChars());
        if (auto e = o.isExpression())
        {
            if (auto se = e.isStringExp())
            {
                return compare(se) == 0;
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
    size_t numberOfCodeUnits(int tynto = 0) const
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
                if (const s = utf_decodeChar(string[0 .. len], u, c))
                {
                    error("%.*s", cast(int)s.length, s.ptr);
                    return 0;
                }
                result += utf_codeLength(encSize, c);
            }
            break;

        case 2:
            for (size_t u = 0; u < len;)
            {
                if (const s = utf_decodeWchar(wstring[0 .. len], u, c))
                {
                    error("%.*s", cast(int)s.length, s.ptr);
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
    dchar getCodeUnit(size_t i) const pure
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
    void setCodeUnit(size_t i, dchar c)
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
            auto se = e.isStringExp();
            assert(se.sz == 1);
            return se;
        }
        return this;
    }

    /**
     * Compare two `StringExp` by length, then value
     *
     * The comparison is not the usual C-style comparison as seen with
     * `strcmp` or `memcmp`, but instead first compare based on the length.
     * This allows both faster lookup and sorting when comparing sparse data.
     *
     * This ordering scheme is relied on by the string-switching feature.
     * Code in Druntime's `core.internal.switch_` relies on this ordering
     * when doing a binary search among case statements.
     *
     * Both `StringExp` should be of the same encoding.
     *
     * Params:
     *   se2 = String expression to compare `this` to
     *
     * Returns:
     *   `0` when `this` is equal to se2, a value greater than `0` if
     *   `this` should be considered greater than `se2`,
     *   and a value less than `0` if `this` is lesser than `se2`.
     */
    int compare(const StringExp se2) const nothrow pure @nogc
    {
        //printf("StringExp::compare()\n");
        const len1 = len;
        const len2 = se2.len;

        assert(this.sz == se2.sz, "Comparing string expressions of different sizes");
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
                    foreach (u; 0 .. len)
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
                    foreach (u; 0 .. len)
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
        return result;
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
    extern (D) const(char)[] toStringz() const
    {
        auto nbytes = len * sz;
        char* s = cast(char*)mem.xmalloc(nbytes + sz);
        writeTo(s, true);
        return s[0 .. nbytes];
    }

    extern (D) const(char)[] peekString() const
    {
        assert(sz == 1);
        return this.string[0 .. len];
    }

    extern (D) const(wchar)[] peekWstring() const
    {
        assert(sz == 2);
        return this.wstring[0 .. len];
    }

    extern (D) const(dchar)[] peekDstring() const
    {
        assert(sz == 4);
        return this.dstring[0 .. len];
    }

    /*******************
     * Get a slice of the data.
     */
    extern (D) const(ubyte)[] peekData() const
    {
        return cast(const(ubyte)[])this.string[0 .. len * sz];
    }

    /*******************
     * Borrow a slice of the data, so the caller can modify
     * it in-place (!)
     */
    extern (D) ubyte[] borrowData()
    {
        return cast(ubyte[])this.string[0 .. len * sz];
    }

    /***********************
     * Set new string data.
     * `this` becomes the new owner of the data.
     */
    extern (D) void setData(void* s, size_t len, ubyte sz)
    {
        this.string = cast(char*)s;
        this.len = len;
        this.sz = sz;
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
        foreach (o; *tup.objects)
        {
            if (Dsymbol s = getDsymbol(o))
            {
                /* If tuple element represents a symbol, translate to DsymbolExp
                 * to supply implicit 'this' if needed later.
                 */
                Expression e = new DsymbolExp(loc, s);
                this.exps.push(e);
            }
            else if (auto eo = o.isExpression())
            {
                auto e = eo.copy();
                e.loc = loc;    // https://issues.dlang.org/show_bug.cgi?id=15669
                this.exps.push(e);
            }
            else if (auto t = o.isType())
            {
                Expression e = new TypeExp(loc, t);
                this.exps.push(e);
            }
            else
            {
                error("`%s` is not an expression", o.toChars());
            }
        }
    }

    static TupleExp create(Loc loc, Expressions* exps)
    {
        return new TupleExp(loc, exps);
    }

    override TupleExp toTupleExp()
    {
        return this;
    }

    override Expression syntaxCopy()
    {
        return new TupleExp(loc, e0 ? e0.syntaxCopy() : null, arraySyntaxCopy(exps));
    }

    override bool equals(const RootObject o) const
    {
        if (this == o)
            return true;
        if (auto e = o.isExpression())
            if (auto te = e.isTupleExp())
            {
                if (exps.dim != te.exps.dim)
                    return false;
                if (e0 && !e0.equals(te.e0) || !e0 && te.e0)
                    return false;
                foreach (i, e1; *exps)
                {
                    auto e2 = (*te.exps)[i];
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


    extern (D) this(const ref Loc loc, Type type, Expressions* elements)
    {
        super(loc, TOK.arrayLiteral, __traits(classInstanceSize, ArrayLiteralExp));
        this.type = type;
        this.elements = elements;
    }

    extern (D) this(const ref Loc loc, Type type, Expression e)
    {
        super(loc, TOK.arrayLiteral, __traits(classInstanceSize, ArrayLiteralExp));
        this.type = type;
        elements = new Expressions();
        elements.push(e);
    }

    extern (D) this(const ref Loc loc, Type type, Expression basis, Expressions* elements)
    {
        super(loc, TOK.arrayLiteral, __traits(classInstanceSize, ArrayLiteralExp));
        this.type = type;
        this.basis = basis;
        this.elements = elements;
    }

    static ArrayLiteralExp create(Loc loc, Expressions* elements)
    {
        return new ArrayLiteralExp(loc, null, elements);
    }

    // Same as create, but doesn't allocate memory.
    static void emplace(UnionExp* pue, Loc loc, Expressions* elements)
    {
        emplaceExp!(ArrayLiteralExp)(pue, loc, null, elements);
    }

    override Expression syntaxCopy()
    {
        return new ArrayLiteralExp(loc,
            null,
            basis ? basis.syntaxCopy() : null,
            arraySyntaxCopy(elements));
    }

    override bool equals(const RootObject o) const
    {
        if (this == o)
            return true;
        auto e = o.isExpression();
        if (!e)
            return false;
        if (auto ae = e.isArrayLiteralExp())
        {
            if (elements.dim != ae.elements.dim)
                return false;
            if (elements.dim == 0 && !type.equals(ae.type))
            {
                return false;
            }

            foreach (i, e1; *elements)
            {
                auto e2 = (*ae.elements)[i];
                auto e1x = e1 ? e1 : basis;
                auto e2x = e2 ? e2 : ae.basis;

                if (e1x != e2x && (!e1x || !e2x || !e1x.equals(e2x)))
                    return false;
            }
            return true;
        }
        return false;
    }

    Expression getElement(size_t i)
    {
        return this[i];
    }

    Expression opIndex(size_t i)
    {
        auto el = (*elements)[i];
        return el ? el : basis;
    }

    override bool isBool(bool result)
    {
        size_t dim = elements ? elements.dim : 0;
        return result ? (dim != 0) : (dim == 0);
    }

    override StringExp toStringExp()
    {
        TY telem = type.nextOf().toBasetype().ty;
        if (telem.isSomeChar || (telem == Tvoid && (!elements || elements.dim == 0)))
        {
            ubyte sz = 1;
            if (telem == Twchar)
                sz = 2;
            else if (telem == Tdchar)
                sz = 4;

            OutBuffer buf;
            if (elements)
            {
                foreach (i; 0 .. elements.dim)
                {
                    auto ch = this[i];
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

            const size_t len = buf.length / sz - 1;
            auto se = new StringExp(loc, buf.extractSlice()[0 .. len * sz], len, sz, prefix);
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

    override bool equals(const RootObject o) const
    {
        if (this == o)
            return true;
        auto e = o.isExpression();
        if (!e)
            return false;
        if (auto ae = e.isAssocArrayLiteralExp())
        {
            if (keys.dim != ae.keys.dim)
                return false;
            size_t count = 0;
            foreach (i, key; *keys)
            {
                foreach (j, akey; *ae.keys)
                {
                    if (key.equals(akey))
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

version (IN_LLVM)
{
    // With the introduction of pointers returned from CTFE, struct literals can
    // now contain pointers to themselves. While in toElem, contains a pointer
    // to the memory used to build the literal for resolving such references.
    void* inProgressMemory; // llvm::Value*
}

    Symbol* sym;            /// back end symbol to initialize with literal

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

    bool useStaticInit;     /// if this is true, use the StructDeclaration's init symbol
    bool isOriginal = false; /// used when moving instances to indicate `this is this.origin`
    OwnedBy ownedByCtfe = OwnedBy.code;

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

    override bool equals(const RootObject o) const
    {
        if (this == o)
            return true;
        auto e = o.isExpression();
        if (!e)
            return false;
        if (auto se = e.isStructLiteralExp())
        {
            if (!type.equals(se.type))
                return false;
            if (elements.dim != se.elements.dim)
                return false;
            foreach (i, e1; *elements)
            {
                auto e2 = (*se.elements)[i];
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
            if (i >= sd.nonHiddenFields())
                return null;

            assert(i < elements.dim);
            e = (*elements)[i];
            if (e)
            {
                //printf("e = %s, e.type = %s\n", e.toChars(), e.type.toChars());

                /* If type is a static array, and e is an initializer for that array,
                 * then the field initializer should be an array literal of e.
                 */
                auto tsa = type.isTypeSArray();
                if (tsa && e.type.castMod(0) != type.castMod(0))
                {
                    const length = cast(size_t)tsa.dim.toInteger();
                    auto z = new Expressions(length);
                    foreach (ref q; *z)
                        q = e.copy();
                    e = new ArrayLiteralExp(loc, type, z);
                }
                else
                {
                    e = e.copy();
                    e.type = type;
                }
                if (useStaticInit && e.type.needsNested())
                    if (auto se = e.isStructLiteralExp())
                    {
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
            foreach (i, v; sd.fields)
            {
                if (offset == v.offset && type.size() == v.type.size())
                {
                    /* context fields might not be filled. */
                    if (i >= sd.nonHiddenFields())
                        return cast(int)i;
                    if (auto e = (*elements)[i])
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
            char[10] buf = void;
            const prefix = "__sl";
            const ident = sd.ident.toString;
            const fullLen = prefix.length + ident.length;
            const len = fullLen < buf.length ? fullLen : buf.length;
            buf[0 .. prefix.length] = prefix;
            buf[prefix.length .. len] = ident[0 .. len - prefix.length];

            auto tmp = copyToTemp(0, buf[0 .. len], this);
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
        return symbolToExp(fd, loc, sc, true);
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
    Dsymbol originalScope; // original scope before inlining

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
version (IN_LLVM)
{
        // For a weak symbol, we only statically know that it is non-null if the
        // offset is non-zero.
        if (var.llvmInternal == LDCPragma.LLVMextern_weak)
            return result && offset != 0;
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
    bool delegateWasExtracted;
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

    override bool equals(const RootObject o) const
    {
        if (this == o)
            return true;
        if (auto ne = o.isExpression().isVarExp())
        {
            if (type.toHeadMutable().equals(ne.type.toHeadMutable()) && var == ne.var)
            {
                return true;
            }
        }
        return false;
    }

    override Modifiable checkModifiable(Scope* sc, int flag)
    {
        //printf("VarExp::checkModifiable %s", toChars());
        assert(type);
        return var.checkModify(loc, sc, null, flag);
    }

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
            error("manifest constant `%s` cannot be modified", var.toChars());
            return new ErrorExp();
        }
        if (var.storage_class & STC.lazy_ && !delegateWasExtracted)
        {
            error("lazy variable `%s` cannot be modified", var.toChars());
            return new ErrorExp();
        }
        if (var.ident == Id.ctfe)
        {
            error("cannot modify compiler-generated variable `__ctfe`");
            return new ErrorExp();
        }
        if (var.ident == Id.dollar) // https://issues.dlang.org/show_bug.cgi?id=13574
        {
            error("cannot modify operator `$`");
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

    override Expression syntaxCopy()
    {
        auto ret = super.syntaxCopy();
        return ret;
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

    override bool equals(const RootObject o) const
    {
        if (this == o)
            return true;
        auto e = o.isExpression();
        if (!e)
            return false;
        if (auto fe = e.isFuncExp())
        {
            return fd == fe.fd;
        }
        return false;
    }

    extern (D) void genIdent(Scope* sc)
    {
        if (fd.ident == Id.empty)
        {
            const(char)[] s;
            if (fd.fes)
                s = "__foreachbody";
            else if (fd.tok == TOK.reserved)
                s = "__lambda";
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
            Identifier id = Identifier.generateId(s, symtab.length() + 1);
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

    extern (D) MATCH matchType(Type to, Scope* sc, FuncExp* presult, int flag = 0)
    {

        static MATCH cannotInfer(Expression e, Type to, int flag)
        {
            if (!flag)
                e.error("cannot infer parameter types from `%s`", to.toChars());
            return MATCH.nomatch;
        }

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
        else if (to.ty == Tpointer && (tof = to.nextOf().isTypeFunction()) !is null)
        {
            if (tok == TOK.delegate_)
            {
                if (!flag)
                    error("cannot match delegate literal to function pointer type `%s`", to.toChars());
                return MATCH.nomatch;
            }
        }

        if (td)
        {
            if (!tof)
            {
                return cannotInfer(this, to, flag);
            }

            // Parameter types inference from 'tof'
            assert(td._scope);
            TypeFunction tf = fd.type.isTypeFunction();
            //printf("\ttof = %s\n", tof.toChars());
            //printf("\ttf  = %s\n", tf.toChars());
            size_t dim = tf.parameterList.length;

            if (tof.parameterList.length != dim || tof.parameterList.varargs != tf.parameterList.varargs)
                return cannotInfer(this, to, flag);

            auto tiargs = new Objects();
            tiargs.reserve(td.parameters.dim);

            foreach (tp; *td.parameters)
            {
                size_t u = 0;
                for (; u < dim; u++)
                {
                    Parameter p = tf.parameterList[u];
                    if (auto ti = p.type.isTypeIdentifier())
                        if (ti && ti.ident == tp.ident)
                        {
                            break;
                        }
                }
                assert(u < dim);
                Parameter pto = tof.parameterList[u];
                Type t = pto.type;
                if (t.ty == Terror)
                    return cannotInfer(this, to, flag);
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
            if (auto ef = ex.isFuncExp())
                return ef.matchType(to, sc, presult, flag);
            else
                return cannotInfer(this, to, flag);
        }

        if (!tof || !tof.next)
            return MATCH.nomatch;

        assert(type && type != Type.tvoid);
        if (fd.type.ty == Terror)
            return MATCH.nomatch;
        auto tfx = fd.type.isTypeFunction();
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

            auto tfy = new TypeFunction(tfx.parameterList, tof.next,
                        tfx.linkage, STC.undefined_);
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
        if (tok == TOK.delegate_ ||
            tok == TOK.reserved && (type.ty == Tdelegate || type.ty == Tpointer && to.ty == Tdelegate))
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
version (IN_LLVM)
{
                (*presult).fd.type = tof; // Also, update function return type.
}
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

    override const(char)* toChars() const
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
    Type tspec;         // can be null
    TemplateParameters* parameters;
    TOK tok;            // ':' or '=='
    TOK tok2;           // 'struct', 'union', etc.

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
            p = new TemplateParameters(parameters.dim);
            foreach (i, el; *parameters)
                (*p)[i] = el.syntaxCopy();
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
extern (C++) abstract class UnaExp : Expression
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
        if (auto edi = e1.isDotIdExp())
            edi.noderef = true;

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

alias fp_t = UnionExp function(const ref Loc loc, Type, Expression, Expression);
alias fp2_t = bool function(const ref Loc loc, TOK, Expression, Expression);

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

    extern (D) final Expression checkOpAssignTypes(Scope* sc)
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

    extern (D) final bool checkIntegralBin()
    {
        bool r1 = e1.checkIntegral();
        bool r2 = e2.checkIntegral();
        return (r1 || r2);
    }

    extern (D) final bool checkArithmeticBin()
    {
        bool r1 = e1.checkArithmetic();
        bool r2 = e2.checkArithmetic();
        return (r1 || r2);
    }

    extern (D) final bool checkSharedAccessBin(Scope* sc)
    {
        const r1 = e1.checkSharedAccess(sc);
        const r2 = e2.checkSharedAccess(sc);
        return (r1 || r2);
    }

    /*********************
     * Mark the operands as will never be dereferenced,
     * which is useful info for @safe checks.
     * Do before semantic() on operands rewrites them.
     */
    final void setNoderefOperands()
    {
        if (auto edi = e1.isDotIdExp())
            edi.noderef = true;
        if (auto edi = e2.isDotIdExp())
            edi.noderef = true;

    }

    final Expression reorderSettingAAElem(Scope* sc)
    {
        BinExp be = this;

        auto ie = be.e1.isIndexExp();
        if (!ie)
            return be;
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

            auto ie1 = ie.e1.isIndexExp();
            if (!ie1 ||
                ie1.e1.type.toBasetype().ty != Taarray)
            {
                break;
            }
            ie = ie1;
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
 * https://dlang.org/spec/expression.html#mixin_expressions
 */
extern (C++) final class CompileExp : Expression
{
    Expressions* exps;

    extern (D) this(const ref Loc loc, Expressions* exps)
    {
        super(loc, TOK.mixin_, __traits(classInstanceSize, CompileExp));
        this.exps = exps;
    }

    override Expression syntaxCopy()
    {
        return new CompileExp(loc, arraySyntaxCopy(exps));
    }

    override bool equals(const RootObject o) const
    {
        if (this == o)
            return true;
        auto e = o.isExpression();
        if (!e)
            return false;
        if (auto ce = e.isCompileExp())
        {
            if (exps.dim != ce.exps.dim)
                return false;
            foreach (i, e1; *exps)
            {
                auto e2 = (*ce.exps)[i];
                if (e1 != e2 && (!e1 || !e2 || !e1.equals(e2)))
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

    override Modifiable checkModifiable(Scope* sc, int flag)
    {
        //printf("DotVarExp::checkModifiable %s %s\n", toChars(), type.toChars());
        if (checkUnsafeAccess(sc, this, false, !flag))
            return Modifiable.initialization;

        if (e1.op == TOK.this_)
            return var.checkModify(loc, sc, e1, flag);

        /* https://issues.dlang.org/show_bug.cgi?id=12764
         * If inside a constructor and an expression of type `this.field.var`
         * is encountered, where `field` is a struct declaration with
         * default construction disabled, we must make sure that
         * assigning to `var` does not imply that `field` was initialized
         */
        if (sc.func && sc.func.isCtorDeclaration())
        {
            // if inside a constructor scope and e1 of this DotVarExp
            // is a DotVarExp, then check if e1.e1 is a `this` identifier
            if (auto dve = e1.isDotVarExp())
            {
                if (dve.e1.op == TOK.this_)
                {
                    scope v = dve.var.isVarDeclaration();
                    /* if v is a struct member field with no initializer, no default construction
                     * and v wasn't intialized before
                     */
                    if (v && v.isField() && !v._init && !v.ctorinit)
                    {
                        if (auto ts = v.type.isTypeStruct())
                        {
                            if (ts.sym.noDefaultCtor)
                            {
                                /* checkModify will consider that this is an initialization
                                 * of v while it is actually an assignment of a field of v
                                 */
                                scope modifyLevel = v.checkModify(loc, sc, dve.e1, flag);
                                // reflect that assigning a field of v is not initialization of v
                                v.ctorinit = false;
                                if (modifyLevel == Modifiable.initialization)
                                    return Modifiable.yes;
                                return modifyLevel;
                            }
                        }
                    }
                }
            }
        }

        //printf("\te1 = %s\n", e1.toChars());
        return e1.checkModifiable(sc, flag);
    }

    override bool isLvalue()
    {
        return true;
    }

    override Expression toLvalue(Scope* sc, Expression e)
    {
        //printf("DotVarExp::toLvalue(%s)\n", toChars());
        if (e1.op == TOK.this_ && sc.ctorflow.fieldinit.length && !(sc.ctorflow.callSuper & CSX.any_ctor))
        {
            if (VarDeclaration vd = var.isVarDeclaration())
            {
                auto ad = vd.isMember2();
                if (ad && ad.fields.dim == sc.ctorflow.fieldinit.length)
                {
                    foreach (i, f; ad.fields)
                    {
                        if (f == vd)
                        {
                            if (!(sc.ctorflow.fieldinit[i].csx & CSX.this_ctor))
                            {
                                /* If the address of vd is taken, assume it is thereby initialized
                                 * https://issues.dlang.org/show_bug.cgi?id=15869
                                 */
                                modifyFieldVar(loc, sc, vd, e1);
                            }
                            break;
                        }
                    }
                }
            }
        }
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
    VarDeclaration vthis2;  // container for multi-context

    extern (D) this(const ref Loc loc, Expression e, FuncDeclaration f, bool hasOverloads = true, VarDeclaration vthis2 = null)
    {
        super(loc, TOK.delegate_, __traits(classInstanceSize, DelegateExp), e);
        this.func = f;
        this.hasOverloads = hasOverloads;
        this.vthis2 = vthis2;
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
    VarDeclaration vthis2;  // container for multi-context

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
        this.arguments = new Expressions();
        if (earg1)
            this.arguments.push(earg1);
    }

    extern (D) this(const ref Loc loc, Expression e, Expression earg1, Expression earg2)
    {
        super(loc, TOK.call, __traits(classInstanceSize, CallExp), e);
        auto arguments = new Expressions(2);
        (*arguments)[0] = earg1;
        (*arguments)[1] = earg2;
        this.arguments = arguments;
    }

    /***********************************************************
    * Instatiates a new function call expression
    * Params:
    *       loc   = location
    *       fd    = the declaration of the function to call
    *       earg1 = the function argument
    */
    extern(D) this(const ref Loc loc, FuncDeclaration fd, Expression earg1)
    {
        this(loc, new VarExp(loc, fd, false), earg1);
        this.f = fd;
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

    /***********************************************************
    * Creates a new function call expression
    * Params:
    *       loc   = location
    *       fd    = the declaration of the function to call
    *       earg1 = the function argument
    */
    static CallExp create(Loc loc, FuncDeclaration fd, Expression earg1)
    {
        return new CallExp(loc, fd, earg1);
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
        auto tf = tb.isTypeFunction();
        if (tf && tf.isref)
        {
            if (auto dve = e1.isDotVarExp())
                if (dve.var.isCtorDeclaration())
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

        if (auto tf = e1.type.isTypeFunction())
        {
            if (tf.isref)
                return this;
        }

        Type tv = type.baseElemOf();
        if (auto ts = tv.isTypeStruct())
        {
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
    if (auto ae = e.isAddrExp())
    {
        auto ae1 = ae.e1;
        if (auto ve = ae1.isVarExp())
        {
            if (hasOverloads)
                *hasOverloads = ve.hasOverloads;
            return ve.var.isFuncDeclaration();
        }
        if (auto dve = ae1.isDotVarExp())
        {
            if (hasOverloads)
                *hasOverloads = dve.hasOverloads;
            return dve.var.isFuncDeclaration();
        }
    }
    else
    {
        if (auto soe = e.isSymOffExp())
        {
            if (hasOverloads)
                *hasOverloads = soe.hasOverloads;
            return soe.var.isFuncDeclaration();
        }
        if (auto dge = e.isDelegateExp())
        {
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

    extern (D) this(const ref Loc loc, Expression e, Type t)
    {
        this(loc, e);
        type = t;
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

    override Modifiable checkModifiable(Scope* sc, int flag)
    {
        if (auto se = e1.isSymOffExp())
        {
            return se.var.checkModify(loc, sc, null, flag);
        }
        else if (auto ae = e1.isAddrExp())
        {
            return ae.e1.checkModifiable(sc, flag);
        }
        return Modifiable.yes;
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

    override Expression addDtorHook(Scope* sc)
    {
        if (to.toBasetype().ty == Tvoid)        // look past the cast(void)
            e1 = e1.addDtorHook(sc);
        return this;
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
    OwnedBy ownedByCtfe = OwnedBy.code;

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

    // Same as create, but doesn't allocate memory.
    static void emplace(UnionExp* pue, Loc loc, Expression e, Type type)
    {
        emplaceExp!(VectorExp)(pue, loc, e, type);
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
 * e1.array property for vectors.
 *
 * https://dlang.org/spec/simd.html#properties
 */
extern (C++) final class VectorArrayExp : UnaExp
{
    extern (D) this(const ref Loc loc, Expression e1)
    {
        super(loc, TOK.vectorArray, __traits(classInstanceSize, VectorArrayExp), e1);
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

    override Modifiable checkModifiable(Scope* sc, int flag)
    {
        //printf("SliceExp::checkModifiable %s\n", toChars());
        if (e1.type.ty == Tsarray || (e1.op == TOK.index && e1.type.ty != Tarray) || e1.op == TOK.slice)
        {
            return e1.checkModifiable(sc, flag);
        }
        return Modifiable.yes;
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

    override Modifiable checkModifiable(Scope* sc, int flag)
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
        if (exp)
            if (auto ce = exp.isCommaExp())
                ce.allowCommaExp = true;
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

    override Modifiable checkModifiable(Scope* sc, int flag)
    {
        if (e1.type.ty == Tsarray ||
            e1.type.ty == Taarray ||
            (e1.op == TOK.index && e1.type.ty != Tarray) ||
            e1.op == TOK.slice)
        {
            return e1.checkModifiable(sc, flag);
        }
        return Modifiable.yes;
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

    extern (D) Expression markSettingAAElem()
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

            if (auto ie = e1.isIndexExp())
            {
                Expression ex = ie.markSettingAAElem();
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
        assert(op == TOK.minusMinus || op == TOK.plusPlus);
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
        assert(op == TOK.preMinusMinus || op == TOK.prePlusPlus);
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

    this(const ref Loc loc, TOK tok, Expression e1, Expression e2)
    {
        super(loc, tok, __traits(classInstanceSize, AssignExp), e1, e2);
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
        super(loc, TOK.construct, e1, e2);
    }

    // Internal use only. If `v` is a reference variable, the assignment
    // will become a reference initialization automatically.
    extern (D) this(const ref Loc loc, VarDeclaration v, Expression e2)
    {
        auto ve = new VarExp(loc, v);
        assert(v.type && ve.type);

        super(loc, TOK.construct, ve, e2);

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
        super(loc, TOK.blit, e1, e2);
    }

    // Internal use only. If `v` is a reference variable, the assinment
    // will become a reference rebinding automatically.
    extern (D) this(const ref Loc loc, VarDeclaration v, Expression e2)
    {
        auto ve = new VarExp(loc, v);
        assert(v.type && ve.type);

        super(loc, TOK.blit, ve, e2);

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
extern (C++) class CatAssignExp : BinAssignExp
{
    extern (D) this(const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, TOK.concatenateAssign, __traits(classInstanceSize, CatAssignExp), e1, e2);
    }

    extern (D) this(const ref Loc loc, TOK tok, Expression e1, Expression e2)
    {
        super(loc, tok, __traits(classInstanceSize, CatAssignExp), e1, e2);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

///
extern (C++) final class CatElemAssignExp : CatAssignExp
{
    extern (D) this(const ref Loc loc, Type type, Expression e1, Expression e2)
    {
        super(loc, TOK.concatenateElemAssign, e1, e2);
        this.type = type;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

///
extern (C++) final class CatDcharAssignExp : CatAssignExp
{
    extern (D) this(const ref Loc loc, Type type, Expression e1, Expression e2)
    {
        super(loc, TOK.concatenateDcharAssign, e1, e2);
        this.type = type;
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

    override Expression resolveLoc(const ref Loc loc, Scope* sc)
    {
        e1 = e1.resolveLoc(loc, sc);
        e2 = e2.resolveLoc(loc, sc);
        return this;
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
        assert(op == TOK.andAnd || op == TOK.orOr);
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
 *      TOK.lessThan, TOK.lessOrEqual, TOK.greaterThan, TOK.greaterOrEqual
 *
 * http://dlang.org/spec/expression.html#relation_expressions
 */
extern (C++) final class CmpExp : BinExp
{
    extern (D) this(TOK op, const ref Loc loc, Expression e1, Expression e2)
    {
        super(loc, op, __traits(classInstanceSize, CmpExp), e1, e2);
        assert(op == TOK.lessThan || op == TOK.lessOrEqual || op == TOK.greaterThan || op == TOK.greaterOrEqual);
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
        assert(op == TOK.identity || op == TOK.notIdentity);
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

    override Modifiable checkModifiable(Scope* sc, int flag)
    {
        if (e1.checkModifiable(sc, flag) != Modifiable.no
            && e2.checkModifiable(sc, flag) != Modifiable.no)
            return Modifiable.yes;
        return Modifiable.no;
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
            alias visit = typeof(super).visit;
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
                            walkPostorder(ei.exp, this);
                    }

                    if (v.edtor)
                        walkPostorder(v.edtor, this);

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
            s = FileName.toAbsolute(loc.isValid() ? loc.filename : sc._module.srcfile.toChars());
        else
            s = loc.isValid() ? loc.filename : sc._module.ident.toChars();

        Expression e = new StringExp(loc, s.toDString());
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
        const auto s = (sc.callsc ? sc.callsc : sc)._module.toPrettyChars().toDString();
        Expression e = new StringExp(loc, s);
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
        Expression e = new StringExp(loc, s.toDString());
        e = e.expressionSemantic(sc);
        e.type = Type.tstring;
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
        FuncDeclaration fd = (sc.callsc && sc.callsc.func)
                        ? sc.callsc.func
                        : sc.func;

        const(char)* s;
        if (fd)
        {
            const funcStr = fd.Dsymbol.toPrettyChars();
            OutBuffer buf;
            functionToBufferWithIdent(fd.type.isTypeFunction(), &buf, funcStr);
            s = buf.extractChars();
        }
        else
        {
            s = "";
        }

        Expression e = new StringExp(loc, s.toDString());
        e = e.expressionSemantic(sc);
        e.type = Type.tstring;
        return e;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/**
 * Objective-C class reference expression.
 *
 * Used to get the metaclass of an Objective-C class, `NSObject.Class`.
 */
extern (C++) final class ObjcClassReferenceExp : Expression
{
    ClassDeclaration classDeclaration;

    extern (D) this(const ref Loc loc, ClassDeclaration classDeclaration)
    {
        super(loc, TOK.objcClassReference,
            __traits(classInstanceSize, ObjcClassReferenceExp));
        this.classDeclaration = classDeclaration;
        type = objc.getRuntimeMetaclass(classDeclaration).getType();
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}
