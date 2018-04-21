/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/escape.d, _escape.d)
 * Documentation:  https://dlang.org/phobos/dmd_escape.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/escape.d
 */

module dmd.escape;

import core.stdc.stdio : printf;

import dmd.aggregate;
import dmd.declaration;
import dmd.dscope;
import dmd.dsymbol;
import dmd.errors;
import dmd.expression;
import dmd.func;
import dmd.globals;
import dmd.identifier;
import dmd.init;
import dmd.mtype;
import dmd.root.rootobject;
import dmd.tokens;
import dmd.visitor;
import dmd.arraytypes;

/****************************************
 * Function parameter par is being initialized to arg,
 * and par may escape.
 * Detect if scoped values can escape this way.
 * Print error messages when these are detected.
 * Params:
 *      sc = used to determine current function and module
 *      fdc = function being called, `null` if called indirectly
 *      par = identifier of function parameter
 *      arg = initializer for param
 *      gag = do not print error messages
 * Returns:
 *      true if pointers to the stack can escape via assignment
 */
bool checkParamArgumentEscape(Scope* sc, FuncDeclaration fdc, Identifier par, Expression arg, bool gag)
{
    //printf("checkParamArgumentEscape(arg: %s par: %s)\n", arg.toChars(), par.toChars());
    //printf("type = %s, %d\n", arg.type.toChars(), arg.type.hasPointers());

    if (!arg.type.hasPointers())
        return false;

    EscapeByResults er;

    escapeByValue(arg, &er);

    if (!er.byref.dim && !er.byvalue.dim && !er.byfunc.dim && !er.byexp.dim)
        return false;

    bool result = false;

    /* 'v' is assigned unsafely to 'par'
     */
    void unsafeAssign(VarDeclaration v, const char* desc)
    {
        if (global.params.vsafe && sc.func.setUnsafe())
        {
            if (!gag)
                error(arg.loc, "%s `%s` assigned to non-scope parameter `%s` calling %s",
                    desc, v.toChars(),
                    par ? par.toChars() : "unnamed",
                    fdc ? fdc.toPrettyChars() : "indirectly");
            result = true;
        }
    }

    foreach (VarDeclaration v; er.byvalue)
    {
        //printf("byvalue %s\n", v.toChars());
        if (v.isDataseg())
            continue;

        Dsymbol p = v.toParent2();

        v.storage_class &= ~STC.maybescope;

        if (v.isScope())
        {
            unsafeAssign(v, "scope variable");
        }
        else if (v.storage_class & STC.variadic && p == sc.func)
        {
            Type tb = v.type.toBasetype();
            if (tb.ty == Tarray || tb.ty == Tsarray)
            {
                unsafeAssign(v, "variadic variable");
            }
        }
        else
        {
            /* v is not 'scope', and is assigned to a parameter that may escape.
             * Therefore, v can never be 'scope'.
             */
            v.doNotInferScope = true;
        }
    }

    foreach (VarDeclaration v; er.byref)
    {
        if (v.isDataseg())
            continue;

        Dsymbol p = v.toParent2();

        v.storage_class &= ~STC.maybescope;

        if ((v.storage_class & (STC.ref_ | STC.out_)) == 0 && p == sc.func)
        {
            unsafeAssign(v, "reference to local variable");
            continue;
        }
    }

    foreach (FuncDeclaration fd; er.byfunc)
    {
        //printf("fd = %s, %d\n", fd.toChars(), fd.tookAddressOf);
        VarDeclarations vars;
        findAllOuterAccessedVariables(fd, &vars);

        foreach (v; vars)
        {
            //printf("v = %s\n", v.toChars());
            assert(!v.isDataseg());     // these are not put in the closureVars[]

            Dsymbol p = v.toParent2();

            v.storage_class &= ~STC.maybescope;

            if ((v.storage_class & (STC.ref_ | STC.out_ | STC.scope_)) && p == sc.func)
            {
                unsafeAssign(v, "reference to local");
                continue;
            }
        }
    }

    foreach (Expression ee; er.byexp)
    {
        if (sc.func.setUnsafe())
        {
            if (!gag)
                error(ee.loc, "reference to stack allocated value returned by `%s` assigned to non-scope parameter `%s`",
                    ee.toChars(),
                    par ? par.toChars() : "unnamed");
            result = true;
        }
    }

    return result;
}

/****************************************
 * Given an AssignExp, determine if the lvalue will cause
 * the contents of the rvalue to escape.
 * Print error messages when these are detected.
 * Infer 'scope' for the lvalue where possible, in order
 * to eliminate the error.
 * Params:
 *      sc = used to determine current function and module
 *      e = AssignExp or CatAssignExp to check for any pointers to the stack
 *      gag = do not print error messages
 * Returns:
 *      true if pointers to the stack can escape via assignment
 */
bool checkAssignEscape(Scope* sc, Expression e, bool gag)
{
    enum log = false;
    if (log) printf("checkAssignEscape(e: %s)\n", e.toChars());
    if (e.op != TOK.assign && e.op != TOK.blit && e.op != TOK.construct &&
        e.op != TOK.concatenateAssign && e.op != TOK.concatenateElemAssign && e.op != TOK.concatenateDcharAssign)
        return false;
    auto ae = cast(BinExp)e;
    Expression e1 = ae.e1;
    Expression e2 = ae.e2;
    //printf("type = %s, %d\n", e1.type.toChars(), e1.type.hasPointers());

    if (!e1.type.hasPointers())
        return false;

    if (e1.op == TOK.slice)
        return false;

    EscapeByResults er;

    escapeByValue(e2, &er);

    if (!er.byref.dim && !er.byvalue.dim && !er.byfunc.dim && !er.byexp.dim)
        return false;

    VarDeclaration va;
    while (e1.op == TOK.dotVariable)
        e1 = (cast(DotVarExp)e1).e1;

    if (e1.op == TOK.variable)
        va = (cast(VarExp)e1).var.isVarDeclaration();
    else if (e1.op == TOK.this_)
        va = (cast(ThisExp)e1).var.isVarDeclaration();
    else if (e1.op == TOK.index)
    {
        auto ie = cast(IndexExp)e1;
        if (ie.e1.op == TOK.variable && ie.e1.type.toBasetype().ty == Tsarray)
            va = (cast(VarExp)ie.e1).var.isVarDeclaration();
    }

    if (va && e.op == TOK.concatenateElemAssign)
    {
        /* https://issues.dlang.org/show_bug.cgi?id=17842
         * Draw an equivalence between:
         *   *q = p;
         * and:
         *   va ~= e;
         * since we are not assigning to va, but are assigning indirectly through va.
         */
        va = null;
    }

    if (log && va) printf("va: %s\n", va.toChars());

    // Try to infer 'scope' for va if in a function not marked @system
    bool inferScope = false;
    if (va && sc.func && sc.func.type && sc.func.type.ty == Tfunction)
        inferScope = (cast(TypeFunction)sc.func.type).trust != TRUST.system;

    bool result = false;
    foreach (VarDeclaration v; er.byvalue)
    {
        if (log) printf("byvalue: %s\n", v.toChars());
        if (v.isDataseg())
            continue;

        if (v == va)
            continue;

        Dsymbol p = v.toParent2();

        if (!(va && va.isScope()))
            v.storage_class &= ~STC.maybescope;

        if (v.isScope())
        {
            if (va && va.isScope() && va.storage_class & STC.return_ && !(v.storage_class & STC.return_) &&
                sc.func.setUnsafe())
            {
                if (!gag)
                    error(ae.loc, "scope variable `%s` assigned to return scope `%s`", v.toChars(), va.toChars());
                result = true;
                continue;
            }

            // If va's lifetime encloses v's, then error
            if (va &&
                (va.enclosesLifetimeOf(v) && !(v.storage_class & (STC.parameter | STC.temp)) ||
                 // va is class reference
                 ae.e1.op == TOK.dotVariable && va.type.toBasetype().ty == Tclass && (va.enclosesLifetimeOf(v) || !va.isScope) ||
                 va.storage_class & STC.ref_ && !(v.storage_class & STC.temp)) &&
                sc.func.setUnsafe())
            {
                if (!gag)
                    error(ae.loc, "scope variable `%s` assigned to `%s` with longer lifetime", v.toChars(), va.toChars());
                result = true;
                continue;
            }

            if (va && !va.isDataseg() && !va.doNotInferScope)
            {
                if (!va.isScope() && inferScope)
                {   //printf("inferring scope for %s\n", va.toChars());
                    va.storage_class |= STC.scope_ | STC.scopeinferred;
                    va.storage_class |= v.storage_class & STC.return_;
                }
                continue;
            }
            if (sc.func.setUnsafe())
            {
                if (!gag)
                    error(ae.loc, "scope variable `%s` assigned to non-scope `%s`", v.toChars(), e1.toChars());
                result = true;
            }
        }
        else if (v.storage_class & STC.variadic && p == sc.func)
        {
            Type tb = v.type.toBasetype();
            if (tb.ty == Tarray || tb.ty == Tsarray)
            {
                if (va && !va.isDataseg() && !va.doNotInferScope)
                {
                    if (!va.isScope() && inferScope)
                    {   //printf("inferring scope for %s\n", va.toChars());
                        va.storage_class |= STC.scope_ | STC.scopeinferred;
                    }
                    continue;
                }
                if (sc.func.setUnsafe())
                {
                    if (!gag)
                        error(ae.loc, "variadic variable `%s` assigned to non-scope `%s`", v.toChars(), e1.toChars());
                    result = true;
                }
            }
        }
        else
        {
            /* v is not 'scope', and we didn't check the scope of where we assigned it to.
             * It may escape via that assignment, therefore, v can never be 'scope'.
             */
            v.doNotInferScope = true;
        }
    }

ByRef:
    foreach (VarDeclaration v; er.byref)
    {
        if (log) printf("byref: %s\n", v.toChars());
        if (v.isDataseg())
            continue;

        Dsymbol p = v.toParent2();

        // If va's lifetime encloses v's, then error
        if (va &&
            (va.enclosesLifetimeOf(v) && !(v.storage_class & STC.parameter) ||
             va.storage_class & STC.ref_ ||
             va.isDataseg()) &&
            sc.func.setUnsafe())
        {
            if (!gag)
                error(ae.loc, "address of variable `%s` assigned to `%s` with longer lifetime", v.toChars(), va.toChars());
            result = true;
            continue;
        }

        if (va &&  v.storage_class & (STC.ref_ | STC.out_))
        {
            Dsymbol pva = va.toParent2();
            for (Dsymbol pv = p; pv; )
            {
                pv = pv.toParent2();
                if (pva == pv)  // if v is nested inside pva
                {
                    if (sc.func.setUnsafe())
                    {
                        if (!gag)
                            error(ae.loc, "reference `%s` assigned to `%s` with longer lifetime", v.toChars(), va.toChars());
                        result = true;
                        continue ByRef;
                    }
                    break;
                }
            }
        }

        if (!(va && va.isScope()))
            v.storage_class &= ~STC.maybescope;

        if ((v.storage_class & (STC.ref_ | STC.out_)) == 0 && p == sc.func)
        {
            if (va && !va.isDataseg() && !va.doNotInferScope)
            {
                if (!va.isScope() && inferScope)
                {   //printf("inferring scope for %s\n", va.toChars());
                    va.storage_class |= STC.scope_ | STC.scopeinferred;
                }
                continue;
            }
            if (sc.func.setUnsafe())
            {
                if (!gag)
                    error(ae.loc, "reference to local variable `%s` assigned to non-scope `%s`", v.toChars(), e1.toChars());
                result = true;
            }
            continue;
        }
    }

    foreach (FuncDeclaration fd; er.byfunc)
    {
        if (log) printf("byfunc: %s, %d\n", fd.toChars(), fd.tookAddressOf);
        VarDeclarations vars;
        findAllOuterAccessedVariables(fd, &vars);

        foreach (v; vars)
        {
            //printf("v = %s\n", v.toChars());
            assert(!v.isDataseg());     // these are not put in the closureVars[]

            Dsymbol p = v.toParent2();

            if (!(va && va.isScope()))
                v.storage_class &= ~STC.maybescope;

            if ((v.storage_class & (STC.ref_ | STC.out_ | STC.scope_)) && p == sc.func)
            {
                if (va && !va.isDataseg() && !va.doNotInferScope)
                {
                    /* Don't infer STC.scope_ for va, because then a closure
                     * won't be generated for sc.func.
                     */
                    //if (!va.isScope() && inferScope)
                        //va.storage_class |= STC.scope_ | STC.scopeinferred;
                    continue;
                }
                if (sc.func.setUnsafe())
                {
                    if (!gag)
                        error(ae.loc, "reference to local `%s` assigned to non-scope `%s` in @safe code", v.toChars(), e1.toChars());
                    result = true;
                }
                continue;
            }
        }
    }

    foreach (Expression ee; er.byexp)
    {
        if (log) printf("byexp: %s\n", ee.toChars());

        /* Do not allow slicing of a static array returned by a function
         */
        if (va && ee.op == TOK.call && ee.type.toBasetype().ty == Tsarray && va.type.toBasetype().ty == Tarray &&
            !(va.storage_class & STC.temp))
        {
            if (!gag)
                deprecation(ee.loc, "slice of static array temporary returned by `%s` assigned to longer lived variable `%s`",
                    ee.toChars(), va.toChars());
            //result = true;
            continue;
        }

        if (va && !va.isDataseg() && !va.doNotInferScope)
        {
            if (!va.isScope() && inferScope)
            {   //printf("inferring scope for %s\n", va.toChars());
                va.storage_class |= STC.scope_ | STC.scopeinferred;
            }
            continue;
        }

        if (sc.func.setUnsafe())
        {
            if (!gag)
                error(ee.loc, "reference to stack allocated value returned by `%s` assigned to non-scope `%s`",
                    ee.toChars(), e1.toChars());
            result = true;
        }
    }

    return result;
}

/************************************
 * Detect cases where pointers to the stack can 'escape' the
 * lifetime of the stack frame when throwing `e`.
 * Print error messages when these are detected.
 * Params:
 *      sc = used to determine current function and module
 *      e = expression to check for any pointers to the stack
 *      gag = do not print error messages
 * Returns:
 *      true if pointers to the stack can escape
 */
bool checkThrowEscape(Scope* sc, Expression e, bool gag)
{
    //printf("[%s] checkThrowEscape, e = %s\n", e.loc.toChars(), e.toChars());
    EscapeByResults er;

    escapeByValue(e, &er);

    if (!er.byref.dim && !er.byvalue.dim && !er.byexp.dim)
        return false;

    bool result = false;
    foreach (VarDeclaration v; er.byvalue)
    {
        //printf("byvalue %s\n", v.toChars());
        if (v.isDataseg())
            continue;

        if (v.isScope() && !v.iscatchvar)       // special case: allow catch var to be rethrown
                                                // despite being `scope`
        {
            if (sc._module && sc._module.isRoot())
            {
                // Only look for errors if in module listed on command line
                if (global.params.vsafe) // https://issues.dlang.org/show_bug.cgi?id=17029
                {
                    if (!gag)
                        error(e.loc, "scope variable `%s` may not be thrown", v.toChars());
                    result = true;
                }
                continue;
            }
        }
        else
        {
            //printf("no infer for %s\n", v.toChars());
            v.doNotInferScope = true;
        }
    }
    return result;
}

/************************************
 * Detect cases where pointers to the stack can 'escape' the
 * lifetime of the stack frame by returning 'e' by value.
 * Print error messages when these are detected.
 * Params:
 *      sc = used to determine current function and module
 *      e = expression to check for any pointers to the stack
 *      gag = do not print error messages
 * Returns:
 *      true if pointers to the stack can escape
 */
bool checkReturnEscape(Scope* sc, Expression e, bool gag)
{
    //printf("[%s] checkReturnEscape, e = %s\n", e.loc.toChars(), e.toChars());
    return checkReturnEscapeImpl(sc, e, false, gag);
}

/************************************
 * Detect cases where returning 'e' by ref can result in a reference to the stack
 * being returned.
 * Print error messages when these are detected.
 * Params:
 *      sc = used to determine current function and module
 *      e = expression to check
 *      gag = do not print error messages
 * Returns:
 *      true if references to the stack can escape
 */
bool checkReturnEscapeRef(Scope* sc, Expression e, bool gag)
{
    version (none)
    {
        printf("[%s] checkReturnEscapeRef, e = %s\n", e.loc.toChars(), e.toChars());
        printf("current function %s\n", sc.func.toChars());
        printf("parent2 function %s\n", sc.func.toParent2().toChars());
    }

    return checkReturnEscapeImpl(sc, e, true, gag);
}

private bool checkReturnEscapeImpl(Scope* sc, Expression e, bool refs, bool gag)
{
    enum log = false;
    if (log) printf("[%s] checkReturnEscapeImpl, refs: %d e: `%s`\n", e.loc.toChars(), refs, e.toChars());
    EscapeByResults er;

    if (refs)
        escapeByRef(e, &er);
    else
        escapeByValue(e, &er);

    if (!er.byref.dim && !er.byvalue.dim && !er.byexp.dim)
        return false;

    bool result = false;
    foreach (VarDeclaration v; er.byvalue)
    {
        if (log) printf("byvalue `%s`\n", v.toChars());
        if (v.isDataseg())
            continue;

        Dsymbol p = v.toParent2();

        if ((v.isScope() || (v.storage_class & STC.maybescope)) &&
            !(v.storage_class & STC.return_) &&
            v.isParameter() &&
            sc.func.flags & FUNCFLAG.returnInprocess &&
            p == sc.func)
        {
            inferReturn(sc.func, v);        // infer addition of 'return'
            continue;
        }

        if (v.isScope())
        {
            if (v.storage_class & STC.return_)
                continue;

            if (sc._module && sc._module.isRoot() &&
                /* This case comes up when the ReturnStatement of a __foreachbody is
                 * checked for escapes by the caller of __foreachbody. Skip it.
                 *
                 * struct S { static int opApply(int delegate(S*) dg); }
                 * S* foo() {
                 *    foreach (S* s; S) // create __foreachbody for body of foreach
                 *        return s;     // s is inferred as 'scope' but incorrectly tested in foo()
                 *    return null; }
                 */
                !(!refs && p.parent == sc.func))
            {
                // Only look for errors if in module listed on command line
                if (global.params.vsafe) // https://issues.dlang.org/show_bug.cgi?id=17029
                {
                    if (!gag)
                        error(e.loc, "scope variable `%s` may not be returned", v.toChars());
                    result = true;
                }
                continue;
            }
        }
        else if (v.storage_class & STC.variadic && p == sc.func)
        {
            Type tb = v.type.toBasetype();
            if (tb.ty == Tarray || tb.ty == Tsarray)
            {
                if (!gag)
                    error(e.loc, "returning `%s` escapes a reference to variadic parameter `%s`", e.toChars(), v.toChars());
                result = false;
            }
        }
        else
        {
            //printf("no infer for %s\n", v.toChars());
            v.doNotInferScope = true;
        }
    }

    foreach (VarDeclaration v; er.byref)
    {
        if (log) printf("byref `%s`\n", v.toChars());

        void escapingRef(VarDeclaration v)
        {
            if (!gag)
            {
                const(char)* msg;
                if (v.storage_class & STC.parameter)
                    msg = "returning `%s` escapes a reference to parameter `%s`, perhaps annotate with `return`";
                else
                    msg = "returning `%s` escapes a reference to local variable `%s`";
                error(e.loc, msg, e.toChars(), v.toChars());
            }
            result = true;
        }

        if (v.isDataseg())
            continue;

        Dsymbol p = v.toParent2();

        if ((v.storage_class & (STC.ref_ | STC.out_)) == 0)
        {
            if (p == sc.func)
            {
                escapingRef(v);
                continue;
            }
            FuncDeclaration fd = p.isFuncDeclaration();
            if (fd && sc.func.flags & FUNCFLAG.returnInprocess)
            {
                /* Code like:
                 *   int x;
                 *   auto dg = () { return &x; }
                 * Making it:
                 *   auto dg = () return { return &x; }
                 * Because dg.ptr points to x, this is returning dt.ptr+offset
                 */
                if (global.params.vsafe)
                    sc.func.storage_class |= STC.return_;
            }

        }

        /* Check for returning a ref variable by 'ref', but should be 'return ref'
         * Infer the addition of 'return', or set result to be the offending expression.
         */
        if ( (v.storage_class & (STC.ref_ | STC.out_)) &&
            !(v.storage_class & (STC.return_ | STC.foreach_)))
        {
            if (sc.func.flags & FUNCFLAG.returnInprocess && p == sc.func)
            {
                inferReturn(sc.func, v);        // infer addition of 'return'
            }
            else if (global.params.useDIP25 &&
                     sc._module && sc._module.isRoot())
            {
                // Only look for errors if in module listed on command line

                if (p == sc.func)
                {
                    //printf("escaping reference to local ref variable %s\n", v.toChars());
                    //printf("storage class = x%llx\n", v.storage_class);
                    escapingRef(v);
                    continue;
                }
                // Don't need to be concerned if v's parent does not return a ref
                FuncDeclaration fd = p.isFuncDeclaration();
                if (fd && fd.type && fd.type.ty == Tfunction)
                {
                    TypeFunction tf = cast(TypeFunction)fd.type;
                    if (tf.isref)
                    {
                        if (!gag)
                            error(e.loc, "escaping reference to outer local variable `%s`", v.toChars());
                        result = true;
                        continue;
                    }
                }

            }
        }
    }

    foreach (Expression ee; er.byexp)
    {
        if (log) printf("byexp %s\n", ee.toChars());
        if (!gag)
            error(ee.loc, "escaping reference to stack allocated value returned by `%s`", ee.toChars());
        result = true;
    }

    return result;
}


/*************************************
 * Variable v needs to have 'return' inferred for it.
 * Params:
 *      fd = function that v is a parameter to
 *      v = parameter that needs to be STC.return_
 */

private void inferReturn(FuncDeclaration fd, VarDeclaration v)
{
    // v is a local in the current function

    //printf("for function '%s' inferring 'return' for variable '%s'\n", fd.toChars(), v.toChars());
    v.storage_class |= STC.return_;

    TypeFunction tf = cast(TypeFunction)fd.type;
    if (v == fd.vthis)
    {
        /* v is the 'this' reference, so mark the function
         */
        fd.storage_class |= STC.return_;
        if (tf.ty == Tfunction)
        {
            //printf("'this' too %p %s\n", tf, sc.func.toChars());
            tf.isreturn = true;
        }
    }
    else
    {
        // Perform 'return' inference on parameter
        if (tf.ty == Tfunction && tf.parameters)
        {
            const dim = Parameter.dim(tf.parameters);
            foreach (const i; 0 .. dim)
            {
                Parameter p = Parameter.getNth(tf.parameters, i);
                if (p.ident == v.ident)
                {
                    p.storageClass |= STC.return_;
                    break;              // there can be only one
                }
            }
        }
    }
}


/****************************************
 * e is an expression to be returned by value, and that value contains pointers.
 * Walk e to determine which variables are possibly being
 * returned by value, such as:
 *      int* function(int* p) { return p; }
 * If e is a form of &p, determine which variables have content
 * which is being returned as ref, such as:
 *      int* function(int i) { return &i; }
 * Multiple variables can be inserted, because of expressions like this:
 *      int function(bool b, int i, int* p) { return b ? &i : p; }
 *
 * No side effects.
 *
 * Params:
 *      e = expression to be returned by value
 *      er = where to place collected data
 */
private void escapeByValue(Expression e, EscapeByResults* er)
{
    //printf("[%s] escapeByValue, e: %s\n", e.loc.toChars(), e.toChars());
    extern (C++) final class EscapeVisitor : Visitor
    {
        alias visit = Visitor.visit;
    public:
        EscapeByResults* er;

        extern (D) this(EscapeByResults* er)
        {
            this.er = er;
        }

        override void visit(Expression e)
        {
        }

        override void visit(AddrExp e)
        {
            escapeByRef(e.e1, er);
        }

        override void visit(SymOffExp e)
        {
            VarDeclaration v = e.var.isVarDeclaration();
            if (v)
                er.byref.push(v);
        }

        override void visit(VarExp e)
        {
            VarDeclaration v = e.var.isVarDeclaration();
            if (v)
                er.byvalue.push(v);
        }

        override void visit(ThisExp e)
        {
            if (e.var)
                er.byvalue.push(e.var);
        }

        override void visit(DotVarExp e)
        {
            auto t = e.e1.type.toBasetype();
            if (t.ty == Tstruct)
                e.e1.accept(this);
        }

        override void visit(DelegateExp e)
        {
            Type t = e.e1.type.toBasetype();
            if (t.ty == Tclass || t.ty == Tpointer)
                escapeByValue(e.e1, er);
            else
                escapeByRef(e.e1, er);
            er.byfunc.push(e.func);
        }

        override void visit(FuncExp e)
        {
            if (e.fd.tok == TOK.delegate_)
                er.byfunc.push(e.fd);
        }

        override void visit(TupleExp e)
        {
            assert(0); // should have been lowered by now
        }

        override void visit(ArrayLiteralExp e)
        {
            Type tb = e.type.toBasetype();
            if (tb.ty == Tsarray || tb.ty == Tarray)
            {
                if (e.basis)
                    e.basis.accept(this);
                foreach (el; *e.elements)
                {
                    if (el)
                        el.accept(this);
                }
            }
        }

        override void visit(StructLiteralExp e)
        {
            if (e.elements)
            {
                foreach (ex; *e.elements)
                {
                    if (ex)
                        ex.accept(this);
                }
            }
        }

        override void visit(NewExp e)
        {
            Type tb = e.newtype.toBasetype();
            if (tb.ty == Tstruct && !e.member && e.arguments)
            {
                foreach (ex; *e.arguments)
                {
                    if (ex)
                        ex.accept(this);
                }
            }
        }

        override void visit(CastExp e)
        {
            Type tb = e.type.toBasetype();
            if (tb.ty == Tarray && e.e1.type.toBasetype().ty == Tsarray)
            {
                escapeByRef(e.e1, er);
            }
            else
                e.e1.accept(this);
        }

        override void visit(SliceExp e)
        {
            if (e.e1.op == TOK.variable)
            {
                VarDeclaration v = (cast(VarExp)e.e1).var.isVarDeclaration();
                Type tb = e.type.toBasetype();
                if (v)
                {
                    if (tb.ty == Tsarray)
                        return;
                    if (v.storage_class & STC.variadic)
                    {
                        er.byvalue.push(v);
                        return;
                    }
                }
            }
            Type t1b = e.e1.type.toBasetype();
            if (t1b.ty == Tsarray)
            {
                Type tb = e.type.toBasetype();
                if (tb.ty != Tsarray)
                    escapeByRef(e.e1, er);
            }
            else
                e.e1.accept(this);
        }

        override void visit(IndexExp e)
        {
            if (e.type.hasPointers())
            {
                e.e1.accept(this);
            }
        }

        override void visit(BinExp e)
        {
            Type tb = e.type.toBasetype();
            if (tb.ty == Tpointer)
            {
                e.e1.accept(this);
                e.e2.accept(this);
            }
        }

        override void visit(BinAssignExp e)
        {
            e.e1.accept(this);
        }

        override void visit(AssignExp e)
        {
            e.e1.accept(this);
        }

        override void visit(CommaExp e)
        {
            e.e2.accept(this);
        }

        override void visit(CondExp e)
        {
            e.e1.accept(this);
            e.e2.accept(this);
        }

        override void visit(CallExp e)
        {
            //printf("CallExp(): %s\n", e.toChars());
            /* Check each argument that is
             * passed as 'return scope'.
             */
            Type t1 = e.e1.type.toBasetype();
            TypeFunction tf;
            TypeDelegate dg;
            if (t1.ty == Tdelegate)
            {
                dg = cast(TypeDelegate)t1;
                tf = cast(TypeFunction)(cast(TypeDelegate)t1).next;
            }
            else if (t1.ty == Tfunction)
                tf = cast(TypeFunction)t1;
            else
                return;

            if (e.arguments && e.arguments.dim)
            {
                /* j=1 if _arguments[] is first argument,
                 * skip it because it is not passed by ref
                 */
                int j = (tf.linkage == LINK.d && tf.varargs == 1);
                for (size_t i = j; i < e.arguments.dim; ++i)
                {
                    Expression arg = (*e.arguments)[i];
                    size_t nparams = Parameter.dim(tf.parameters);
                    if (i - j < nparams && i >= j)
                    {
                        Parameter p = Parameter.getNth(tf.parameters, i - j);
                        const stc = tf.parameterStorageClass(p);
                        if ((stc & (STC.scope_)) && (stc & STC.return_))
                            arg.accept(this);
                        else if ((stc & (STC.ref_)) && (stc & STC.return_))
                            escapeByRef(arg, er);
                    }
                }
            }
            // If 'this' is returned, check it too
            if (e.e1.op == TOK.dotVariable && t1.ty == Tfunction)
            {
                DotVarExp dve = cast(DotVarExp)e.e1;
                FuncDeclaration fd = dve.var.isFuncDeclaration();
                AggregateDeclaration ad;
                if (global.params.vsafe && tf.isreturn && fd && (ad = fd.isThis()) !is null)
                {
                    if (ad.isClassDeclaration() || tf.isscope)       // this is 'return scope'
                        dve.e1.accept(this);
                    else if (ad.isStructDeclaration()) // this is 'return ref'
                        escapeByRef(dve.e1, er);
                }
                else if (dve.var.storage_class & STC.return_ || tf.isreturn)
                {
                    if (dve.var.storage_class & STC.scope_)
                        dve.e1.accept(this);
                    else if (dve.var.storage_class & STC.ref_)
                        escapeByRef(dve.e1, er);
                }
            }

            /* If returning the result of a delegate call, the .ptr
             * field of the delegate must be checked.
             */
            if (dg)
            {
                if (tf.isreturn)
                    e.e1.accept(this);
            }
        }
    }

    scope EscapeVisitor v = new EscapeVisitor(er);
    e.accept(v);
}


/****************************************
 * e is an expression to be returned by 'ref'.
 * Walk e to determine which variables are possibly being
 * returned by ref, such as:
 *      ref int function(int i) { return i; }
 * If e is a form of *p, determine which variables have content
 * which is being returned as ref, such as:
 *      ref int function(int* p) { return *p; }
 * Multiple variables can be inserted, because of expressions like this:
 *      ref int function(bool b, int i, int* p) { return b ? i : *p; }
 *
 * No side effects.
 *
 * Params:
 *      e = expression to be returned by 'ref'
 *      er = where to place collected data
 */
private void escapeByRef(Expression e, EscapeByResults* er)
{
    //printf("[%s] escapeByRef, e: %s\n", e.loc.toChars(), e.toChars());
    extern (C++) final class EscapeRefVisitor : Visitor
    {
        alias visit = Visitor.visit;
    public:
        EscapeByResults* er;

        extern (D) this(EscapeByResults* er)
        {
            this.er = er;
        }

        override void visit(Expression e)
        {
        }

        override void visit(VarExp e)
        {
            auto v = e.var.isVarDeclaration();
            if (v)
            {
                if (v.storage_class & STC.ref_ && v.storage_class & (STC.foreach_ | STC.temp) && v._init)
                {
                    /* If compiler generated ref temporary
                     *   (ref v = ex; ex)
                     * look at the initializer instead
                     */
                    if (ExpInitializer ez = v._init.isExpInitializer())
                    {
                        assert(ez.exp && ez.exp.op == TOK.construct);
                        Expression ex = (cast(ConstructExp)ez.exp).e2;
                        ex.accept(this);
                    }
                }
                else
                    er.byref.push(v);
            }
        }

        override void visit(ThisExp e)
        {
            if (e.var)
                er.byref.push(e.var);
        }

        override void visit(PtrExp e)
        {
            escapeByValue(e.e1, er);
        }

        override void visit(IndexExp e)
        {
            Type tb = e.e1.type.toBasetype();
            if (e.e1.op == TOK.variable)
            {
                VarDeclaration v = (cast(VarExp)e.e1).var.isVarDeclaration();
                if (tb.ty == Tarray || tb.ty == Tsarray)
                {
                    if (v && v.storage_class & STC.variadic)
                    {
                        er.byref.push(v);
                        return;
                    }
                }
            }
            if (tb.ty == Tsarray)
            {
                e.e1.accept(this);
            }
            else if (tb.ty == Tarray)
            {
                escapeByValue(e.e1, er);
            }
        }

        override void visit(DotVarExp e)
        {
            Type t1b = e.e1.type.toBasetype();
            if (t1b.ty == Tclass)
                escapeByValue(e.e1, er);
            else
                e.e1.accept(this);
        }

        override void visit(BinAssignExp e)
        {
            e.e1.accept(this);
        }

        override void visit(AssignExp e)
        {
            e.e1.accept(this);
        }

        override void visit(CommaExp e)
        {
            e.e2.accept(this);
        }

        override void visit(CondExp e)
        {
            e.e1.accept(this);
            e.e2.accept(this);
        }

        override void visit(CallExp e)
        {
            /* If the function returns by ref, check each argument that is
             * passed as 'return ref'.
             */
            Type t1 = e.e1.type.toBasetype();
            TypeFunction tf;
            if (t1.ty == Tdelegate)
                tf = cast(TypeFunction)(cast(TypeDelegate)t1).next;
            else if (t1.ty == Tfunction)
                tf = cast(TypeFunction)t1;
            else
                return;
            if (tf.isref)
            {
                if (e.arguments && e.arguments.dim)
                {
                    /* j=1 if _arguments[] is first argument,
                     * skip it because it is not passed by ref
                     */
                    int j = (tf.linkage == LINK.d && tf.varargs == 1);
                    for (size_t i = j; i < e.arguments.dim; ++i)
                    {
                        Expression arg = (*e.arguments)[i];
                        size_t nparams = Parameter.dim(tf.parameters);
                        if (i - j < nparams && i >= j)
                        {
                            Parameter p = Parameter.getNth(tf.parameters, i - j);
                            const stc = tf.parameterStorageClass(p);
                            if ((stc & (STC.out_ | STC.ref_)) && (stc & STC.return_))
                                arg.accept(this);
                            else if ((stc & STC.scope_) && (stc & STC.return_))
                            {
                                if (arg.op == TOK.delegate_)
                                {
                                    DelegateExp de = cast(DelegateExp)arg;
                                    if (de.func.isNested())
                                        er.byexp.push(de);
                                }
                                else
                                    escapeByValue(arg, er);
                            }
                        }
                    }
                }
                // If 'this' is returned by ref, check it too
                if (e.e1.op == TOK.dotVariable && t1.ty == Tfunction)
                {
                    DotVarExp dve = cast(DotVarExp)e.e1;
                    if (dve.var.storage_class & STC.return_ || tf.isreturn)
                    {
                        if (dve.var.storage_class & STC.scope_ || tf.isscope)
                            escapeByValue(dve.e1, er);
                        else if (dve.var.storage_class & STC.ref_ || tf.isref)
                            dve.e1.accept(this);
                    }
                }
                // If it's a delegate, check it too
                if (e.e1.op == TOK.variable && t1.ty == Tdelegate)
                {
                    escapeByValue(e.e1, er);
                }
            }
            else
                er.byexp.push(e);
        }
    }

    scope EscapeRefVisitor v = new EscapeRefVisitor(er);
    e.accept(v);
}


/************************************
 * Aggregate the data collected by the escapeBy??() functions.
 */
private struct EscapeByResults
{
    VarDeclarations byref;      // array into which variables being returned by ref are inserted
    VarDeclarations byvalue;    // array into which variables with values containing pointers are inserted
    FuncDeclarations byfunc;    // nested functions that are turned into delegates
    Expressions byexp;          // array into which temporaries being returned by ref are inserted
}

/*************************
 * Find all variables accessed by this delegate that are
 * in functions enclosing it.
 * Params:
 *      fd = function
 *      vars = array to append found variables to
 */
void findAllOuterAccessedVariables(FuncDeclaration fd, VarDeclarations* vars)
{
    //printf("findAllOuterAccessedVariables(fd: %s)\n", fd.toChars());
    for (auto p = fd.parent; p; p = p.parent)
    {
        auto fdp = p.isFuncDeclaration();
        if (fdp)
        {
            foreach (v; fdp.closureVars)
            {
                foreach (const fdv; v.nestedrefs)
                {
                    if (fdv == fd)
                    {
                        //printf("accessed: %s, type %s\n", v.toChars(), v.type.toChars());
                        vars.push(v);
                    }
                }
            }
        }
    }
}
