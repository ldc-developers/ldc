// Compiler implementation of the D programming language
// Copyright (c) 1999-2016 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.dinterpret;

import core.stdc.stdio;
import core.stdc.string;
import ddmd.apply;
import ddmd.arraytypes;
import ddmd.attrib;
import ddmd.builtin;
import ddmd.constfold;
import ddmd.ctfeexpr;
import ddmd.dclass;
import ddmd.declaration;
import ddmd.dstruct;
import ddmd.dsymbol;
import ddmd.dtemplate;
import ddmd.errors;
import ddmd.expression;
import ddmd.func;
import ddmd.globals;
import ddmd.id;
import ddmd.identifier;
import ddmd.init;
import ddmd.mtype;
import ddmd.root.array;
import ddmd.root.rootobject;
import ddmd.statement;
import ddmd.tokens;
import ddmd.utf;
import ddmd.visitor;

enum CtfeGoal : int
{
    ctfeNeedRvalue,     // Must return an Rvalue (== CTFE value)
    ctfeNeedLvalue,     // Must return an Lvalue (== CTFE reference)
    ctfeNeedNothing,    // The return value is not required
}

alias ctfeNeedRvalue = CtfeGoal.ctfeNeedRvalue;
alias ctfeNeedLvalue = CtfeGoal.ctfeNeedLvalue;
alias ctfeNeedNothing = CtfeGoal.ctfeNeedNothing;

//debug = LOG;
//debug = LOGASSIGN;
//debug = LOGCOMPILE;
//debug = SHOWPERFORMANCE;

// Maximum allowable recursive function calls in CTFE
enum CTFE_RECURSION_LIMIT = 1000;

/**
 The values of all CTFE variables
 */
struct CtfeStack
{
private:
    /* The stack. Every declaration we encounter is pushed here,
     * together with the VarDeclaration, and the previous
     * stack address of that variable, so that we can restore it
     * when we leave the stack frame.
     * Note that when a function is forward referenced, the interpreter must
     * run semantic3, and that may start CTFE again with a NULL istate. Thus
     * the stack might not be empty when CTFE begins.
     *
     * Ctfe Stack addresses are just 0-based integers, but we save
     * them as 'void *' because Array can only do pointers.
     */
    Expressions values;         // values on the stack
    VarDeclarations vars;       // corresponding variables
    Array!(void*) savedId;      // id of the previous state of that var

    Array!(void*) frames;       // all previous frame pointers
    Expressions savedThis;      // all previous values of localThis

    /* Global constants get saved here after evaluation, so we never
     * have to redo them. This saves a lot of time and memory.
     */
    Expressions globalValues;   // values of global constants

    size_t framepointer;        // current frame pointer
    size_t maxStackPointer;     // most stack we've ever used
    Expression localThis;       // value of 'this', or NULL if none

public:
    extern (C++) size_t stackPointer()
    {
        return values.dim;
    }

    // The current value of 'this', or NULL if none
    extern (C++) Expression getThis()
    {
        return localThis;
    }

    // Largest number of stack positions we've used
    extern (C++) size_t maxStackUsage()
    {
        return maxStackPointer;
    }

    // Start a new stack frame, using the provided 'this'.
    extern (C++) void startFrame(Expression thisexp)
    {
        frames.push(cast(void*)cast(size_t)framepointer);
        savedThis.push(localThis);
        framepointer = stackPointer();
        localThis = thisexp;
    }

    extern (C++) void endFrame()
    {
        size_t oldframe = cast(size_t)frames[frames.dim - 1];
        localThis = savedThis[savedThis.dim - 1];
        popAll(framepointer);
        framepointer = oldframe;
        frames.setDim(frames.dim - 1);
        savedThis.setDim(savedThis.dim - 1);
    }

    extern (C++) bool isInCurrentFrame(VarDeclaration v)
    {
        if (v.isDataseg() && !v.isCTFE())
            return false; // It's a global
        return v.ctfeAdrOnStack >= framepointer;
    }

    extern (C++) Expression getValue(VarDeclaration v)
    {
        if ((v.isDataseg() || v.storage_class & STCmanifest) && !v.isCTFE())
        {
            assert(v.ctfeAdrOnStack >= 0 && v.ctfeAdrOnStack < globalValues.dim);
            return globalValues[v.ctfeAdrOnStack];
        }
        assert(v.ctfeAdrOnStack >= 0 && v.ctfeAdrOnStack < stackPointer());
        return values[v.ctfeAdrOnStack];
    }

    extern (C++) void setValue(VarDeclaration v, Expression e)
    {
        assert(!v.isDataseg() || v.isCTFE());
        assert(v.ctfeAdrOnStack >= 0 && v.ctfeAdrOnStack < stackPointer());
        values[v.ctfeAdrOnStack] = e;
    }

    extern (C++) void push(VarDeclaration v)
    {
        assert(!v.isDataseg() || v.isCTFE());
        if (v.ctfeAdrOnStack != cast(size_t)-1 && v.ctfeAdrOnStack >= framepointer)
        {
            // Already exists in this frame, reuse it.
            values[v.ctfeAdrOnStack] = null;
            return;
        }
        savedId.push(cast(void*)cast(size_t)v.ctfeAdrOnStack);
        v.ctfeAdrOnStack = cast(int)values.dim;
        vars.push(v);
        values.push(null);
    }

    extern (C++) void pop(VarDeclaration v)
    {
        assert(!v.isDataseg() || v.isCTFE());
        assert(!(v.storage_class & (STCref | STCout)));
        int oldid = v.ctfeAdrOnStack;
        v.ctfeAdrOnStack = cast(int)cast(size_t)savedId[oldid];
        if (v.ctfeAdrOnStack == values.dim - 1)
        {
            values.pop();
            vars.pop();
            savedId.pop();
        }
    }

    extern (C++) void popAll(size_t stackpointer)
    {
        if (stackPointer() > maxStackPointer)
            maxStackPointer = stackPointer();
        assert(values.dim >= stackpointer);
        for (size_t i = stackpointer; i < values.dim; ++i)
        {
            VarDeclaration v = vars[i];
            v.ctfeAdrOnStack = cast(int)cast(size_t)savedId[i];
        }
        values.setDim(stackpointer);
        vars.setDim(stackpointer);
        savedId.setDim(stackpointer);
    }

    extern (C++) void saveGlobalConstant(VarDeclaration v, Expression e)
    {
        assert(v._init && (v.isConst() || v.isImmutable() || v.storage_class & STCmanifest) && !v.isCTFE());
        v.ctfeAdrOnStack = cast(int)globalValues.dim;
        globalValues.push(e);
    }
}

struct InterState
{
    InterState* caller;     // calling function's InterState
    FuncDeclaration fd;     // function being interpreted
    Statement start;        // if !=NULL, start execution at this statement

    /* target of CTFEExp result; also
     * target of labelled CTFEExp or
     * CTFEExp. (null if no label).
     */
    Statement gotoTarget;
}

extern (C++) __gshared CtfeStack ctfeStack;

// CTFE diagnostic information
extern (C++) void printCtfePerformanceStats()
{
    debug (SHOWPERFORMANCE)
    {
        printf("        ---- CTFE Performance ----\n");
        printf("max call depth = %d\tmax stack = %d\n", CtfeStatus.maxCallDepth, ctfeStack.maxStackUsage());
        printf("array allocs = %d\tassignments = %d\n\n", CtfeStatus.numArrayAllocs, CtfeStatus.numAssignments);
    }
}

/***********************************************************
 * CTFE-object code for a single function
 *
 * Currently only counts the number of local variables in the function
 */
struct CompiledCtfeFunction
{
    FuncDeclaration func; // Function being compiled, NULL if global scope
    int numVars; // Number of variables declared in this function
    Loc callingloc;

    extern (D) this(FuncDeclaration f)
    {
        func = f;
    }

    extern (C++) void onDeclaration(VarDeclaration v)
    {
        //printf("%s CTFE declare %s\n", v.loc.toChars(), v.toChars());
        ++numVars;
    }

    extern (C++) void onExpression(Expression e)
    {
        extern (C++) final class VarWalker : StoppableVisitor
        {
            alias visit = super.visit;
        public:
            CompiledCtfeFunction* ccf;

            extern (D) this(CompiledCtfeFunction* ccf)
            {
                this.ccf = ccf;
            }

            override void visit(Expression e)
            {
            }

            override void visit(ErrorExp e)
            {
                // Currently there's a front-end bug: silent errors
                // can occur inside delegate literals inside is(typeof()).
                // Suppress the check in this case.
                if (global.gag && ccf.func)
                {
                    stop = 1;
                    return;
                }
                .error(e.loc, "CTFE internal error: ErrorExp in %s\n", ccf.func ? ccf.func.loc.toChars() : ccf.callingloc.toChars());
                assert(0);
            }

            override void visit(DeclarationExp e)
            {
                VarDeclaration v = e.declaration.isVarDeclaration();
                if (!v)
                    return;
                TupleDeclaration td = v.toAlias().isTupleDeclaration();
                if (td)
                {
                    if (!td.objects)
                        return;
                    for (size_t i = 0; i < td.objects.dim; ++i)
                    {
                        RootObject o = td.objects.tdata()[i];
                        Expression ex = isExpression(o);
                        DsymbolExp s = (ex && ex.op == TOKdsymbol) ? cast(DsymbolExp)ex : null;
                        assert(s);
                        VarDeclaration v2 = s.s.isVarDeclaration();
                        assert(v2);
                        if (!v2.isDataseg() || v2.isCTFE())
                            ccf.onDeclaration(v2);
                    }
                }
                else if (!(v.isDataseg() || v.storage_class & STCmanifest) || v.isCTFE())
                    ccf.onDeclaration(v);
                Dsymbol s = v.toAlias();
                if (s == v && !v.isStatic() && v._init)
                {
                    ExpInitializer ie = v._init.isExpInitializer();
                    if (ie)
                        ccf.onExpression(ie.exp);
                }
            }

            override void visit(IndexExp e)
            {
                if (e.lengthVar)
                    ccf.onDeclaration(e.lengthVar);
            }

            override void visit(SliceExp e)
            {
                if (e.lengthVar)
                    ccf.onDeclaration(e.lengthVar);
            }
        }

        scope VarWalker v = new VarWalker(&this);
        walkPostorder(e, v);
    }
}

extern (C++) final class CtfeCompiler : Visitor
{
    alias visit = super.visit;
public:
    CompiledCtfeFunction* ccf;

    extern (D) this(CompiledCtfeFunction* ccf)
    {
        this.ccf = ccf;
    }

    override void visit(Statement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s Statement::ctfeCompile %s\n", s.loc.toChars(), s.toChars());
        }
        assert(0);
    }

    override void visit(ExpStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s ExpStatement::ctfeCompile\n", s.loc.toChars());
        }
        if (s.exp)
            ccf.onExpression(s.exp);
    }

    override void visit(CompoundStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s CompoundStatement::ctfeCompile\n", s.loc.toChars());
        }
        for (size_t i = 0; i < s.statements.dim; i++)
        {
            Statement sx = (*s.statements)[i];
            if (sx)
                ctfeCompile(sx);
        }
    }

    override void visit(UnrolledLoopStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s UnrolledLoopStatement::ctfeCompile\n", s.loc.toChars());
        }
        for (size_t i = 0; i < s.statements.dim; i++)
        {
            Statement sx = (*s.statements)[i];
            if (sx)
                ctfeCompile(sx);
        }
    }

    override void visit(IfStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s IfStatement::ctfeCompile\n", s.loc.toChars());
        }
        ccf.onExpression(s.condition);
        if (s.ifbody)
            ctfeCompile(s.ifbody);
        if (s.elsebody)
            ctfeCompile(s.elsebody);
    }

    override void visit(ScopeStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s ScopeStatement::ctfeCompile\n", s.loc.toChars());
        }
        if (s.statement)
            ctfeCompile(s.statement);
    }

    override void visit(OnScopeStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s OnScopeStatement::ctfeCompile\n", s.loc.toChars());
        }
        // rewritten to try/catch/finally
        assert(0);
    }

    override void visit(DoStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s DoStatement::ctfeCompile\n", s.loc.toChars());
        }
        ccf.onExpression(s.condition);
        if (s._body)
            ctfeCompile(s._body);
    }

    override void visit(WhileStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s WhileStatement::ctfeCompile\n", s.loc.toChars());
        }
        // rewritten to ForStatement
        assert(0);
    }

    override void visit(ForStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s ForStatement::ctfeCompile\n", s.loc.toChars());
        }
        if (s._init)
            ctfeCompile(s._init);
        if (s.condition)
            ccf.onExpression(s.condition);
        if (s.increment)
            ccf.onExpression(s.increment);
        if (s._body)
            ctfeCompile(s._body);
    }

    override void visit(ForeachStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s ForeachStatement::ctfeCompile\n", s.loc.toChars());
        }
        // rewritten for ForStatement
        assert(0);
    }

    override void visit(SwitchStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s SwitchStatement::ctfeCompile\n", s.loc.toChars());
        }
        ccf.onExpression(s.condition);
        // Note that the body contains the the Case and Default
        // statements, so we only need to compile the expressions
        for (size_t i = 0; i < s.cases.dim; i++)
        {
            ccf.onExpression((*s.cases)[i].exp);
        }
        if (s._body)
            ctfeCompile(s._body);
    }

    override void visit(CaseStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s CaseStatement::ctfeCompile\n", s.loc.toChars());
        }
        if (s.statement)
            ctfeCompile(s.statement);
    }

    override void visit(DefaultStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s DefaultStatement::ctfeCompile\n", s.loc.toChars());
        }
        if (s.statement)
            ctfeCompile(s.statement);
    }

    override void visit(GotoDefaultStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s GotoDefaultStatement::ctfeCompile\n", s.loc.toChars());
        }
    }

    override void visit(GotoCaseStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s GotoCaseStatement::ctfeCompile\n", s.loc.toChars());
        }
    }

    override void visit(SwitchErrorStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s SwitchErrorStatement::ctfeCompile\n", s.loc.toChars());
        }
    }

    override void visit(ReturnStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s ReturnStatement::ctfeCompile\n", s.loc.toChars());
        }
        if (s.exp)
            ccf.onExpression(s.exp);
    }

    override void visit(BreakStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s BreakStatement::ctfeCompile\n", s.loc.toChars());
        }
    }

    override void visit(ContinueStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s ContinueStatement::ctfeCompile\n", s.loc.toChars());
        }
    }

    override void visit(WithStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s WithStatement::ctfeCompile\n", s.loc.toChars());
        }
        // If it is with(Enum) {...}, just execute the body.
        if (s.exp.op == TOKscope || s.exp.op == TOKtype)
        {
        }
        else
        {
            ccf.onDeclaration(s.wthis);
            ccf.onExpression(s.exp);
        }
        if (s._body)
            ctfeCompile(s._body);
    }

    override void visit(TryCatchStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s TryCatchStatement::ctfeCompile\n", s.loc.toChars());
        }
        if (s._body)
            ctfeCompile(s._body);
        for (size_t i = 0; i < s.catches.dim; i++)
        {
            Catch ca = (*s.catches)[i];
            if (ca.var)
                ccf.onDeclaration(ca.var);
            if (ca.handler)
                ctfeCompile(ca.handler);
        }
    }

    override void visit(TryFinallyStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s TryFinallyStatement::ctfeCompile\n", s.loc.toChars());
        }
        if (s._body)
            ctfeCompile(s._body);
        if (s.finalbody)
            ctfeCompile(s.finalbody);
    }

    override void visit(ThrowStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s ThrowStatement::ctfeCompile\n", s.loc.toChars());
        }
        ccf.onExpression(s.exp);
    }

    override void visit(GotoStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s GotoStatement::ctfeCompile\n", s.loc.toChars());
        }
    }

    override void visit(LabelStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s LabelStatement::ctfeCompile\n", s.loc.toChars());
        }
        if (s.statement)
            ctfeCompile(s.statement);
    }

    override void visit(ImportStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s ImportStatement::ctfeCompile\n", s.loc.toChars());
        }
        // Contains no variables or executable code
    }

    override void visit(ForeachRangeStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s ForeachRangeStatement::ctfeCompile\n", s.loc.toChars());
        }
        // rewritten for ForStatement
        assert(0);
    }

    override void visit(AsmStatement s)
    {
        debug (LOGCOMPILE)
        {
            printf("%s AsmStatement::ctfeCompile\n", s.loc.toChars());
        }
        // we can't compile asm statements
    }

    void ctfeCompile(Statement s)
    {
        s.accept(this);
    }
}

/*************************************
 * Compile this function for CTFE.
 * At present, this merely allocates variables.
 */
extern (C++) void ctfeCompile(FuncDeclaration fd)
{
    debug (LOGCOMPILE)
    {
        printf("\n%s FuncDeclaration::ctfeCompile %s\n", fd.loc.toChars(), fd.toChars());
    }
    assert(!fd.ctfeCode);
    assert(!fd.semantic3Errors);
    assert(fd.semanticRun == PASSsemantic3done);

    fd.ctfeCode = new CompiledCtfeFunction(fd);
    if (fd.parameters)
    {
        Type tb = fd.type.toBasetype();
        assert(tb.ty == Tfunction);
        for (size_t i = 0; i < fd.parameters.dim; i++)
        {
            VarDeclaration v = (*fd.parameters)[i];
            fd.ctfeCode.onDeclaration(v);
        }
    }
    if (fd.vresult)
        fd.ctfeCode.onDeclaration(fd.vresult);
    scope CtfeCompiler v = new CtfeCompiler(fd.ctfeCode);
    v.ctfeCompile(fd.fbody);
}

/*************************************
 * Entry point for CTFE.
 * A compile-time result is required. Give an error if not possible.
 *
 * `e` must be semantically valid expression. In other words, it should not
 * contain any `ErrorExp`s in it. But, CTFE interpretation will cross over
 * functions and may invoke a function that contains `ErrorStatement` in its body.
 * If that, the "CTFE failed because of previous errors" error is raised.
 */
extern (C++) Expression ctfeInterpret(Expression e)
{
    if (e.op == TOKerror)
        return e;
    assert(e.type); // https://issues.dlang.org/show_bug.cgi?id=14642
    //assert(e.type.ty != Terror);    // FIXME
    if (e.type.ty == Terror)
        return new ErrorExp();

    // This code is outside a function, but still needs to be compiled
    // (there are compiler-generated temporary variables such as __dollar).
    // However, this will only be run once and can then be discarded.
    auto ctfeCodeGlobal = CompiledCtfeFunction(null);
    ctfeCodeGlobal.callingloc = e.loc;
    ctfeCodeGlobal.onExpression(e);

    Expression result = interpret(e, null);

    if (!CTFEExp.isCantExp(result))
        result = scrubReturnValue(e.loc, result);
    if (CTFEExp.isCantExp(result))
        result = new ErrorExp();

    return result;
}

/* Run CTFE on the expression, but allow the expression to be a TypeExp
 *  or a tuple containing a TypeExp. (This is required by pragma(msg)).
 */
extern (C++) Expression ctfeInterpretForPragmaMsg(Expression e)
{
    if (e.op == TOKerror || e.op == TOKtype)
        return e;

    // It's also OK for it to be a function declaration (happens only with
    // __traits(getOverloads))
    if (e.op == TOKvar && (cast(VarExp)e).var.isFuncDeclaration())
    {
        return e;
    }

    if (e.op != TOKtuple)
        return e.ctfeInterpret();

    // Tuples need to be treated seperately, since they are
    // allowed to contain a TypeExp in this case.

    TupleExp tup = cast(TupleExp)e;
    Expressions* expsx = null;
    for (size_t i = 0; i < tup.exps.dim; ++i)
    {
        Expression g = (*tup.exps)[i];
        Expression h = g;
        h = ctfeInterpretForPragmaMsg(g);
        if (h != g)
        {
            if (!expsx)
            {
                expsx = new Expressions();
                expsx.setDim(tup.exps.dim);
                for (size_t j = 0; j < tup.exps.dim; j++)
                    (*expsx)[j] = (*tup.exps)[j];
            }
            (*expsx)[i] = h;
        }
    }
    if (expsx)
    {
        auto te = new TupleExp(e.loc, expsx);
        expandTuples(te.exps);
        te.type = new TypeTuple(te.exps);
        return te;
    }
    return e;
}

/*************************************
 * Attempt to interpret a function given the arguments.
 * Input:
 *      istate     state for calling function (NULL if none)
 *      arguments  function arguments
 *      thisarg    'this', if a needThis() function, NULL if not.
 *
 * Return result expression if successful, TOKcantexp if not,
 * or CTFEExp if function returned void.
 */
extern (C++) Expression interpret(FuncDeclaration fd, InterState* istate, Expressions* arguments, Expression thisarg)
{
    debug (LOG)
    {
        printf("\n********\n%s FuncDeclaration::interpret(istate = %p) %s\n", fd.loc.toChars(), istate, fd.toChars());
    }
    if (fd.semanticRun == PASSsemantic3)
    {
        fd.error("circular dependency. Functions cannot be interpreted while being compiled");
        return CTFEExp.cantexp;
    }
    if (!fd.functionSemantic3())
        return CTFEExp.cantexp;
    if (fd.semanticRun < PASSsemantic3done)
        return CTFEExp.cantexp;

    // CTFE-compile the function
    if (!fd.ctfeCode)
        ctfeCompile(fd);

    Type tb = fd.type.toBasetype();
    assert(tb.ty == Tfunction);
    TypeFunction tf = cast(TypeFunction)tb;
    if (tf.varargs && arguments && ((fd.parameters && arguments.dim != fd.parameters.dim) || (!fd.parameters && arguments.dim)))
    {
        fd.error("C-style variadic functions are not yet implemented in CTFE");
        return CTFEExp.cantexp;
    }

    // Nested functions always inherit the 'this' pointer from the parent,
    // except for delegates. (Note that the 'this' pointer may be null).
    // Func literals report isNested() even if they are in global scope,
    // so we need to check that the parent is a function.
    if (fd.isNested() && fd.toParent2().isFuncDeclaration() && !thisarg && istate)
        thisarg = ctfeStack.getThis();

    if (fd.needThis() && !thisarg)
    {
        // error, no this. Prevent segfault.
        // Here should be unreachable by the strict 'this' check in front-end.
        fd.error("need 'this' to access member %s", fd.toChars());
        return CTFEExp.cantexp;
    }

    // Place to hold all the arguments to the function while
    // we are evaluating them.
    Expressions eargs;
    size_t dim = arguments ? arguments.dim : 0;
    assert((fd.parameters ? fd.parameters.dim : 0) == dim);

    /* Evaluate all the arguments to the function,
     * store the results in eargs[]
     */
    eargs.setDim(dim);
    for (size_t i = 0; i < dim; i++)
    {
        Expression earg = (*arguments)[i];
        Parameter fparam = Parameter.getNth(tf.parameters, i);

        if (fparam.storageClass & (STCout | STCref))
        {
            if (!istate && (fparam.storageClass & STCout))
            {
                // initializing an out parameter involves writing to it.
                earg.error("global %s cannot be passed as an 'out' parameter at compile time", earg.toChars());
                return CTFEExp.cantexp;
            }
            // Convert all reference arguments into lvalue references
            earg = interpret(earg, istate, ctfeNeedLvalue);
            if (CTFEExp.isCantExp(earg))
                return earg;
        }
        else if (fparam.storageClass & STClazy)
        {
        }
        else
        {
            /* Value parameters
             */
            Type ta = fparam.type.toBasetype();
            if (ta.ty == Tsarray && earg.op == TOKaddress)
            {
                /* Static arrays are passed by a simple pointer.
                 * Skip past this to get at the actual arg.
                 */
                earg = (cast(AddrExp)earg).e1;
            }
            earg = interpret(earg, istate);
            if (CTFEExp.isCantExp(earg))
                return earg;

            /* Struct literals are passed by value, but we don't need to
             * copy them if they are passed as const
             */
            if (earg.op == TOKstructliteral && !(fparam.storageClass & (STCconst | STCimmutable)))
                earg = copyLiteral(earg).copy();
        }
        if (earg.op == TOKthrownexception)
        {
            if (istate)
                return earg;
            (cast(ThrownExceptionExp)earg).generateUncaughtError();
            return CTFEExp.cantexp;
        }
        eargs[i] = earg;
    }

    // Now that we've evaluated all the arguments, we can start the frame
    // (this is the moment when the 'call' actually takes place).
    InterState istatex;
    istatex.caller = istate;
    istatex.fd = fd;
    ctfeStack.startFrame(thisarg);
    if (fd.vthis && thisarg)
    {
        ctfeStack.push(fd.vthis);
        setValue(fd.vthis, thisarg);
    }

    for (size_t i = 0; i < dim; i++)
    {
        Expression earg = eargs[i];
        Parameter fparam = Parameter.getNth(tf.parameters, i);
        VarDeclaration v = (*fd.parameters)[i];
        debug (LOG)
        {
            printf("arg[%d] = %s\n", i, earg.toChars());
        }
        ctfeStack.push(v);

        if ((fparam.storageClass & (STCout | STCref)) && earg.op == TOKvar && (cast(VarExp)earg).var.toParent2() == fd)
        {
            VarDeclaration vx = (cast(VarExp)earg).var.isVarDeclaration();
            if (!vx)
            {
                fd.error("cannot interpret %s as a ref parameter", earg.toChars());
                return CTFEExp.cantexp;
            }

            /* vx is a variable that is declared in fd.
             * It means that fd is recursively called. e.g.
             *
             *  void fd(int n, ref int v = dummy) {
             *      int vx;
             *      if (n == 1) fd(2, vx);
             *  }
             *  fd(1);
             *
             * The old value of vx on the stack in fd(1)
             * should be saved at the start of fd(2, vx) call.
             */
            int oldadr = vx.ctfeAdrOnStack;

            ctfeStack.push(vx);
            assert(!hasValue(vx)); // vx is made uninitialized

            // https://issues.dlang.org/show_bug.cgi?id=14299
            // v.ctfeAdrOnStack should be saved already
            // in the stack before the overwrite.
            v.ctfeAdrOnStack = oldadr;
            assert(hasValue(v)); // ref parameter v should refer existing value.
        }
        else
        {
            // Value parameters and non-trivial references
            setValueWithoutChecking(v, earg);
        }
        debug (LOG)
        {
            printf("interpreted arg[%d] = %s\n", i, earg.toChars());
            showCtfeExpr(earg);
        }
        debug (LOGASSIGN)
        {
            printf("interpreted arg[%d] = %s\n", i, earg.toChars());
            showCtfeExpr(earg);
        }
    }

    if (fd.vresult)
        ctfeStack.push(fd.vresult);

    // Enter the function
    ++CtfeStatus.callDepth;
    if (CtfeStatus.callDepth > CtfeStatus.maxCallDepth)
        CtfeStatus.maxCallDepth = CtfeStatus.callDepth;

    Expression e = null;
    while (1)
    {
        if (CtfeStatus.callDepth > CTFE_RECURSION_LIMIT)
        {
            // This is a compiler error. It must not be suppressed.
            global.gag = 0;
            fd.error("CTFE recursion limit exceeded");
            e = CTFEExp.cantexp;
            break;
        }
        e = interpret(fd.fbody, &istatex);
        if (CTFEExp.isCantExp(e))
        {
            debug (LOG)
            {
                printf("function body failed to interpret\n");
            }
        }

        if (istatex.start)
        {
            fd.error("CTFE internal error: failed to resume at statement %s", istatex.start.toChars());
            return CTFEExp.cantexp;
        }

        /* This is how we deal with a recursive statement AST
         * that has arbitrary goto statements in it.
         * Bubble up a 'result' which is the target of the goto
         * statement, then go recursively down the AST looking
         * for that statement, then execute starting there.
         */
        if (CTFEExp.isGotoExp(e))
        {
            istatex.start = istatex.gotoTarget; // set starting statement
            istatex.gotoTarget = null;
        }
        else
        {
            assert(!e || (e.op != TOKcontinue && e.op != TOKbreak));
            break;
        }
    }
    // If fell off the end of a void function, return void
    if (!e && tf.next.ty == Tvoid)
        e = CTFEExp.voidexp;
    if (tf.isref && e.op == TOKvar && (cast(VarExp)e).var == fd.vthis)
        e = thisarg;
    assert(e !is null);

    // Leave the function
    --CtfeStatus.callDepth;

    ctfeStack.endFrame();

    // If it generated an uncaught exception, report error.
    if (!istate && e.op == TOKthrownexception)
    {
        (cast(ThrownExceptionExp)e).generateUncaughtError();
        e = CTFEExp.cantexp;
    }

    return e;
}

extern (C++) final class Interpreter : Visitor
{
    alias visit = super.visit;
public:
    InterState* istate;
    CtfeGoal goal;
    Expression result;

    extern (D) this(InterState* istate, CtfeGoal goal)
    {
        this.istate = istate;
        this.goal = goal;
    }

    // If e is TOKthrowexception or TOKcantexp,
    // set it to 'result' and returns true.
    bool exceptionOrCant(Expression e)
    {
        if (exceptionOrCantInterpret(e))
        {
            result = e;
            return true;
        }
        return false;
    }

    static Expressions* copyArrayOnWrite(Expressions* exps, Expressions* original)
    {
        if (exps is original)
        {
            if (!original)
                exps = new Expressions();
            else
                exps = original.copy();
            ++CtfeStatus.numArrayAllocs;
        }
        return exps;
    }

    /******************************** Statement ***************************/

    override void visit(Statement s)
    {
        debug (LOG)
        {
            printf("%s Statement::interpret()\n", s.loc.toChars());
        }
        if (istate.start)
        {
            if (istate.start != s)
                return;
            istate.start = null;
        }

        s.error("statement %s cannot be interpreted at compile time", s.toChars());
        result = CTFEExp.cantexp;
    }

    override void visit(ExpStatement s)
    {
        debug (LOG)
        {
            printf("%s ExpStatement::interpret(%s)\n", s.loc.toChars(), s.exp ? s.exp.toChars() : "");
        }
        if (istate.start)
        {
            if (istate.start != s)
                return;
            istate.start = null;
        }

        Expression e = interpret(s.exp, istate, ctfeNeedNothing);
        if (exceptionOrCant(e))
            return;
    }

    override void visit(CompoundStatement s)
    {
        debug (LOG)
        {
            printf("%s CompoundStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start == s)
            istate.start = null;

        size_t dim = s.statements ? s.statements.dim : 0;
        for (size_t i = 0; i < dim; i++)
        {
            Statement sx = (*s.statements)[i];
            result = interpret(sx, istate);
            if (result)
                break;
        }
        debug (LOG)
        {
            printf("%s -CompoundStatement::interpret() %p\n", s.loc.toChars(), result);
        }
    }

    override void visit(UnrolledLoopStatement s)
    {
        debug (LOG)
        {
            printf("%s UnrolledLoopStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start == s)
            istate.start = null;

        size_t dim = s.statements ? s.statements.dim : 0;
        for (size_t i = 0; i < dim; i++)
        {
            Statement sx = (*s.statements)[i];
            Expression e = interpret(sx, istate);
            if (!e) // suceeds to interpret, or goto target was not found
                continue;
            if (exceptionOrCant(e))
                return;
            if (e.op == TOKbreak)
            {
                if (istate.gotoTarget && istate.gotoTarget != s)
                {
                    result = e; // break at a higher level
                    return;
                }
                istate.gotoTarget = null;
                result = null;
                return;
            }
            if (e.op == TOKcontinue)
            {
                if (istate.gotoTarget && istate.gotoTarget != s)
                {
                    result = e; // continue at a higher level
                    return;
                }
                istate.gotoTarget = null;
                continue;
            }

            // expression from return statement, or thrown exception
            result = e;
            break;
        }
    }

    override void visit(IfStatement s)
    {
        debug (LOG)
        {
            printf("%s IfStatement::interpret(%s)\n", s.loc.toChars(), s.condition.toChars());
        }
        if (istate.start == s)
            istate.start = null;
        if (istate.start)
        {
            Expression e = null;
            e = interpret(s.ifbody, istate);
            if (!e && istate.start)
                e = interpret(s.elsebody, istate);
            result = e;
            return;
        }

        Expression e = interpret(s.condition, istate);
        assert(e);
        if (exceptionOrCant(e))
            return;

        if (isTrueBool(e))
            result = interpret(s.ifbody, istate);
        else if (e.isBool(false))
            result = interpret(s.elsebody, istate);
        else
        {
            // no error, or assert(0)?
            result = CTFEExp.cantexp;
        }
    }

    override void visit(ScopeStatement s)
    {
        debug (LOG)
        {
            printf("%s ScopeStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start == s)
            istate.start = null;

        result = interpret(s.statement, istate);
    }

    /**
     Given an expression e which is about to be returned from the current
     function, generate an error if it contains pointers to local variables.
     Return true if it is safe to return, false if an error was generated.

     Only checks expressions passed by value (pointers to local variables
     may already be stored in members of classes, arrays, or AAs which
     were passed as mutable function parameters).
     */
    static bool stopPointersEscaping(Loc loc, Expression e)
    {
        if (!e.type.hasPointers())
            return true;
        if (isPointer(e.type))
        {
            Expression x = e;
            if (e.op == TOKaddress)
                x = (cast(AddrExp)e).e1;
            VarDeclaration v;
            while (x.op == TOKvar && (v = (cast(VarExp)x).var.isVarDeclaration()) !is null)
            {
                if (v.storage_class & STCref)
                {
                    x = getValue(v);
                    if (e.op == TOKaddress)
                        (cast(AddrExp)e).e1 = x;
                    continue;
                }
                if (ctfeStack.isInCurrentFrame(v))
                {
                    error(loc, "returning a pointer to a local stack variable");
                    return false;
                }
                else
                    break;
            }
            // TODO: If it is a TOKdotvar or TOKindex, we should check that it is not
            // pointing to a local struct or static array.
        }
        if (e.op == TOKstructliteral)
        {
            StructLiteralExp se = cast(StructLiteralExp)e;
            return stopPointersEscapingFromArray(loc, se.elements);
        }
        if (e.op == TOKarrayliteral)
        {
            return stopPointersEscapingFromArray(loc, (cast(ArrayLiteralExp)e).elements);
        }
        if (e.op == TOKassocarrayliteral)
        {
            AssocArrayLiteralExp aae = cast(AssocArrayLiteralExp)e;
            if (!stopPointersEscapingFromArray(loc, aae.keys))
                return false;
            return stopPointersEscapingFromArray(loc, aae.values);
        }
        return true;
    }

    // Check all members of an array for escaping local variables. Return false if error
    static bool stopPointersEscapingFromArray(Loc loc, Expressions* elems)
    {
        for (size_t i = 0; i < elems.dim; i++)
        {
            Expression m = (*elems)[i];
            if (!m)
                continue;
            if (!stopPointersEscaping(loc, m))
                return false;
        }
        return true;
    }

    override void visit(ReturnStatement s)
    {
        debug (LOG)
        {
            printf("%s ReturnStatement::interpret(%s)\n", s.loc.toChars(), s.exp ? s.exp.toChars() : "");
        }
        if (istate.start)
        {
            if (istate.start != s)
                return;
            istate.start = null;
        }

        if (!s.exp)
        {
            result = CTFEExp.voidexp;
            return;
        }

        assert(istate && istate.fd && istate.fd.type && istate.fd.type.ty == Tfunction);
        TypeFunction tf = cast(TypeFunction)istate.fd.type;

        /* If the function returns a ref AND it's been called from an assignment,
         * we need to return an lvalue. Otherwise, just do an (rvalue) interpret.
         */
        if (tf.isref)
        {
            result = interpret(s.exp, istate, ctfeNeedLvalue);
            return;
        }
        if (tf.next && tf.next.ty == Tdelegate && istate.fd.closureVars.dim > 0)
        {
            // To support this, we need to copy all the closure vars
            // into the delegate literal.
            s.error("closures are not yet supported in CTFE");
            result = CTFEExp.cantexp;
            return;
        }

        // We need to treat pointers specially, because TOKsymoff can be used to
        // return a value OR a pointer
        Expression e = interpret(s.exp, istate);
        if (exceptionOrCant(e))
            return;

        // Disallow returning pointers to stack-allocated variables (bug 7876)
        if (!stopPointersEscaping(s.loc, e))
        {
            result = CTFEExp.cantexp;
            return;
        }

        if (needToCopyLiteral(e))
            e = copyLiteral(e).copy();
        debug (LOGASSIGN)
        {
            printf("RETURN %s\n", s.loc.toChars());
            showCtfeExpr(e);
        }
        result = e;
    }

    static Statement findGotoTarget(InterState* istate, Identifier ident)
    {
        Statement target = null;
        if (ident)
        {
            LabelDsymbol label = istate.fd.searchLabel(ident);
            assert(label && label.statement);
            LabelStatement ls = label.statement;
            target = ls.gotoTarget ? ls.gotoTarget : ls.statement;
        }
        return target;
    }

    override void visit(BreakStatement s)
    {
        debug (LOG)
        {
            printf("%s BreakStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start)
        {
            if (istate.start != s)
                return;
            istate.start = null;
        }

        istate.gotoTarget = findGotoTarget(istate, s.ident);
        result = CTFEExp.breakexp;
    }

    override void visit(ContinueStatement s)
    {
        debug (LOG)
        {
            printf("%s ContinueStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start)
        {
            if (istate.start != s)
                return;
            istate.start = null;
        }

        istate.gotoTarget = findGotoTarget(istate, s.ident);
        result = CTFEExp.continueexp;
    }

    override void visit(WhileStatement s)
    {
        debug (LOG)
        {
            printf("WhileStatement::interpret()\n");
        }
        assert(0); // rewritten to ForStatement
    }

    override void visit(DoStatement s)
    {
        debug (LOG)
        {
            printf("%s DoStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start == s)
            istate.start = null;

        while (1)
        {
            Expression e = interpret(s._body, istate);
            if (!e && istate.start) // goto target was not found
                return;
            assert(!istate.start);

            if (exceptionOrCant(e))
                return;
            if (e && e.op == TOKbreak)
            {
                if (istate.gotoTarget && istate.gotoTarget != s)
                {
                    result = e; // break at a higher level
                    return;
                }
                istate.gotoTarget = null;
                break;
            }
            if (e && e.op == TOKcontinue)
            {
                if (istate.gotoTarget && istate.gotoTarget != s)
                {
                    result = e; // continue at a higher level
                    return;
                }
                istate.gotoTarget = null;
                e = null;
            }
            if (e)
            {
                result = e; // bubbled up from ReturnStatement
                return;
            }

            e = interpret(s.condition, istate);
            if (exceptionOrCant(e))
                return;
            if (!e.isConst())
            {
                result = CTFEExp.cantexp;
                return;
            }
            if (e.isBool(false))
                break;
            assert(isTrueBool(e));
        }
        assert(result is null);
    }

    override void visit(ForStatement s)
    {
        debug (LOG)
        {
            printf("%s ForStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start == s)
            istate.start = null;

        Expression ei = interpret(s._init, istate);
        if (exceptionOrCant(ei))
            return;
        assert(!ei); // s.init never returns from function, or jumps out from it

        while (1)
        {
            if (s.condition && !istate.start)
            {
                Expression e = interpret(s.condition, istate);
                if (exceptionOrCant(e))
                    return;
                if (e.isBool(false))
                    break;
                assert(isTrueBool(e));
            }

            Expression e = interpret(s._body, istate);
            if (!e && istate.start) // goto target was not found
                return;
            assert(!istate.start);

            if (exceptionOrCant(e))
                return;
            if (e && e.op == TOKbreak)
            {
                if (istate.gotoTarget && istate.gotoTarget != s)
                {
                    result = e; // break at a higher level
                    return;
                }
                istate.gotoTarget = null;
                break;
            }
            if (e && e.op == TOKcontinue)
            {
                if (istate.gotoTarget && istate.gotoTarget != s)
                {
                    result = e; // continue at a higher level
                    return;
                }
                istate.gotoTarget = null;
                e = null;
            }
            if (e)
            {
                result = e; // bubbled up from ReturnStatement
                return;
            }

            e = interpret(s.increment, istate, ctfeNeedNothing);
            if (exceptionOrCant(e))
                return;
        }
        assert(result is null);
    }

    override void visit(ForeachStatement s)
    {
        assert(0); // rewritten to ForStatement
    }

    override void visit(ForeachRangeStatement s)
    {
        assert(0); // rewritten to ForStatement
    }

    override void visit(SwitchStatement s)
    {
        debug (LOG)
        {
            printf("%s SwitchStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start == s)
            istate.start = null;
        if (istate.start)
        {
            Expression e = interpret(s._body, istate);
            if (istate.start) // goto target was not found
                return;
            if (exceptionOrCant(e))
                return;
            if (e && e.op == TOKbreak)
            {
                if (istate.gotoTarget && istate.gotoTarget != s)
                {
                    result = e; // break at a higher level
                    return;
                }
                istate.gotoTarget = null;
                e = null;
            }
            result = e;
            return;
        }

        Expression econdition = interpret(s.condition, istate);
        if (exceptionOrCant(econdition))
            return;

        Statement scase = null;
        size_t dim = s.cases ? s.cases.dim : 0;
        for (size_t i = 0; i < dim; i++)
        {
            CaseStatement cs = (*s.cases)[i];
            Expression ecase = interpret(cs.exp, istate);
            if (exceptionOrCant(ecase))
                return;
            if (ctfeEqual(cs.exp.loc, TOKequal, econdition, ecase))
            {
                scase = cs;
                break;
            }
        }
        if (!scase)
        {
            if (s.hasNoDefault)
                s.error("no default or case for %s in switch statement", econdition.toChars());
            scase = s.sdefault;
        }

        assert(scase);

        /* Jump to scase
         */
        istate.start = scase;
        Expression e = interpret(s._body, istate);
        assert(!istate.start); // jump must not fail
        if (e && e.op == TOKbreak)
        {
            if (istate.gotoTarget && istate.gotoTarget != s)
            {
                result = e; // break at a higher level
                return;
            }
            istate.gotoTarget = null;
            e = null;
        }
        result = e;
    }

    override void visit(CaseStatement s)
    {
        debug (LOG)
        {
            printf("%s CaseStatement::interpret(%s) this = %p\n", s.loc.toChars(), s.exp.toChars(), s);
        }
        if (istate.start == s)
            istate.start = null;

        result = interpret(s.statement, istate);
    }

    override void visit(DefaultStatement s)
    {
        debug (LOG)
        {
            printf("%s DefaultStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start == s)
            istate.start = null;

        result = interpret(s.statement, istate);
    }

    override void visit(GotoStatement s)
    {
        debug (LOG)
        {
            printf("%s GotoStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start)
        {
            if (istate.start != s)
                return;
            istate.start = null;
        }

        assert(s.label && s.label.statement);
        istate.gotoTarget = s.label.statement;
        result = CTFEExp.gotoexp;
    }

    override void visit(GotoCaseStatement s)
    {
        debug (LOG)
        {
            printf("%s GotoCaseStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start)
        {
            if (istate.start != s)
                return;
            istate.start = null;
        }

        assert(s.cs);
        istate.gotoTarget = s.cs;
        result = CTFEExp.gotoexp;
    }

    override void visit(GotoDefaultStatement s)
    {
        debug (LOG)
        {
            printf("%s GotoDefaultStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start)
        {
            if (istate.start != s)
                return;
            istate.start = null;
        }

        assert(s.sw && s.sw.sdefault);
        istate.gotoTarget = s.sw.sdefault;
        result = CTFEExp.gotoexp;
    }

    override void visit(LabelStatement s)
    {
        debug (LOG)
        {
            printf("%s LabelStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start == s)
            istate.start = null;

        result = interpret(s.statement, istate);
    }

    override void visit(TryCatchStatement s)
    {
        debug (LOG)
        {
            printf("%s TryCatchStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start == s)
            istate.start = null;
        if (istate.start)
        {
            Expression e = null;
            e = interpret(s._body, istate);
            for (size_t i = 0; i < s.catches.dim; i++)
            {
                if (e || !istate.start) // goto target was found
                    break;
                Catch ca = (*s.catches)[i];
                e = interpret(ca.handler, istate);
            }
            result = e;
            return;
        }

        Expression e = interpret(s._body, istate);

        // An exception was thrown
        if (e && e.op == TOKthrownexception)
        {
            ThrownExceptionExp ex = cast(ThrownExceptionExp)e;
            Type extype = ex.thrown.originalClass().type;

            // Search for an appropriate catch clause.
            for (size_t i = 0; i < s.catches.dim; i++)
            {
                Catch ca = (*s.catches)[i];
                Type catype = ca.type;
                if (!catype.equals(extype) && !catype.isBaseOf(extype, null))
                    continue;

                // Execute the handler
                if (ca.var)
                {
                    ctfeStack.push(ca.var);
                    setValue(ca.var, ex.thrown);
                }
                e = interpret(ca.handler, istate);
                if (CTFEExp.isGotoExp(e))
                {
                    /* This is an optimization that relies on the locality of the jump target.
                     * If the label is in the same catch handler, the following scan
                     * would find it quickly and can reduce jump cost.
                     * Otherwise, the catch block may be unnnecessary scanned again
                     * so it would make CTFE speed slower.
                     */
                    InterState istatex = *istate;
                    istatex.start = istate.gotoTarget; // set starting statement
                    istatex.gotoTarget = null;
                    Expression eh = interpret(ca.handler, &istatex);
                    if (!istatex.start)
                    {
                        istate.gotoTarget = null;
                        e = eh;
                    }
                }
                break;
            }
        }
        result = e;
    }

    static bool isAnErrorException(ClassDeclaration cd)
    {
        return cd == ClassDeclaration.errorException || ClassDeclaration.errorException.isBaseOf(cd, null);
    }

    static ThrownExceptionExp chainExceptions(ThrownExceptionExp oldest, ThrownExceptionExp newest)
    {
        debug (LOG)
        {
            printf("Collided exceptions %s %s\n", oldest.thrown.toChars(), newest.thrown.toChars());
        }
        // Little sanity check to make sure it's really a Throwable
        ClassReferenceExp boss = oldest.thrown;
        assert((*boss.value.elements)[4].type.ty == Tclass); // Throwable.next
        ClassReferenceExp collateral = newest.thrown;
        if (isAnErrorException(collateral.originalClass()) && !isAnErrorException(boss.originalClass()))
        {
            // The new exception bypass the existing chain
            assert((*collateral.value.elements)[5].type.ty == Tclass);
            (*collateral.value.elements)[5] = boss;
            return newest;
        }
        while ((*boss.value.elements)[4].op == TOKclassreference)
        {
            boss = cast(ClassReferenceExp)(*boss.value.elements)[4];
        }
        (*boss.value.elements)[4] = collateral;
        return oldest;
    }

    override void visit(TryFinallyStatement s)
    {
        debug (LOG)
        {
            printf("%s TryFinallyStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start == s)
            istate.start = null;
        if (istate.start)
        {
            Expression e = null;
            e = interpret(s._body, istate);
            // Jump into/out from finalbody is disabled in semantic analysis.
            // and jump inside will be handled by the ScopeStatement == finalbody.
            result = e;
            return;
        }

        Expression ex = interpret(s._body, istate);
        if (CTFEExp.isCantExp(ex))
        {
            result = ex;
            return;
        }
        while (CTFEExp.isGotoExp(ex))
        {
            // If the goto target is within the body, we must not interpret the finally statement,
            // because that will call destructors for objects within the scope, which we should not do.
            InterState istatex = *istate;
            istatex.start = istate.gotoTarget; // set starting statement
            istatex.gotoTarget = null;
            Expression bex = interpret(s._body, &istatex);
            if (istatex.start)
            {
                // The goto target is outside the current scope.
                break;
            }
            // The goto target was within the body.
            if (CTFEExp.isCantExp(bex))
            {
                result = bex;
                return;
            }
            *istate = istatex;
            ex = bex;
        }

        Expression ey = interpret(s.finalbody, istate);
        if (CTFEExp.isCantExp(ey))
        {
            result = ey;
            return;
        }
        if (ey && ey.op == TOKthrownexception)
        {
            // Check for collided exceptions
            if (ex && ex.op == TOKthrownexception)
                ex = chainExceptions(cast(ThrownExceptionExp)ex, cast(ThrownExceptionExp)ey);
            else
                ex = ey;
        }
        result = ex;
    }

    override void visit(ThrowStatement s)
    {
        debug (LOG)
        {
            printf("%s ThrowStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start)
        {
            if (istate.start != s)
                return;
            istate.start = null;
        }

        Expression e = interpret(s.exp, istate);
        if (exceptionOrCant(e))
            return;

        assert(e.op == TOKclassreference);
        result = new ThrownExceptionExp(s.loc, cast(ClassReferenceExp)e);
    }

    override void visit(OnScopeStatement s)
    {
        assert(0);
    }

    override void visit(WithStatement s)
    {
        debug (LOG)
        {
            printf("%s WithStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start == s)
            istate.start = null;
        if (istate.start)
        {
            result = s._body ? interpret(s._body, istate) : null;
            return;
        }

        // If it is with(Enum) {...}, just execute the body.
        if (s.exp.op == TOKscope || s.exp.op == TOKtype)
        {
            result = interpret(s._body, istate);
            return;
        }

        Expression e = interpret(s.exp, istate);
        if (exceptionOrCant(e))
            return;

        if (s.wthis.type.ty == Tpointer && s.exp.type.ty != Tpointer)
        {
            e = new AddrExp(s.loc, e);
            e.type = s.wthis.type;
        }
        ctfeStack.push(s.wthis);
        setValue(s.wthis, e);
        e = interpret(s._body, istate);
        if (CTFEExp.isGotoExp(e))
        {
            /* This is an optimization that relies on the locality of the jump target.
             * If the label is in the same WithStatement, the following scan
             * would find it quickly and can reduce jump cost.
             * Otherwise, the statement body may be unnnecessary scanned again
             * so it would make CTFE speed slower.
             */
            InterState istatex = *istate;
            istatex.start = istate.gotoTarget; // set starting statement
            istatex.gotoTarget = null;
            Expression ex = interpret(s._body, &istatex);
            if (!istatex.start)
            {
                istate.gotoTarget = null;
                e = ex;
            }
        }
        ctfeStack.pop(s.wthis);
        result = e;
    }

    override void visit(AsmStatement s)
    {
        debug (LOG)
        {
            printf("%s AsmStatement::interpret()\n", s.loc.toChars());
        }
        if (istate.start)
        {
            if (istate.start != s)
                return;
            istate.start = null;
        }
        s.error("asm statements cannot be interpreted at compile time");
        result = CTFEExp.cantexp;
    }

    override void visit(ImportStatement s)
    {
        debug (LOG)
        {
            printf("ImportStatement::interpret()\n");
        }
        if (istate.start)
        {
            if (istate.start != s)
                return;
            istate.start = null;
        }
    }

    /******************************** Expression ***************************/

    override void visit(Expression e)
    {
        debug (LOG)
        {
            printf("%s Expression::interpret() '%s' %s\n", e.loc.toChars(), Token.toChars(e.op), e.toChars());
            printf("type = %s\n", e.type.toChars());
            e.print();
        }
        e.error("cannot interpret %s at compile time", e.toChars());
        result = CTFEExp.cantexp;
    }

    override void visit(ThisExp e)
    {
        debug (LOG)
        {
            printf("%s ThisExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        if (goal == ctfeNeedLvalue)
        {
            if (istate.fd.vthis)
            {
                result = new VarExp(e.loc, istate.fd.vthis);
                result.type = e.type;
            }
            else
                result = e;
            return;
        }

        result = ctfeStack.getThis();
        if (result)
        {
            assert(result.op == TOKstructliteral || result.op == TOKclassreference);
            return;
        }
        e.error("value of 'this' is not known at compile time");
        result = CTFEExp.cantexp;
    }

    override void visit(NullExp e)
    {
        result = e;
    }

    override void visit(IntegerExp e)
    {
        debug (LOG)
        {
            printf("%s IntegerExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        result = e;
    }

    override void visit(RealExp e)
    {
        debug (LOG)
        {
            printf("%s RealExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        result = e;
    }

    override void visit(ComplexExp e)
    {
        result = e;
    }

    override void visit(StringExp e)
    {
        debug (LOG)
        {
            printf("%s StringExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        /* Attempts to modify string literals are prevented
         * in BinExp::interpretAssignCommon.
         */
        result = e;
    }

    override void visit(FuncExp e)
    {
        debug (LOG)
        {
            printf("%s FuncExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        result = e;
    }

    override void visit(SymOffExp e)
    {
        debug (LOG)
        {
            printf("%s SymOffExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        if (e.var.isFuncDeclaration() && e.offset == 0)
        {
            result = e;
            return;
        }
        if (isTypeInfo_Class(e.type) && e.offset == 0)
        {
            result = e;
            return;
        }
        if (e.type.ty != Tpointer)
        {
            // Probably impossible
            e.error("cannot interpret %s at compile time", e.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        Type pointee = (cast(TypePointer)e.type).next;
        if (e.var.isThreadlocal())
        {
            e.error("cannot take address of thread-local variable %s at compile time", e.var.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        // Check for taking an address of a shared variable.
        // If the shared variable is an array, the offset might not be zero.
        Type fromType = null;
        if (e.var.type.ty == Tarray || e.var.type.ty == Tsarray)
        {
            fromType = (cast(TypeArray)e.var.type).next;
        }
        if (e.var.isDataseg() && ((e.offset == 0 && isSafePointerCast(e.var.type, pointee)) || (fromType && isSafePointerCast(fromType, pointee))))
        {
            result = e;
            return;
        }

        Expression val = getVarExp(e.loc, istate, e.var, goal);
        if (exceptionOrCant(val))
            return;
        if (val.type.ty == Tarray || val.type.ty == Tsarray)
        {
            // Check for unsupported type painting operations
            Type elemtype = (cast(TypeArray)val.type).next;
            d_uns64 elemsize = elemtype.size();

            // It's OK to cast from fixed length to dynamic array, eg &int[3] to int[]*
            if (val.type.ty == Tsarray && pointee.ty == Tarray && elemsize == pointee.nextOf().size())
            {
                result = new AddrExp(e.loc, val);
                result.type = e.type;
                return;
            }

            // It's OK to cast from fixed length to fixed length array, eg &int[n] to int[d]*.
            if (val.type.ty == Tsarray && pointee.ty == Tsarray && elemsize == pointee.nextOf().size())
            {
                size_t d = cast(size_t)(cast(TypeSArray)pointee).dim.toInteger();
                Expression elwr = new IntegerExp(e.loc, e.offset / elemsize, Type.tsize_t);
                Expression eupr = new IntegerExp(e.loc, e.offset / elemsize + d, Type.tsize_t);

                // Create a CTFE pointer &val[ofs..ofs+d]
                result = new SliceExp(e.loc, val, elwr, eupr);
                result.type = pointee;
                result = new AddrExp(e.loc, result);
                result.type = e.type;
                return;
            }

            if (!isSafePointerCast(elemtype, pointee))
            {
                // It's also OK to cast from &string to string*.
                if (e.offset == 0 && isSafePointerCast(e.var.type, pointee))
                {
                    // Create a CTFE pointer &var
                    result = new VarExp(e.loc, e.var);
                    result.type = elemtype;
                    result = new AddrExp(e.loc, result);
                    result.type = e.type;
                    return;
                }
                e.error("reinterpreting cast from %s to %s is not supported in CTFE", val.type.toChars(), e.type.toChars());
                result = CTFEExp.cantexp;
                return;
            }

            dinteger_t sz = pointee.size();
            dinteger_t indx = e.offset / sz;
            assert(sz * indx == e.offset);
            Expression aggregate = null;
            if (val.op == TOKarrayliteral || val.op == TOKstring)
            {
                aggregate = val;
            }
            else if (val.op == TOKslice)
            {
                aggregate = (cast(SliceExp)val).e1;
                Expression lwr = interpret((cast(SliceExp)val).lwr, istate);
                indx += lwr.toInteger();
            }
            if (aggregate)
            {
                // Create a CTFE pointer &aggregate[ofs]
                auto ofs = new IntegerExp(e.loc, indx, Type.tsize_t);
                result = new IndexExp(e.loc, aggregate, ofs);
                result.type = elemtype;
                result = new AddrExp(e.loc, result);
                result.type = e.type;
                return;
            }
        }
        else if (e.offset == 0 && isSafePointerCast(e.var.type, pointee))
        {
            // Create a CTFE pointer &var
            auto ve = new VarExp(e.loc, e.var);
            ve.type = e.var.type;
            result = new AddrExp(e.loc, ve);
            result.type = e.type;
            return;
        }

        e.error("cannot convert &%s to %s at compile time", e.var.type.toChars(), e.type.toChars());
        result = CTFEExp.cantexp;
    }

    override void visit(AddrExp e)
    {
        debug (LOG)
        {
            printf("%s AddrExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        if (e.e1.op == TOKvar && (cast(VarExp)e.e1).var.isDataseg())
        {
            // Normally this is already done by optimize()
            // Do it here in case optimize(WANTvalue) wasn't run before CTFE
            result = new SymOffExp(e.loc, (cast(VarExp)e.e1).var, 0);
            result.type = e.type;
            return;
        }
        result = interpret(e.e1, istate, ctfeNeedLvalue);
        if (result.op == TOKvar && (cast(VarExp)result).var == istate.fd.vthis)
            result = interpret(result, istate);
        if (exceptionOrCant(result))
            return;

        // Return a simplified address expression
        result = new AddrExp(e.loc, result);
        result.type = e.type;
    }

    override void visit(DelegateExp e)
    {
        debug (LOG)
        {
            printf("%s DelegateExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        // TODO: Really we should create a CTFE-only delegate expression
        // of a pointer and a funcptr.

        // If it is &nestedfunc, just return it
        // TODO: We should save the context pointer
        if (e.e1.op == TOKvar && (cast(VarExp)e.e1).var == e.func)
        {
            result = e;
            return;
        }

        result = interpret(e.e1, istate);
        if (exceptionOrCant(result))
            return;
        if (result == e.e1)
        {
            // If it has already been CTFE'd, just return it
            result = e;
        }
        else
        {
            result = new DelegateExp(e.loc, result, e.func, false);
            result.type = e.type;
        }
    }

    static Expression getVarExp(Loc loc, InterState* istate, Declaration d, CtfeGoal goal)
    {
        Expression e = CTFEExp.cantexp;
        if (VarDeclaration v = d.isVarDeclaration())
        {
            /* Magic variable __ctfe always returns true when interpreting
             */
            if (v.ident == Id.ctfe)
                return new IntegerExp(loc, 1, Type.tbool);

            if (!v.originalType && v._scope) // semantic() not yet run
            {
                v.semantic(v._scope);
                if (v.type.ty == Terror)
                    return CTFEExp.cantexp;
            }

            if ((v.isConst() || v.isImmutable() || v.storage_class & STCmanifest) && !hasValue(v) && v._init && !v.isCTFE())
            {
                if (v.inuse)
                {
                    error(loc, "circular initialization of %s '%s'", v.kind(), v.toPrettyChars());
                    return CTFEExp.cantexp;
                }
                if (v._scope)
                {
                    v.inuse++;
                    v._init = v._init.semantic(v._scope, v.type, INITinterpret); // might not be run on aggregate members
                    v.inuse--;
                }
                e = v._init.toExpression(v.type);
                if (!e)
                    return CTFEExp.cantexp;
                assert(e.type);

                if (e.op == TOKconstruct || e.op == TOKblit)
                {
                    AssignExp ae = cast(AssignExp)e;
                    e = ae.e2;
                }

                if (e.op == TOKerror)
                {
                    // FIXME: Ultimately all errors should be detected in prior semantic analysis stage.
                }
                else if (v.isDataseg() || (v.storage_class & STCmanifest))
                {
                    /* https://issues.dlang.org/show_bug.cgi?id=14304
                     * e is a value that is not yet owned by CTFE.
                     * Mark as "cached", and use it directly during interpretation.
                     */
                    e = scrubCacheValue(v.loc, e);
                    ctfeStack.saveGlobalConstant(v, e);
                }
                else
                {
                    v.inuse++;
                    e = interpret(e, istate);
                    v.inuse--;
                    if (CTFEExp.isCantExp(e) && !global.gag && !CtfeStatus.stackTraceCallsToSuppress)
                        errorSupplemental(loc, "while evaluating %s.init", v.toChars());
                    if (exceptionOrCantInterpret(e))
                        return e;
                }
            }
            else if (v.isCTFE() && !hasValue(v))
            {
                if (v._init && v.type.size() != 0)
                {
                    if (v._init.isVoidInitializer())
                    {
                        // var should have been initialized when it was created
                        error(loc, "CTFE internal error: trying to access uninitialized var");
                        assert(0);
                    }
                    e = v._init.toExpression();
                }
                else
                    e = v.type.defaultInitLiteral(e.loc);

                e = interpret(e, istate);
            }
            else if (!(v.isDataseg() || v.storage_class & STCmanifest) && !v.isCTFE() && !istate)
            {
                error(loc, "variable %s cannot be read at compile time", v.toChars());
                return CTFEExp.cantexp;
            }
            else
            {
                e = hasValue(v) ? getValue(v) : null;
                if (!e && !v.isCTFE() && v.isDataseg())
                {
                    error(loc, "static variable %s cannot be read at compile time", v.toChars());
                    return CTFEExp.cantexp;
                }
                if (!e)
                {
                    assert(!(v._init && v._init.isVoidInitializer()));
                    // CTFE initiated from inside a function
                    error(loc, "variable %s cannot be read at compile time", v.toChars());
                    return CTFEExp.cantexp;
                }
                if (e.op == TOKvoid)
                {
                    VoidInitExp ve = cast(VoidInitExp)e;
                    error(loc, "cannot read uninitialized variable %s in ctfe", v.toPrettyChars());
                    errorSupplemental(ve.var.loc, "%s was uninitialized and used before set", ve.var.toChars());
                    return CTFEExp.cantexp;
                }
                if (goal != ctfeNeedLvalue && (v.isRef() || v.isOut()))
                    e = interpret(e, istate, goal);
            }
            if (!e)
                e = CTFEExp.cantexp;
        }
        else if (SymbolDeclaration s = d.isSymbolDeclaration())
        {
            // Struct static initializers, for example
            e = s.dsym.type.defaultInitLiteral(loc);
            if (e.op == TOKerror)
                error(loc, "CTFE failed because of previous errors in %s.init", s.toChars());
            e = e.semantic(null);
            if (e.op == TOKerror)
                e = CTFEExp.cantexp;
            else // Convert NULL to CTFEExp
                e = interpret(e, istate, goal);
        }
        else
            error(loc, "cannot interpret declaration %s at compile time", d.toChars());
        return e;
    }

    override void visit(VarExp e)
    {
        debug (LOG)
        {
            printf("%s VarExp::interpret() %s, goal = %d\n", e.loc.toChars(), e.toChars(), goal);
        }
        if (e.var.isFuncDeclaration())
        {
            result = e;
            return;
        }

        if (goal == ctfeNeedLvalue)
        {
            VarDeclaration v = e.var.isVarDeclaration();
            if (v && !v.isDataseg() && !v.isCTFE() && !istate)
            {
                e.error("variable %s cannot be read at compile time", v.toChars());
                result = CTFEExp.cantexp;
                return;
            }
            if (v && !hasValue(v))
            {
                if (!v.isCTFE() && v.isDataseg())
                    e.error("static variable %s cannot be read at compile time", v.toChars());
                else // CTFE initiated from inside a function
                    e.error("variable %s cannot be read at compile time", v.toChars());
                result = CTFEExp.cantexp;
                return;
            }
            if (v && (v.storage_class & (STCout | STCref)) && hasValue(v))
            {
                // Strip off the nest of ref variables
                Expression ev = getValue(v);
                if (ev.op == TOKvar || ev.op == TOKindex || ev.op == TOKdotvar)
                {
                    result = interpret(ev, istate, goal);
                    return;
                }
            }
            result = e;
            return;
        }
        result = getVarExp(e.loc, istate, e.var, goal);
        if (exceptionOrCant(result))
            return;
        if ((e.var.storage_class & (STCref | STCout)) == 0 && e.type.baseElemOf().ty != Tstruct)
        {
            /* Ultimately, STCref|STCout check should be enough to see the
             * necessity of type repainting. But currently front-end paints
             * non-ref struct variables by the const type.
             *
             *  auto foo(ref const S cs);
             *  S s;
             *  foo(s); // VarExp('s') will have const(S)
             */
            // A VarExp may include an implicit cast. It must be done explicitly.
            result = paintTypeOntoLiteral(e.type, result);
        }
    }

    override void visit(DeclarationExp e)
    {
        debug (LOG)
        {
            printf("%s DeclarationExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        Dsymbol s = e.declaration;
        if (VarDeclaration v = s.isVarDeclaration())
        {
            if (TupleDeclaration td = v.toAlias().isTupleDeclaration())
            {
                result = null;

                // Reserve stack space for all tuple members
                if (!td.objects)
                    return;
                for (size_t i = 0; i < td.objects.dim; ++i)
                {
                    RootObject o = (*td.objects)[i];
                    Expression ex = isExpression(o);
                    DsymbolExp ds = (ex && ex.op == TOKdsymbol) ? cast(DsymbolExp)ex : null;
                    VarDeclaration v2 = ds ? ds.s.isVarDeclaration() : null;
                    assert(v2);
                    if (v2.isDataseg() && !v2.isCTFE())
                        continue;

                    ctfeStack.push(v2);
                    if (v2._init)
                    {
                        Expression einit;
                        if (ExpInitializer ie = v2._init.isExpInitializer())
                        {
                            einit = interpret(ie.exp, istate, goal);
                            if (exceptionOrCant(einit))
                                return;
                        }
                        else if (v2._init.isVoidInitializer())
                        {
                            einit = voidInitLiteral(v2.type, v2).copy();
                        }
                        else
                        {
                            e.error("declaration %s is not yet implemented in CTFE", e.toChars());
                            result = CTFEExp.cantexp;
                            return;
                        }
                        setValue(v2, einit);
                    }
                }
                return;
            }
            if (v.isStatic())
            {
                // Just ignore static variables which aren't read or written yet
                result = null;
                return;
            }
            if (!(v.isDataseg() || v.storage_class & STCmanifest) || v.isCTFE())
                ctfeStack.push(v);
            if (v._init)
            {
                if (ExpInitializer ie = v._init.isExpInitializer())
                {
                    result = interpret(ie.exp, istate, goal);
                }
                else if (v._init.isVoidInitializer())
                {
                    result = voidInitLiteral(v.type, v).copy();
                    // There is no AssignExp for void initializers,
                    // so set it here.
                    setValue(v, result);
                }
                else
                {
                    e.error("declaration %s is not yet implemented in CTFE", e.toChars());
                    result = CTFEExp.cantexp;
                }
            }
            else if (v.type.size() == 0)
            {
                // Zero-length arrays don't need an initializer
                result = v.type.defaultInitLiteral(e.loc);
            }
            else
            {
                e.error("variable %s cannot be modified at compile time", v.toChars());
                result = CTFEExp.cantexp;
            }
            return;
        }
        if (s.isAttribDeclaration() || s.isTemplateMixin() || s.isTupleDeclaration())
        {
            // Check for static struct declarations, which aren't executable
            AttribDeclaration ad = e.declaration.isAttribDeclaration();
            if (ad && ad.decl && ad.decl.dim == 1)
            {
                Dsymbol sparent = (*ad.decl)[0];
                if (sparent.isAggregateDeclaration() || sparent.isTemplateDeclaration() || sparent.isAliasDeclaration())
                {
                    result = null;
                    return; // static (template) struct declaration. Nothing to do.
                }
            }

            // These can be made to work, too lazy now
            e.error("declaration %s is not yet implemented in CTFE", e.toChars());
            result = CTFEExp.cantexp;
            return;
        }

        // Others should not contain executable code, so are trivial to evaluate
        result = null;
        debug (LOG)
        {
            printf("-DeclarationExp::interpret(%s): %p\n", e.toChars(), result);
        }
    }

    override void visit(TypeidExp e)
    {
        debug (LOG)
        {
            printf("%s TypeidExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        if (Type t = isType(e.obj))
        {
            result = e;
            return;
        }
        if (Expression ex = isExpression(e.obj))
        {
            result = interpret(ex, istate);
            if (exceptionOrCant(ex))
                return;

            if (result.op == TOKnull)
            {
                e.error("null pointer dereference evaluating typeid. '%s' is null", ex.toChars());
                result = CTFEExp.cantexp;
                return;
            }
            if (result.op != TOKclassreference)
            {
                e.error("CTFE internal error: determining classinfo");
                result = CTFEExp.cantexp;
                return;
            }

            ClassDeclaration cd = (cast(ClassReferenceExp)result).originalClass();
            assert(cd);

            result = new TypeidExp(e.loc, cd.type);
            result.type = e.type;
            return;
        }
        visit(cast(Expression)e);
    }

    override void visit(TupleExp e)
    {
        debug (LOG)
        {
            printf("%s TupleExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        if (exceptionOrCant(interpret(e.e0, istate, ctfeNeedNothing)))
            return;

        auto expsx = e.exps;
        for (size_t i = 0; i < expsx.dim; i++)
        {
            Expression exp = (*expsx)[i];
            Expression ex = interpret(exp, istate);
            if (exceptionOrCant(ex))
                return;

            // A tuple of assignments can contain void (Bug 5676).
            if (goal == ctfeNeedNothing)
                continue;
            if (ex.op == TOKvoidexp)
            {
                e.error("CTFE internal error: void element %s in tuple", exp.toChars());
                assert(0);
            }

            /* If any changes, do Copy On Write
             */
            if (ex !is exp)
            {
                expsx = copyArrayOnWrite(expsx, e.exps);
                (*expsx)[i] = ex;
            }
        }

        if (expsx !is e.exps)
        {
            expandTuples(expsx);
            auto te = new TupleExp(e.loc, expsx);
            te.type = new TypeTuple(te.exps);
            result = te;
        }
        else
            result = e;
    }

    override void visit(ArrayLiteralExp e)
    {
        debug (LOG)
        {
            printf("%s ArrayLiteralExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        if (e.ownedByCtfe >= OWNEDctfe) // We've already interpreted all the elements
        {
            result = e;
            return;
        }

        Type tn = e.type.toBasetype().nextOf().toBasetype();
        bool wantCopy = (tn.ty == Tsarray || tn.ty == Tstruct);

        auto basis = interpret(e.basis, istate);
        if (exceptionOrCant(basis))
            return;

        auto expsx = e.elements;
        size_t dim = expsx ? expsx.dim : 0;
        for (size_t i = 0; i < dim; i++)
        {
            Expression exp = (*expsx)[i];
            Expression ex;
            if (!exp)
            {
                ex = copyLiteral(basis).copy();
            }
            else
            {
                // segfault bug 6250
                assert(exp.op != TOKindex || (cast(IndexExp)exp).e1 != e);

                ex = interpret(exp, istate);
                if (exceptionOrCant(ex))
                    return;

                /* Each elements should have distinct CTFE memory.
                 *  int[1] z = 7;
                 *  int[1][] pieces = [z,z];    // here
                 */
                if (wantCopy)
                    ex = copyLiteral(ex).copy();
            }

            /* If any changes, do Copy On Write
             */
            if (ex !is exp)
            {
                expsx = copyArrayOnWrite(expsx, e.elements);
                (*expsx)[i] = ex;
            }
        }

        if (expsx !is e.elements)
        {
            // todo: all tuple expansions should go in semantic phase.
            expandTuples(expsx);
            if (expsx.dim != dim)
            {
                e.error("CTFE internal error: invalid array literal");
                result = CTFEExp.cantexp;
                return;
            }
            auto ale = new ArrayLiteralExp(e.loc, basis, expsx);
            ale.type = e.type;
            ale.ownedByCtfe = OWNEDctfe;
            result = ale;
        }
        else if ((cast(TypeNext)e.type).next.mod & (MODconst | MODimmutable))
        {
            // If it's immutable, we don't need to dup it
            result = e;
        }
        else
            result = copyLiteral(e).copy();
    }

    override void visit(AssocArrayLiteralExp e)
    {
        debug (LOG)
        {
            printf("%s AssocArrayLiteralExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        if (e.ownedByCtfe >= OWNEDctfe) // We've already interpreted all the elements
        {
            result = e;
            return;
        }

        auto keysx = e.keys;
        auto valuesx = e.values;
        for (size_t i = 0; i < keysx.dim; i++)
        {
            auto ekey = (*keysx)[i];
            auto evalue = (*valuesx)[i];

            auto ek = interpret(ekey, istate);
            if (exceptionOrCant(ek))
                return;
            auto ev = interpret(evalue, istate);
            if (exceptionOrCant(ev))
                return;

            /* If any changes, do Copy On Write
             */
            if (ek !is ekey ||
                ev !is evalue)
            {
                keysx = copyArrayOnWrite(keysx, e.keys);
                valuesx = copyArrayOnWrite(valuesx, e.values);
                (*keysx)[i] = ek;
                (*valuesx)[i] = ev;
            }
        }
        if (keysx !is e.keys)
            expandTuples(keysx);
        if (valuesx !is e.values)
            expandTuples(valuesx);
        if (keysx.dim != valuesx.dim)
        {
            e.error("CTFE internal error: invalid AA");
            result = CTFEExp.cantexp;
            return;
        }

        /* Remove duplicate keys
         */
        for (size_t i = 1; i < keysx.dim; i++)
        {
            auto ekey = (*keysx)[i - 1];
            for (size_t j = i; j < keysx.dim; j++)
            {
                auto ekey2 = (*keysx)[j];
                if (!ctfeEqual(e.loc, TOKequal, ekey, ekey2))
                    continue;

                // Remove ekey
                keysx = copyArrayOnWrite(keysx, e.keys);
                valuesx = copyArrayOnWrite(valuesx, e.values);
                keysx.remove(i - 1);
                valuesx.remove(i - 1);

                i -= 1; // redo the i'th iteration
                break;
            }
        }

        if (keysx !is e.keys ||
            valuesx !is e.values)
        {
            assert(keysx !is e.keys &&
                   valuesx !is e.values);
            auto aae = new AssocArrayLiteralExp(e.loc, keysx, valuesx);
            aae.type = e.type;
            aae.ownedByCtfe = OWNEDctfe;
            result = aae;
        }
        else
            result = copyLiteral(e).copy();
    }

    override void visit(StructLiteralExp e)
    {
        debug (LOG)
        {
            printf("%s StructLiteralExp::interpret() %s ownedByCtfe = %d\n", e.loc.toChars(), e.toChars(), e.ownedByCtfe);
        }
        if (e.ownedByCtfe >= OWNEDctfe)
        {
            result = e;
            return;
        }

        size_t dim = e.elements ? e.elements.dim : 0;
        auto expsx = e.elements;

        if (dim != e.sd.fields.dim)
        {
            // guaranteed by AggregateDeclaration.fill and TypeStruct.defaultInitLiteral
            assert(e.sd.isNested() && dim == e.sd.fields.dim - 1);

            /* If a nested struct has no initialized hidden pointer,
             * set it to null to match the runtime behaviour.
             */
            auto ne = new NullExp(e.loc);
            ne.type = e.sd.vthis.type;

            expsx = copyArrayOnWrite(expsx, e.elements);
            expsx.push(ne);
            ++dim;
        }
        assert(dim == e.sd.fields.dim);

        foreach (i; 0 .. dim)
        {
            auto v = e.sd.fields[i];
            Expression exp = (*expsx)[i];
            Expression ex;
            if (!exp)
            {
                ex = voidInitLiteral(v.type, v).copy();
            }
            else
            {
                ex = interpret(exp, istate);
                if (exceptionOrCant(ex))
                    return;
                if ((v.type.ty != ex.type.ty) && v.type.ty == Tsarray)
                {
                    // Block assignment from inside struct literals
                    auto tsa = cast(TypeSArray)v.type;
                    auto len = cast(size_t)tsa.dim.toInteger();
                    ex = createBlockDuplicatedArrayLiteral(ex.loc, v.type, ex, len);
                }
            }

            /* If any changes, do Copy On Write
             */
            if (ex !is exp)
            {
                expsx = copyArrayOnWrite(expsx, e.elements);
                (*expsx)[i] = ex;
            }
        }

        if (expsx !is e.elements)
        {
            expandTuples(expsx);
            if (expsx.dim != e.sd.fields.dim)
            {
                e.error("CTFE internal error: invalid struct literal");
                result = CTFEExp.cantexp;
                return;
            }
            auto sle = new StructLiteralExp(e.loc, e.sd, expsx);
            sle.type = e.type;
            sle.ownedByCtfe = OWNEDctfe;
            result = sle;
        }
        else
            result = copyLiteral(e).copy();
    }

    // Create an array literal of type 'newtype' with dimensions given by
    // 'arguments'[argnum..$]
    static Expression recursivelyCreateArrayLiteral(Loc loc, Type newtype, InterState* istate, Expressions* arguments, int argnum)
    {
        Expression lenExpr = interpret((*arguments)[argnum], istate);
        if (exceptionOrCantInterpret(lenExpr))
            return lenExpr;
        size_t len = cast(size_t)lenExpr.toInteger();
        Type elemType = (cast(TypeArray)newtype).next;
        if (elemType.ty == Tarray && argnum < arguments.dim - 1)
        {
            Expression elem = recursivelyCreateArrayLiteral(loc, elemType, istate, arguments, argnum + 1);
            if (exceptionOrCantInterpret(elem))
                return elem;

            auto elements = new Expressions();
            elements.setDim(len);
            for (size_t i = 0; i < len; i++)
                (*elements)[i] = copyLiteral(elem).copy();
            auto ae = new ArrayLiteralExp(loc, elements);
            ae.type = newtype;
            ae.ownedByCtfe = OWNEDctfe;
            return ae;
        }
        assert(argnum == arguments.dim - 1);
        if (elemType.ty == Tchar || elemType.ty == Twchar || elemType.ty == Tdchar)
        {
            const ch = cast(dchar)elemType.defaultInitLiteral(loc).toInteger();
            const sz = cast(ubyte)elemType.size();
            return createBlockDuplicatedStringLiteral(loc, newtype, ch, len, sz);
        }
        else
        {
            auto el = interpret(elemType.defaultInitLiteral(loc), istate);
            return createBlockDuplicatedArrayLiteral(loc, newtype, el, len);
        }
    }

    override void visit(NewExp e)
    {
        debug (LOG)
        {
            printf("%s NewExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        if (e.allocator)
        {
            e.error("member allocators not supported by CTFE");
            result = CTFEExp.cantexp;
            return;
        }

        result = interpret(e.argprefix, istate, ctfeNeedNothing);
        if (exceptionOrCant(result))
            return;

        if (e.newtype.ty == Tarray && e.arguments)
        {
            result = recursivelyCreateArrayLiteral(e.loc, e.newtype, istate, e.arguments, 0);
            return;
        }
        if (e.newtype.toBasetype().ty == Tstruct)
        {
            if (e.member)
            {
                Expression se = e.newtype.defaultInitLiteral(e.loc);
                se = interpret(se, istate);
                if (exceptionOrCant(se))
                    return;
                result = interpret(e.member, istate, e.arguments, se);

                // Repaint as same as CallExp::interpret() does.
                result.loc = e.loc;
            }
            else
            {
                StructDeclaration sd = (cast(TypeStruct)e.newtype.toBasetype()).sym;
                auto exps = new Expressions();
                exps.reserve(sd.fields.dim);
                if (e.arguments)
                {
                    exps.setDim(e.arguments.dim);
                    for (size_t i = 0; i < exps.dim; i++)
                    {
                        Expression ex = (*e.arguments)[i];
                        ex = interpret(ex, istate);
                        if (exceptionOrCant(ex))
                            return;
                        (*exps)[i] = ex;
                    }
                }
                sd.fill(e.loc, exps, false);

                auto se = new StructLiteralExp(e.loc, sd, exps, e.newtype);
                se.type = e.newtype;
                se.ownedByCtfe = OWNEDctfe;
                result = interpret(se, istate);
            }
            if (exceptionOrCant(result))
                return;
            result = new AddrExp(e.loc, result);
            result.type = e.type;
            return;
        }
        if (e.newtype.toBasetype().ty == Tclass)
        {
            ClassDeclaration cd = (cast(TypeClass)e.newtype.toBasetype()).sym;
            size_t totalFieldCount = 0;
            for (ClassDeclaration c = cd; c; c = c.baseClass)
                totalFieldCount += c.fields.dim;
            auto elems = new Expressions();
            elems.setDim(totalFieldCount);
            size_t fieldsSoFar = totalFieldCount;
            for (ClassDeclaration c = cd; c; c = c.baseClass)
            {
                fieldsSoFar -= c.fields.dim;
                for (size_t i = 0; i < c.fields.dim; i++)
                {
                    VarDeclaration v = c.fields[i];
                    if (v.inuse)
                    {
                        e.error("circular reference to '%s'", v.toPrettyChars());
                        result = CTFEExp.cantexp;
                        return;
                    }
                    Expression m;
                    if (v._init)
                    {
                        if (v._init.isVoidInitializer())
                            m = voidInitLiteral(v.type, v).copy();
                        else
                            m = v.getConstInitializer(true);
                    }
                    else
                        m = v.type.defaultInitLiteral(e.loc);
                    if (exceptionOrCant(m))
                        return;
                    (*elems)[fieldsSoFar + i] = copyLiteral(m).copy();
                }
            }
            // Hack: we store a ClassDeclaration instead of a StructDeclaration.
            // We probably won't get away with this.
            auto se = new StructLiteralExp(e.loc, cast(StructDeclaration)cd, elems, e.newtype);
            se.ownedByCtfe = OWNEDctfe;
            Expression eref = new ClassReferenceExp(e.loc, se, e.type);
            if (e.member)
            {
                // Call constructor
                if (!e.member.fbody)
                {
                    Expression ctorfail = evaluateIfBuiltin(istate, e.loc, e.member, e.arguments, eref);
                    if (ctorfail)
                    {
                        if (exceptionOrCant(ctorfail))
                            return;
                        result = eref;
                        return;
                    }
                    e.member.error("%s cannot be constructed at compile time, because the constructor has no available source code", e.newtype.toChars());
                    result = CTFEExp.cantexp;
                    return;
                }
                Expression ctorfail = interpret(e.member, istate, e.arguments, eref);
                if (exceptionOrCant(ctorfail))
                    return;

                /* https://issues.dlang.org/show_bug.cgi?id=14465
                 * Repaint the loc, because a super() call
                 * in the constructor modifies the loc of ClassReferenceExp
                 * in CallExp::interpret().
                 */
                eref.loc = e.loc;
            }
            result = eref;
            return;
        }
        if (e.newtype.toBasetype().isscalar())
        {
            Expression newval;
            if (e.arguments && e.arguments.dim)
                newval = (*e.arguments)[0];
            else
                newval = e.newtype.defaultInitLiteral(e.loc);
            newval = interpret(newval, istate);
            if (exceptionOrCant(newval))
                return;

            // Create a CTFE pointer &[newval][0]
            auto elements = new Expressions();
            elements.setDim(1);
            (*elements)[0] = newval;
            auto ae = new ArrayLiteralExp(e.loc, elements);
            ae.type = e.newtype.arrayOf();
            ae.ownedByCtfe = OWNEDctfe;

            result = new IndexExp(e.loc, ae, new IntegerExp(Loc(), 0, Type.tsize_t));
            result.type = e.newtype;
            result = new AddrExp(e.loc, result);
            result.type = e.type;
            return;
        }
        e.error("cannot interpret %s at compile time", e.toChars());
        result = CTFEExp.cantexp;
    }

    override void visit(UnaExp e)
    {
        debug (LOG)
        {
            printf("%s UnaExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        Expression e1 = interpret(e.e1, istate);
        if (exceptionOrCant(e1))
            return;
        UnionExp ue;
        switch (e.op)
        {
        case TOKneg:
            ue = Neg(e.type, e1);
            break;

        case TOKtilde:
            ue = Com(e.type, e1);
            break;

        case TOKnot:
            ue = Not(e.type, e1);
            break;

        case TOKvector:
            result = e;
            return; // do nothing

        default:
            assert(0);
        }
        result = ue.copy();
    }

    override void visit(DotTypeExp e)
    {
        debug (LOG)
        {
            printf("%s DotTypeExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        Expression e1 = interpret(e.e1, istate);
        if (exceptionOrCant(e1))
            return;
        if (e1 == e.e1)
            result = e; // optimize: reuse this CTFE reference
        else
        {
            result = e.copy();
            (cast(DotTypeExp)result).e1 = e1;
        }
    }

    void interpretCommon(BinExp e, fp_t fp)
    {
        debug (LOG)
        {
            printf("%s BinExp::interpretCommon() %s\n", e.loc.toChars(), e.toChars());
        }
        if (e.e1.type.ty == Tpointer && e.e2.type.ty == Tpointer && e.op == TOKmin)
        {
            Expression e1 = interpret(e.e1, istate);
            if (exceptionOrCant(e1))
                return;
            Expression e2 = interpret(e.e2, istate);
            if (exceptionOrCant(e2))
                return;
            result = pointerDifference(e.loc, e.type, e1, e2).copy();
            return;
        }
        if (e.e1.type.ty == Tpointer && e.e2.type.isintegral())
        {
            Expression e1 = interpret(e.e1, istate);
            if (exceptionOrCant(e1))
                return;
            Expression e2 = interpret(e.e2, istate);
            if (exceptionOrCant(e2))
                return;
            result = pointerArithmetic(e.loc, e.op, e.type, e1, e2).copy();
            return;
        }
        if (e.e2.type.ty == Tpointer && e.e1.type.isintegral() && e.op == TOKadd)
        {
            Expression e1 = interpret(e.e1, istate);
            if (exceptionOrCant(e1))
                return;
            Expression e2 = interpret(e.e2, istate);
            if (exceptionOrCant(e2))
                return;
            result = pointerArithmetic(e.loc, e.op, e.type, e2, e1).copy();
            return;
        }
        if (e.e1.type.ty == Tpointer || e.e2.type.ty == Tpointer)
        {
            e.error("pointer expression %s cannot be interpreted at compile time", e.toChars());
            result = CTFEExp.cantexp;
            return;
        }

        bool evalOperand(Expression ex, out Expression er)
        {
            er = interpret(ex, istate);
            if (exceptionOrCant(er))
                return false;
            if (er.isConst() != 1)
            {
                if (er.op == TOKarrayliteral)
                    // Until we get it to work, issue a reasonable error message
                    e.error("cannot interpret array literal expression %s at compile time", e.toChars());
                else
                    e.error("CTFE internal error: non-constant value %s", ex.toChars());
                result = CTFEExp.cantexp;
                return false;
            }
            return true;
        }

        Expression e1;
        if (!evalOperand(e.e1, e1))
            return;

        Expression e2;
        if (!evalOperand(e.e2, e2))
            return;

        if (e.op == TOKshr || e.op == TOKshl || e.op == TOKushr)
        {
            sinteger_t i2 = e2.toInteger();
            d_uns64 sz = e1.type.size() * 8;
            if (i2 < 0 || i2 >= sz)
            {
                e.error("shift by %lld is outside the range 0..%llu", i2, cast(ulong)sz - 1);
                result = CTFEExp.cantexp;
                return;
            }
        }
        result = (*fp)(e.loc, e.type, e1, e2).copy();
        if (CTFEExp.isCantExp(result))
            e.error("%s cannot be interpreted at compile time", e.toChars());
    }

    void interpretCompareCommon(BinExp e, fp2_t fp)
    {
        debug (LOG)
        {
            printf("%s BinExp::interpretCompareCommon() %s\n", e.loc.toChars(), e.toChars());
        }
        if (e.e1.type.ty == Tpointer && e.e2.type.ty == Tpointer)
        {
            Expression e1 = interpret(e.e1, istate);
            if (exceptionOrCant(e1))
                return;
            Expression e2 = interpret(e.e2, istate);
            if (exceptionOrCant(e2))
                return;
            //printf("e1 = %s %s, e2 = %s %s\n", e1.type.toChars(), e1.toChars(), e2.type.toChars(), e2.toChars());
            dinteger_t ofs1, ofs2;
            Expression agg1 = getAggregateFromPointer(e1, &ofs1);
            Expression agg2 = getAggregateFromPointer(e2, &ofs2);
            //printf("agg1 = %p %s, agg2 = %p %s\n", agg1, agg1.toChars(), agg2, agg2.toChars());
            int cmp = comparePointers(e.loc, e.op, e.type, agg1, ofs1, agg2, ofs2);
            if (cmp == -1)
            {
                char dir = (e.op == TOKgt || e.op == TOKge) ? '<' : '>';
                e.error("the ordering of pointers to unrelated memory blocks is indeterminate in CTFE. To check if they point to the same memory block, use both > and < inside && or ||, eg (%s && %s %c= %s + 1)", e.toChars(), e.e1.toChars(), dir, e.e2.toChars());
                result = CTFEExp.cantexp;
                return;
            }
            result = new IntegerExp(e.loc, cmp, e.type);
            return;
        }
        Expression e1 = interpret(e.e1, istate);
        if (exceptionOrCant(e1))
            return;
        if (!isCtfeComparable(e1))
        {
            e.error("cannot compare %s at compile time", e1.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        Expression e2 = interpret(e.e2, istate);
        if (exceptionOrCant(e2))
            return;
        if (!isCtfeComparable(e2))
        {
            e.error("cannot compare %s at compile time", e2.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        int cmp = (*fp)(e.loc, e.op, e1, e2);
        result = new IntegerExp(e.loc, cmp, e.type);
    }

    override void visit(BinExp e)
    {
        switch (e.op)
        {
        case TOKadd:
            interpretCommon(e, &Add);
            return;

        case TOKmin:
            interpretCommon(e, &Min);
            return;

        case TOKmul:
            interpretCommon(e, &Mul);
            return;

        case TOKdiv:
            interpretCommon(e, &Div);
            return;

        case TOKmod:
            interpretCommon(e, &Mod);
            return;

        case TOKshl:
            interpretCommon(e, &Shl);
            return;

        case TOKshr:
            interpretCommon(e, &Shr);
            return;

        case TOKushr:
            interpretCommon(e, &Ushr);
            return;

        case TOKand:
            interpretCommon(e, &And);
            return;

        case TOKor:
            interpretCommon(e, &Or);
            return;

        case TOKxor:
            interpretCommon(e, &Xor);
            return;

        case TOKpow:
            interpretCommon(e, &Pow);
            return;

        case TOKequal:
        case TOKnotequal:
            interpretCompareCommon(e, &ctfeEqual);
            return;

        case TOKidentity:
        case TOKnotidentity:
            interpretCompareCommon(e, &ctfeIdentity);
            return;

        case TOKlt:
        case TOKle:
        case TOKgt:
        case TOKge:
            interpretCompareCommon(e, &ctfeCmp);
            return;

        default:
            printf("be = '%s' %s at [%s]\n", Token.toChars(e.op), e.toChars(), e.loc.toChars());
            assert(0);
        }
    }

    /* Helper functions for BinExp::interpretAssignCommon
     */
    // Returns the variable which is eventually modified, or NULL if an rvalue.
    // thisval is the current value of 'this'.
    static VarDeclaration findParentVar(Expression e)
    {
        for (;;)
        {
            if (e.op == TOKvar)
                break;
            if (e.op == TOKindex)
                e = (cast(IndexExp)e).e1;
            else if (e.op == TOKdotvar)
                e = (cast(DotVarExp)e).e1;
            else if (e.op == TOKdotti)
                e = (cast(DotTemplateInstanceExp)e).e1;
            else if (e.op == TOKslice)
                e = (cast(SliceExp)e).e1;
            else
                return null;
        }
        VarDeclaration v = (cast(VarExp)e).var.isVarDeclaration();
        assert(v);
        return v;
    }

    void interpretAssignCommon(BinExp e, fp_t fp, int post = 0)
    {
        debug (LOG)
        {
            printf("%s BinExp::interpretAssignCommon() %s\n", e.loc.toChars(), e.toChars());
        }
        result = CTFEExp.cantexp;

        Expression e1 = e.e1;
        if (!istate)
        {
            e.error("value of %s is not known at compile time", e1.toChars());
            return;
        }

        ++CtfeStatus.numAssignments;

        /* Before we begin, we need to know if this is a reference assignment
         * (dynamic array, AA, or class) or a value assignment.
         * Determining this for slice assignments are tricky: we need to know
         * if it is a block assignment (a[] = e) rather than a direct slice
         * assignment (a[] = b[]). Note that initializers of multi-dimensional
         * static arrays can have 2D block assignments (eg, int[7][7] x = 6;).
         * So we need to recurse to determine if it is a block assignment.
         */
        bool isBlockAssignment = false;
        if (e1.op == TOKslice)
        {
            // a[] = e can have const e. So we compare the naked types.
            Type tdst = e1.type.toBasetype();
            Type tsrc = e.e2.type.toBasetype();
            while (tdst.ty == Tsarray || tdst.ty == Tarray)
            {
                tdst = (cast(TypeArray)tdst).next.toBasetype();
                if (tsrc.equivalent(tdst))
                {
                    isBlockAssignment = true;
                    break;
                }
            }
        }

        // ---------------------------------------
        //      Deal with reference assignment
        // ---------------------------------------
        // If it is a construction of a ref variable, it is a ref assignment
        if ((e.op == TOKconstruct || e.op == TOKblit) &&
            ((cast(AssignExp)e).memset & MemorySet.referenceInit))
        {
            assert(!fp);

            Expression newval = interpret(e.e2, istate, ctfeNeedLvalue);
            if (exceptionOrCant(newval))
                return;

            VarDeclaration v = (cast(VarExp)e1).var.isVarDeclaration();
            setValue(v, newval);

            // Get the value to return. Note that 'newval' is an Lvalue,
            // so if we need an Rvalue, we have to interpret again.
            if (goal == ctfeNeedRvalue)
                result = interpret(newval, istate);
            else
                result = e1; // VarExp is a CTFE reference
            return;
        }

        if (fp)
        {
            while (e1.op == TOKcast)
            {
                CastExp ce = cast(CastExp)e1;
                e1 = ce.e1;
            }
        }

        // ---------------------------------------
        //      Interpret left hand side
        // ---------------------------------------
        AssocArrayLiteralExp existingAA = null;
        Expression lastIndex = null;
        Expression oldval = null;
        if (e1.op == TOKindex && (cast(IndexExp)e1).e1.type.toBasetype().ty == Taarray)
        {
            // ---------------------------------------
            //      Deal with AA index assignment
            // ---------------------------------------
            /* This needs special treatment if the AA doesn't exist yet.
             * There are two special cases:
             * (1) If the AA is itself an index of another AA, we may need to create
             *     multiple nested AA literals before we can insert the new value.
             * (2) If the ultimate AA is null, no insertion happens at all. Instead,
             *     we create nested AA literals, and change it into a assignment.
             */
            IndexExp ie = cast(IndexExp)e1;
            int depth = 0; // how many nested AA indices are there?
            while (ie.e1.op == TOKindex && (cast(IndexExp)ie.e1).e1.type.toBasetype().ty == Taarray)
            {
                assert(ie.modifiable);
                ie = cast(IndexExp)ie.e1;
                ++depth;
            }

            // Get the AA value to be modified.
            Expression aggregate = interpret(ie.e1, istate);
            if (exceptionOrCant(aggregate))
                return;
            if (aggregate.op == TOKassocarrayliteral)
            {
                existingAA = cast(AssocArrayLiteralExp)aggregate;

                // Normal case, ultimate parent AA already exists
                // We need to walk from the deepest index up, checking that an AA literal
                // already exists on each level.
                lastIndex = interpret((cast(IndexExp)e1).e2, istate);
                lastIndex = resolveSlice(lastIndex); // only happens with AA assignment
                if (exceptionOrCant(lastIndex))
                    return;

                while (depth > 0)
                {
                    // Walk the syntax tree to find the indexExp at this depth
                    IndexExp xe = cast(IndexExp)e1;
                    for (int d = 0; d < depth; ++d)
                        xe = cast(IndexExp)xe.e1;

                    Expression ekey = interpret(xe.e2, istate);
                    if (exceptionOrCant(ekey))
                        return;
                    ekey = resolveSlice(ekey); // only happens with AA assignment

                    // Look up this index in it up in the existing AA, to get the next level of AA.
                    AssocArrayLiteralExp newAA = cast(AssocArrayLiteralExp)findKeyInAA(e.loc, existingAA, ekey);
                    if (exceptionOrCant(newAA))
                        return;
                    if (!newAA)
                    {
                        // Doesn't exist yet, create an empty AA...
                        auto keysx = new Expressions();
                        auto valuesx = new Expressions();
                        newAA = new AssocArrayLiteralExp(e.loc, keysx, valuesx);
                        newAA.type = xe.type;
                        newAA.ownedByCtfe = OWNEDctfe;
                        //... and insert it into the existing AA.
                        existingAA.keys.push(ekey);
                        existingAA.values.push(newAA);
                    }
                    existingAA = newAA;
                    --depth;
                }

                if (fp)
                {
                    oldval = findKeyInAA(e.loc, existingAA, lastIndex);
                    if (!oldval)
                        oldval = copyLiteral(e.e1.type.defaultInitLiteral(e.loc)).copy();
                }
            }
            else
            {
                /* The AA is currently null. 'aggregate' is actually a reference to
                 * whatever contains it. It could be anything: var, dotvarexp, ...
                 * We rewrite the assignment from:
                 *     aa[i][j] op= newval;
                 * into:
                 *     aa = [i:[j:T.init]];
                 *     aa[j] op= newval;
                 */
                oldval = copyLiteral(e.e1.type.defaultInitLiteral(e.loc)).copy();

                Expression newaae = oldval;
                while (e1.op == TOKindex && (cast(IndexExp)e1).e1.type.toBasetype().ty == Taarray)
                {
                    Expression ekey = interpret((cast(IndexExp)e1).e2, istate);
                    if (exceptionOrCant(ekey))
                        return;
                    ekey = resolveSlice(ekey); // only happens with AA assignment

                    auto keysx = new Expressions();
                    auto valuesx = new Expressions();
                    keysx.push(ekey);
                    valuesx.push(newaae);

                    auto aae = new AssocArrayLiteralExp(e.loc, keysx, valuesx);
                    aae.type = (cast(IndexExp)e1).e1.type;
                    aae.ownedByCtfe = OWNEDctfe;
                    if (!existingAA)
                    {
                        existingAA = aae;
                        lastIndex = ekey;
                    }
                    newaae = aae;
                    e1 = (cast(IndexExp)e1).e1;
                }

                // We must set to aggregate with newaae
                e1 = interpret(e1, istate, ctfeNeedLvalue);
                if (exceptionOrCant(e1))
                    return;
                e1 = assignToLvalue(e, e1, newaae);
                if (exceptionOrCant(e1))
                    return;
            }
            assert(existingAA && lastIndex);
            e1 = null; // stomp
        }
        else if (e1.op == TOKarraylength)
        {
            oldval = interpret(e1, istate);
            if (exceptionOrCant(oldval))
                return;
        }
        else if (e.op == TOKconstruct || e.op == TOKblit)
        {
            // Unless we have a simple var assignment, we're
            // only modifying part of the variable. So we need to make sure
            // that the parent variable exists.
            VarDeclaration ultimateVar = findParentVar(e1);
            if (e1.op == TOKvar)
            {
                VarDeclaration v = (cast(VarExp)e1).var.isVarDeclaration();
                assert(v);
                if (v.storage_class & STCout)
                    goto L1;
            }
            else if (ultimateVar && !getValue(ultimateVar))
            {
                Expression ex = interpret(ultimateVar.type.defaultInitLiteral(e.loc), istate);
                if (exceptionOrCant(ex))
                    return;
                setValue(ultimateVar, ex);
            }
            else
                goto L1;
        }
        else
        {
        L1:
            e1 = interpret(e1, istate, ctfeNeedLvalue);
            if (exceptionOrCant(e1))
                return;

            if (e1.op == TOKindex && (cast(IndexExp)e1).e1.type.toBasetype().ty == Taarray)
            {
                IndexExp ie = cast(IndexExp)e1;
                assert(ie.e1.op == TOKassocarrayliteral);
                existingAA = cast(AssocArrayLiteralExp)ie.e1;
                lastIndex = ie.e2;
            }
        }

        // ---------------------------------------
        //      Interpret right hand side
        // ---------------------------------------
        Expression newval = interpret(e.e2, istate);
        if (exceptionOrCant(newval))
            return;
        if (e.op == TOKblit && newval.op == TOKint64)
        {
            Type tbn = e.type.baseElemOf();
            if (tbn.ty == Tstruct)
            {
                /* Look for special case of struct being initialized with 0.
                 */
                newval = e.type.defaultInitLiteral(e.loc);
                if (newval.op == TOKerror)
                {
                    result = CTFEExp.cantexp;
                    return;
                }
                newval = interpret(newval, istate); // copy and set ownedByCtfe flag
                if (exceptionOrCant(newval))
                    return;
            }
        }

        // ----------------------------------------------------
        //  Deal with read-modify-write assignments.
        //  Set 'newval' to the final assignment value
        //  Also determine the return value (except for slice
        //  assignments, which are more complicated)
        // ----------------------------------------------------
        if (fp)
        {
            if (!oldval)
            {
                // Load the left hand side after interpreting the right hand side.
                oldval = interpret(e1, istate);
                if (exceptionOrCant(oldval))
                    return;
            }

            if (e.e1.type.ty != Tpointer)
            {
                // ~= can create new values (see bug 6052)
                if (e.op == TOKcatass)
                {
                    // We need to dup it and repaint the type. For a dynamic array
                    // we can skip duplication, because it gets copied later anyway.
                    if (newval.type.ty != Tarray)
                    {
                        newval = copyLiteral(newval).copy();
                        newval.type = e.e2.type; // repaint type
                    }
                    else
                    {
                        newval = paintTypeOntoLiteral(e.e2.type, newval);
                        newval = resolveSlice(newval);
                    }
                }
                oldval = resolveSlice(oldval);

                newval = (*fp)(e.loc, e.type, oldval, newval).copy();
            }
            else if (e.e2.type.isintegral() &&
                     (e.op == TOKaddass ||
                      e.op == TOKminass ||
                      e.op == TOKplusplus ||
                      e.op == TOKminusminus))
            {
                newval = pointerArithmetic(e.loc, e.op, e.type, oldval, newval).copy();
            }
            else
            {
                e.error("pointer expression %s cannot be interpreted at compile time", e.toChars());
                result = CTFEExp.cantexp;
                return;
            }
            if (exceptionOrCant(newval))
            {
                if (CTFEExp.isCantExp(newval))
                    e.error("cannot interpret %s at compile time", e.toChars());
                return;
            }
        }

        if (existingAA)
        {
            if (existingAA.ownedByCtfe != OWNEDctfe)
            {
                e.error("cannot modify read-only constant %s", existingAA.toChars());
                result = CTFEExp.cantexp;
                return;
            }

            //printf("\t+L%d existingAA = %s, lastIndex = %s, oldval = %s, newval = %s\n",
            //    __LINE__, existingAA.toChars(), lastIndex.toChars(), oldval ? oldval.toChars() : NULL, newval.toChars());
            assignAssocArrayElement(e.loc, existingAA, lastIndex, newval);

            // Determine the return value
            result = ctfeCast(e.loc, e.type, e.type, fp && post ? oldval : newval);
            return;
        }
        if (e1.op == TOKarraylength)
        {
            /* Change the assignment from:
             *  arr.length = n;
             * into:
             *  arr = new_length_array; (result is n)
             */

            // Determine the return value
            result = ctfeCast(e.loc, e.type, e.type, fp && post ? oldval : newval);
            if (exceptionOrCant(result))
                return;

            size_t oldlen = cast(size_t)oldval.toInteger();
            size_t newlen = cast(size_t)newval.toInteger();
            if (oldlen == newlen) // no change required -- we're done!
                return;

            // We have changed it into a reference assignment
            // Note that returnValue is still the new length.
            e1 = (cast(ArrayLengthExp)e1).e1;
            Type t = e1.type.toBasetype();
            if (t.ty != Tarray)
            {
                e.error("%s is not yet supported at compile time", e.toChars());
                result = CTFEExp.cantexp;
                return;
            }
            e1 = interpret(e1, istate, ctfeNeedLvalue);
            if (exceptionOrCant(e1))
                return;

            if (oldlen != 0) // Get the old array literal.
                oldval = interpret(e1, istate);
            newval = changeArrayLiteralLength(e.loc, cast(TypeArray)t, oldval, oldlen, newlen).copy();

            e1 = assignToLvalue(e, e1, newval);
            if (exceptionOrCant(e1))
                return;

            return;
        }

        if (!isBlockAssignment)
        {
            newval = ctfeCast(e.loc, e.type, e.type, newval);
            if (exceptionOrCant(newval))
                return;

            // Determine the return value
            if (goal == ctfeNeedLvalue) // https://issues.dlang.org/show_bug.cgi?id=14371
                result = e1;
            else
                result = ctfeCast(e.loc, e.type, e.type, fp && post ? oldval : newval);
            if (exceptionOrCant(result))
                return;
        }
        if (exceptionOrCant(newval))
            return;

        debug (LOGASSIGN)
        {
            printf("ASSIGN: %s=%s\n", e1.toChars(), newval.toChars());
            showCtfeExpr(newval);
        }

        /* Block assignment or element-wise assignment.
         */
        if (e1.op == TOKslice ||
            e1.op == TOKvector ||
            e1.op == TOKarrayliteral ||
            e1.op == TOKstring ||
            e1.op == TOKnull && e1.type.toBasetype().ty == Tarray)
        {
            // Note that slice assignments don't support things like ++, so
            // we don't need to remember 'returnValue'.
            result = interpretAssignToSlice(e, e1, newval, isBlockAssignment);
            if (exceptionOrCant(result))
                return;
            if (e.e1.op == TOKslice)
            {
                Expression e1x = interpret((cast(SliceExp)e.e1).e1, istate, ctfeNeedLvalue);
                if (e1x.op == TOKdotvar)
                {
                    auto dve = cast(DotVarExp)e1x;
                    auto ex = dve.e1;
                    auto sle = ex.op == TOKstructliteral ? (cast(StructLiteralExp)ex)
                             : ex.op == TOKclassreference ? (cast(ClassReferenceExp)ex).value
                             : null;
                    auto v = dve.var.isVarDeclaration();
                    if (!sle || !v)
                    {
                        e.error("CTFE internal error: dotvar slice assignment");
                        result = CTFEExp.cantexp;
                        return;
                    }
                    stompOverlappedFields(sle, v);
                }
            }
            return;
        }
        assert(result);

        /* Assignment to a CTFE reference.
         */
        if (Expression ex = assignToLvalue(e, e1, newval))
            result = ex;

        return;
    }

    /* Set all sibling fields which overlap with v to VoidExp.
     */
    void stompOverlappedFields(StructLiteralExp sle, VarDeclaration v)
    {
        if (!v.overlapped)
            return;
        foreach (size_t i, v2; sle.sd.fields)
        {
            if (v is v2 || !v.isOverlappedWith(v2))
                continue;
            auto e = (*sle.elements)[i];
            if (e.op != TOKvoid)
                (*sle.elements)[i] = voidInitLiteral(e.type, v).copy();
        }
    }

    Expression assignToLvalue(BinExp e, Expression e1, Expression newval)
    {
        VarDeclaration vd = null;
        Expression* payload = null; // dead-store to prevent spurious warning
        Expression oldval;

        if (e1.op == TOKvar)
        {
            vd = (cast(VarExp)e1).var.isVarDeclaration();
            oldval = getValue(vd);
        }
        else if (e1.op == TOKdotvar)
        {
            /* Assignment to member variable of the form:
             *  e.v = newval
             */
            auto ex = (cast(DotVarExp)e1).e1;
            auto sle = ex.op == TOKstructliteral ? (cast(StructLiteralExp)ex)
                     : ex.op == TOKclassreference ? (cast(ClassReferenceExp)ex).value
                     : null;
            auto v = (cast(DotVarExp)e1).var.isVarDeclaration();
            if (!sle || !v)
            {
                e.error("CTFE internal error: dotvar assignment");
                return CTFEExp.cantexp;
            }
            if (sle.ownedByCtfe != OWNEDctfe)
            {
                e.error("cannot modify read-only constant %s", sle.toChars());
                return CTFEExp.cantexp;
            }

            int fieldi = ex.op == TOKstructliteral ? findFieldIndexByName(sle.sd, v)
                       : (cast(ClassReferenceExp)ex).findFieldIndexByName(v);
            if (fieldi == -1)
            {
                e.error("CTFE internal error: cannot find field %s in %s", v.toChars(), ex.toChars());
                return CTFEExp.cantexp;
            }
            assert(0 <= fieldi && fieldi < sle.elements.dim);

            // If it's a union, set all other members of this union to void
            stompOverlappedFields(sle, v);

            payload = &(*sle.elements)[fieldi];
            oldval = *payload;
        }
        else if (e1.op == TOKindex)
        {
            IndexExp ie = cast(IndexExp)e1;
            assert(ie.e1.type.toBasetype().ty != Taarray);

            Expression aggregate;
            uinteger_t indexToModify;
            if (!resolveIndexing(ie, istate, &aggregate, &indexToModify, true))
            {
                return CTFEExp.cantexp;
            }
            size_t index = cast(size_t)indexToModify;

            if (aggregate.op == TOKstring)
            {
                StringExp existingSE = cast(StringExp)aggregate;
                if (existingSE.ownedByCtfe != OWNEDctfe)
                {
                    e.error("cannot modify read-only string literal %s", ie.e1.toChars());
                    return CTFEExp.cantexp;
                }
                existingSE.setCodeUnit(index, cast(dchar)newval.toInteger());
                return null;
            }
            if (aggregate.op != TOKarrayliteral)
            {
                e.error("index assignment %s is not yet supported in CTFE ", e.toChars());
                return CTFEExp.cantexp;
            }

            ArrayLiteralExp existingAE = cast(ArrayLiteralExp)aggregate;
            if (existingAE.ownedByCtfe != OWNEDctfe)
            {
                e.error("cannot modify read-only constant %s", existingAE.toChars());
                return CTFEExp.cantexp;
            }

            payload = &(*existingAE.elements)[index];
            oldval = *payload;
        }
        else
        {
            e.error("%s cannot be evaluated at compile time", e.toChars());
            return CTFEExp.cantexp;
        }

        Type t1b = e1.type.toBasetype();
        bool wantCopy = t1b.baseElemOf().ty == Tstruct;

        if (newval.op == TOKstructliteral && oldval)
        {
            newval = copyLiteral(newval).copy();
            assignInPlace(oldval, newval);
        }
        else if (wantCopy && e.op == TOKassign)
        {
            // Currently postblit/destructor calls on static array are done
            // in the druntime internal functions so they don't appear in AST.
            // Therefore interpreter should handle them specially.

            assert(oldval);
            version (all) // todo: instead we can directly access to each elements of the slice
            {
                newval = resolveSlice(newval);
                if (CTFEExp.isCantExp(newval))
                {
                    e.error("CTFE internal error: assignment %s", e.toChars());
                    return CTFEExp.cantexp;
                }
            }
            assert(oldval.op == TOKarrayliteral);
            assert(newval.op == TOKarrayliteral);

            Expressions* oldelems = (cast(ArrayLiteralExp)oldval).elements;
            Expressions* newelems = (cast(ArrayLiteralExp)newval).elements;
            assert(oldelems.dim == newelems.dim);

            Type elemtype = oldval.type.nextOf();
            for (size_t i = 0; i < newelems.dim; i++)
            {
                Expression oldelem = (*oldelems)[i];
                Expression newelem = paintTypeOntoLiteral(elemtype, (*newelems)[i]);
                // https://issues.dlang.org/show_bug.cgi?id=9245
                if (e.e2.isLvalue())
                {
                    if (Expression ex = evaluatePostblit(istate, newelem))
                        return ex;
                }
                // https://issues.dlang.org/show_bug.cgi?id=13661
                if (Expression ex = evaluateDtor(istate, oldelem))
                    return ex;
                (*oldelems)[i] = newelem;
            }
        }
        else
        {
            // e1 has its own payload, so we have to create a new literal.
            if (wantCopy)
                newval = copyLiteral(newval).copy();

            if (t1b.ty == Tsarray && e.op == TOKconstruct && e.e2.isLvalue())
            {
                // https://issues.dlang.org/show_bug.cgi?id=9245
                if (Expression ex = evaluatePostblit(istate, newval))
                    return ex;
            }

            oldval = newval;
        }

        if (vd)
            setValue(vd, oldval);
        else
            *payload = oldval;

        // Blit assignment should return the newly created value.
        if (e.op == TOKblit)
            return oldval;

        return null;
    }

    /*************
     * Deal with assignments of the form:
     *  dest[] = newval
     *  dest[low..upp] = newval
     * where newval has already been interpreted
     *
     * This could be a slice assignment or a block assignment, and
     * dest could be either an array literal, or a string.
     *
     * Returns TOKcantexp on failure. If there are no errors,
     * it returns aggregate[low..upp], except that as an optimisation,
     * if goal == ctfeNeedNothing, it will return NULL
     */
    Expression interpretAssignToSlice(BinExp e, Expression e1, Expression newval, bool isBlockAssignment)
    {
        dinteger_t lowerbound;
        dinteger_t upperbound;
        dinteger_t firstIndex;

        Expression aggregate;

        if (e1.op == TOKvector)
            e1 = (cast(VectorExp)e1).e1;
        if (e1.op == TOKslice)
        {
            // ------------------------------
            //   aggregate[] = newval
            //   aggregate[low..upp] = newval
            // ------------------------------
            SliceExp se = cast(SliceExp)e1;
            version (all) // should be move in interpretAssignCommon as the evaluation of e1
            {
                Expression oldval = interpret(se.e1, istate);

                // Set the $ variable
                uinteger_t dollar = resolveArrayLength(oldval);
                if (se.lengthVar)
                {
                    Expression dollarExp = new IntegerExp(e1.loc, dollar, Type.tsize_t);
                    ctfeStack.push(se.lengthVar);
                    setValue(se.lengthVar, dollarExp);
                }
                Expression lwr = interpret(se.lwr, istate);
                if (exceptionOrCantInterpret(lwr))
                {
                    if (se.lengthVar)
                        ctfeStack.pop(se.lengthVar);
                    return lwr;
                }
                Expression upr = interpret(se.upr, istate);
                if (exceptionOrCantInterpret(upr))
                {
                    if (se.lengthVar)
                        ctfeStack.pop(se.lengthVar);
                    return upr;
                }
                if (se.lengthVar)
                    ctfeStack.pop(se.lengthVar); // $ is defined only in [L..U]

                const dim = dollar;
                lowerbound = lwr ? lwr.toInteger() : 0;
                upperbound = upr ? upr.toInteger() : dim;

                if (lowerbound < 0 || dim < upperbound)
                {
                    e.error("array bounds [0..%llu] exceeded in slice [%llu..%llu]",
                        ulong(dim), ulong(lowerbound), ulong(upperbound));
                    return CTFEExp.cantexp;
                }
            }
            aggregate = oldval;
            firstIndex = lowerbound;

            if (aggregate.op == TOKslice)
            {
                // Slice of a slice --> change the bounds
                SliceExp oldse = cast(SliceExp)aggregate;
                if (oldse.upr.toInteger() < upperbound + oldse.lwr.toInteger())
                {
                    e.error("slice [%llu..%llu] exceeds array bounds [0..%llu]",
                        ulong(lowerbound), ulong(upperbound), oldse.upr.toInteger() - oldse.lwr.toInteger());
                    return CTFEExp.cantexp;
                }
                aggregate = oldse.e1;
                firstIndex = lowerbound + oldse.lwr.toInteger();
            }
        }
        else
        {
            if (e1.op == TOKarrayliteral)
            {
                lowerbound = 0;
                upperbound = (cast(ArrayLiteralExp)e1).elements.dim;
            }
            else if (e1.op == TOKstring)
            {
                lowerbound = 0;
                upperbound = (cast(StringExp)e1).len;
            }
            else if (e1.op == TOKnull)
            {
                lowerbound = 0;
                upperbound = 0;
            }
            else
                assert(0);

            aggregate = e1;
            firstIndex = lowerbound;
        }
        if (upperbound == lowerbound)
            return newval;

        // For slice assignment, we check that the lengths match.
        if (!isBlockAssignment)
        {
            const srclen = resolveArrayLength(newval);
            if (srclen != (upperbound - lowerbound))
            {
                e.error("array length mismatch assigning [0..%llu] to [%llu..%llu]",
                    ulong(srclen), ulong(lowerbound), ulong(upperbound));
                return CTFEExp.cantexp;
            }
        }

        if (aggregate.op == TOKstring)
        {
            StringExp existingSE = cast(StringExp)aggregate;
            if (existingSE.ownedByCtfe != OWNEDctfe)
            {
                e.error("cannot modify read-only string literal %s", existingSE.toChars());
                return CTFEExp.cantexp;
            }

            if (newval.op == TOKslice)
            {
                auto se = cast(SliceExp)newval;
                auto aggr2 = se.e1;
                const srclower = se.lwr.toInteger();
                const srcupper = se.upr.toInteger();

                if (aggregate == aggr2 &&
                    lowerbound < srcupper && srclower < upperbound)
                {
                    e.error("overlapping slice assignment [%llu..%llu] = [%llu..%llu]",
                        ulong(lowerbound), ulong(upperbound), ulong(srclower), ulong(srcupper));
                    return CTFEExp.cantexp;
                }
                version (all) // todo: instead we can directly access to each elements of the slice
                {
                    Expression orignewval = newval;
                    newval = resolveSlice(newval);
                    if (CTFEExp.isCantExp(newval))
                    {
                        e.error("CTFE internal error: slice %s", orignewval.toChars());
                        return CTFEExp.cantexp;
                    }
                }
                assert(newval.op != TOKslice);
            }
            if (newval.op == TOKstring)
            {
                sliceAssignStringFromString(existingSE, cast(StringExp)newval, cast(size_t)firstIndex);
                return newval;
            }
            if (newval.op == TOKarrayliteral)
            {
                /* Mixed slice: it was initialized as a string literal.
                 * Now a slice of it is being set with an array literal.
                 */
                sliceAssignStringFromArrayLiteral(existingSE, cast(ArrayLiteralExp)newval, cast(size_t)firstIndex);
                return newval;
            }

            // String literal block slice assign
            const value = cast(dchar)newval.toInteger();
            foreach (i; 0 .. upperbound - lowerbound)
            {
                existingSE.setCodeUnit(cast(size_t)(i + firstIndex), value);
            }
            if (goal == ctfeNeedNothing)
                return null; // avoid creating an unused literal
            auto retslice = new SliceExp(e.loc, existingSE, new IntegerExp(e.loc, firstIndex, Type.tsize_t), new IntegerExp(e.loc, firstIndex + upperbound - lowerbound, Type.tsize_t));
            retslice.type = e.type;
            return interpret(retslice, istate);
        }
        if (aggregate.op == TOKarrayliteral)
        {
            ArrayLiteralExp existingAE = cast(ArrayLiteralExp)aggregate;
            if (existingAE.ownedByCtfe != OWNEDctfe)
            {
                e.error("cannot modify read-only constant %s", existingAE.toChars());
                return CTFEExp.cantexp;
            }

            if (newval.op == TOKslice && !isBlockAssignment)
            {
                auto se = cast(SliceExp)newval;
                auto aggr2 = se.e1;
                const srclower = se.lwr.toInteger();
                const srcupper = se.upr.toInteger();
                const wantCopy = (newval.type.toBasetype().nextOf().baseElemOf().ty == Tstruct);

                //printf("oldval = %p %s[%d..%u]\nnewval = %p %s[%llu..%llu] wantCopy = %d\n",
                //    aggregate, aggregate.toChars(), lowerbound, upperbound,
                //    aggr2, aggr2.toChars(), srclower, srcupper, wantCopy);
                if (wantCopy)
                {
                    // Currently overlapping for struct array is allowed.
                    // The order of elements processing depends on the overlapping.
                    // https://issues.dlang.org/show_bug.cgi?id=14024
                    assert(aggr2.op == TOKarrayliteral);
                    Expressions* oldelems = existingAE.elements;
                    Expressions* newelems = (cast(ArrayLiteralExp)aggr2).elements;

                    Type elemtype = aggregate.type.nextOf();
                    bool needsPostblit = e.e2.isLvalue();

                    if (aggregate == aggr2 && srclower < lowerbound && lowerbound < srcupper)
                    {
                        // reverse order
                        for (auto i = upperbound - lowerbound; 0 < i--;)
                        {
                            Expression oldelem = (*oldelems)[cast(size_t)(i + firstIndex)];
                            Expression newelem = (*newelems)[cast(size_t)(i + srclower)];
                            newelem = copyLiteral(newelem).copy();
                            newelem.type = elemtype;
                            if (needsPostblit)
                            {
                                if (Expression x = evaluatePostblit(istate, newelem))
                                    return x;
                            }
                            if (Expression x = evaluateDtor(istate, oldelem))
                                return x;
                            (*oldelems)[cast(size_t)(lowerbound + i)] = newelem;
                        }
                    }
                    else
                    {
                        // normal order
                        for (auto i = 0; i < upperbound - lowerbound; i++)
                        {
                            Expression oldelem = (*oldelems)[cast(size_t)(i + firstIndex)];
                            Expression newelem = (*newelems)[cast(size_t)(i + srclower)];
                            newelem = copyLiteral(newelem).copy();
                            newelem.type = elemtype;
                            if (needsPostblit)
                            {
                                if (Expression x = evaluatePostblit(istate, newelem))
                                    return x;
                            }
                            if (Expression x = evaluateDtor(istate, oldelem))
                                return x;
                            (*oldelems)[cast(size_t)(lowerbound + i)] = newelem;
                        }
                    }

                    //assert(0);
                    return newval; // oldval?
                }
                if (aggregate == aggr2 &&
                    lowerbound < srcupper && srclower < upperbound)
                {
                    e.error("overlapping slice assignment [%llu..%llu] = [%llu..%llu]",
                        ulong(lowerbound), ulong(upperbound), ulong(srclower), ulong(srcupper));
                    return CTFEExp.cantexp;
                }
                version (all) // todo: instead we can directly access to each elements of the slice
                {
                    Expression orignewval = newval;
                    newval = resolveSlice(newval);
                    if (CTFEExp.isCantExp(newval))
                    {
                        e.error("CTFE internal error: slice %s", orignewval.toChars());
                        return CTFEExp.cantexp;
                    }
                }
                // no overlapping
                //length?
                assert(newval.op != TOKslice);
            }
            if (newval.op == TOKstring && !isBlockAssignment)
            {
                /* Mixed slice: it was initialized as an array literal of chars/integers.
                 * Now a slice of it is being set with a string.
                 */
                sliceAssignArrayLiteralFromString(existingAE, cast(StringExp)newval, cast(size_t)firstIndex);
                return newval;
            }
            if (newval.op == TOKarrayliteral && !isBlockAssignment)
            {
                Expressions* oldelems = existingAE.elements;
                Expressions* newelems = (cast(ArrayLiteralExp)newval).elements;
                Type elemtype = existingAE.type.nextOf();
                bool needsPostblit = e.op != TOKblit && e.e2.isLvalue();
                for (size_t j = 0; j < newelems.dim; j++)
                {
                    Expression newelem = (*newelems)[j];
                    newelem = paintTypeOntoLiteral(elemtype, newelem);
                    if (needsPostblit)
                    {
                        Expression x = evaluatePostblit(istate, newelem);
                        if (exceptionOrCantInterpret(x))
                            return x;
                    }
                    (*oldelems)[cast(size_t)(j + firstIndex)] = newelem;
                }
                return newval;
            }

            /* Block assignment, initialization of static arrays
             *   x[] = newval
             *  x may be a multidimensional static array. (Note that this
             *  only happens with array literals, never with strings).
             */
            struct RecursiveBlock
            {
                InterState* istate;
                Expression newval;
                bool refCopy;
                bool needsPostblit;
                bool needsDtor;

                extern (C++) Expression assignTo(ArrayLiteralExp ae)
                {
                    return assignTo(ae, 0, ae.elements.dim);
                }

                extern (C++) Expression assignTo(ArrayLiteralExp ae, size_t lwr, size_t upr)
                {
                    Expressions* w = ae.elements;
                    assert(ae.type.ty == Tsarray || ae.type.ty == Tarray);
                    bool directblk = (cast(TypeArray)ae.type).next.equivalent(newval.type);
                    for (size_t k = lwr; k < upr; k++)
                    {
                        if (!directblk && (*w)[k].op == TOKarrayliteral)
                        {
                            // Multidimensional array block assign
                            if (Expression ex = assignTo(cast(ArrayLiteralExp)(*w)[k]))
                                return ex;
                        }
                        else if (refCopy)
                        {
                            (*w)[k] = newval;
                        }
                        else if (!needsPostblit && !needsDtor)
                        {
                            assignInPlace((*w)[k], newval);
                        }
                        else
                        {
                            Expression oldelem = (*w)[k];
                            Expression tmpelem = needsDtor ? copyLiteral(oldelem).copy() : null;
                            assignInPlace(oldelem, newval);
                            if (needsPostblit)
                            {
                                if (Expression ex = evaluatePostblit(istate, oldelem))
                                    return ex;
                            }
                            if (needsDtor)
                            {
                                // https://issues.dlang.org/show_bug.cgi?id=14860
                                if (Expression ex = evaluateDtor(istate, tmpelem))
                                    return ex;
                            }
                        }
                    }
                    return null;
                }
            }

            Type tn = newval.type.toBasetype();
            bool wantRef = (tn.ty == Tarray || isAssocArray(tn) || tn.ty == Tclass);
            bool cow = newval.op != TOKstructliteral && newval.op != TOKarrayliteral && newval.op != TOKstring;
            Type tb = tn.baseElemOf();
            StructDeclaration sd = (tb.ty == Tstruct ? (cast(TypeStruct)tb).sym : null);

            RecursiveBlock rb;
            rb.istate = istate;
            rb.newval = newval;
            rb.refCopy = wantRef || cow;
            rb.needsPostblit = sd && sd.postblit && e.op != TOKblit && e.e2.isLvalue();
            rb.needsDtor = sd && sd.dtor && e.op == TOKassign;
            if (Expression ex = rb.assignTo(existingAE, cast(size_t)lowerbound, cast(size_t)upperbound))
                return ex;

            if (goal == ctfeNeedNothing)
                return null; // avoid creating an unused literal
            auto retslice = new SliceExp(e.loc, existingAE, new IntegerExp(e.loc, firstIndex, Type.tsize_t), new IntegerExp(e.loc, firstIndex + upperbound - lowerbound, Type.tsize_t));
            retslice.type = e.type;
            return interpret(retslice, istate);
        }

        e.error("slice operation %s = %s cannot be evaluated at compile time", e1.toChars(), newval.toChars());
        return CTFEExp.cantexp;
    }

    override void visit(AssignExp e)
    {
        interpretAssignCommon(e, null);
    }

    override void visit(BinAssignExp e)
    {
        switch (e.op)
        {
        case TOKaddass:
            interpretAssignCommon(e, &Add);
            return;

        case TOKminass:
            interpretAssignCommon(e, &Min);
            return;

        case TOKcatass:
            interpretAssignCommon(e, &ctfeCat);
            return;

        case TOKmulass:
            interpretAssignCommon(e, &Mul);
            return;

        case TOKdivass:
            interpretAssignCommon(e, &Div);
            return;

        case TOKmodass:
            interpretAssignCommon(e, &Mod);
            return;

        case TOKshlass:
            interpretAssignCommon(e, &Shl);
            return;

        case TOKshrass:
            interpretAssignCommon(e, &Shr);
            return;

        case TOKushrass:
            interpretAssignCommon(e, &Ushr);
            return;

        case TOKandass:
            interpretAssignCommon(e, &And);
            return;

        case TOKorass:
            interpretAssignCommon(e, &Or);
            return;

        case TOKxorass:
            interpretAssignCommon(e, &Xor);
            return;

        case TOKpowass:
            interpretAssignCommon(e, &Pow);
            return;

        default:
            assert(0);
        }
    }

    override void visit(PostExp e)
    {
        debug (LOG)
        {
            printf("%s PostExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        if (e.op == TOKplusplus)
            interpretAssignCommon(e, &Add, 1);
        else
            interpretAssignCommon(e, &Min, 1);
        debug (LOG)
        {
            if (CTFEExp.isCantExp(result))
                printf("PostExp::interpret() CANT\n");
        }
    }

    /* Return 1 if e is a p1 > p2 or p1 >= p2 pointer comparison;
     *       -1 if e is a p1 < p2 or p1 <= p2 pointer comparison;
     *        0 otherwise
     */
    static int isPointerCmpExp(Expression e, Expression* p1, Expression* p2)
    {
        int ret = 1;
        while (e.op == TOKnot)
        {
            ret *= -1;
            e = (cast(NotExp)e).e1;
        }
        switch (e.op)
        {
        case TOKlt:
        case TOKle:
            ret *= -1;
            goto case; /+ fall through +/
        case TOKgt:
        case TOKge:
            *p1 = (cast(BinExp)e).e1;
            *p2 = (cast(BinExp)e).e2;
            if (!(isPointer((*p1).type) && isPointer((*p2).type)))
                ret = 0;
            break;

        default:
            ret = 0;
            break;
        }
        return ret;
    }

    /** Negate a relational operator, eg >= becomes <
     */
    static TOK reverseRelation(TOK op)
    {
        switch (op)
        {
        case TOKge:
            return TOKlt;

        case TOKgt:
            return TOKle;

        case TOKle:
            return TOKgt;

        case TOKlt:
            return TOKge;

        default:
            assert(0);
        }
    }

    /** If this is a four pointer relation, evaluate it, else return NULL.
     *
     *  This is an expression of the form (p1 > q1 && p2 < q2) or (p1 < q1 || p2 > q2)
     *  where p1, p2 are expressions yielding pointers to memory block p,
     *  and q1, q2 are expressions yielding pointers to memory block q.
     *  This expression is valid even if p and q are independent memory
     *  blocks and are therefore not normally comparable; the && form returns true
     *  if [p1..p2] lies inside [q1..q2], and false otherwise; the || form returns
     *  true if [p1..p2] lies outside [q1..q2], and false otherwise.
     *
     *  Within the expression, any ordering of p1, p2, q1, q2 is permissible;
     *  the comparison operators can be any of >, <, <=, >=, provided that
     *  both directions (p > q and p < q) are checked. Additionally the
     *  relational sub-expressions can be negated, eg
     *  (!(q1 < p1) && p2 <= q2) is valid.
     */
    void interpretFourPointerRelation(BinExp e)
    {
        assert(e.op == TOKandand || e.op == TOKoror);

        /*  It can only be an isInside expression, if both e1 and e2 are
         *  directional pointer comparisons.
         *  Note that this check can be made statically; it does not depends on
         *  any runtime values. This allows a JIT implementation to compile a
         *  special AndAndPossiblyInside, keeping the normal AndAnd case efficient.
         */

        // Save the pointer expressions and the comparison directions,
        // so we can use them later.
        Expression p1 = null;
        Expression p2 = null;
        Expression p3 = null;
        Expression p4 = null;
        int dir1 = isPointerCmpExp(e.e1, &p1, &p2);
        int dir2 = isPointerCmpExp(e.e2, &p3, &p4);
        if (dir1 == 0 || dir2 == 0)
        {
            result = null;
            return;
        }

        //printf("FourPointerRelation %s\n", toChars());

        // Evaluate the first two pointers
        p1 = interpret(p1, istate);
        if (exceptionOrCant(p1))
            return;
        p2 = interpret(p2, istate);
        if (exceptionOrCant(p2))
            return;
        dinteger_t ofs1, ofs2;
        Expression agg1 = getAggregateFromPointer(p1, &ofs1);
        Expression agg2 = getAggregateFromPointer(p2, &ofs2);

        if (!pointToSameMemoryBlock(agg1, agg2) && agg1.op != TOKnull && agg2.op != TOKnull)
        {
            // Here it is either CANT_INTERPRET,
            // or an IsInside comparison returning false.
            p3 = interpret(p3, istate);
            if (CTFEExp.isCantExp(p3))
                return;
            // Note that it is NOT legal for it to throw an exception!
            Expression except = null;
            if (exceptionOrCantInterpret(p3))
                except = p3;
            else
            {
                p4 = interpret(p4, istate);
                if (CTFEExp.isCantExp(p4))
                {
                    result = p4;
                    return;
                }
                if (exceptionOrCantInterpret(p4))
                    except = p4;
            }
            if (except)
            {
                e.error("comparison %s of pointers to unrelated memory blocks remains indeterminate at compile time because exception %s was thrown while evaluating %s", e.e1.toChars(), except.toChars(), e.e2.toChars());
                result = CTFEExp.cantexp;
                return;
            }
            dinteger_t ofs3, ofs4;
            Expression agg3 = getAggregateFromPointer(p3, &ofs3);
            Expression agg4 = getAggregateFromPointer(p4, &ofs4);
            // The valid cases are:
            // p1 > p2 && p3 > p4  (same direction, also for < && <)
            // p1 > p2 && p3 < p4  (different direction, also < && >)
            // Changing any > into >= doesnt affect the result
            if ((dir1 == dir2 && pointToSameMemoryBlock(agg1, agg4) && pointToSameMemoryBlock(agg2, agg3)) || (dir1 != dir2 && pointToSameMemoryBlock(agg1, agg3) && pointToSameMemoryBlock(agg2, agg4)))
            {
                // it's a legal two-sided comparison
                result = new IntegerExp(e.loc, (e.op == TOKandand) ? 0 : 1, e.type);
                return;
            }
            // It's an invalid four-pointer comparison. Either the second
            // comparison is in the same direction as the first, or else
            // more than two memory blocks are involved (either two independent
            // invalid comparisons are present, or else agg3 == agg4).
            e.error("comparison %s of pointers to unrelated memory blocks is indeterminate at compile time, even when combined with %s.", e.e1.toChars(), e.e2.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        // The first pointer expression didn't need special treatment, so we
        // we need to interpret the entire expression exactly as a normal && or ||.
        // This is easy because we haven't evaluated e2 at all yet, and we already
        // know it will return a bool.
        // But we mustn't evaluate the pointer expressions in e1 again, in case
        // they have side-effects.
        bool nott = false;
        Expression ex = e.e1;
        while (ex.op == TOKnot)
        {
            nott = !nott;
            ex = (cast(NotExp)ex).e1;
        }
        TOK cmpop = ex.op;
        if (nott)
            cmpop = reverseRelation(cmpop);
        int cmp = comparePointers(e.loc, cmpop, e.e1.type, agg1, ofs1, agg2, ofs2);
        // We already know this is a valid comparison.
        assert(cmp >= 0);
        if (e.op == TOKandand && cmp == 1 || e.op == TOKoror && cmp == 0)
        {
            result = interpret(e.e2, istate);
            return;
        }
        result = new IntegerExp(e.loc, (e.op == TOKandand) ? 0 : 1, e.type);
    }

    override void visit(AndAndExp e)
    {
        debug (LOG)
        {
            printf("%s AndAndExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        // Check for an insidePointer expression, evaluate it if so
        interpretFourPointerRelation(e);
        if (result)
            return;

        result = interpret(e.e1, istate);
        if (exceptionOrCant(result))
            return;

        int res;
        if (result.isBool(false))
            res = 0;
        else if (isTrueBool(result))
        {
            result = interpret(e.e2, istate);
            if (exceptionOrCant(result))
                return;
            if (result.op == TOKvoidexp)
            {
                assert(e.type.ty == Tvoid);
                result = null;
                return;
            }
            if (result.isBool(false))
                res = 0;
            else if (isTrueBool(result))
                res = 1;
            else
            {
                result.error("%s does not evaluate to a boolean", result.toChars());
                result = CTFEExp.cantexp;
                return;
            }
        }
        else
        {
            result.error("%s cannot be interpreted as a boolean", result.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        if (goal != ctfeNeedNothing)
            result = new IntegerExp(e.loc, res, e.type);
    }

    override void visit(OrOrExp e)
    {
        debug (LOG)
        {
            printf("%s OrOrExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        // Check for an insidePointer expression, evaluate it if so
        interpretFourPointerRelation(e);
        if (result)
            return;

        result = interpret(e.e1, istate);
        if (exceptionOrCant(result))
            return;

        int res;
        if (isTrueBool(result))
            res = 1;
        else if (result.isBool(false))
        {
            result = interpret(e.e2, istate);
            if (exceptionOrCant(result))
                return;
            if (result.op == TOKvoidexp)
            {
                assert(e.type.ty == Tvoid);
                result = null;
                return;
            }
            if (result.isBool(false))
                res = 0;
            else if (isTrueBool(result))
                res = 1;
            else
            {
                result.error("%s cannot be interpreted as a boolean", result.toChars());
                result = CTFEExp.cantexp;
                return;
            }
        }
        else
        {
            result.error("%s cannot be interpreted as a boolean", result.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        if (goal != ctfeNeedNothing)
            result = new IntegerExp(e.loc, res, e.type);
    }

    // Print a stack trace, starting from callingExp which called fd.
    // To shorten the stack trace, try to detect recursion.
    void showCtfeBackTrace(CallExp callingExp, FuncDeclaration fd)
    {
        if (CtfeStatus.stackTraceCallsToSuppress > 0)
        {
            --CtfeStatus.stackTraceCallsToSuppress;
            return;
        }
        errorSupplemental(callingExp.loc, "called from here: %s", callingExp.toChars());
        // Quit if it's not worth trying to compress the stack trace
        if (CtfeStatus.callDepth < 6 || global.params.verbose)
            return;
        // Recursion happens if the current function already exists in the call stack.
        int numToSuppress = 0;
        int recurseCount = 0;
        int depthSoFar = 0;
        InterState* lastRecurse = istate;
        for (InterState* cur = istate; cur; cur = cur.caller)
        {
            if (cur.fd == fd)
            {
                ++recurseCount;
                numToSuppress = depthSoFar;
                lastRecurse = cur;
            }
            ++depthSoFar;
        }
        // We need at least three calls to the same function, to make compression worthwhile
        if (recurseCount < 2)
            return;
        // We found a useful recursion.  Print all the calls involved in the recursion
        errorSupplemental(fd.loc, "%d recursive calls to function %s", recurseCount, fd.toChars());
        for (InterState* cur = istate; cur.fd != fd; cur = cur.caller)
        {
            errorSupplemental(cur.fd.loc, "recursively called from function %s", cur.fd.toChars());
        }
        // We probably didn't enter the recursion in this function.
        // Go deeper to find the real beginning.
        InterState* cur = istate;
        while (lastRecurse.caller && cur.fd == lastRecurse.caller.fd)
        {
            cur = cur.caller;
            lastRecurse = lastRecurse.caller;
            ++numToSuppress;
        }
        CtfeStatus.stackTraceCallsToSuppress = numToSuppress;
    }

    override void visit(CallExp e)
    {
        debug (LOG)
        {
            printf("%s CallExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        Expression pthis = null;
        FuncDeclaration fd = null;

        Expression ecall = interpret(e.e1, istate);
        if (exceptionOrCant(ecall))
            return;

        if (ecall.op == TOKdotvar)
        {
            DotVarExp dve = cast(DotVarExp)ecall;

            // Calling a member function
            pthis = dve.e1;
            fd = dve.var.isFuncDeclaration();
            assert(fd);

            if (pthis.op == TOKdottype)
                pthis = (cast(DotTypeExp)dve.e1).e1;
        }
        else if (ecall.op == TOKvar)
        {
            fd = (cast(VarExp)ecall).var.isFuncDeclaration();
            assert(fd);

            if (fd.ident == Id._ArrayPostblit || fd.ident == Id._ArrayDtor)
            {
                assert(e.arguments.dim == 1);
                Expression ea = (*e.arguments)[0];
                //printf("1 ea = %s %s\n", ea.type.toChars(), ea.toChars());
                if (ea.op == TOKslice)
                    ea = (cast(SliceExp)ea).e1;
                if (ea.op == TOKcast)
                    ea = (cast(CastExp)ea).e1;

                //printf("2 ea = %s, %s %s\n", ea.type.toChars(), Token::toChars(ea.op), ea.toChars());
                if (ea.op == TOKvar || ea.op == TOKsymoff)
                    result = getVarExp(e.loc, istate, (cast(SymbolExp)ea).var, ctfeNeedRvalue);
                else if (ea.op == TOKaddress)
                    result = interpret((cast(AddrExp)ea).e1, istate);
                else
                    assert(0);
                if (CTFEExp.isCantExp(result))
                    return;

                if (fd.ident == Id._ArrayPostblit)
                    result = evaluatePostblit(istate, result);
                else
                    result = evaluateDtor(istate, result);
                if (!result)
                    result = CTFEExp.voidexp;
                return;
            }
        }
        else if (ecall.op == TOKsymoff)
        {
            SymOffExp soe = cast(SymOffExp)ecall;
            fd = soe.var.isFuncDeclaration();
            assert(fd && soe.offset == 0);
        }
        else if (ecall.op == TOKdelegate)
        {
            // Calling a delegate
            fd = (cast(DelegateExp)ecall).func;
            pthis = (cast(DelegateExp)ecall).e1;

            // Special handling for: &nestedfunc --> DelegateExp(VarExp(nestedfunc), nestedfunc)
            if (pthis.op == TOKvar && (cast(VarExp)pthis).var == fd)
                pthis = null; // context is not necessary for CTFE
        }
        else if (ecall.op == TOKfunction)
        {
            // Calling a delegate literal
            fd = (cast(FuncExp)ecall).fd;
        }
        else
        {
            // delegate.funcptr()
            // others
            e.error("cannot call %s at compile time", e.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        if (!fd)
        {
            e.error("CTFE internal error: cannot evaluate %s at compile time", e.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        if (pthis)
        {
            // Member function call

            // Currently this is satisfied because closure is not yet supported.
            assert(!fd.isNested());

            if (pthis.op == TOKtypeid)
            {
                pthis.error("static variable %s cannot be read at compile time", pthis.toChars());
                result = CTFEExp.cantexp;
                return;
            }
            assert(pthis);

            if (pthis.op == TOKnull)
            {
                assert(pthis.type.toBasetype().ty == Tclass);
                e.error("function call through null class reference %s", pthis.toChars());
                result = CTFEExp.cantexp;
                return;
            }
            assert(pthis.op == TOKstructliteral || pthis.op == TOKclassreference);

            if (fd.isVirtual() && !e.directcall)
            {
                // Make a virtual function call.
                // Get the function from the vtable of the original class
                assert(pthis.op == TOKclassreference);
                ClassDeclaration cd = (cast(ClassReferenceExp)pthis).originalClass();

                // We can't just use the vtable index to look it up, because
                // vtables for interfaces don't get populated until the glue layer.
                fd = cd.findFunc(fd.ident, cast(TypeFunction)fd.type);
                assert(fd);
            }
        }

        if (fd && fd.semanticRun >= PASSsemantic3done && fd.semantic3Errors)
        {
            e.error("CTFE failed because of previous errors in %s", fd.toChars());
            result = CTFEExp.cantexp;
            return;
        }

        // Check for built-in functions
        result = evaluateIfBuiltin(istate, e.loc, fd, e.arguments, pthis);
        if (result)
            return;

        if (!fd.fbody)
        {
            e.error("%s cannot be interpreted at compile time, because it has no available source code", fd.toChars());
            result = CTFEExp.cantexp;
            return;
        }

        result = interpret(fd, istate, e.arguments, pthis);
        if (result.op == TOKvoidexp)
            return;
        if (!exceptionOrCantInterpret(result))
        {
            if (goal != ctfeNeedLvalue) // Peel off CTFE reference if it's unnesessary
                result = interpret(result, istate);
        }
        if (!exceptionOrCantInterpret(result))
        {
            result = paintTypeOntoLiteral(e.type, result);
            result.loc = e.loc;
        }
        else if (CTFEExp.isCantExp(result) && !global.gag)
            showCtfeBackTrace(e, fd); // Print a stack trace.
    }

    override void visit(CommaExp e)
    {
        debug (LOG)
        {
            printf("%s CommaExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        CommaExp firstComma = e;
        while (firstComma.e1.op == TOKcomma)
            firstComma = cast(CommaExp)firstComma.e1;

        // If it creates a variable, and there's no context for
        // the variable to be created in, we need to create one now.
        InterState istateComma;
        if (!istate && firstComma.e1.op == TOKdeclaration)
        {
            ctfeStack.startFrame(null);
            istate = &istateComma;
        }

        result = CTFEExp.cantexp;

        // If the comma returns a temporary variable, it needs to be an lvalue
        // (this is particularly important for struct constructors)
        if (e.e1.op == TOKdeclaration &&
            e.e2.op == TOKvar &&
            (cast(DeclarationExp)e.e1).declaration == (cast(VarExp)e.e2).var &&
            (cast(VarExp)e.e2).var.storage_class & STCctfe)
        {
            VarExp ve = cast(VarExp)e.e2;
            VarDeclaration v = ve.var.isVarDeclaration();
            ctfeStack.push(v);
            if (!v._init && !getValue(v))
            {
                setValue(v, copyLiteral(v.type.defaultInitLiteral(e.loc)).copy());
            }
            if (!getValue(v))
            {
                Expression newval = v._init.toExpression();
                // Bug 4027. Copy constructors are a weird case where the
                // initializer is a void function (the variable is modified
                // through a reference parameter instead).
                newval = interpret(newval, istate);
                if (exceptionOrCant(newval))
                    goto Lfin;
                if (newval.op != TOKvoidexp)
                {
                    // v isn't necessarily null.
                    setValueWithoutChecking(v, copyLiteral(newval).copy());
                }
            }
            result = interpret(e.e2, istate, goal);
        }
        else
        {
            result = interpret(e.e1, istate, ctfeNeedNothing);
            if (exceptionOrCant(result))
                goto Lfin;
            result = interpret(e.e2, istate, goal);
        }
    Lfin:
        // If we created a temporary stack frame, end it now.
        if (istate == &istateComma)
            ctfeStack.endFrame();
    }

    override void visit(CondExp e)
    {
        debug (LOG)
        {
            printf("%s CondExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        if (isPointer(e.econd.type))
        {
            result = interpret(e.econd, istate);
            if (exceptionOrCant(result))
                return;
            if (result.op != TOKnull)
                result = new IntegerExp(e.loc, 1, Type.tbool);
        }
        else
            result = interpret(e.econd, istate);
        if (exceptionOrCant(result))
            return;
        if (isTrueBool(result))
            result = interpret(e.e1, istate, goal);
        else if (result.isBool(false))
            result = interpret(e.e2, istate, goal);
        else
        {
            e.error("%s does not evaluate to boolean result at compile time", e.econd.toChars());
            result = CTFEExp.cantexp;
        }
    }

    override void visit(ArrayLengthExp e)
    {
        debug (LOG)
        {
            printf("%s ArrayLengthExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        Expression e1 = interpret(e.e1, istate);
        assert(e1);
        if (exceptionOrCant(e1))
            return;
        if (e1.op != TOKstring && e1.op != TOKarrayliteral && e1.op != TOKslice && e1.op != TOKnull)
        {
            e.error("%s cannot be evaluated at compile time", e.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        result = new IntegerExp(e.loc, resolveArrayLength(e1), e.type);
    }

    override void visit(DelegatePtrExp e)
    {
        debug (LOG)
        {
            printf("%s DelegatePtrExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        Expression e1 = interpret(e.e1, istate);
        assert(e1);
        if (exceptionOrCant(e1))
            return;
        e.error("%s cannot be evaluated at compile time", e.toChars());
        result = CTFEExp.cantexp;
    }

    override void visit(DelegateFuncptrExp e)
    {
        debug (LOG)
        {
            printf("%s DelegateFuncptrExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        Expression e1 = interpret(e.e1, istate);
        assert(e1);
        if (exceptionOrCant(e1))
            return;
        e.error("%s cannot be evaluated at compile time", e.toChars());
        result = CTFEExp.cantexp;
    }

    static bool resolveIndexing(IndexExp e, InterState* istate, Expression* pagg, uinteger_t* pidx, bool modify)
    {
        assert(e.e1.type.toBasetype().ty != Taarray);

        if (e.e1.type.toBasetype().ty == Tpointer)
        {
            // Indexing a pointer. Note that there is no $ in this case.
            Expression e1 = interpret(e.e1, istate);
            if (exceptionOrCantInterpret(e1))
                return false;

            Expression e2 = interpret(e.e2, istate);
            if (exceptionOrCantInterpret(e2))
                return false;
            sinteger_t indx = e2.toInteger();

            dinteger_t ofs;
            Expression agg = getAggregateFromPointer(e1, &ofs);

            if (agg.op == TOKnull)
            {
                e.error("cannot index through null pointer %s", e.e1.toChars());
                return false;
            }
            if (agg.op == TOKint64)
            {
                e.error("cannot index through invalid pointer %s of value %s", e.e1.toChars(), e1.toChars());
                return false;
            }
            // Pointer to a non-array variable
            if (agg.op == TOKsymoff)
            {
                e.error("mutable variable %s cannot be %s at compile time, even through a pointer", cast(char*)(modify ? "modified" : "read"), (cast(SymOffExp)agg).var.toChars());
                return false;
            }

            if (agg.op == TOKarrayliteral || agg.op == TOKstring)
            {
                dinteger_t len = resolveArrayLength(agg);
                if (ofs + indx >= len)
                {
                    e.error("pointer index [%lld] exceeds allocated memory block [0..%lld]", ofs + indx, len);
                    return false;
                }
            }
            else
            {
                if (ofs + indx != 0)
                {
                    e.error("pointer index [%lld] lies outside memory block [0..1]", ofs + indx);
                    return false;
                }
            }
            *pagg = agg;
            *pidx = ofs + indx;
            return true;
        }

        Expression e1 = interpret(e.e1, istate);
        if (exceptionOrCantInterpret(e1))
            return false;
        if (e1.op == TOKnull)
        {
            e.error("cannot index null array %s", e.e1.toChars());
            return false;
        }
        if (e1.op == TOKvector)
            e1 = (cast(VectorExp)e1).e1;

        // Set the $ variable, and find the array literal to modify
        if (e1.op != TOKarrayliteral && e1.op != TOKstring && e1.op != TOKslice)
        {
            e.error("cannot determine length of %s at compile time", e.e1.toChars());
            return false;
        }

        dinteger_t len = resolveArrayLength(e1);
        if (e.lengthVar)
        {
            Expression dollarExp = new IntegerExp(e.loc, len, Type.tsize_t);
            ctfeStack.push(e.lengthVar);
            setValue(e.lengthVar, dollarExp);
        }
        Expression e2 = interpret(e.e2, istate);
        if (e.lengthVar)
            ctfeStack.pop(e.lengthVar); // $ is defined only inside []
        if (exceptionOrCantInterpret(e2))
            return false;
        if (e2.op != TOKint64)
        {
            e.error("CTFE internal error: non-integral index [%s]", e.e2.toChars());
            return false;
        }

        if (e1.op == TOKslice)
        {
            // Simplify index of slice: agg[lwr..upr][indx] --> agg[indx']
            uinteger_t index = e2.toInteger();
            uinteger_t ilwr = (cast(SliceExp)e1).lwr.toInteger();
            uinteger_t iupr = (cast(SliceExp)e1).upr.toInteger();

            if (index > iupr - ilwr)
            {
                e.error("index %llu exceeds array length %llu", index, iupr - ilwr);
                return false;
            }
            *pagg = (cast(SliceExp)e1).e1;
            *pidx = index + ilwr;
        }
        else
        {
            *pagg = e1;
            *pidx = e2.toInteger();
            if (len <= *pidx)
            {
                e.error("array index %lld is out of bounds [0..%lld]", *pidx, len);
                return false;
            }
        }
        return true;
    }

    override void visit(IndexExp e)
    {
        debug (LOG)
        {
            printf("%s IndexExp::interpret() %s, goal = %d\n", e.loc.toChars(), e.toChars(), goal);
        }
        if (e.e1.type.toBasetype().ty == Tpointer)
        {
            Expression agg;
            uinteger_t indexToAccess;
            if (!resolveIndexing(e, istate, &agg, &indexToAccess, false))
            {
                result = CTFEExp.cantexp;
                return;
            }
            if (agg.op == TOKarrayliteral || agg.op == TOKstring)
            {
                if (goal == ctfeNeedLvalue)
                {
                    // if we need a reference, IndexExp shouldn't be interpreting
                    // the expression to a value, it should stay as a reference
                    result = new IndexExp(e.loc, agg, new IntegerExp(e.e2.loc, indexToAccess, e.e2.type));
                    result.type = e.type;
                    return;
                }
                result = ctfeIndex(e.loc, e.type, agg, indexToAccess);
                return;
            }
            else
            {
                assert(indexToAccess == 0);
                result = interpret(agg, istate, goal);
                if (exceptionOrCant(result))
                    return;
                result = paintTypeOntoLiteral(e.type, result);
                return;
            }
        }

        if (e.e1.type.toBasetype().ty == Taarray)
        {
            Expression e1 = interpret(e.e1, istate);
            if (exceptionOrCant(e1))
                return;
            if (e1.op == TOKnull)
            {
                if (goal == ctfeNeedLvalue && e1.type.ty == Taarray && e.modifiable)
                {
                    assert(0); // does not reach here?
                }
                e.error("cannot index null array %s", e.e1.toChars());
                result = CTFEExp.cantexp;
                return;
            }
            Expression e2 = interpret(e.e2, istate);
            if (exceptionOrCant(e2))
                return;

            if (goal == ctfeNeedLvalue)
            {
                // Pointer or reference of a scalar type
                if (e1 == e.e1 && e2 == e.e2)
                    result = e;
                else
                {
                    result = new IndexExp(e.loc, e1, e2);
                    result.type = e.type;
                }
                return;
            }

            assert(e1.op == TOKassocarrayliteral);
            e2 = resolveSlice(e2);
            result = findKeyInAA(e.loc, cast(AssocArrayLiteralExp)e1, e2);
            if (!result)
            {
                e.error("key %s not found in associative array %s", e2.toChars(), e.e1.toChars());
                result = CTFEExp.cantexp;
            }
            return;
        }

        Expression agg;
        uinteger_t indexToAccess;
        if (!resolveIndexing(e, istate, &agg, &indexToAccess, false))
        {
            result = CTFEExp.cantexp;
            return;
        }

        if (goal == ctfeNeedLvalue)
        {
            Expression e2 = new IntegerExp(e.e2.loc, indexToAccess, Type.tsize_t);
            result = new IndexExp(e.loc, agg, e2);
            result.type = e.type;
            return;
        }

        result = ctfeIndex(e.loc, e.type, agg, indexToAccess);
        if (exceptionOrCant(result))
            return;
        if (result.op == TOKvoid)
        {
            e.error("%s is used before initialized", e.toChars());
            errorSupplemental(result.loc, "originally uninitialized here");
            result = CTFEExp.cantexp;
            return;
        }
        result = paintTypeOntoLiteral(e.type, result);
    }

    override void visit(SliceExp e)
    {
        debug (LOG)
        {
            printf("%s SliceExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        if (e.e1.type.toBasetype().ty == Tpointer)
        {
            // Slicing a pointer. Note that there is no $ in this case.
            Expression e1 = interpret(e.e1, istate);
            if (exceptionOrCant(e1))
                return;
            if (e1.op == TOKint64)
            {
                e.error("cannot slice invalid pointer %s of value %s", e.e1.toChars(), e1.toChars());
                result = CTFEExp.cantexp;
                return;
            }

            /* Evaluate lower and upper bounds of slice
             */
            Expression lwr = interpret(e.lwr, istate);
            if (exceptionOrCant(lwr))
                return;
            Expression upr = interpret(e.upr, istate);
            if (exceptionOrCant(upr))
                return;
            uinteger_t ilwr = lwr.toInteger();
            uinteger_t iupr = upr.toInteger();

            dinteger_t ofs;
            Expression agg = getAggregateFromPointer(e1, &ofs);
            ilwr += ofs;
            iupr += ofs;
            if (agg.op == TOKnull)
            {
                if (iupr == ilwr)
                {
                    result = new NullExp(e.loc);
                    result.type = e.type;
                    return;
                }
                e.error("cannot slice null pointer %s", e.e1.toChars());
                result = CTFEExp.cantexp;
                return;
            }
            if (agg.op == TOKsymoff)
            {
                e.error("slicing pointers to static variables is not supported in CTFE");
                result = CTFEExp.cantexp;
                return;
            }
            if (agg.op != TOKarrayliteral && agg.op != TOKstring)
            {
                e.error("pointer %s cannot be sliced at compile time (it does not point to an array)", e.e1.toChars());
                result = CTFEExp.cantexp;
                return;
            }
            assert(agg.op == TOKarrayliteral || agg.op == TOKstring);
            dinteger_t len = ArrayLength(Type.tsize_t, agg).exp().toInteger();
            //Type *pointee = ((TypePointer *)agg.type)->next;
            if (iupr > (len + 1) || iupr < ilwr)
            {
                e.error("pointer slice [%lld..%lld] exceeds allocated memory block [0..%lld]", ilwr, iupr, len);
                result = CTFEExp.cantexp;
                return;
            }
            if (ofs != 0)
            {
                lwr = new IntegerExp(e.loc, ilwr, lwr.type);
                upr = new IntegerExp(e.loc, iupr, upr.type);
            }
            result = new SliceExp(e.loc, agg, lwr, upr);
            result.type = e.type;
            return;
        }

        Expression e1 = interpret(e.e1, istate);
        if (exceptionOrCant(e1))
            return;

        if (!e.lwr)
        {
            result = paintTypeOntoLiteral(e.type, e1);
            return;
        }

        /* Set the $ variable
         */
        if (e1.op != TOKarrayliteral && e1.op != TOKstring && e1.op != TOKnull && e1.op != TOKslice)
        {
            e.error("cannot determine length of %s at compile time", e1.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        uinteger_t dollar = resolveArrayLength(e1);
        if (e.lengthVar)
        {
            auto dollarExp = new IntegerExp(e.loc, dollar, Type.tsize_t);
            ctfeStack.push(e.lengthVar);
            setValue(e.lengthVar, dollarExp);
        }

        /* Evaluate lower and upper bounds of slice
         */
        Expression lwr = interpret(e.lwr, istate);
        if (exceptionOrCant(lwr))
        {
            if (e.lengthVar)
                ctfeStack.pop(e.lengthVar);
            return;
        }
        Expression upr = interpret(e.upr, istate);
        if (exceptionOrCant(upr))
        {
            if (e.lengthVar)
                ctfeStack.pop(e.lengthVar);
            return;
        }
        if (e.lengthVar)
            ctfeStack.pop(e.lengthVar); // $ is defined only inside [L..U]

        uinteger_t ilwr = lwr.toInteger();
        uinteger_t iupr = upr.toInteger();
        if (e1.op == TOKnull)
        {
            if (ilwr == 0 && iupr == 0)
            {
                result = e1;
                return;
            }
            e1.error("slice [%llu..%llu] is out of bounds", ilwr, iupr);
            result = CTFEExp.cantexp;
            return;
        }
        if (e1.op == TOKslice)
        {
            SliceExp se = cast(SliceExp)e1;
            // Simplify slice of slice:
            //  aggregate[lo1..up1][lwr..upr] ---> aggregate[lwr'..upr']
            uinteger_t lo1 = se.lwr.toInteger();
            uinteger_t up1 = se.upr.toInteger();
            if (ilwr > iupr || iupr > up1 - lo1)
            {
                e.error("slice[%llu..%llu] exceeds array bounds[%llu..%llu]", ilwr, iupr, lo1, up1);
                result = CTFEExp.cantexp;
                return;
            }
            ilwr += lo1;
            iupr += lo1;
            result = new SliceExp(e.loc, se.e1, new IntegerExp(e.loc, ilwr, lwr.type), new IntegerExp(e.loc, iupr, upr.type));
            result.type = e.type;
            return;
        }
        if (e1.op == TOKarrayliteral || e1.op == TOKstring)
        {
            if (iupr < ilwr || dollar < iupr)
            {
                e.error("slice [%lld..%lld] exceeds array bounds [0..%lld]", ilwr, iupr, dollar);
                result = CTFEExp.cantexp;
                return;
            }
        }
        result = new SliceExp(e.loc, e1, lwr, upr);
        result.type = e.type;
    }

    override void visit(InExp e)
    {
        debug (LOG)
        {
            printf("%s InExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        Expression e1 = interpret(e.e1, istate);
        if (exceptionOrCant(e1))
            return;
        Expression e2 = interpret(e.e2, istate);
        if (exceptionOrCant(e2))
            return;
        if (e2.op == TOKnull)
        {
            result = new NullExp(e.loc, e.type);
            return;
        }
        if (e2.op != TOKassocarrayliteral)
        {
            e.error("%s cannot be interpreted at compile time", e.toChars());
            result = CTFEExp.cantexp;
            return;
        }

        e1 = resolveSlice(e1);
        result = findKeyInAA(e.loc, cast(AssocArrayLiteralExp)e2, e1);
        if (exceptionOrCant(result))
            return;
        if (!result)
        {
            result = new NullExp(e.loc, e.type);
        }
        else
        {
            // Create a CTFE pointer &aa[index]
            result = new IndexExp(e.loc, e2, e1);
            result.type = e.type.nextOf();
            result = new AddrExp(e.loc, result);
            result.type = e.type;
        }
    }

    override void visit(CatExp e)
    {
        debug (LOG)
        {
            printf("%s CatExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        Expression e1 = interpret(e.e1, istate);
        if (exceptionOrCant(e1))
            return;
        Expression e2 = interpret(e.e2, istate);
        if (exceptionOrCant(e2))
            return;

        e1 = resolveSlice(e1);
        e2 = resolveSlice(e2);
        result = ctfeCat(e.loc, e.type, e1, e2).copy();
        if (CTFEExp.isCantExp(result))
        {
            e.error("%s cannot be interpreted at compile time", e.toChars());
            return;
        }
        // We know we still own it, because we interpreted both e1 and e2
        if (result.op == TOKarrayliteral)
        {
            ArrayLiteralExp ale = cast(ArrayLiteralExp)result;
            ale.ownedByCtfe = OWNEDctfe;

            // https://issues.dlang.org/show_bug.cgi?id=14686
            for (size_t i = 0; i < ale.elements.dim; i++)
            {
                Expression ex = evaluatePostblit(istate, (*ale.elements)[i]);
                if (exceptionOrCant(ex))
                    return;
            }
        }
        if (result.op == TOKstring)
            (cast(StringExp)result).ownedByCtfe = OWNEDctfe;
    }

    override void visit(DeleteExp e)
    {
        debug (LOG)
        {
            printf("%s DeleteExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        result = interpret(e.e1, istate);
        if (exceptionOrCant(result))
            return;

        if (result.op == TOKnull)
        {
            result = CTFEExp.voidexp;
            return;
        }

        auto tb = e.e1.type.toBasetype();
        switch (tb.ty)
        {
        case Tclass:
            if (result.op != TOKclassreference)
            {
                e.error("delete on invalid class reference '%s'", result.toChars());
                result = CTFEExp.cantexp;
                return;
            }

            auto cre = cast(ClassReferenceExp)result;
            auto cd = cre.originalClass();
            if (cd.aggDelete)
            {
                e.error("member deallocators not supported by CTFE");
                result = CTFEExp.cantexp;
                return;
            }

            if (cd.dtor)
            {
                result = interpret(cd.dtor, istate, null, cre);
                if (exceptionOrCant(result))
                    return;
            }
            break;

        case Tpointer:
            tb = (cast(TypePointer)tb).next.toBasetype();
            if (tb.ty == Tstruct)
            {
                if (result.op != TOKaddress ||
                    (cast(AddrExp)result).e1.op != TOKstructliteral)
                {
                    e.error("delete on invalid struct pointer '%s'", result.toChars());
                    result = CTFEExp.cantexp;
                    return;
                }

                auto sd = (cast(TypeStruct)tb).sym;
                auto sle = cast(StructLiteralExp)(cast(AddrExp)result).e1;
                if (sd.aggDelete)
                {
                    e.error("member deallocators not supported by CTFE");
                    result = CTFEExp.cantexp;
                    return;
                }

                if (sd.dtor)
                {
                    result = interpret(sd.dtor, istate, null, sle);
                    if (exceptionOrCant(result))
                        return;
                }
            }
            break;

        case Tarray:
            auto tv = tb.nextOf().baseElemOf();
            if (tv.ty == Tstruct)
            {
                if (result.op != TOKarrayliteral)
                {
                    e.error("delete on invalid struct array '%s'", result.toChars());
                    result = CTFEExp.cantexp;
                    return;
                }

                auto sd = (cast(TypeStruct)tv).sym;
                if (sd.aggDelete)
                {
                    e.error("member deallocators not supported by CTFE");
                    result = CTFEExp.cantexp;
                    return;
                }

                if (sd.dtor)
                {
                    auto ale = cast(ArrayLiteralExp)result;
                    foreach (el; *ale.elements)
                    {
                        result = interpret(sd.dtor, istate, null, el);
                        if (exceptionOrCant(result))
                            return;
                    }
                }
            }
            break;

        default:
            assert(0);
        }
        result = CTFEExp.voidexp;
    }

    override void visit(CastExp e)
    {
        debug (LOG)
        {
            printf("%s CastExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        Expression e1 = interpret(e.e1, istate, goal);
        if (exceptionOrCant(e1))
            return;
        // If the expression has been cast to void, do nothing.
        if (e.to.ty == Tvoid)
        {
            result = CTFEExp.voidexp;
            return;
        }
        if (e.to.ty == Tpointer && e1.op != TOKnull)
        {
            Type pointee = (cast(TypePointer)e.type).next;
            // Implement special cases of normally-unsafe casts
            if (e1.op == TOKint64)
            {
                // Happens with Windows HANDLEs, for example.
                result = paintTypeOntoLiteral(e.to, e1);
                return;
            }

            bool castToSarrayPointer = false;
            bool castBackFromVoid = false;
            if (e1.type.ty == Tarray || e1.type.ty == Tsarray || e1.type.ty == Tpointer)
            {
                // Check for unsupported type painting operations
                // For slices, we need the type being sliced,
                // since it may have already been type painted
                Type elemtype = e1.type.nextOf();
                if (e1.op == TOKslice)
                    elemtype = (cast(SliceExp)e1).e1.type.nextOf();

                // Allow casts from X* to void *, and X** to void** for any X.
                // But don't allow cast from X* to void**.
                // So, we strip all matching * from source and target to find X.
                // Allow casts to X* from void* only if the 'void' was originally an X;
                // we check this later on.
                Type ultimatePointee = pointee;
                Type ultimateSrc = elemtype;
                while (ultimatePointee.ty == Tpointer && ultimateSrc.ty == Tpointer)
                {
                    ultimatePointee = ultimatePointee.nextOf();
                    ultimateSrc = ultimateSrc.nextOf();
                }
                if (ultimatePointee.ty == Tsarray && ultimatePointee.nextOf().equivalent(ultimateSrc))
                {
                    castToSarrayPointer = true;
                }
                else if (ultimatePointee.ty != Tvoid && ultimateSrc.ty != Tvoid && !isSafePointerCast(elemtype, pointee))
                {
                    e.error("reinterpreting cast from %s* to %s* is not supported in CTFE", elemtype.toChars(), pointee.toChars());
                    result = CTFEExp.cantexp;
                    return;
                }
                if (ultimateSrc.ty == Tvoid)
                    castBackFromVoid = true;
            }

            if (e1.op == TOKslice)
            {
                if ((cast(SliceExp)e1).e1.op == TOKnull)
                {
                    result = paintTypeOntoLiteral(e.type, (cast(SliceExp)e1).e1);
                    return;
                }
                // Create a CTFE pointer &aggregate[1..2]
                result = new IndexExp(e.loc, (cast(SliceExp)e1).e1, (cast(SliceExp)e1).lwr);
                result.type = e.type.nextOf();
                result = new AddrExp(e.loc, result);
                result.type = e.type;
                return;
            }
            if (e1.op == TOKarrayliteral || e1.op == TOKstring)
            {
                // Create a CTFE pointer &[1,2,3][0] or &"abc"[0]
                result = new IndexExp(e.loc, e1, new IntegerExp(e.loc, 0, Type.tsize_t));
                result.type = e.type.nextOf();
                result = new AddrExp(e.loc, result);
                result.type = e.type;
                return;
            }
            if (e1.op == TOKindex && !(cast(IndexExp)e1).e1.type.equals(e1.type))
            {
                // type painting operation
                IndexExp ie = cast(IndexExp)e1;
                result = new IndexExp(e1.loc, ie.e1, ie.e2);
                if (castBackFromVoid)
                {
                    // get the original type. For strings, it's just the type...
                    Type origType = ie.e1.type.nextOf();
                    // ..but for arrays of type void*, it's the type of the element
                    Expression xx = null;
                    if (ie.e1.op == TOKarrayliteral && ie.e2.op == TOKint64)
                    {
                        ArrayLiteralExp ale = cast(ArrayLiteralExp)ie.e1;
                        size_t indx = cast(size_t)ie.e2.toInteger();
                        if (indx < ale.elements.dim)
                            xx = (*ale.elements)[indx];
                    }
                    if (xx && xx.op == TOKindex)
                        origType = (cast(IndexExp)xx).e1.type.nextOf();
                    else if (xx && xx.op == TOKaddress)
                        origType = (cast(AddrExp)xx).e1.type;
                    else if (xx && xx.op == TOKvar)
                        origType = (cast(VarExp)xx).var.type;
                    if (!isSafePointerCast(origType, pointee))
                    {
                        e.error("using void* to reinterpret cast from %s* to %s* is not supported in CTFE", origType.toChars(), pointee.toChars());
                        result = CTFEExp.cantexp;
                        return;
                    }
                }
                result.type = e.type;
                return;
            }
            if (e1.op == TOKaddress)
            {
                Type origType = (cast(AddrExp)e1).e1.type;
                if (isSafePointerCast(origType, pointee))
                {
                    result = new AddrExp(e.loc, (cast(AddrExp)e1).e1);
                    result.type = e.type;
                    return;
                }
                if (castToSarrayPointer && pointee.toBasetype().ty == Tsarray && (cast(AddrExp)e1).e1.op == TOKindex)
                {
                    // &val[idx]
                    dinteger_t dim = (cast(TypeSArray)pointee.toBasetype()).dim.toInteger();
                    IndexExp ie = cast(IndexExp)(cast(AddrExp)e1).e1;
                    Expression lwr = ie.e2;
                    Expression upr = new IntegerExp(ie.e2.loc, ie.e2.toInteger() + dim, Type.tsize_t);

                    // Create a CTFE pointer &val[idx..idx+dim]
                    result = new SliceExp(e.loc, ie.e1, lwr, upr);
                    result.type = pointee;
                    result = new AddrExp(e.loc, result);
                    result.type = e.type;
                    return;
                }
            }
            if (e1.op == TOKvar || e1.op == TOKsymoff)
            {
                // type painting operation
                Type origType = (cast(SymbolExp)e1).var.type;
                if (castBackFromVoid && !isSafePointerCast(origType, pointee))
                {
                    e.error("using void* to reinterpret cast from %s* to %s* is not supported in CTFE", origType.toChars(), pointee.toChars());
                    result = CTFEExp.cantexp;
                    return;
                }
                if (e1.op == TOKvar)
                    result = new VarExp(e.loc, (cast(VarExp)e1).var);
                else
                    result = new SymOffExp(e.loc, (cast(SymOffExp)e1).var, (cast(SymOffExp)e1).offset);
                result.type = e.to;
                return;
            }

            // Check if we have a null pointer (eg, inside a struct)
            e1 = interpret(e1, istate);
            if (e1.op != TOKnull)
            {
                e.error("pointer cast from %s to %s is not supported at compile time", e1.type.toChars(), e.to.toChars());
                result = CTFEExp.cantexp;
                return;
            }
        }
        if (e.to.ty == Tsarray && e.e1.type.ty == Tvector)
        {
            // Special handling for: cast(float[4])__vector([w, x, y, z])
            e1 = interpret(e.e1, istate);
            if (exceptionOrCant(e1))
                return;
            assert(e1.op == TOKvector);
            e1 = (cast(VectorExp)e1).e1;
        }
        if (e.to.ty == Tarray && e1.op == TOKslice)
        {
            // Note that the slice may be void[], so when checking for dangerous
            // casts, we need to use the original type, which is se.e1.
            SliceExp se = cast(SliceExp)e1;
            if (!isSafePointerCast(se.e1.type.nextOf(), e.to.nextOf()))
            {
                e.error("array cast from %s to %s is not supported at compile time", se.e1.type.toChars(), e.to.toChars());
                result = CTFEExp.cantexp;
                return;
            }
            e1 = new SliceExp(e1.loc, se.e1, se.lwr, se.upr);
            e1.type = e.to;
            result = e1;
            return;
        }
        // Disallow array type painting, except for conversions between built-in
        // types of identical size.
        if ((e.to.ty == Tsarray || e.to.ty == Tarray) && (e1.type.ty == Tsarray || e1.type.ty == Tarray) && !isSafePointerCast(e1.type.nextOf(), e.to.nextOf()))
        {
            e.error("array cast from %s to %s is not supported at compile time", e1.type.toChars(), e.to.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        if (e.to.ty == Tsarray)
            e1 = resolveSlice(e1);
        if (e.to.toBasetype().ty == Tbool && e1.type.ty == Tpointer)
        {
            result = new IntegerExp(e.loc, e1.op != TOKnull, e.to);
            return;
        }
        result = ctfeCast(e.loc, e.type, e.to, e1);
    }

    override void visit(AssertExp e)
    {
        debug (LOG)
        {
            printf("%s AssertExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        Expression e1 = interpret(e.e1, istate);
        if (exceptionOrCant(e1))
            return;
        if (isTrueBool(e1))
        {
        }
        else if (e1.isBool(false))
        {
            if (e.msg)
            {
                result = interpret(e.msg, istate);
                if (exceptionOrCant(result))
                    return;
                e.error("%s", result.toChars());
            }
            else
                e.error("%s failed", e.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        else
        {
            e.error("%s is not a compile time boolean expression", e1.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        result = e1;
        return;
    }

    override void visit(PtrExp e)
    {
        debug (LOG)
        {
            printf("%s PtrExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        // Check for int<->float and long<->double casts.
        if (e.e1.op == TOKsymoff && (cast(SymOffExp)e.e1).offset == 0 && (cast(SymOffExp)e.e1).var.isVarDeclaration() && isFloatIntPaint(e.type, (cast(SymOffExp)e.e1).var.type))
        {
            // *(cast(int*)&v), where v is a float variable
            result = paintFloatInt(getVarExp(e.loc, istate, (cast(SymOffExp)e.e1).var, ctfeNeedRvalue), e.type);
            return;
        }
        if (e.e1.op == TOKcast && (cast(CastExp)e.e1).e1.op == TOKaddress)
        {
            // *(cast(int*)&x), where x is a float expression
            Expression x = (cast(AddrExp)(cast(CastExp)e.e1).e1).e1;
            if (isFloatIntPaint(e.type, x.type))
            {
                result = paintFloatInt(interpret(x, istate), e.type);
                return;
            }
        }

        // Constant fold *(&structliteral + offset)
        if (e.e1.op == TOKadd)
        {
            AddExp ae = cast(AddExp)e.e1;
            if (ae.e1.op == TOKaddress && ae.e2.op == TOKint64)
            {
                AddrExp ade = cast(AddrExp)ae.e1;
                Expression ex = interpret(ade.e1, istate);
                if (exceptionOrCant(ex))
                    return;
                if (ex.op == TOKstructliteral)
                {
                    StructLiteralExp se = cast(StructLiteralExp)ex;
                    dinteger_t offset = ae.e2.toInteger();
                    result = se.getField(e.type, cast(uint)offset);
                    if (result)
                        return;
                }
            }
        }

        // It's possible we have an array bounds error. We need to make sure it
        // errors with this line number, not the one where the pointer was set.
        result = interpret(e.e1, istate);
        if (exceptionOrCant(result))
            return;

        if (result.op == TOKfunction)
            return;
        if (result.op == TOKsymoff)
        {
            SymOffExp soe = cast(SymOffExp)result;
            if (soe.offset == 0 && soe.var.isFuncDeclaration())
                return;
            e.error("cannot dereference pointer to static variable %s at compile time", soe.var.toChars());
            result = CTFEExp.cantexp;
            return;
        }

        if (result.op != TOKaddress)
        {
            if (result.op == TOKnull)
                e.error("dereference of null pointer '%s'", e.e1.toChars());
            else
                e.error("dereference of invalid pointer '%s'", result.toChars());
            result = CTFEExp.cantexp;
            return;
        }

        // *(&x) ==> x
        result = (cast(AddrExp)result).e1;

        if (result.op == TOKslice && e.type.toBasetype().ty == Tsarray)
        {
            /* aggr[lwr..upr]
             * upr may exceed the upper boundary of aggr, but the check is deferred
             * until those out-of-bounds elements will be touched.
             */
            return;
        }
        result = interpret(result, istate, goal);
        if (exceptionOrCant(result))
            return;

        debug (LOG)
        {
            if (CTFEExp.isCantExp(result))
                printf("PtrExp::interpret() %s = CTFEExp::cantexp\n", e.toChars());
        }
    }

    override void visit(DotVarExp e)
    {
        debug (LOG)
        {
            printf("%s DotVarExp::interpret() %s, goal = %d\n", e.loc.toChars(), e.toChars(), goal);
        }
        Expression ex = interpret(e.e1, istate);
        if (exceptionOrCant(ex))
            return;

        if (FuncDeclaration f = e.var.isFuncDeclaration())
        {
            if (ex == e.e1)
                result = e; // optimize: reuse this CTFE reference
            else
            {
                result = new DotVarExp(e.loc, ex, f, false);
                result.type = e.type;
            }
            return;
        }

        VarDeclaration v = e.var.isVarDeclaration();
        if (!v)
        {
            e.error("CTFE internal error: %s", e.toChars());
            result = CTFEExp.cantexp;
            return;
        }

        if (ex.op == TOKnull)
        {
            if (ex.type.toBasetype().ty == Tclass)
                e.error("class '%s' is null and cannot be dereferenced", e.e1.toChars());
            else
                e.error("CTFE internal error: null this '%s'", e.e1.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        if (ex.op != TOKstructliteral && ex.op != TOKclassreference)
        {
            e.error("%s.%s is not yet implemented at compile time", e.e1.toChars(), e.var.toChars());
            result = CTFEExp.cantexp;
            return;
        }

        StructLiteralExp se;
        int i;

        // We can't use getField, because it makes a copy
        if (ex.op == TOKclassreference)
        {
            se = (cast(ClassReferenceExp)ex).value;
            i = (cast(ClassReferenceExp)ex).findFieldIndexByName(v);
        }
        else
        {
            se = cast(StructLiteralExp)ex;
            i = findFieldIndexByName(se.sd, v);
        }
        if (i == -1)
        {
            e.error("couldn't find field %s of type %s in %s", v.toChars(), e.type.toChars(), se.toChars());
            result = CTFEExp.cantexp;
            return;
        }

        if (goal == ctfeNeedLvalue)
        {
            Expression ev = (*se.elements)[i];
            if (!ev || ev.op == TOKvoid)
                (*se.elements)[i] = voidInitLiteral(e.type, v).copy();
            // just return the (simplified) dotvar expression as a CTFE reference
            if (e.e1 == ex)
                result = e;
            else
            {
                result = new DotVarExp(e.loc, ex, v);
                result.type = e.type;
            }
            return;
        }

        result = (*se.elements)[i];
        if (!result)
        {
            e.error("Internal Compiler Error: null field %s", v.toChars());
            result = CTFEExp.cantexp;
            return;
        }
        if (result.op == TOKvoid)
        {
            VoidInitExp ve = cast(VoidInitExp)result;
            const(char)* s = ve.var.toChars();
            if (v.overlapped)
            {
                e.error("reinterpretation through overlapped field %s is not allowed in CTFE", s);
                result = CTFEExp.cantexp;
                return;
            }
            e.error("cannot read uninitialized variable %s in CTFE", s);
            result = CTFEExp.cantexp;
            return;
        }

        if (v.type.ty != result.type.ty && v.type.ty == Tsarray)
        {
            // Block assignment from inside struct literals
            auto tsa = cast(TypeSArray)v.type;
            auto len = cast(size_t)tsa.dim.toInteger();
            result = createBlockDuplicatedArrayLiteral(ex.loc, v.type, ex, len);
            (*se.elements)[i] = result;
        }
        debug (LOG)
        {
            if (CTFEExp.isCantExp(result))
                printf("DotVarExp::interpret() %s = CTFEExp::cantexp\n", e.toChars());
        }
    }

    override void visit(RemoveExp e)
    {
        debug (LOG)
        {
            printf("%s RemoveExp::interpret() %s\n", e.loc.toChars(), e.toChars());
        }
        Expression agg = interpret(e.e1, istate);
        if (exceptionOrCant(agg))
            return;
        Expression index = interpret(e.e2, istate);
        if (exceptionOrCant(index))
            return;
        if (agg.op == TOKnull)
        {
            result = CTFEExp.voidexp;
            return;
        }

        assert(agg.op == TOKassocarrayliteral);
        AssocArrayLiteralExp aae = cast(AssocArrayLiteralExp)agg;
        Expressions* keysx = aae.keys;
        Expressions* valuesx = aae.values;
        size_t removed = 0;
        for (size_t j = 0; j < valuesx.dim; ++j)
        {
            Expression ekey = (*keysx)[j];
            int eq = ctfeEqual(e.loc, TOKequal, ekey, index);
            if (eq)
                ++removed;
            else if (removed != 0)
            {
                (*keysx)[j - removed] = ekey;
                (*valuesx)[j - removed] = (*valuesx)[j];
            }
        }
        valuesx.dim = valuesx.dim - removed;
        keysx.dim = keysx.dim - removed;
        result = new IntegerExp(e.loc, removed ? 1 : 0, Type.tbool);
    }

    override void visit(ClassReferenceExp e)
    {
        //printf("ClassReferenceExp::interpret() %s\n", e.value.toChars());
        result = e;
    }

    override void visit(VoidInitExp e)
    {
        e.error("CTFE internal error: trying to read uninitialized variable");
        assert(0);
    }

    override void visit(ThrownExceptionExp e)
    {
        assert(0); // This should never be interpreted
    }
}

extern (C++) Expression interpret(Expression e, InterState* istate, CtfeGoal goal = ctfeNeedRvalue)
{
    if (!e)
        return null;
    scope Interpreter v = new Interpreter(istate, goal);
    e.accept(v);
    Expression ex = v.result;
    assert(goal == ctfeNeedNothing || ex !is null);
    return ex;
}

/***********************************
 * Interpret the statement.
 * Returns:
 *      NULL    continue to next statement
 *      TOKcantexp      cannot interpret statement at compile time
 *      !NULL   expression from return statement, or thrown exception
 */
extern (C++) Expression interpret(Statement s, InterState* istate)
{
    if (!s)
        return null;
    scope Interpreter v = new Interpreter(istate, ctfeNeedNothing);
    s.accept(v);
    return v.result;
}

/* All results destined for use outside of CTFE need to have their CTFE-specific
 * features removed.
 * In particular, all slices must be resolved.
 */
extern (C++) Expression scrubReturnValue(Loc loc, Expression e)
{
    if (e.op == TOKclassreference)
    {
        StructLiteralExp se = (cast(ClassReferenceExp)e).value;
        se.ownedByCtfe = OWNEDcode;
        if (!(se.stageflags & stageScrub))
        {
            int old = se.stageflags;
            se.stageflags |= stageScrub;
            if (Expression ex = scrubArray(loc, se.elements, true))
                return ex;
            se.stageflags = old;
        }
    }
    if (e.op == TOKvoid)
    {
        error(loc, "uninitialized variable '%s' cannot be returned from CTFE", (cast(VoidInitExp)e).var.toChars());
        return new ErrorExp();
    }
    e = resolveSlice(e);
    if (e.op == TOKstructliteral)
    {
        StructLiteralExp se = cast(StructLiteralExp)e;
        se.ownedByCtfe = OWNEDcode;
        if (!(se.stageflags & stageScrub))
        {
            int old = se.stageflags;
            se.stageflags |= stageScrub;
            if (Expression ex = scrubArray(loc, se.elements, true))
                return ex;
            se.stageflags = old;
        }
    }
    if (e.op == TOKstring)
    {
        (cast(StringExp)e).ownedByCtfe = OWNEDcode;
    }
    if (e.op == TOKarrayliteral)
    {
        (cast(ArrayLiteralExp)e).ownedByCtfe = OWNEDcode;
        if (Expression ex = scrubArray(loc, (cast(ArrayLiteralExp)e).elements))
            return ex;
    }
    if (e.op == TOKassocarrayliteral)
    {
        AssocArrayLiteralExp aae = cast(AssocArrayLiteralExp)e;
        aae.ownedByCtfe = OWNEDcode;
        if (Expression ex = scrubArray(loc, aae.keys))
            return ex;
        if (Expression ex = scrubArray(loc, aae.values))
            return ex;
        aae.type = toBuiltinAAType(aae.type);
    }
    return e;
}

// Return true if every element is either void,
// or is an array literal or struct literal of void elements.
extern (C++) bool isEntirelyVoid(Expressions* elems)
{
    for (size_t i = 0; i < elems.dim; i++)
    {
        Expression m = (*elems)[i];
        // It can be NULL for performance reasons,
        // see StructLiteralExp::interpret().
        if (!m)
            continue;

        if (!(m.op == TOKvoid) && !(m.op == TOKarrayliteral && isEntirelyVoid((cast(ArrayLiteralExp)m).elements)) && !(m.op == TOKstructliteral && isEntirelyVoid((cast(StructLiteralExp)m).elements)))
        {
            return false;
        }
    }
    return true;
}

// Scrub all members of an array. Return false if error
extern (C++) Expression scrubArray(Loc loc, Expressions* elems, bool structlit = false)
{
    for (size_t i = 0; i < elems.dim; i++)
    {
        Expression m = (*elems)[i];
        // It can be NULL for performance reasons,
        // see StructLiteralExp::interpret().
        if (!m)
            continue;

        // A struct .init may contain void members.
        // Static array members are a weird special case (bug 10994).
        if (structlit && ((m.op == TOKvoid) || (m.op == TOKarrayliteral && m.type.ty == Tsarray && isEntirelyVoid((cast(ArrayLiteralExp)m).elements)) || (m.op == TOKstructliteral && isEntirelyVoid((cast(StructLiteralExp)m).elements))))
        {
            m = null;
        }
        else
        {
            m = scrubReturnValue(loc, m);
            if (CTFEExp.isCantExp(m) || m.op == TOKerror)
                return m;
        }
        (*elems)[i] = m;
    }
    return null;
}

extern (C++) Expression scrubCacheValue(Loc loc, Expression e)
{
    if (e.op == TOKclassreference)
    {
        StructLiteralExp sle = (cast(ClassReferenceExp)e).value;
        sle.ownedByCtfe = OWNEDcache;
        if (!(sle.stageflags & stageScrub))
        {
            int old = sle.stageflags;
            sle.stageflags |= stageScrub;
            if (Expression ex = scrubArrayCache(loc, sle.elements))
                return ex;
            sle.stageflags = old;
        }
    }
    if (e.op == TOKstructliteral)
    {
        StructLiteralExp sle = cast(StructLiteralExp)e;
        sle.ownedByCtfe = OWNEDcache;
        if (!(sle.stageflags & stageScrub))
        {
            int old = sle.stageflags;
            sle.stageflags |= stageScrub;
            if (Expression ex = scrubArrayCache(loc, sle.elements))
                return ex;
            sle.stageflags = old;
        }
    }
    if (e.op == TOKstring)
    {
        (cast(StringExp)e).ownedByCtfe = OWNEDcache;
    }
    if (e.op == TOKarrayliteral)
    {
        (cast(ArrayLiteralExp)e).ownedByCtfe = OWNEDcache;
        if (Expression ex = scrubArrayCache(loc, (cast(ArrayLiteralExp)e).elements))
            return ex;
    }
    if (e.op == TOKassocarrayliteral)
    {
        AssocArrayLiteralExp aae = cast(AssocArrayLiteralExp)e;
        aae.ownedByCtfe = OWNEDcache;
        if (Expression ex = scrubArrayCache(loc, aae.keys))
            return ex;
        if (Expression ex = scrubArrayCache(loc, aae.values))
            return ex;
    }
    return e;
}

extern (C++) Expression scrubArrayCache(Loc loc, Expressions* elems)
{
    for (size_t i = 0; i < elems.dim; i++)
    {
        Expression m = (*elems)[i];
        if (!m)
            continue;
        (*elems)[i] = scrubCacheValue(loc, m);
    }
    return null;
}

/******************************* Special Functions ***************************/

extern (C++) Expression interpret_length(InterState* istate, Expression earg)
{
    //printf("interpret_length()\n");
    earg = interpret(earg, istate);
    if (exceptionOrCantInterpret(earg))
        return earg;
    dinteger_t len = 0;
    if (earg.op == TOKassocarrayliteral)
        len = (cast(AssocArrayLiteralExp)earg).keys.dim;
    else
        assert(earg.op == TOKnull);
    Expression e = new IntegerExp(earg.loc, len, Type.tsize_t);
    return e;
}

extern (C++) Expression interpret_keys(InterState* istate, Expression earg, Type returnType)
{
    debug (LOG)
    {
        printf("interpret_keys()\n");
    }
    earg = interpret(earg, istate);
    if (exceptionOrCantInterpret(earg))
        return earg;
    if (earg.op == TOKnull)
        return new NullExp(earg.loc, returnType);
    if (earg.op != TOKassocarrayliteral && earg.type.toBasetype().ty != Taarray)
        return null;
    assert(earg.op == TOKassocarrayliteral);
    AssocArrayLiteralExp aae = cast(AssocArrayLiteralExp)earg;
    auto ae = new ArrayLiteralExp(aae.loc, aae.keys);
    ae.ownedByCtfe = aae.ownedByCtfe;
    ae.type = returnType;
    return copyLiteral(ae).copy();
}

extern (C++) Expression interpret_values(InterState* istate, Expression earg, Type returnType)
{
    debug (LOG)
    {
        printf("interpret_values()\n");
    }
    earg = interpret(earg, istate);
    if (exceptionOrCantInterpret(earg))
        return earg;
    if (earg.op == TOKnull)
        return new NullExp(earg.loc, returnType);
    if (earg.op != TOKassocarrayliteral && earg.type.toBasetype().ty != Taarray)
        return null;
    assert(earg.op == TOKassocarrayliteral);
    AssocArrayLiteralExp aae = cast(AssocArrayLiteralExp)earg;
    auto ae = new ArrayLiteralExp(aae.loc, aae.values);
    ae.ownedByCtfe = aae.ownedByCtfe;
    ae.type = returnType;
    //printf("result is %s\n", e.toChars());
    return copyLiteral(ae).copy();
}

extern (C++) Expression interpret_dup(InterState* istate, Expression earg)
{
    debug (LOG)
    {
        printf("interpret_dup()\n");
    }
    earg = interpret(earg, istate);
    if (exceptionOrCantInterpret(earg))
        return earg;
    if (earg.op == TOKnull)
        return new NullExp(earg.loc, earg.type);
    if (earg.op != TOKassocarrayliteral && earg.type.toBasetype().ty != Taarray)
        return null;
    assert(earg.op == TOKassocarrayliteral);
    AssocArrayLiteralExp aae = cast(AssocArrayLiteralExp)copyLiteral(earg).copy();
    for (size_t i = 0; i < aae.keys.dim; i++)
    {
        if (Expression e = evaluatePostblit(istate, (*aae.keys)[i]))
            return e;
        if (Expression e = evaluatePostblit(istate, (*aae.values)[i]))
            return e;
    }
    aae.type = earg.type.mutableOf(); // repaint type from const(int[int]) to const(int)[int]
    //printf("result is %s\n", aae.toChars());
    return aae;
}

// signature is int delegate(ref Value) OR int delegate(ref Key, ref Value)
extern (C++) Expression interpret_aaApply(InterState* istate, Expression aa, Expression deleg)
{
    aa = interpret(aa, istate);
    if (exceptionOrCantInterpret(aa))
        return aa;
    if (aa.op != TOKassocarrayliteral)
        return new IntegerExp(deleg.loc, 0, Type.tsize_t);

    FuncDeclaration fd = null;
    Expression pthis = null;
    if (deleg.op == TOKdelegate)
    {
        fd = (cast(DelegateExp)deleg).func;
        pthis = (cast(DelegateExp)deleg).e1;
    }
    else if (deleg.op == TOKfunction)
        fd = (cast(FuncExp)deleg).fd;

    assert(fd && fd.fbody);
    assert(fd.parameters);
    size_t numParams = fd.parameters.dim;
    assert(numParams == 1 || numParams == 2);

    Parameter fparam = Parameter.getNth((cast(TypeFunction)fd.type).parameters, numParams - 1);
    bool wantRefValue = 0 != (fparam.storageClass & (STCout | STCref));

    Expressions args;
    args.setDim(numParams);

    AssocArrayLiteralExp ae = cast(AssocArrayLiteralExp)aa;
    if (!ae.keys || ae.keys.dim == 0)
        return new IntegerExp(deleg.loc, 0, Type.tsize_t);
    Expression eresult;

    for (size_t i = 0; i < ae.keys.dim; ++i)
    {
        Expression ekey = (*ae.keys)[i];
        Expression evalue = (*ae.values)[i];
        if (wantRefValue)
        {
            Type t = evalue.type;
            evalue = new IndexExp(deleg.loc, ae, ekey);
            evalue.type = t;
        }
        args[numParams - 1] = evalue;
        if (numParams == 2)
            args[0] = ekey;

        eresult = interpret(fd, istate, &args, pthis);
        if (exceptionOrCantInterpret(eresult))
            return eresult;

        assert(eresult.op == TOKint64);
        if ((cast(IntegerExp)eresult).getInteger() != 0)
            return eresult;
    }
    return eresult;
}

// Helper function: given a function of type A[] f(...),
// return A[].
extern (C++) Type returnedArrayType(FuncDeclaration fd)
{
    assert(fd.type.ty == Tfunction);
    assert(fd.type.nextOf().ty == Tarray);
    return (cast(TypeFunction)fd.type).nextOf();
}

/* Decoding UTF strings for foreach loops. Duplicates the functionality of
 * the twelve _aApplyXXn functions in aApply.d in the runtime.
 */
extern (C++) Expression foreachApplyUtf(InterState* istate, Expression str, Expression deleg, bool rvs)
{
    debug (LOG)
    {
        printf("foreachApplyUtf(%s, %s)\n", str.toChars(), deleg.toChars());
    }
    FuncDeclaration fd = null;
    Expression pthis = null;
    if (deleg.op == TOKdelegate)
    {
        fd = (cast(DelegateExp)deleg).func;
        pthis = (cast(DelegateExp)deleg).e1;
    }
    else if (deleg.op == TOKfunction)
        fd = (cast(FuncExp)deleg).fd;

    assert(fd && fd.fbody);
    assert(fd.parameters);
    size_t numParams = fd.parameters.dim;
    assert(numParams == 1 || numParams == 2);
    Type charType = (*fd.parameters)[numParams - 1].type;
    Type indexType = numParams == 2 ? (*fd.parameters)[0].type : Type.tsize_t;
    size_t len = cast(size_t)resolveArrayLength(str);
    if (len == 0)
        return new IntegerExp(deleg.loc, 0, indexType);

    str = resolveSlice(str);

    StringExp se = null;
    ArrayLiteralExp ale = null;
    if (str.op == TOKstring)
        se = cast(StringExp)str;
    else if (str.op == TOKarrayliteral)
        ale = cast(ArrayLiteralExp)str;
    else
    {
        str.error("CTFE internal error: cannot foreach %s", str.toChars());
        return CTFEExp.cantexp;
    }
    Expressions args;
    args.setDim(numParams);

    Expression eresult = null; // ded-store to prevent spurious warning

    // Buffers for encoding; also used for decoding array literals
    char[4] utf8buf;
    wchar[2] utf16buf;

    size_t start = rvs ? len : 0;
    size_t end = rvs ? 0 : len;
    for (size_t indx = start; indx != end;)
    {
        // Step 1: Decode the next dchar from the string.

        const(char)* errmsg = null; // Used for reporting decoding errors
        dchar rawvalue; // Holds the decoded dchar
        size_t currentIndex = indx; // The index of the decoded character

        if (ale)
        {
            // If it is an array literal, copy the code points into the buffer
            size_t buflen = 1; // #code points in the buffer
            size_t n = 1; // #code points in this char
            size_t sz = cast(size_t)ale.type.nextOf().size();

            switch (sz)
            {
            case 1:
                if (rvs)
                {
                    // find the start of the string
                    --indx;
                    buflen = 1;
                    while (indx > 0 && buflen < 4)
                    {
                        Expression r = (*ale.elements)[indx];
                        assert(r.op == TOKint64);
                        char x = cast(char)(cast(IntegerExp)r).getInteger();
                        if ((x & 0xC0) != 0x80)
                            break;
                        ++buflen;
                    }
                }
                else
                    buflen = (indx + 4 > len) ? len - indx : 4;
                for (size_t i = 0; i < buflen; ++i)
                {
                    Expression r = (*ale.elements)[indx + i];
                    assert(r.op == TOKint64);
                    utf8buf[i] = cast(char)(cast(IntegerExp)r).getInteger();
                }
                n = 0;
                errmsg = utf_decodeChar(&utf8buf[0], buflen, n, rawvalue);
                break;

            case 2:
                if (rvs)
                {
                    // find the start of the string
                    --indx;
                    buflen = 1;
                    Expression r = (*ale.elements)[indx];
                    assert(r.op == TOKint64);
                    ushort x = cast(ushort)(cast(IntegerExp)r).getInteger();
                    if (indx > 0 && x >= 0xDC00 && x <= 0xDFFF)
                    {
                        --indx;
                        ++buflen;
                    }
                }
                else
                    buflen = (indx + 2 > len) ? len - indx : 2;
                for (size_t i = 0; i < buflen; ++i)
                {
                    Expression r = (*ale.elements)[indx + i];
                    assert(r.op == TOKint64);
                    utf16buf[i] = cast(ushort)(cast(IntegerExp)r).getInteger();
                }
                n = 0;
                errmsg = utf_decodeWchar(&utf16buf[0], buflen, n, rawvalue);
                break;

            case 4:
                {
                    if (rvs)
                        --indx;
                    Expression r = (*ale.elements)[indx];
                    assert(r.op == TOKint64);
                    rawvalue = cast(dchar)(cast(IntegerExp)r).getInteger();
                    n = 1;
                }
                break;

            default:
                assert(0);
            }
            if (!rvs)
                indx += n;
        }
        else
        {
            // String literals
            size_t saveindx; // used for reverse iteration

            switch (se.sz)
            {
            case 1:
                if (rvs)
                {
                    // find the start of the string
                    --indx;
                    while (indx > 0 && ((se.getCodeUnit(indx) & 0xC0) == 0x80))
                        --indx;
                    saveindx = indx;
                }
                errmsg = utf_decodeChar(se.string, se.len, indx, rawvalue);
                if (rvs)
                    indx = saveindx;
                break;

            case 2:
                if (rvs)
                {
                    // find the start
                    --indx;
                    auto wc = se.getCodeUnit(indx);
                    if (wc >= 0xDC00 && wc <= 0xDFFF)
                        --indx;
                    saveindx = indx;
                }
                errmsg = utf_decodeWchar(se.wstring, se.len, indx, rawvalue);
                if (rvs)
                    indx = saveindx;
                break;

            case 4:
                if (rvs)
                    --indx;
                rawvalue = se.getCodeUnit(indx);
                if (!rvs)
                    ++indx;
                break;

            default:
                assert(0);
            }
        }
        if (errmsg)
        {
            deleg.error("%s", errmsg);
            return CTFEExp.cantexp;
        }

        // Step 2: encode the dchar in the target encoding

        int charlen = 1; // How many codepoints are involved?
        switch (charType.size())
        {
        case 1:
            charlen = utf_codeLengthChar(rawvalue);
            utf_encodeChar(&utf8buf[0], rawvalue);
            break;
        case 2:
            charlen = utf_codeLengthWchar(rawvalue);
            utf_encodeWchar(&utf16buf[0], rawvalue);
            break;
        case 4:
            break;
        default:
            assert(0);
        }
        if (rvs)
            currentIndex = indx;

        // Step 3: call the delegate once for each code point

        // The index only needs to be set once
        if (numParams == 2)
            args[0] = new IntegerExp(deleg.loc, currentIndex, indexType);

        Expression val = null;

        for (int k = 0; k < charlen; ++k)
        {
            dchar codepoint;
            switch (charType.size())
            {
            case 1:
                codepoint = utf8buf[k];
                break;
            case 2:
                codepoint = utf16buf[k];
                break;
            case 4:
                codepoint = rawvalue;
                break;
            default:
                assert(0);
            }
            val = new IntegerExp(str.loc, codepoint, charType);

            args[numParams - 1] = val;

            eresult = interpret(fd, istate, &args, pthis);
            if (exceptionOrCantInterpret(eresult))
                return eresult;
            assert(eresult.op == TOKint64);
            if ((cast(IntegerExp)eresult).getInteger() != 0)
                return eresult;
        }
    }
    return eresult;
}

/* If this is a built-in function, return the interpreted result,
 * Otherwise, return NULL.
 */
extern (C++) Expression evaluateIfBuiltin(InterState* istate, Loc loc, FuncDeclaration fd, Expressions* arguments, Expression pthis)
{
    Expression e = null;
    size_t nargs = arguments ? arguments.dim : 0;
    if (!pthis)
    {
        if (isBuiltin(fd) == BUILTINyes)
        {
            Expressions args;
            args.setDim(nargs);
            for (size_t i = 0; i < args.dim; i++)
            {
                Expression earg = (*arguments)[i];
                earg = interpret(earg, istate);
                if (exceptionOrCantInterpret(earg))
                    return earg;
                args[i] = earg;
            }
            e = eval_builtin(loc, fd, &args);
            if (!e)
            {
                error(loc, "cannot evaluate unimplemented builtin %s at compile time", fd.toChars());
                e = CTFEExp.cantexp;
            }
        }
    }
    if (!pthis)
    {
        Expression firstarg = nargs > 0 ? (*arguments)[0] : null;
        if (firstarg && firstarg.type.toBasetype().ty == Taarray)
        {
            TypeAArray firstAAtype = cast(TypeAArray)firstarg.type;
            const id = fd.ident.toChars();
            if (nargs == 1 && fd.ident == Id.aaLen)
                return interpret_length(istate, firstarg);
            if (nargs == 3 && !strcmp(id, "_aaApply"))
                return interpret_aaApply(istate, firstarg, arguments.data[2]);
            if (nargs == 3 && !strcmp(id, "_aaApply2"))
                return interpret_aaApply(istate, firstarg, arguments.data[2]);
            if (nargs == 1 && !strcmp(id, "keys") && !strcmp(fd.toParent2().ident.toChars(), "object"))
                return interpret_keys(istate, firstarg, firstAAtype.index.arrayOf());
            if (nargs == 1 && !strcmp(id, "values") && !strcmp(fd.toParent2().ident.toChars(), "object"))
                return interpret_values(istate, firstarg, firstAAtype.nextOf().arrayOf());
            if (nargs == 1 && !strcmp(id, "rehash") && !strcmp(fd.toParent2().ident.toChars(), "object"))
                return interpret(firstarg, istate);
            if (nargs == 1 && !strcmp(id, "dup") && !strcmp(fd.toParent2().ident.toChars(), "object"))
                return interpret_dup(istate, firstarg);
        }
    }
    if (pthis && !fd.fbody && fd.isCtorDeclaration() && fd.parent && fd.parent.parent && fd.parent.parent.ident == Id.object)
    {
        if (pthis.op == TOKclassreference && fd.parent.ident == Id.Throwable)
        {
            // At present, the constructors just copy their arguments into the struct.
            // But we might need some magic if stack tracing gets added to druntime.
            StructLiteralExp se = (cast(ClassReferenceExp)pthis).value;
            assert(arguments.dim <= se.elements.dim);
            for (size_t i = 0; i < arguments.dim; ++i)
            {
                e = interpret((*arguments)[i], istate);
                if (exceptionOrCantInterpret(e))
                    return e;
                (*se.elements)[i] = e;
            }
            return CTFEExp.voidexp;
        }
    }
    if (nargs == 1 && !pthis && (fd.ident == Id.criticalenter || fd.ident == Id.criticalexit))
    {
        // Support synchronized{} as a no-op
        return CTFEExp.voidexp;
    }
    if (!pthis)
    {
        const idlen = fd.ident.toString().length;
        const id = fd.ident.toChars();
        if (nargs == 2 && (idlen == 10 || idlen == 11) && !strncmp(id, "_aApply", 7))
        {
            // Functions from aApply.d and aApplyR.d in the runtime
            bool rvs = (idlen == 11); // true if foreach_reverse
            char c = id[idlen - 3]; // char width: 'c', 'w', or 'd'
            char s = id[idlen - 2]; // string width: 'c', 'w', or 'd'
            char n = id[idlen - 1]; // numParams: 1 or 2.
            // There are 12 combinations
            if ((n == '1' || n == '2') && (c == 'c' || c == 'w' || c == 'd') && (s == 'c' || s == 'w' || s == 'd') && c != s)
            {
                Expression str = (*arguments)[0];
                str = interpret(str, istate);
                if (exceptionOrCantInterpret(str))
                    return str;
                return foreachApplyUtf(istate, str, (*arguments)[1], rvs);
            }
        }
    }
    return e;
}

extern (C++) Expression evaluatePostblit(InterState* istate, Expression e)
{
    Type tb = e.type.baseElemOf();
    if (tb.ty != Tstruct)
        return null;
    StructDeclaration sd = (cast(TypeStruct)tb).sym;
    if (!sd.postblit)
        return null;

    if (e.op == TOKarrayliteral)
    {
        ArrayLiteralExp ale = cast(ArrayLiteralExp)e;
        for (size_t i = 0; i < ale.elements.dim; i++)
        {
            e = evaluatePostblit(istate, (*ale.elements)[i]);
            if (e)
                return e;
        }
        return null;
    }
    if (e.op == TOKstructliteral)
    {
        // e.__postblit()
        e = interpret(sd.postblit, istate, null, e);
        if (exceptionOrCantInterpret(e))
            return e;
        return null;
    }
    assert(0);
}

extern (C++) Expression evaluateDtor(InterState* istate, Expression e)
{
    Type tb = e.type.baseElemOf();
    if (tb.ty != Tstruct)
        return null;
    StructDeclaration sd = (cast(TypeStruct)tb).sym;
    if (!sd.dtor)
        return null;

    if (e.op == TOKarrayliteral)
    {
        ArrayLiteralExp alex = cast(ArrayLiteralExp)e;
        for (size_t i = alex.elements.dim; 0 < i--;)
            e = evaluateDtor(istate, (*alex.elements)[i]);
    }
    else if (e.op == TOKstructliteral)
    {
        // e.__dtor()
        e = interpret(sd.dtor, istate, null, e);
    }
    else
        assert(0);
    if (exceptionOrCantInterpret(e))
        return e;
    return null;
}

/*************************** CTFE Sanity Checks ***************************/
/* Setter functions for CTFE variable values.
 * These functions exist to check for compiler CTFE bugs.
 */
extern (C++) bool hasValue(VarDeclaration vd)
{
    if (vd.ctfeAdrOnStack == cast(size_t)-1)
        return false;
    return null !is getValue(vd);
}

extern (C++) Expression getValue(VarDeclaration vd)
{
    return ctfeStack.getValue(vd);
}

extern (C++) void setValueNull(VarDeclaration vd)
{
    ctfeStack.setValue(vd, null);
}

// Don't check for validity
extern (C++) void setValueWithoutChecking(VarDeclaration vd, Expression newval)
{
    ctfeStack.setValue(vd, newval);
}

extern (C++) void setValue(VarDeclaration vd, Expression newval)
{
    version (none)
    {
        if (!((vd.storage_class & (STCout | STCref)) ? isCtfeReferenceValid(newval) : isCtfeValueValid(newval)))
        {
            printf("[%s] vd = %s %s, newval = %s\n", vd.loc.toChars(), vd.type.toChars(), vd.toChars(), newval.toChars());
        }
    }
    assert((vd.storage_class & (STCout | STCref)) ? isCtfeReferenceValid(newval) : isCtfeValueValid(newval));
    ctfeStack.setValue(vd, newval);
}
