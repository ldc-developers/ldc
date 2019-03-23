/***
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/func.d, _func.d)
 * Documentation:  https://dlang.org/phobos/dmd_func.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/func.d
 */

module dmd.func;

import core.stdc.stdio;
import core.stdc.string;
import dmd.aggregate;
import dmd.arraytypes;
import dmd.blockexit;
import dmd.gluelayer;
import dmd.dclass;
import dmd.declaration;
import dmd.delegatize;
import dmd.dinterpret;
import dmd.dmodule;
import dmd.dscope;
import dmd.dstruct;
import dmd.dsymbol;
import dmd.dsymbolsem;
import dmd.dtemplate;
import dmd.errors;
import dmd.escape;
import dmd.expression;
import dmd.globals;
import dmd.hdrgen;
import dmd.id;
import dmd.identifier;
import dmd.init;
import dmd.mtype;
import dmd.objc;
import dmd.root.outbuffer;
import dmd.root.rootobject;
import dmd.semantic2;
import dmd.semantic3;
import dmd.statement_rewrite_walker;
import dmd.statement;
import dmd.statementsem;
import dmd.tokens;
import dmd.visitor;

/// Inline Status
enum ILS : int
{
    uninitialized,       /// not computed yet
    no,                  /// cannot inline
    yes,                 /// can inline
}

enum BUILTIN : int
{
    unknown = -1,    /// not known if this is a builtin
    no,              /// this is not a builtin
    yes,             /// this is a builtin
}


/* Tweak all return statements and dtor call for nrvo_var, for correct NRVO.
 */
extern (C++) final class NrvoWalker : StatementRewriteWalker
{
    alias visit = typeof(super).visit;
public:
    FuncDeclaration fd;
    Scope* sc;

    override void visit(ReturnStatement s)
    {
        // See if all returns are instead to be replaced with a goto returnLabel;
        if (fd.returnLabel)
        {
            /* Rewrite:
             *  return exp;
             * as:
             *  vresult = exp; goto Lresult;
             */
            auto gs = new GotoStatement(s.loc, Id.returnLabel);
            gs.label = fd.returnLabel;

            Statement s1 = gs;
            if (s.exp)
                s1 = new CompoundStatement(s.loc, new ExpStatement(s.loc, s.exp), gs);

            replaceCurrent(s1);
        }
    }

    override void visit(TryFinallyStatement s)
    {
        DtorExpStatement des;
        if (fd.nrvo_can && s.finalbody && (des = s.finalbody.isDtorExpStatement()) !is null &&
            fd.nrvo_var == des.var)
        {
            if (!(global.params.useExceptions && ClassDeclaration.throwable))
            {
                /* Don't need to call destructor at all, since it is nrvo
                 */
                replaceCurrent(s._body);
                s._body.accept(this);
                return;
            }

            /* Normally local variable dtors are called regardless exceptions.
             * But for nrvo_var, its dtor should be called only when exception is thrown.
             *
             * Rewrite:
             *      try { s.body; } finally { nrvo_var.edtor; }
             *      // equivalent with:
             *      //    s.body; scope(exit) nrvo_var.edtor;
             * as:
             *      try { s.body; } catch(Throwable __o) { nrvo_var.edtor; throw __o; }
             *      // equivalent with:
             *      //    s.body; scope(failure) nrvo_var.edtor;
             */
            Statement sexception = new DtorExpStatement(Loc.initial, fd.nrvo_var.edtor, fd.nrvo_var);
            Identifier id = Identifier.generateId("__o");

            Statement handler = new PeelStatement(sexception);
            if (sexception.blockExit(fd, false) & BE.fallthru)
            {
                auto ts = new ThrowStatement(Loc.initial, new IdentifierExp(Loc.initial, id));
                ts.internalThrow = true;
                handler = new CompoundStatement(Loc.initial, handler, ts);
            }

            auto catches = new Catches();
            auto ctch = new Catch(Loc.initial, getThrowable(), id, handler);
            ctch.internalCatch = true;
            ctch.catchSemantic(sc); // Run semantic to resolve identifier '__o'
            catches.push(ctch);

            Statement s2 = new TryCatchStatement(Loc.initial, s._body, catches);
            fd.eh_none = false;
            replaceCurrent(s2);
            s2.accept(this);
        }
        else
            StatementRewriteWalker.visit(s);
    }
}

enum FUNCFLAG : uint
{
    purityInprocess  = 1,      /// working on determining purity
    safetyInprocess  = 2,      /// working on determining safety
    nothrowInprocess = 4,      /// working on determining nothrow
    nogcInprocess    = 8,      /// working on determining @nogc
    returnInprocess  = 0x10,   /// working on inferring 'return' for parameters
    inlineScanned    = 0x20,   /// function has been scanned for inline possibilities
    inferScope       = 0x40,   /// infer 'scope' for parameters
    hasCatches       = 0x80,   /// function has try-catch statements
    compileTimeOnly  = 0x100,  /// is a compile time only function; no code will be generated for it
}

/***********************************************************
 * Tuple of result identifier (possibly null) and statement.
 * This is used to store out contracts: out(id){ ensure }
 */
extern (C++) struct Ensure
{
    Identifier id;
    Statement ensure;

    Ensure syntaxCopy()
    {
        return Ensure(id, ensure.syntaxCopy());
    }

    /*****************************************
     * Do syntax copy of an array of Ensure's.
     */
    static Ensures* arraySyntaxCopy(Ensures* a)
    {
        Ensures* b = null;
        if (a)
        {
            b = a.copy();
            foreach (i, e; *a)
            {
                (*b)[i] = e.syntaxCopy();
            }
        }
        return b;
    }

}

/***********************************************************
 */
extern (C++) class FuncDeclaration : Declaration
{
    /// All hidden parameters bundled.
    static struct HiddenParameters
    {
        /**
         * The `this` parameter for methods or nested functions.
         *
         * For methods, it would be the class object or struct value the
         * method is called on. For nested functions it would be the enclosing
         * function's stack frame.
         */
        VarDeclaration vthis;

        /// The selector parameter for Objective-C methods.
        VarDeclaration selectorParameter;
    }

    Types* fthrows;                     /// Array of Type's of exceptions (not used)
    Statements* frequires;              /// in contracts
    Ensures* fensures;                  /// out contracts
    Statement frequire;                 /// lowered in contract
    Statement fensure;                  /// lowered out contract
    Statement fbody;                    /// function body

    FuncDeclarations foverrides;        /// functions this function overrides
    FuncDeclaration fdrequire;          /// function that does the in contract
    FuncDeclaration fdensure;           /// function that does the out contract

    const(char)* mangleString;          /// mangled symbol created from mangleExact()

    version (IN_LLVM)
    {
        // Argument lists for the __require/__ensure calls. NULL if not a virtual
        // function with contracts.
        Expressions* fdrequireParams;
        Expressions* fdensureParams;

        const(char)* intrinsicName;
        uint priority;

        // true if overridden with the pragma(LDC_allow_inline); statement
        bool allowInlining = false;

        // true if set with the pragma(LDC_never_inline); statement
        bool neverInline = false;

        // Whether to emit instrumentation code if -fprofile-instr-generate is specified,
        // the value is set with pragma(LDC_profile_instr, true|false)
        bool emitInstrumentation = true;
    }

    VarDeclaration vresult;             /// result variable for out contracts
    LabelDsymbol returnLabel;           /// where the return goes

    // used to prevent symbols in different
    // scopes from having the same name
    DsymbolTable localsymtab;
    VarDeclaration vthis;               /// 'this' parameter (member and nested)
    VarDeclaration v_arguments;         /// '_arguments' parameter
    ObjcSelector* selector;             /// Objective-C method selector (member function only)
    VarDeclaration selectorParameter;   /// Objective-C implicit selector parameter

    VarDeclaration v_argptr;            /// '_argptr' variable
    VarDeclarations* parameters;        /// Array of VarDeclaration's for parameters
    DsymbolTable labtab;                /// statement label symbol table
    Dsymbol overnext;                   /// next in overload list
    FuncDeclaration overnext0;          /// next in overload list (only used during IFTI)
    Loc endloc;                         /// location of closing curly bracket
    int vtblIndex = -1;                 /// for member functions, index into vtbl[]
    bool naked;                         /// true if naked
    bool generated;                     /// true if function was generated by the compiler rather than
                                        /// supplied by the user
    ubyte isCrtCtorDtor;                /// has attribute pragma(crt_constructor(1)/crt_destructor(2))
                                        /// not set before the glue layer

    ILS inlineStatusStmt = ILS.uninitialized;
    ILS inlineStatusExp = ILS.uninitialized;
    PINLINE inlining = PINLINE.default_;

    CompiledCtfeFunctionPimpl ctfeCode; /// Local data (i.e. CompileCtfeFunction*) for module dinterpret
    int inlineNest;                     /// !=0 if nested inline
    bool isArrayOp;                     /// true if array operation
    bool eh_none;                       /// true if no exception unwinding is needed

    bool semantic3Errors;               /// true if errors in semantic3 this function's frame ptr
    ForeachStatement fes;               /// if foreach body, this is the foreach
    BaseClass* interfaceVirtual;        /// if virtual, but only appears in base interface vtbl[]
    bool introducing;                   /// true if 'introducing' function
    /** if !=NULL, then this is the type
    of the 'introducing' function
    this one is overriding
    */
    Type tintro;

    bool inferRetType;                  /// true if return type is to be inferred
    StorageClass storage_class2;        /// storage class for template onemember's

    // Things that should really go into Scope

    /// 1 if there's a return exp; statement
    /// 2 if there's a throw statement
    /// 4 if there's an assert(0)
    /// 8 if there's inline asm
    /// 16 if there are multiple return statements
    int hasReturnExp;

    // Support for NRVO (named return value optimization)
    bool nrvo_can = true;               /// true means we can do NRVO
    VarDeclaration nrvo_var;            /// variable to replace with shidden
    Symbol* shidden;                    /// hidden pointer passed to function

    ReturnStatements* returns;

    GotoStatements* gotos;              /// Gotos with forward references

    /// set if this is a known, builtin function we can evaluate at compile time
    BUILTIN builtin = BUILTIN.unknown;

    /// set if someone took the address of this function
    int tookAddressOf;

    bool requiresClosure;               // this function needs a closure

    /// local variables in this function which are referenced by nested functions
    VarDeclarations closureVars;
    /// Sibling nested functions which called this one
    FuncDeclarations siblingCallers;

    FuncDeclarations *inlinedNestedCallees;

    uint flags;                        /// FUNCFLAG.xxxxx

    extern (D) this(const ref Loc loc, const ref Loc endloc, Identifier id, StorageClass storage_class, Type type)
    {
        super(id);
        //printf("FuncDeclaration(id = '%s', type = %p)\n", id.toChars(), type);
        //printf("storage_class = x%x\n", storage_class);
        this.storage_class = storage_class;
        this.type = type;
        if (type)
        {
            // Normalize storage_class, because function-type related attributes
            // are already set in the 'type' in parsing phase.
            this.storage_class &= ~(STC.TYPECTOR | STC.FUNCATTR);
        }
        this.loc = loc;
        this.endloc = endloc;
        /* The type given for "infer the return type" is a TypeFunction with
         * NULL for the return type.
         */
        inferRetType = (type && type.nextOf() is null);
    }

    static FuncDeclaration create(const ref Loc loc, const ref Loc endloc, Identifier id, StorageClass storage_class, Type type)
    {
        return new FuncDeclaration(loc, endloc, id, storage_class, type);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        //printf("FuncDeclaration::syntaxCopy('%s')\n", toChars());
        FuncDeclaration f = s ? cast(FuncDeclaration)s : new FuncDeclaration(loc, endloc, ident, storage_class, type.syntaxCopy());
        f.frequires = frequires ? Statement.arraySyntaxCopy(frequires) : null;
        f.fensures = fensures ? Ensure.arraySyntaxCopy(fensures) : null;
        f.fbody = fbody ? fbody.syntaxCopy() : null;
        assert(!fthrows); // deprecated
        version (IN_LLVM)
        {
            f.intrinsicName = intrinsicName ? strdup(intrinsicName) : null;
        }
        return f;
    }

    version (IN_LLVM)
    {
        final private Parameters* outToRef(Parameters* params)
        {
            auto result = new Parameters();

            int outToRefDg(size_t n, Parameter p)
            {
                if (p.storageClass & STC.out_)
                {
                    p = p.syntaxCopy();
                    p.storageClass &= ~STC.out_;
                    p.storageClass |= STC.ref_;
                }
                result.push(p);
                return 0;
            }

            Parameter._foreach(params, &outToRefDg);
            return result;
        }
    }

    /****************************************************
     * Resolve forward reference of function signature -
     * parameter types, return type, and attributes.
     * Returns false if any errors exist in the signature.
     */
    final bool functionSemantic()
    {
        if (!_scope)
            return !errors;

        if (!originalType) // semantic not yet run
        {
            TemplateInstance spec = isSpeculative();
            uint olderrs = global.errors;
            uint oldgag = global.gag;
            if (global.gag && !spec)
                global.gag = 0;
            dsymbolSemantic(this, _scope);
            global.gag = oldgag;
            if (spec && global.errors != olderrs)
                spec.errors = (global.errors - olderrs != 0);
            if (olderrs != global.errors) // if errors compiling this function
                return false;
        }

        // if inferring return type, sematic3 needs to be run
        // - When the function body contains any errors, we cannot assume
        //   the inferred return type is valid.
        //   So, the body errors should become the function signature error.
        if (inferRetType && type && !type.nextOf())
            return functionSemantic3();

        TemplateInstance ti;
        if (isInstantiated() && !isVirtualMethod() &&
            ((ti = parent.isTemplateInstance()) is null || ti.isTemplateMixin() || ti.tempdecl.ident == ident))
        {
            AggregateDeclaration ad = isMember2();
            if (ad && ad.sizeok != Sizeok.done)
            {
                /* Currently dmd cannot resolve forward references per methods,
                 * then setting SIZOKfwd is too conservative and would break existing code.
                 * So, just stop method attributes inference until ad.dsymbolSemantic() done.
                 */
                //ad.sizeok = Sizeok.fwd;
            }
            else
                return functionSemantic3() || !errors;
        }

        if (storage_class & STC.inference)
            return functionSemantic3() || !errors;

        return !errors;
    }

    /****************************************************
     * Resolve forward reference of function body.
     * Returns false if any errors exist in the body.
     */
    final bool functionSemantic3()
    {
        if (semanticRun < PASS.semantic3 && _scope)
        {
            /* Forward reference - we need to run semantic3 on this function.
             * If errors are gagged, and it's not part of a template instance,
             * we need to temporarily ungag errors.
             */
            TemplateInstance spec = isSpeculative();
            uint olderrs = global.errors;
            uint oldgag = global.gag;
version (IN_LLVM)
{
            if (global.gag && !spec && !global.gaggedForInlining)
                global.gag = 0;
}
else
{
            if (global.gag && !spec)
                global.gag = 0;
}
            semantic3(this, _scope);
            global.gag = oldgag;

            // If it is a speculatively-instantiated template, and errors occur,
            // we need to mark the template as having errors.
            if (spec && global.errors != olderrs)
                spec.errors = (global.errors - olderrs != 0);
            if (olderrs != global.errors) // if errors compiling this function
                return false;
        }

        return !errors && !semantic3Errors;
    }

    /****************************************************
     * Check that this function type is properly resolved.
     * If not, report "forward reference error" and return true.
     */
    extern (D) final bool checkForwardRef(const ref Loc loc)
    {
        if (!functionSemantic())
            return true;

        /* No deco means the functionSemantic() call could not resolve
         * forward referenes in the type of this function.
         */
        if (!type.deco)
        {
            bool inSemantic3 = (inferRetType && semanticRun >= PASS.semantic3);
            .error(loc, "forward reference to %s`%s`",
                (inSemantic3 ? "inferred return type of function " : "").ptr,
                toChars());
            return true;
        }
        return false;
    }

    // called from semantic3
    /**
     * Creates and returns the hidden parameters for this function declaration.
     *
     * Hidden parameters include the `this` parameter of a class, struct or
     * nested function and the selector parameter for Objective-C methods.
     */
    final HiddenParameters declareThis(Scope* sc, AggregateDeclaration ad)
    {
        if (ad)
        {
            //printf("declareThis() %s\n", toChars());
            Type thandle = ad.handleType();
            assert(thandle);
            thandle = thandle.addMod(type.mod);
            thandle = thandle.addStorageClass(storage_class);
            VarDeclaration v = new ThisDeclaration(loc, thandle);
            v.storage_class |= STC.parameter;
            if (thandle.ty == Tstruct)
            {
                v.storage_class |= STC.ref_;
                // if member function is marked 'inout', then 'this' is 'return ref'
                if (type.ty == Tfunction && (cast(TypeFunction)type).iswild & 2)
                    v.storage_class |= STC.return_;
            }
            if (type.ty == Tfunction)
            {
                TypeFunction tf = cast(TypeFunction)type;
                if (tf.isreturn)
                    v.storage_class |= STC.return_;
                if (tf.isscope)
                    v.storage_class |= STC.scope_;
            }
            if (flags & FUNCFLAG.inferScope && !(v.storage_class & STC.scope_))
                v.storage_class |= STC.maybescope;

            v.dsymbolSemantic(sc);
            if (!sc.insert(v))
                assert(0);
            v.parent = this;
            return HiddenParameters(v, objc.createSelectorParameter(this, sc));
        }
        if (isNested())
        {
            /* The 'this' for a nested function is the link to the
             * enclosing function's stack frame.
             * Note that nested functions and member functions are disjoint.
             */
            VarDeclaration v = new VarDeclaration(loc, Type.tvoid.pointerTo(), Id.capture, null);
            v.storage_class |= STC.parameter | STC.nodtor;
            if (type.ty == Tfunction)
            {
                TypeFunction tf = cast(TypeFunction)type;
                if (tf.isreturn)
                    v.storage_class |= STC.return_;
                if (tf.isscope)
                    v.storage_class |= STC.scope_;
            }
            if (flags & FUNCFLAG.inferScope && !(v.storage_class & STC.scope_))
                v.storage_class |= STC.maybescope;

            v.dsymbolSemantic(sc);
            if (!sc.insert(v))
                assert(0);
            v.parent = this;
            return HiddenParameters(v);
        }
        return HiddenParameters.init;
    }

    override final bool equals(RootObject o)
    {
        if (this == o)
            return true;

        Dsymbol s = isDsymbol(o);
        if (s)
        {
            alias fd1 = this;
            auto  fd2 = s.isFuncDeclaration();
            if (!fd2)
                return false;

            auto fa1 = fd1.isFuncAliasDeclaration();
            auto fa2 = fd2.isFuncAliasDeclaration();
            if (fa1 && fa2)
            {
                return fa1.toAliasFunc().equals(fa2.toAliasFunc()) && fa1.hasOverloads == fa2.hasOverloads;
            }

            if (fa1 && (fd1 = fa1.toAliasFunc()).isUnique() && !fa1.hasOverloads)
                fa1 = null;
            if (fa2 && (fd2 = fa2.toAliasFunc()).isUnique() && !fa2.hasOverloads)
                fa2 = null;
            if ((fa1 !is null) != (fa2 !is null))
                return false;

            return fd1.toParent().equals(fd2.toParent()) && fd1.ident.equals(fd2.ident) && fd1.type.equals(fd2.type);
        }
        return false;
    }

    /****************************************************
     * Determine if 'this' overrides fd.
     * Return !=0 if it does.
     */
    final int overrides(FuncDeclaration fd)
    {
        int result = 0;
        if (fd.ident == ident)
        {
            int cov = type.covariant(fd.type);
            if (cov)
            {
                ClassDeclaration cd1 = toParent().isClassDeclaration();
                ClassDeclaration cd2 = fd.toParent().isClassDeclaration();
                if (cd1 && cd2 && cd2.isBaseOf(cd1, null))
                    result = 1;
            }
        }
        return result;
    }

    /*************************************************
     * Find index of function in vtbl[0..dim] that
     * this function overrides.
     * Prefer an exact match to a covariant one.
     * Params:
     *      vtbl     = vtable to use
     *      dim      = maximal vtable dimension
     *      fix17349 = enable fix https://issues.dlang.org/show_bug.cgi?id=17349
     * Returns:
     *      -1      didn't find one
     *      -2      can't determine because of forward references
     */
    final int findVtblIndex(Dsymbols* vtbl, int dim, bool fix17349 = true)
    {
        //printf("findVtblIndex() %s\n", toChars());
        FuncDeclaration mismatch = null;
        StorageClass mismatchstc = 0;
        int mismatchvi = -1;
        int exactvi = -1;
        int bestvi = -1;
        for (int vi = 0; vi < dim; vi++)
        {
            FuncDeclaration fdv = (*vtbl)[vi].isFuncDeclaration();
            if (fdv && fdv.ident == ident)
            {
                if (type.equals(fdv.type)) // if exact match
                {
                    if (fdv.parent.isClassDeclaration())
                    {
                        if (fdv.isFuture())
                        {
                            bestvi = vi;
                            continue;           // keep looking
                        }
                        return vi; // no need to look further
                    }

                    if (exactvi >= 0)
                    {
                        error("cannot determine overridden function");
                        return exactvi;
                    }
                    exactvi = vi;
                    bestvi = vi;
                    continue;
                }

                StorageClass stc = 0;
                int cov = type.covariant(fdv.type, &stc, fix17349);
                //printf("\tbaseclass cov = %d\n", cov);
                switch (cov)
                {
                case 0:
                    // types are distinct
                    break;

                case 1:
                    bestvi = vi; // covariant, but not identical
                    break;
                    // keep looking for an exact match

                case 2:
                    mismatchvi = vi;
                    mismatchstc = stc;
                    mismatch = fdv; // overrides, but is not covariant
                    break;
                    // keep looking for an exact match

                case 3:
                    return -2; // forward references

                default:
                    assert(0);
                }
            }
        }
        if (bestvi == -1 && mismatch)
        {
            //type.print();
            //mismatch.type.print();
            //printf("%s %s\n", type.deco, mismatch.type.deco);
            //printf("stc = %llx\n", mismatchstc);
            if (mismatchstc)
            {
                // Fix it by modifying the type to add the storage classes
                type = type.addStorageClass(mismatchstc);
                bestvi = mismatchvi;
            }
        }
        return bestvi;
    }

    /*********************************
     * If function a function in a base class,
     * return that base class.
     * Returns:
     *  base class if overriding, null if not
     */
    final BaseClass* overrideInterface()
    {
        if (ClassDeclaration cd = toParent2().isClassDeclaration())
        {
            foreach (b; cd.interfaces)
            {
                auto v = findVtblIndex(&b.sym.vtbl, cast(int)b.sym.vtbl.dim);
                if (v >= 0)
                    return b;
            }
        }
        return null;
    }

    /****************************************************
     * Overload this FuncDeclaration with the new one f.
     * Return true if successful; i.e. no conflict.
     */
    override bool overloadInsert(Dsymbol s)
    {
        //printf("FuncDeclaration::overloadInsert(s = %s) this = %s\n", s.toChars(), toChars());
        assert(s != this);
        AliasDeclaration ad = s.isAliasDeclaration();
        if (ad)
        {
            if (overnext)
                return overnext.overloadInsert(ad);
            if (!ad.aliassym && ad.type.ty != Tident && ad.type.ty != Tinstance && ad.type.ty != Ttypeof)
            {
                //printf("\tad = '%s'\n", ad.type.toChars());
                return false;
            }
            overnext = ad;
            //printf("\ttrue: no conflict\n");
            return true;
        }
        TemplateDeclaration td = s.isTemplateDeclaration();
        if (td)
        {
            if (!td.funcroot)
                td.funcroot = this;
            if (overnext)
                return overnext.overloadInsert(td);
            overnext = td;
            return true;
        }
        FuncDeclaration fd = s.isFuncDeclaration();
        if (!fd)
            return false;

        version (none)
        {
            /* Disable this check because:
             *  const void foo();
             * semantic() isn't run yet on foo(), so the const hasn't been
             * applied yet.
             */
            if (type)
            {
                printf("type = %s\n", type.toChars());
                printf("fd.type = %s\n", fd.type.toChars());
            }
            // fd.type can be NULL for overloaded constructors
            if (type && fd.type && fd.type.covariant(type) && fd.type.mod == type.mod && !isFuncAliasDeclaration())
            {
                //printf("\tfalse: conflict %s\n", kind());
                return false;
            }
        }

        if (overnext)
        {
            td = overnext.isTemplateDeclaration();
            if (td)
                fd.overloadInsert(td);
            else
                return overnext.overloadInsert(fd);
        }
        overnext = fd;
        //printf("\ttrue: no conflict\n");
        return true;
    }

    /********************************************
     * Find function in overload list that exactly matches t.
     */
    final FuncDeclaration overloadExactMatch(Type t)
    {
        FuncDeclaration fd;
        overloadApply(this, (Dsymbol s)
        {
            auto f = s.isFuncDeclaration();
            if (!f)
                return 0;
            if (t.equals(f.type))
            {
                fd = f;
                return 1;
            }

            /* Allow covariant matches, as long as the return type
             * is just a const conversion.
             * This allows things like pure functions to match with an impure function type.
             */
            if (t.ty == Tfunction)
            {
                auto tf = cast(TypeFunction)f.type;
                if (tf.covariant(t) == 1 &&
                    tf.nextOf().implicitConvTo(t.nextOf()) >= MATCH.constant)
                {
                    fd = f;
                    return 1;
                }
            }
            return 0;
        });
        return fd;
    }

    /********************************************
     * Find function in overload list that matches to the 'this' modifier.
     * There's four result types.
     *
     * 1. If the 'tthis' matches only one candidate, it's an "exact match".
     *    Returns the function and 'hasOverloads' is set to false.
     *      eg. If 'tthis" is mutable and there's only one mutable method.
     * 2. If there's two or more match candidates, but a candidate function will be
     *    a "better match".
     *    Returns the better match function but 'hasOverloads' is set to true.
     *      eg. If 'tthis' is mutable, and there's both mutable and const methods,
     *          the mutable method will be a better match.
     * 3. If there's two or more match candidates, but there's no better match,
     *    Returns null and 'hasOverloads' is set to true to represent "ambiguous match".
     *      eg. If 'tthis' is mutable, and there's two or more mutable methods.
     * 4. If there's no candidates, it's "no match" and returns null with error report.
     *      e.g. If 'tthis' is const but there's no const methods.
     */
    final FuncDeclaration overloadModMatch(const ref Loc loc, Type tthis, ref bool hasOverloads)
    {
        //printf("FuncDeclaration::overloadModMatch('%s')\n", toChars());
        Match m;
        m.last = MATCH.nomatch;
        overloadApply(this, (Dsymbol s)
        {
            auto f = s.isFuncDeclaration();
            if (!f || f == m.lastf) // skip duplicates
                return 0;
            m.anyf = f;

            auto tf = f.type.toTypeFunction();
            //printf("tf = %s\n", tf.toChars());

            MATCH match;
            if (tthis) // non-static functions are preferred than static ones
            {
                if (f.needThis())
                    match = f.isCtorDeclaration() ? MATCH.exact : MODmethodConv(tthis.mod, tf.mod);
                else
                    match = MATCH.constant; // keep static function in overload candidates
            }
            else // static functions are preferred than non-static ones
            {
                if (f.needThis())
                    match = MATCH.convert;
                else
                    match = MATCH.exact;
            }
            if (match == MATCH.nomatch)
                return 0;

            if (match > m.last) goto LcurrIsBetter;
            if (match < m.last) goto LlastIsBetter;

            // See if one of the matches overrides the other.
            if (m.lastf.overrides(f)) goto LlastIsBetter;
            if (f.overrides(m.lastf)) goto LcurrIsBetter;

        Lambiguous:
            //printf("\tambiguous\n");
            m.nextf = f;
            m.count++;
            return 0;

        LlastIsBetter:
            //printf("\tlastbetter\n");
            m.count++; // count up
            return 0;

        LcurrIsBetter:
            //printf("\tisbetter\n");
            if (m.last <= MATCH.convert)
            {
                // clear last secondary matching
                m.nextf = null;
                m.count = 0;
            }
            m.last = match;
            m.lastf = f;
            m.count++; // count up
            return 0;
        });

        if (m.count == 1)       // exact match
        {
            hasOverloads = false;
        }
        else if (m.count > 1)   // better or ambiguous match
        {
            hasOverloads = true;
        }
        else                    // no match
        {
            hasOverloads = true;
            auto tf = this.type.toTypeFunction();
            assert(tthis);
            assert(!MODimplicitConv(tthis.mod, tf.mod)); // modifier mismatch
            {
                OutBuffer thisBuf, funcBuf;
                MODMatchToBuffer(&thisBuf, tthis.mod, tf.mod);
                MODMatchToBuffer(&funcBuf, tf.mod, tthis.mod);
                .error(loc, "%smethod %s is not callable using a %sobject",
                    funcBuf.peekString(), this.toPrettyChars(), thisBuf.peekString());
            }
        }
        return m.lastf;
    }

    /********************************************
     * find function template root in overload list
     */
    final TemplateDeclaration findTemplateDeclRoot()
    {
        FuncDeclaration f = this;
        while (f && f.overnext)
        {
            //printf("f.overnext = %p %s\n", f.overnext, f.overnext.toChars());
            TemplateDeclaration td = f.overnext.isTemplateDeclaration();
            if (td)
                return td;
            f = f.overnext.isFuncDeclaration();
        }
        return null;
    }

    /********************************************
     * Returns true if function was declared
     * directly or indirectly in a unittest block
     */
    final bool inUnittest()
    {
        Dsymbol f = this;
        do
        {
            if (f.isUnitTestDeclaration())
                return true;
            f = f.toParent();
        }
        while (f);
        return false;
    }

    /*************************************
     * Determine partial specialization order of 'this' vs g.
     * This is very similar to TemplateDeclaration::leastAsSpecialized().
     * Returns:
     *      match   'this' is at least as specialized as g
     *      0       g is more specialized than 'this'
     */
    final MATCH leastAsSpecialized(FuncDeclaration g)
    {
        enum LOG_LEASTAS = 0;
        static if (LOG_LEASTAS)
        {
            printf("%s.leastAsSpecialized(%s)\n", toChars(), g.toChars());
            printf("%s, %s\n", type.toChars(), g.type.toChars());
        }

        /* This works by calling g() with f()'s parameters, and
         * if that is possible, then f() is at least as specialized
         * as g() is.
         */

        TypeFunction tf = type.toTypeFunction();
        TypeFunction tg = g.type.toTypeFunction();
        size_t nfparams = tf.parameterList.length;

        /* If both functions have a 'this' pointer, and the mods are not
         * the same and g's is not const, then this is less specialized.
         */
        if (needThis() && g.needThis() && tf.mod != tg.mod)
        {
            if (isCtorDeclaration())
            {
                if (!MODimplicitConv(tg.mod, tf.mod))
                    return MATCH.nomatch;
            }
            else
            {
                if (!MODimplicitConv(tf.mod, tg.mod))
                    return MATCH.nomatch;
            }
        }

        /* Create a dummy array of arguments out of the parameters to f()
         */
        Expressions args = Expressions(nfparams);
        for (size_t u = 0; u < nfparams; u++)
        {
            Parameter p = tf.parameterList[u];
            Expression e;
            if (p.storageClass & (STC.ref_ | STC.out_))
            {
                e = new IdentifierExp(Loc.initial, p.ident);
                e.type = p.type;
            }
            else
                e = p.type.defaultInitLiteral(Loc.initial);
            args[u] = e;
        }

        MATCH m = tg.callMatch(null, &args, 1);
        if (m > MATCH.nomatch)
        {
            /* A variadic parameter list is less specialized than a
             * non-variadic one.
             */
            if (tf.parameterList.varargs && !tg.parameterList.varargs)
                goto L1; // less specialized

            static if (LOG_LEASTAS)
            {
                printf("  matches %d, so is least as specialized\n", m);
            }
            return m;
        }
    L1:
        static if (LOG_LEASTAS)
        {
            printf("  doesn't match, so is not as specialized\n");
        }
        return MATCH.nomatch;
    }

    /********************************
     * Labels are in a separate scope, one per function.
     */
    final LabelDsymbol searchLabel(Identifier ident)
    {
        Dsymbol s;
        if (!labtab)
            labtab = new DsymbolTable(); // guess we need one

        s = labtab.lookup(ident);
        if (!s)
        {
            s = new LabelDsymbol(ident);
            labtab.insert(s);
        }
        return cast(LabelDsymbol)s;
    }

    /*****************************************
     * Determine lexical level difference from 'this' to nested function 'fd'.
     * Error if this cannot call fd.
     * Returns:
     *      0       same level
     *      >0      decrease nesting by number
     *      -1      increase nesting by 1 (fd is nested within 'this')
     *      -2      error
     */
    final int getLevel(const ref Loc loc, Scope* sc, FuncDeclaration fd)
    {
        int level;
        Dsymbol s;
        Dsymbol fdparent;

        //printf("FuncDeclaration::getLevel(fd = '%s')\n", fd.toChars());
        fdparent = fd.toParent2();
        if (fdparent == this)
            return -1;
        s = this;
        level = 0;
        while (fd != s && fdparent != s.toParent2())
        {
            //printf("\ts = %s, '%s'\n", s.kind(), s.toChars());
            FuncDeclaration thisfd = s.isFuncDeclaration();
            if (thisfd)
            {
                if (!thisfd.isNested() && !thisfd.vthis && !sc.intypeof)
                    goto Lerr;
            }
            else
            {
                AggregateDeclaration thiscd = s.isAggregateDeclaration();
                if (thiscd)
                {
                    /* AggregateDeclaration::isNested returns true only when
                     * it has a hidden pointer.
                     * But, calling the function belongs unrelated lexical scope
                     * is still allowed inside typeof.
                     *
                     * struct Map(alias fun) {
                     *   typeof({ return fun(); }) RetType;
                     *   // No member function makes Map struct 'not nested'.
                     * }
                     */
                    if (!thiscd.isNested() && !sc.intypeof)
                        goto Lerr;
                }
                else
                    goto Lerr;
            }

            s = s.toParent2();
            assert(s);
            level++;
        }
        return level;

    Lerr:
        // Don't give error if in template constraint
        if (!(sc.flags & SCOPE.constraint))
        {
            const(char)* xstatic = isStatic() ? "static " : "";
            // better diagnostics for static functions
            .error(loc, "%s%s %s cannot access frame of function %s", xstatic, kind(), toPrettyChars(), fd.toPrettyChars());
            return -2;
        }
        return 1;
    }

    override const(char)* toPrettyChars(bool QualifyTypes = false)
    {
        if (isMain())
            return "D main";
        else
            return Dsymbol.toPrettyChars(QualifyTypes);
    }

    /** for diagnostics, e.g. 'int foo(int x, int y) pure' */
    final const(char)* toFullSignature()
    {
        OutBuffer buf;
        functionToBufferWithIdent(type.toTypeFunction(), &buf, toChars());
        return buf.extractString();
    }

    final bool isMain() const
    {
        return ident == Id.main && linkage != LINK.c && !isMember() && !isNested();
    }

    final bool isCMain() const
    {
        return ident == Id.main && linkage == LINK.c && !isMember() && !isNested();
    }

    final bool isWinMain() const
    {
        //printf("FuncDeclaration::isWinMain() %s\n", toChars());
        version (none)
        {
            bool x = ident == Id.WinMain && linkage != LINK.c && !isMember();
            printf("%s\n", x ? "yes" : "no");
            return x;
        }
        else
        {
            return ident == Id.WinMain && linkage != LINK.c && !isMember();
        }
    }

    final bool isDllMain() const
    {
        return ident == Id.DllMain && linkage != LINK.c && !isMember();
    }

    final bool isRtInit() const
    {
        return ident == Id.rt_init && linkage == LINK.c && !isMember() && !isNested();
    }

    override final bool isExport() const
    {
        return protection.kind == Prot.Kind.export_;
    }

    override final bool isImportedSymbol() const
    {
        //printf("isImportedSymbol()\n");
        //printf("protection = %d\n", protection);
        return (protection.kind == Prot.Kind.export_) && !fbody;
    }

    override final bool isCodeseg() const pure nothrow @nogc @safe
    {
        return true; // functions are always in the code segment
    }

    override final bool isOverloadable()
    {
        return true; // functions can be overloaded
    }

    /***********************************
     * Override so it can work even if semantic() hasn't yet
     * been run.
     */
    override final bool isAbstract()
    {
        if (storage_class & STC.abstract_)
            return true;
        if (semanticRun >= PASS.semanticdone)
            return false;

        if (_scope)
        {
           if (_scope.stc & STC.abstract_)
                return true;
           parent = _scope.parent;
           Dsymbol parent = toParent();
           if (parent.isInterfaceDeclaration())
                return true;
        }
        return false;
    }

    /**********************************
     * Decide if attributes for this function can be inferred from examining
     * the function body.
     * Returns:
     *  true if can
     */
    final bool canInferAttributes(Scope* sc)
    {
        if (!fbody)
            return false;

        if (isVirtualMethod())
            return false;               // since they may be overridden

        if (sc.func &&
            /********** this is for backwards compatibility for the moment ********/
            (!isMember() || sc.func.isSafeBypassingInference() && !isInstantiated()))
            return true;

        if (isFuncLiteralDeclaration() ||               // externs are not possible with literals
            (storage_class & STC.inference) ||           // do attribute inference
            (inferRetType && !isCtorDeclaration()))
            return true;

        if (isInstantiated())
        {
            auto ti = parent.isTemplateInstance();
            if (ti is null || ti.isTemplateMixin() || ti.tempdecl.ident == ident)
                return true;
        }

        return false;
    }

    /*****************************************
     * Initialize for inferring the attributes of this function.
     */
    final void initInferAttributes()
    {
        //printf("initInferAttributes() for %s (%s)\n", toPrettyChars(), ident.toChars());
        TypeFunction tf = type.toTypeFunction();
        if (tf.purity == PURE.impure) // purity not specified
            flags |= FUNCFLAG.purityInprocess;

        if (tf.trust == TRUST.default_)
            flags |= FUNCFLAG.safetyInprocess;

        if (!tf.isnothrow)
            flags |= FUNCFLAG.nothrowInprocess;

        if (!tf.isnogc)
            flags |= FUNCFLAG.nogcInprocess;

        if (!isVirtual() || introducing)
            flags |= FUNCFLAG.returnInprocess;

        // Initialize for inferring STC.scope_
        if (global.params.vsafe)
            flags |= FUNCFLAG.inferScope;
    }

    final PURE isPure()
    {
        //printf("FuncDeclaration::isPure() '%s'\n", toChars());
        TypeFunction tf = type.toTypeFunction();
        if (flags & FUNCFLAG.purityInprocess)
            setImpure();
        if (tf.purity == PURE.fwdref)
            tf.purityLevel();
        PURE purity = tf.purity;
        if (purity > PURE.weak && isNested())
            purity = PURE.weak;
        if (purity > PURE.weak && needThis())
        {
            // The attribute of the 'this' reference affects purity strength
            if (type.mod & MODFlags.immutable_)
            {
            }
            else if (type.mod & (MODFlags.const_ | MODFlags.wild) && purity >= PURE.const_)
                purity = PURE.const_;
            else
                purity = PURE.weak;
        }
        tf.purity = purity;
        // ^ This rely on the current situation that every FuncDeclaration has a
        //   unique TypeFunction.
        return purity;
    }

    final PURE isPureBypassingInference()
    {
        if (flags & FUNCFLAG.purityInprocess)
            return PURE.fwdref;
        else
            return isPure();
    }

    /**************************************
     * The function is doing something impure,
     * so mark it as impure.
     * If there's a purity error, return true.
     */
    final bool setImpure()
    {
        if (flags & FUNCFLAG.purityInprocess)
        {
            flags &= ~FUNCFLAG.purityInprocess;
            if (fes)
                fes.func.setImpure();
        }
        else if (isPure())
            return true;
        return false;
    }

    final bool isSafe()
    {
        if (flags & FUNCFLAG.safetyInprocess)
            setUnsafe();
        return type.toTypeFunction().trust == TRUST.safe;
    }

    final bool isSafeBypassingInference()
    {
        return !(flags & FUNCFLAG.safetyInprocess) && isSafe();
    }

    final bool isTrusted()
    {
        if (flags & FUNCFLAG.safetyInprocess)
            setUnsafe();
        return type.toTypeFunction().trust == TRUST.trusted;
    }

    /**************************************
     * The function is doing something unsave,
     * so mark it as unsafe.
     * If there's a safe error, return true.
     */
    final bool setUnsafe()
    {
        if (flags & FUNCFLAG.safetyInprocess)
        {
            flags &= ~FUNCFLAG.safetyInprocess;
            type.toTypeFunction().trust = TRUST.system;
            if (fes)
                fes.func.setUnsafe();
        }
        else if (isSafe())
            return true;
        return false;
    }

    final bool isNogc()
    {
        //printf("isNogc() %s, inprocess: %d\n", toChars(), !!(flags & FUNCFLAG.nogcInprocess));
        if (flags & FUNCFLAG.nogcInprocess)
            setGC();
        return type.toTypeFunction().isnogc;
    }

    final bool isNogcBypassingInference()
    {
        return !(flags & FUNCFLAG.nogcInprocess) && isNogc();
    }

    /**************************************
     * The function is doing something that may allocate with the GC,
     * so mark it as not nogc (not no-how).
     * Returns:
     *      true if function is marked as @nogc, meaning a user error occurred
     */
    final bool setGC()
    {
        //printf("setGC() %s\n", toChars());
        if (flags & FUNCFLAG.nogcInprocess && semanticRun < PASS.semantic3 && _scope)
        {
            this.semantic2(_scope);
            this.semantic3(_scope);
        }

        if (flags & FUNCFLAG.nogcInprocess)
        {
            flags &= ~FUNCFLAG.nogcInprocess;
            type.toTypeFunction().isnogc = false;
            if (fes)
                fes.func.setGC();
        }
        else if (isNogc())
            return true;
        return false;
    }

    final void printGCUsage(const ref Loc loc, const(char)* warn)
    {
        if (!global.params.vgc)
            return;

        Module m = getModule();
        if (m && m.isRoot() && !inUnittest())
        {
            message(loc, "vgc: %s", warn);
        }
    }

    /********************************************
     * See if pointers from function parameters, mutable globals, or uplevel functions
     * could leak into return value.
     * Returns:
     *   true if the function return value is isolated from
     *   any inputs to the function
     */
    final bool isReturnIsolated()
    {
        TypeFunction tf = type.toTypeFunction();
        assert(tf.next);

        Type treti = tf.next;
        if (tf.isref)
            return isTypeIsolatedIndirect(treti);              // check influence from parameters

        return isTypeIsolated(treti);
    }

    /********************
     * See if pointers from function parameters, mutable globals, or uplevel functions
     * could leak into type `t`.
     * Params:
     *   t = type to check if it is isolated
     * Returns:
     *   true if `t` is isolated from
     *   any inputs to the function
     */
    final bool isTypeIsolated(Type t)
    {
        //printf("isTypeIsolated(t: %s)\n", t.toChars());

        t = t.baseElemOf();
        switch (t.ty)
        {
            case Tarray:
            case Tpointer:
                return isTypeIsolatedIndirect(t.nextOf()); // go down one level

            case Taarray:
            case Tclass:
                return isTypeIsolatedIndirect(t);

            case Tstruct:
                /* Drill down and check the struct's fields
                 */
                auto sym = t.toDsymbol(null).isStructDeclaration();
                foreach (v; sym.fields)
                {
                    Type tmi = v.type.addMod(t.mod);
                    //printf("\tt = %s, tmi = %s\n", t.toChars(), tmi.toChars());
                    if (!isTypeIsolated(tmi))
                        return false;
                }
                return true;

            default:
                return true;
        }
    }

    /********************************************
     * Params:
     *    t = type of object to test one level of indirection down
     * Returns:
     *    true if an object typed `t` has no indirections
     *    which could have come from the function's parameters, mutable
     *    globals, or uplevel functions.
     */
    private bool isTypeIsolatedIndirect(Type t)
    {
        //printf("isTypeIsolatedIndirect(t: %s)\n", t.toChars());
        assert(t);

        /* Since `t` is one level down from an indirection, it could pick
         * up a reference to a mutable global or an outer function, so
         * return false.
         */
        if (!isPureBypassingInference() || isNested())
            return false;

        TypeFunction tf = type.toTypeFunction();

        //printf("isTypeIsolatedIndirect(%s) t = %s\n", tf.toChars(), t.toChars());

        size_t dim = tf.parameterList.length;
        for (size_t i = 0; i < dim; i++)
        {
            Parameter fparam = tf.parameterList[i];
            Type tp = fparam.type;
            if (!tp)
                continue;

            if (fparam.storageClass & (STC.lazy_ | STC.out_ | STC.ref_))
            {
                if (!traverseIndirections(tp, t))
                    return false;
                continue;
            }

            /* Goes down one level of indirection, then calls traverseIndirection() on
             * the result.
             * Returns:
             *  true if t is isolated from tp
             */
            static bool traverse(Type tp, Type t)
            {
                tp = tp.baseElemOf();
                switch (tp.ty)
                {
                    case Tarray:
                    case Tpointer:
                        return traverseIndirections(tp.nextOf(), t);

                    case Taarray:
                    case Tclass:
                        return traverseIndirections(tp, t);

                    case Tstruct:
                        /* Drill down and check the struct's fields
                         */
                        auto sym = tp.toDsymbol(null).isStructDeclaration();
                        foreach (v; sym.fields)
                        {
                            Type tprmi = v.type.addMod(tp.mod);
                            //printf("\ttp = %s, tprmi = %s\n", tp.toChars(), tprmi.toChars());
                            if (!traverse(tprmi, t))
                                return false;
                        }
                        return true;

                    default:
                        return true;
                }
            }

            if (!traverse(tp, t))
                return false;
        }
        // The 'this' reference is a parameter, too
        if (AggregateDeclaration ad = isCtorDeclaration() ? null : isThis())
        {
            Type tthis = ad.getType().addMod(tf.mod);
            //printf("\ttthis = %s\n", tthis.toChars());
            if (!traverseIndirections(tthis, t))
                return false;
        }

        return true;
    }

    /****************************************
     * Determine if function needs a static frame pointer.
     * Returns:
     *  `true` if function is really nested within other function.
     * Contracts:
     *  If isNested() returns true, isThis() should return false.
     */
    bool isNested() const
    {
        auto f = toAliasFunc();
        //printf("\ttoParent2() = '%s'\n", f.toParent2().toChars());
        return ((f.storage_class & STC.static_) == 0) &&
                (f.linkage == LINK.d) &&
                (f.toParent2().isFuncDeclaration() !is null);
    }

    /****************************************
     * Determine if function is a non-static member function
     * that has an implicit 'this' expression.
     * Returns:
     *  The aggregate it is a member of, or null.
     * Contracts:
     *  If isThis() returns true, isNested() should return false.
     */
    override inout(AggregateDeclaration) isThis() inout
    {
        //printf("+FuncDeclaration::isThis() '%s'\n", toChars());
        auto ad = (storage_class & STC.static_) ? objc.isThis(this) : isMember2();
        //printf("-FuncDeclaration::isThis() %p\n", ad);
        return ad;
    }

    override final bool needThis()
    {
        //printf("FuncDeclaration::needThis() '%s'\n", toChars());
        return toAliasFunc().isThis() !is null;
    }

    // Determine if a function is pedantically virtual
    final bool isVirtualMethod()
    {
        if (toAliasFunc() != this)
            return toAliasFunc().isVirtualMethod();

        //printf("FuncDeclaration::isVirtualMethod() %s\n", toChars());
        if (!isVirtual())
            return false;
        // If it's a final method, and does not override anything, then it is not virtual
        if (isFinalFunc() && foverrides.dim == 0)
        {
            return false;
        }
        return true;
    }

    // Determine if function goes into virtual function pointer table
    bool isVirtual() const
    {
        if (toAliasFunc() != this)
            return toAliasFunc().isVirtual();

        auto p = toParent();

        if (!isMember || !p.isClassDeclaration)
            return false;
                                                             // https://issues.dlang.org/show_bug.cgi?id=19654
        if (p.isClassDeclaration.classKind == ClassKind.objc && !p.isInterfaceDeclaration)
            return objc.isVirtual(this);

        version (none)
        {
            printf("FuncDeclaration::isVirtual(%s)\n", toChars());
            printf("isMember:%p isStatic:%d private:%d ctor:%d !Dlinkage:%d\n", isMember(), isStatic(), protection == Prot.Kind.private_, isCtorDeclaration(), linkage != LINK.d);
            printf("result is %d\n", isMember() && !(isStatic() || protection == Prot.Kind.private_ || protection == Prot.Kind.package_) && p.isClassDeclaration() && !(p.isInterfaceDeclaration() && isFinalFunc()));
        }
        return !(isStatic() || protection.kind == Prot.Kind.private_ || protection.kind == Prot.Kind.package_) && !(p.isInterfaceDeclaration() && isFinalFunc());
    }

    final bool isFinalFunc() const
    {
        if (toAliasFunc() != this)
            return toAliasFunc().isFinalFunc();

        version (none)
        {{
            auto cd = toParent().isClassDeclaration();
            printf("FuncDeclaration::isFinalFunc(%s), %x\n", toChars(), Declaration.isFinal());
            printf("%p %d %d %d\n", isMember(), isStatic(), Declaration.isFinal(), ((cd = toParent().isClassDeclaration()) !is null && cd.storage_class & STC.final_));
            printf("result is %d\n", isMember() && (Declaration.isFinal() || (cd !is null && cd.storage_class & STC.final_)));
            if (cd)
                printf("\tmember of %s\n", cd.toChars());
        }}
        if (!isMember())
            return false;
        if (Declaration.isFinal())
            return true;
        auto cd = toParent().isClassDeclaration();
        return (cd !is null) && (cd.storage_class & STC.final_);
    }

    bool addPreInvariant()
    {
        auto ad = isThis();
        ClassDeclaration cd = ad ? ad.isClassDeclaration() : null;
        return (ad && !(cd && cd.isCPPclass()) && global.params.useInvariants == CHECKENABLE.on && (protection.kind == Prot.Kind.protected_ || protection.kind == Prot.Kind.public_ || protection.kind == Prot.Kind.export_) && !naked);
    }

    bool addPostInvariant()
    {
        auto ad = isThis();
        ClassDeclaration cd = ad ? ad.isClassDeclaration() : null;
        return (ad && !(cd && cd.isCPPclass()) && ad.inv && global.params.useInvariants == CHECKENABLE.on && (protection.kind == Prot.Kind.protected_ || protection.kind == Prot.Kind.public_ || protection.kind == Prot.Kind.export_) && !naked);
    }

    override const(char)* kind() const
    {
        return generated ? "generated function" : "function";
    }

    /********************************************
     * If there are no overloads of function f, return that function,
     * otherwise return NULL.
     */
    final FuncDeclaration isUnique()
    {
        FuncDeclaration result = null;
        overloadApply(this, (Dsymbol s)
        {
            auto f = s.isFuncDeclaration();
            if (!f)
                return 0;
            if (result)
            {
                result = null;
                return 1; // ambiguous, done
            }
            else
            {
                result = f;
                return 0;
            }
        });
        return result;
    }

    /*********************************************
     * In the current function, we are calling 'this' function.
     * 1. Check to see if the current function can call 'this' function, issue error if not.
     * 2. If the current function is not the parent of 'this' function, then add
     *    the current function to the list of siblings of 'this' function.
     * 3. If the current function is a literal, and it's accessing an uplevel scope,
     *    then mark it as a delegate.
     * Returns true if error occurs.
     */
    extern (D) final bool checkNestedReference(Scope* sc, const ref Loc loc)
    {
        //printf("FuncDeclaration::checkNestedReference() %s\n", toPrettyChars());

        if (auto fld = this.isFuncLiteralDeclaration())
        {
            if (fld.tok == TOK.reserved)
            {
                fld.tok = TOK.function_;
                fld.vthis = null;
            }
        }

        if (!parent || parent == sc.parent)
            return false;
        if (ident == Id.require || ident == Id.ensure)
            return false;
        if (!isThis() && !isNested())
            return false;

        // The current function
        FuncDeclaration fdthis = sc.parent.isFuncDeclaration();
        if (!fdthis)
            return false; // out of function scope

        Dsymbol p = toParent2();

        // Function literals from fdthis to p must be delegates
        ensureStaticLinkTo(fdthis, p);

        if (isNested())
        {
            // The function that this function is in
            FuncDeclaration fdv = p.isFuncDeclaration();
            if (!fdv)
                return false;
            if (fdv == fdthis)
                return false;

            //printf("this = %s in [%s]\n", this.toChars(), this.loc.toChars());
            //printf("fdv  = %s in [%s]\n", fdv .toChars(), fdv .loc.toChars());
            //printf("fdthis = %s in [%s]\n", fdthis.toChars(), fdthis.loc.toChars());

            // Add this function to the list of those which called us
            if (fdthis != this)
            {
                bool found = false;
                for (size_t i = 0; i < siblingCallers.dim; ++i)
                {
                    if (siblingCallers[i] == fdthis)
                        found = true;
                }
                if (!found)
                {
                    //printf("\tadding sibling %s\n", fdthis.toPrettyChars());
                    if (!sc.intypeof && !(sc.flags & SCOPE.compile))
                        siblingCallers.push(fdthis);
                }
            }

            int lv = fdthis.getLevel(loc, sc, fdv);
            if (lv == -2)
                return true; // error
            if (lv == -1)
                return false; // downlevel call
            if (lv == 0)
                return false; // same level call

            // Uplevel call
        }
        return false;
    }

    /*******************************
     * Look at all the variables in this function that are referenced
     * by nested functions, and determine if a closure needs to be
     * created for them.
     */
    final bool needsClosure()
    {
        /* Need a closure for all the closureVars[] if any of the
         * closureVars[] are accessed by a
         * function that escapes the scope of this function.
         * We take the conservative approach and decide that a function needs
         * a closure if it:
         * 1) is a virtual function
         * 2) has its address taken
         * 3) has a parent that escapes
         * 4) calls another nested function that needs a closure
         *
         * Note that since a non-virtual function can be called by
         * a virtual one, if that non-virtual function accesses a closure
         * var, the closure still has to be taken. Hence, we check for isThis()
         * instead of isVirtual(). (thanks to David Friedman)
         *
         * When the function returns a local struct or class, `requiresClosure`
         * is already set to `true` upon entering this function when the
         * struct/class refers to a local variable and a closure is needed.
         */

        //printf("FuncDeclaration::needsClosure() %s\n", toChars());

        if (requiresClosure)
            goto Lyes;

        for (size_t i = 0; i < closureVars.dim; i++)
        {
            VarDeclaration v = closureVars[i];
            //printf("\tv = %s\n", v.toChars());

            for (size_t j = 0; j < v.nestedrefs.dim; j++)
            {
                FuncDeclaration f = v.nestedrefs[j];
                assert(f != this);

                //printf("\t\tf = %p, %s, isVirtual=%d, isThis=%p, tookAddressOf=%d\n", f, f.toChars(), f.isVirtual(), f.isThis(), f.tookAddressOf);

                /* Look to see if f escapes. We consider all parents of f within
                 * this, and also all siblings which call f; if any of them escape,
                 * so does f.
                 * Mark all affected functions as requiring closures.
                 */
                for (Dsymbol s = f; s && s != this; s = s.parent)
                {
                    FuncDeclaration fx = s.isFuncDeclaration();
                    if (!fx)
                        continue;
                    if (fx.isThis() || fx.tookAddressOf)
                    {
                        //printf("\t\tfx = %s, isVirtual=%d, isThis=%p, tookAddressOf=%d\n", fx.toChars(), fx.isVirtual(), fx.isThis(), fx.tookAddressOf);

                        /* Mark as needing closure any functions between this and f
                         */
                        markAsNeedingClosure((fx == f) ? fx.parent : fx, this);

                        requiresClosure = true;
                    }

                    /* We also need to check if any sibling functions that
                     * called us, have escaped. This is recursive: we need
                     * to check the callers of our siblings.
                     */
                    if (checkEscapingSiblings(fx, this))
                        requiresClosure = true;

                    /* https://issues.dlang.org/show_bug.cgi?id=12406
                     * Iterate all closureVars to mark all descendant
                     * nested functions that access to the closing context of this function.
                     */
                }
            }
        }
        if (requiresClosure)
            goto Lyes;

        return false;

    Lyes:
        //printf("\tneeds closure\n");
        return true;
    }

    /***********************************************
     * Check that the function contains any closure.
     * If it's @nogc, report suitable errors.
     * This is mostly consistent with FuncDeclaration::needsClosure().
     *
     * Returns:
     *      true if any errors occur.
     */
    extern (D) final bool checkClosure()
    {
        if (!needsClosure())
            return false;

        if (setGC())
        {
            error("is `@nogc` yet allocates closures with the GC");
            if (global.gag)     // need not report supplemental errors
                return true;
        }
        else
        {
            printGCUsage(loc, "using closure causes GC allocation");
            return false;
        }

        FuncDeclarations a;
        foreach (v; closureVars)
        {
            foreach (f; v.nestedrefs)
            {
                assert(f !is this);

            LcheckAncestorsOfANestedRef:
                for (Dsymbol s = f; s && s !is this; s = s.parent)
                {
                    auto fx = s.isFuncDeclaration();
                    if (!fx)
                        continue;
                    if (fx.isThis() ||
                        fx.tookAddressOf ||
                        checkEscapingSiblings(fx, this))
                    {
                        foreach (f2; a)
                        {
                            if (f2 == f)
                                break LcheckAncestorsOfANestedRef;
                        }
                        a.push(f);
                        .errorSupplemental(f.loc, "%s closes over variable %s at %s",
                            f.toPrettyChars(), v.toChars(), v.loc.toChars());
                        break LcheckAncestorsOfANestedRef;
                    }
                }
            }
        }

        return true;
    }

    /***********************************************
     * Determine if function's variables are referenced by a function
     * nested within it.
     */
    final bool hasNestedFrameRefs()
    {
        if (closureVars.dim)
            return true;

        /* If a virtual function has contracts, assume its variables are referenced
         * by those contracts, even if they aren't. Because they might be referenced
         * by the overridden or overriding function's contracts.
         * This can happen because frequire and fensure are implemented as nested functions,
         * and they can be called directly by an overriding function and the overriding function's
         * context had better match, or
         * https://issues.dlang.org/show_bug.cgi?id=7335 will bite.
         */
        if (fdrequire || fdensure)
            return true;

        if (foverrides.dim && isVirtualMethod())
        {
            for (size_t i = 0; i < foverrides.dim; i++)
            {
                FuncDeclaration fdv = foverrides[i];
                if (fdv.hasNestedFrameRefs())
                    return true;
            }
        }
        return false;
    }

    /****************************************************
     * Check whether result variable can be built.
     * Returns:
     *     `true` if the function has a return type that
     *     is different from `void`.
     */
    extern (D) private bool canBuildResultVar()
    {
        auto f = cast(TypeFunction)type;
        return f && f.nextOf() && f.nextOf().toBasetype().ty != Tvoid;
    }

    /****************************************************
     * Declare result variable lazily.
     */
    final void buildResultVar(Scope* sc, Type tret)
    {
        if (!vresult)
        {
            Loc loc = fensure ? fensure.loc : this.loc;

            /* If inferRetType is true, tret may not be a correct return type yet.
             * So, in here it may be a temporary type for vresult, and after
             * fbody.dsymbolSemantic() running, vresult.type might be modified.
             */
            vresult = new VarDeclaration(loc, tret, Id.result, null);
            vresult.storage_class |= STC.nodtor | STC.temp;
            if (!isVirtual())
                vresult.storage_class |= STC.const_;
            vresult.storage_class |= STC.result;

            // set before the semantic() for checkNestedReference()
            vresult.parent = this;
        }

        if (sc && vresult.semanticRun == PASS.init)
        {
            TypeFunction tf = type.toTypeFunction();
            if (tf.isref)
                vresult.storage_class |= STC.ref_;
            vresult.type = tret;

            vresult.dsymbolSemantic(sc);

            if (!sc.insert(vresult))
                error("out result %s is already defined", vresult.toChars());
            assert(vresult.parent == this);
        }
    }

    /****************************************************
     * Merge into this function the 'in' contracts of all it overrides.
     * 'in's are OR'd together, i.e. only one of them needs to pass.
     */
    // IN_LLVM replaced: final Statement mergeFrequire(Statement sf)
    final Statement mergeFrequire(Statement sf, Expressions *params = null)
    {
        version (IN_LLVM)
        {
            if (params is null)
                params = fdrequireParams;
        }

        /* If a base function and its override both have an IN contract, then
         * only one of them needs to succeed. This is done by generating:
         *
         * void derived.in() {
         *  try {
         *    base.in();
         *  }
         *  catch () {
         *    ... body of derived.in() ...
         *  }
         * }
         *
         * So if base.in() doesn't throw, derived.in() need not be executed, and the contract is valid.
         * If base.in() throws, then derived.in()'s body is executed.
         */

version (IN_LLVM)
{
        /* In LDC, we can't rely on these codegen hacks - we explicitly pass
         * parameters on to the contract functions.
         */
} else {
        /* Implementing this is done by having the overriding function call
         * nested functions (the fdrequire functions) nested inside the overridden
         * function. This requires that the stack layout of the calling function's
         * parameters and 'this' pointer be in the same place (as the nested
         * function refers to them).
         * This is easy for the parameters, as they are all on the stack in the same
         * place by definition, since it's an overriding function. The problem is
         * getting the 'this' pointer in the same place, since it is a local variable.
         * We did some hacks in the code generator to make this happen:
         *  1. always generate exception handler frame, or at least leave space for it
         *     in the frame (Windows 32 SEH only)
         *  2. always generate an EBP style frame
         *  3. since 'this' is passed in a register that is subsequently copied into
         *     a stack local, allocate that local immediately following the exception
         *     handler block, so it is always at the same offset from EBP.
         */
}
        foreach (fdv; foverrides)
        {
            /* The semantic pass on the contracts of the overridden functions must
             * be completed before code generation occurs.
             * https://issues.dlang.org/show_bug.cgi?id=3602
             */
            if (fdv.frequires && fdv.semanticRun != PASS.semantic3done)
            {
                assert(fdv._scope);
                Scope* sc = fdv._scope.push();
                sc.stc &= ~STC.override_;
                fdv.semantic3(sc);
                sc.pop();
            }

version (IN_LLVM)
            sf = fdv.mergeFrequire(sf, params);
else
            sf = fdv.mergeFrequire(sf);
            if (sf && fdv.fdrequire)
            {
                //printf("fdv.frequire: %s\n", fdv.frequire.toChars());
                /* Make the call:
                 *   try { __require(); }
                 *   catch (Throwable) { frequire; }
                 */
version (IN_LLVM)
                Expression e = new CallExp(loc, new VarExp(loc, fdv.fdrequire, false), params);
else
{
                Expression eresult = null;
                Expression e = new CallExp(loc, new VarExp(loc, fdv.fdrequire, false), eresult);
}
                Statement s2 = new ExpStatement(loc, e);

                auto c = new Catch(loc, getThrowable(), null, sf);
                c.internalCatch = true;
                auto catches = new Catches();
                catches.push(c);
                sf = new TryCatchStatement(loc, s2, catches);
            }
            else
                return null;
        }
        return sf;
    }

    /****************************************************
     * Determine whether an 'out' contract is declared inside
     * the given function or any of its overrides.
     * Params:
     *      fd = the function to search
     * Returns:
     *      true    found an 'out' contract
     */
    static bool needsFensure(FuncDeclaration fd)
    {
        if (fd.fensures)
            return true;

        foreach (fdv; fd.foverrides)
        {
            if (needsFensure(fdv))
                return true;
        }
        return false;
    }

    /****************************************************
     * Rewrite contracts as statements.
     */
    final void buildEnsureRequire()
    {

        if (frequires)
        {
            /*   in { statements1... }
             *   in { statements2... }
             *   ...
             * becomes:
             *   in { { statements1... } { statements2... } ... }
             */
            assert(frequires.dim);
            auto loc = (*frequires)[0].loc;
            auto s = new Statements;
            foreach (r; *frequires)
            {
                s.push(new ScopeStatement(r.loc, r, r.loc));
            }
            frequire = new CompoundStatement(loc, s);
        }

        if (fensures)
        {
            /*   out(id1) { statements1... }
             *   out(id2) { statements2... }
             *   ...
             * becomes:
             *   out(__result) { { ref id1 = __result; { statements1... } }
             *                   { ref id2 = __result; { statements2... } } ... }
             */
            assert(fensures.dim);
            auto loc = (*fensures)[0].ensure.loc;
            auto s = new Statements;
            foreach (r; *fensures)
            {
                if (r.id && canBuildResultVar())
                {
                    auto rloc = r.ensure.loc;
                    auto resultId = new IdentifierExp(rloc, Id.result);
                    auto init = new ExpInitializer(rloc, resultId);
                    auto stc = STC.ref_ | STC.temp | STC.result;
                    auto decl = new VarDeclaration(rloc, null, r.id, init, stc);
                    auto sdecl = new ExpStatement(rloc, decl);
                    s.push(new ScopeStatement(rloc, new CompoundStatement(rloc, sdecl, r.ensure), rloc));
                }
                else
                {
                    s.push(r.ensure);
                }
            }
            fensure = new CompoundStatement(loc, s);
        }

        if (!isVirtual())
            return;

        /* Rewrite contracts as nested functions, then call them. Doing it as nested
         * functions means that overriding functions can call them.
         */
        TypeFunction f = cast(TypeFunction) type;

        if (frequire)
        {
            version (IN_LLVM)
            {
                /* In LDC, we can't rely on the codegen hacks DMD has to be able
                 * to just magically call the contract function parameterless with
                 * the parameters being picked up from the outer stack frame.
                 *
                 * Thus, we actually pass all the function parameters to the
                 * __require call, rewriting out parameters to ref ones because
                 * they have already been zeroed in the outer function.
                 *
                 * Also set fdrequireParams here.
                 */
                Loc loc = frequire.loc;
                fdrequireParams = new Expressions();
                if (parameters)
                {
                    foreach (vd; *parameters)
                        fdrequireParams.push(new VarExp(loc, vd));
                }
                auto fparams = outToRef(f.parameterList.parameters);
                auto tf = new TypeFunction(ParameterList(fparams), Type.tvoid, LINK.d);
            }
            else
            {
                /*   in { ... }
                 * becomes:
                 *   void __require() { ... }
                 *   __require();
                 */
                Loc loc = frequire.loc;
                auto tf = new TypeFunction(ParameterList(), Type.tvoid, LINK.d);
            }
            tf.isnothrow = f.isnothrow;
            tf.isnogc = f.isnogc;
            tf.purity = f.purity;
            tf.trust = f.trust;
            auto fd = new FuncDeclaration(loc, loc, Id.require, STC.undefined_, tf);
            fd.fbody = frequire;
            Statement s1 = new ExpStatement(loc, fd);
            version (IN_LLVM)
            {
                Expression e = new CallExp(loc, new VarExp(loc, fd, false), fdrequireParams);
            }
            else
            {
                Expression e = new CallExp(loc, new VarExp(loc, fd, false), cast(Expressions*)null);
            }
            Statement s2 = new ExpStatement(loc, e);
            frequire = new CompoundStatement(loc, s1, s2);
            fdrequire = fd;
        }

        version (IN_LLVM)
        {
            /* We need to set fdensureParams here and not in the block below to
             * have the parameters available when calling a base class ensure(),
             * even if this function doesn't have an out contract.
             */
            fdensureParams = new Expressions();
            if (canBuildResultVar())
                fdensureParams.push(new IdentifierExp(loc, Id.result));
            if (parameters)
            {
                foreach (vd; *parameters)
                    fdensureParams.push(new VarExp(loc, vd));
            }
        }
        if (fensure)
        {
            version (IN_LLVM)
            {
                /* Same as for in contracts, see above. */
                Loc loc = fensure.loc;
                auto fparams = outToRef(f.parameterList.parameters);
            }
            else
            {
                /*   out (result) { ... }
                 * becomes:
                 *   void __ensure(ref tret result) { ... }
                 *   __ensure(result);
                 */
                Loc loc = fensure.loc;
                auto fparams = new Parameters();
            }
            Parameter p = null;
            if (canBuildResultVar())
            {
                p = new Parameter(STC.ref_ | STC.const_, f.nextOf(), Id.result, null, null);
                version (IN_LLVM)
                    fparams.insert(0, p);
                else
                    fparams.push(p);
            }
            auto tf = new TypeFunction(ParameterList(fparams), Type.tvoid, LINK.d);
            tf.isnothrow = f.isnothrow;
            tf.isnogc = f.isnogc;
            tf.purity = f.purity;
            tf.trust = f.trust;
            auto fd = new FuncDeclaration(loc, loc, Id.ensure, STC.undefined_, tf);
            fd.fbody = fensure;
            Statement s1 = new ExpStatement(loc, fd);
            version (IN_LLVM)
            {
                Expression e = new CallExp(loc, new VarExp(loc, fd, false), fdensureParams);
            }
            else
            {
                Expression eresult = null;
                if (canBuildResultVar())
                    eresult = new IdentifierExp(loc, Id.result);
                Expression e = new CallExp(loc, new VarExp(loc, fd, false), eresult);
            }
            Statement s2 = new ExpStatement(loc, e);
            fensure = new CompoundStatement(loc, s1, s2);
            fdensure = fd;
        }
    }

    /****************************************************
     * Merge into this function the 'out' contracts of all it overrides.
     * 'out's are AND'd together, i.e. all of them need to pass.
     */
    // IN_LLVM replaced: final Statement mergeFensure(Statement sf, Identifier oid)
    final Statement mergeFensure(Statement sf, Identifier oid, Expressions *params = null)
    {
        version (IN_LLVM)
        {
            if (params is null)
                params = fdensureParams;
        }

        /* Same comments as for mergeFrequire(), except that we take care
         * of generating a consistent reference to the 'result' local by
         * explicitly passing 'result' to the nested function as a reference
         * argument.
         * This won't work for the 'this' parameter as it would require changing
         * the semantic code for the nested function so that it looks on the parameter
         * list for the 'this' pointer, something that would need an unknown amount
         * of tweaking of various parts of the compiler that I'd rather leave alone.
         */
        foreach (fdv; foverrides)
        {
            /* The semantic pass on the contracts of the overridden functions must
             * be completed before code generation occurs.
             * https://issues.dlang.org/show_bug.cgi?id=3602 and
             * https://issues.dlang.org/show_bug.cgi?id=5230
             */
            if (needsFensure(fdv) && fdv.semanticRun != PASS.semantic3done)
            {
                assert(fdv._scope);
                Scope* sc = fdv._scope.push();
                sc.stc &= ~STC.override_;
                fdv.semantic3(sc);
                sc.pop();
            }

version (IN_LLVM)
            sf = fdv.mergeFensure(sf, oid, params);
else
            sf = fdv.mergeFensure(sf, oid);
            if (fdv.fdensure)
            {
                //printf("fdv.fensure: %s\n", fdv.fensure.toChars());
                // Make the call: __ensure(result)
                Expression eresult = null;
                if (canBuildResultVar())
                {
version (IN_LLVM)
                    eresult = (*params)[0];
else
                    eresult = new IdentifierExp(loc, oid);

                    Type t1 = fdv.type.nextOf().toBasetype();
version (IN_LLVM)
{
                    // We actually check for matching types in CommaExp::toElem,
                    // 'testcontract' breaks without this.
                    t1 = t1.constOf();
}
                    Type t2 = this.type.nextOf().toBasetype();
                    if (t1.isBaseOf(t2, null))
                    {
                        /* Making temporary reference variable is necessary
                         * in covariant return.
                         * https://issues.dlang.org/show_bug.cgi?id=5204
                         * https://issues.dlang.org/show_bug.cgi?id=10479
                         */
                        auto ei = new ExpInitializer(Loc.initial, eresult);
                        auto v = new VarDeclaration(Loc.initial, t1, Identifier.generateId("__covres"), ei);
                        v.storage_class |= STC.temp;
                        auto de = new DeclarationExp(Loc.initial, v);
                        auto ve = new VarExp(Loc.initial, v);
                        eresult = new CommaExp(Loc.initial, de, ve);
                    }
                }
version (IN_LLVM)
{
                if (eresult !is null)
                    (*params)[0] = eresult;
                Expression e = new CallExp(loc, new VarExp(loc, fdv.fdensure, false), params);
}
else
{
                Expression e = new CallExp(loc, new VarExp(loc, fdv.fdensure, false), eresult);
}
                Statement s2 = new ExpStatement(loc, e);

                if (sf)
                {
                    sf = new CompoundStatement(sf.loc, s2, sf);
                }
                else
                    sf = s2;
            }
        }
        return sf;
    }

    /*********************************************
     * Returns: the function's parameter list, and whether
     * it is variadic or not.
     */
    final ParameterList getParameterList()
    {
        if (type)
        {
            TypeFunction fdtype = type.isTypeFunction();
            return fdtype.parameterList;
        }

        return ParameterList(null, VarArg.none);
    }

    /**********************************
     * Generate a FuncDeclaration for a runtime library function.
     */
    static FuncDeclaration genCfunc(Parameters* fparams, Type treturn, const(char)* name, StorageClass stc = 0)
    {
        return genCfunc(fparams, treturn, Identifier.idPool(name, cast(uint)strlen(name)), stc);
    }

    static FuncDeclaration genCfunc(Parameters* fparams, Type treturn, Identifier id, StorageClass stc = 0)
    {
        FuncDeclaration fd;
        TypeFunction tf;
        Dsymbol s;
        __gshared DsymbolTable st = null;

        //printf("genCfunc(name = '%s')\n", id.toChars());
        //printf("treturn\n\t"); treturn.print();

        // See if already in table
        if (!st)
            st = new DsymbolTable();
        s = st.lookup(id);
        if (s)
        {
            fd = s.isFuncDeclaration();
            assert(fd);
            assert(fd.type.nextOf().equals(treturn));
        }
        else
        {
            tf = new TypeFunction(ParameterList(fparams), treturn, LINK.c, stc);
            fd = new FuncDeclaration(Loc.initial, Loc.initial, id, STC.static_, tf);
            fd.protection = Prot(Prot.Kind.public_);
            fd.linkage = LINK.c;

            st.insert(fd);
        }
        return fd;
    }

    /******************
     * Check parameters and return type of D main() function.
     * Issue error messages.
     */
    extern (D) final void checkDmain()
    {
        TypeFunction tf = type.toTypeFunction();
        const nparams = tf.parameterList.length;
        bool argerr;
        if (nparams == 1)
        {
            auto fparam0 = tf.parameterList[0];
            auto t = fparam0.type.toBasetype();
            if (t.ty != Tarray ||
                t.nextOf().ty != Tarray ||
                t.nextOf().nextOf().ty != Tchar ||
                fparam0.storageClass & (STC.out_ | STC.ref_ | STC.lazy_))
            {
                argerr = true;
            }
        }

        if (!tf.nextOf())
            error("must return `int` or `void`");
        else if (tf.nextOf().ty != Tint32 && tf.nextOf().ty != Tvoid)
            error("must return `int` or `void`, not `%s`", tf.nextOf().toChars());
        else if (tf.parameterList.varargs || nparams >= 2 || argerr)
            error("parameters must be `main()` or `main(string[] args)`");
    }

    override final inout(FuncDeclaration) isFuncDeclaration() inout
    {
        return this;
    }

    inout(FuncDeclaration) toAliasFunc() inout
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/********************************************************
 * Generate Expression to call the invariant.
 * Input:
 *      ad      aggregate with the invariant
 *      vthis   variable with 'this'
 * Returns:
 *      void expression that calls the invariant
 */
Expression addInvariant(const ref Loc loc, Scope* sc, AggregateDeclaration ad, VarDeclaration vthis)
{
    Expression e = null;
    // Call invariant directly only if it exists
    FuncDeclaration inv = ad.inv;
    ClassDeclaration cd = ad.isClassDeclaration();

    while (!inv && cd)
    {
        cd = cd.baseClass;
        if (!cd)
            break;
        inv = cd.inv;
    }
    if (inv)
    {
        version (all)
        {
            // Workaround for https://issues.dlang.org/show_bug.cgi?id=13394
            // For the correct mangling,
            // run attribute inference on inv if needed.
            inv.functionSemantic();
        }

        //e = new DsymbolExp(Loc.initial, inv);
        //e = new CallExp(Loc.initial, e);
        //e = e.semantic(sc2);

        /* https://issues.dlang.org/show_bug.cgi?id=13113
         * Currently virtual invariant calls completely
         * bypass attribute enforcement.
         * Change the behavior of pre-invariant call by following it.
         */
        e = new ThisExp(Loc.initial);
        e.type = vthis.type;
        e = new DotVarExp(Loc.initial, e, inv, false);
        e.type = inv.type;
        e = new CallExp(Loc.initial, e);
        e.type = Type.tvoid;
    }
    return e;
}

/***************************************************
 * Visit each overloaded function/template in turn, and call dg(s) on it.
 * Exit when no more, or dg(s) returns nonzero.
 *
 * Params:
 *  fstart = symbol to start from
 *  dg = the delegate to be called on the overload
 *  sc = the initial scope from the calling context
 *
 * Returns:
 *      ==0     continue
 *      !=0     done
 */
extern (D) int overloadApply(Dsymbol fstart, scope int delegate(Dsymbol) dg, Scope* sc = null)
{
    Dsymbol next;
    for (Dsymbol d = fstart; d; d = next)
    {
        import dmd.access : checkSymbolAccess;
        if (auto od = d.isOverDeclaration())
        {
            if (od.hasOverloads)
            {
                /* The scope is needed here to check whether a function in
                   an overload set was added by means of a private alias (or a
                   selective import). If the scope where the alias is created
                   is imported somewhere, the overload set is visible, but the private
                   alias is not.
                 */
                if (sc)
                {
                    if (checkSymbolAccess(sc, od))
                    {
                        if (int r = overloadApply(od.aliassym, dg, sc))
                            return r;
                    }
                }
                else if (int r = overloadApply(od.aliassym, dg, sc))
                    return r;
            }
            else
            {
                if (int r = dg(od.aliassym))
                    return r;
            }
            next = od.overnext;
        }
        else if (auto fa = d.isFuncAliasDeclaration())
        {
            if (fa.hasOverloads)
            {
                if (int r = overloadApply(fa.funcalias, dg, sc))
                    return r;
            }
            else if (auto fd = fa.toAliasFunc())
            {
                if (int r = dg(fd))
                    return r;
            }
            else
            {
                d.error("is aliased to a function");
                break;
            }
            next = fa.overnext;
        }
        else if (auto ad = d.isAliasDeclaration())
        {
            if (sc)
            {
                if (checkSymbolAccess(sc, ad))
                    next = ad.toAlias();
            }
            else
               next = ad.toAlias();
            if (next == ad)
                break;
            if (next == fstart)
                break;
        }
        else if (auto td = d.isTemplateDeclaration())
        {
            if (int r = dg(td))
                return r;
            next = td.overnext;
        }
        else if (auto fd = d.isFuncDeclaration())
        {
            if (int r = dg(fd))
                return r;
            next = fd.overnext;
        }
        else
        {
            d.error("is aliased to a function");
            break;
            // BUG: should print error message?
        }
    }
    return 0;
}

/**
Checks for mismatching modifiers between `lhsMod` and `rhsMod` and prints the
mismatching modifiers to `buf`.

The modifiers of the `lhsMod` mismatching the ones with the `rhsMod` are printed, i.e.
lhs(shared) vs. rhs() prints "`shared`", wheras lhs() vs rhs(shared) prints "non-shared".

Params:
    buf = output buffer to write to
    lhsMod = modifier on the left-hand side
    lhsMod = modifier on the right-hand side

Returns:

A tuple with `isMutable` and `isNotShared` set
if the `lhsMod` is missing those modifiers (compared to rhs).
*/
auto MODMatchToBuffer(OutBuffer* buf, ubyte lhsMod, ubyte rhsMod)
{
    static struct Mismatches
    {
        bool isNotShared;
        bool isMutable;
    }

    Mismatches mismatches;

    bool bothMutable = ((lhsMod & rhsMod) == 0);
    bool sharedMismatch = ((lhsMod ^ rhsMod) & MODFlags.shared_) != 0;
    bool sharedMismatchOnly = ((lhsMod ^ rhsMod) == MODFlags.shared_);

    if (lhsMod & MODFlags.shared_)
        buf.writestring("`shared` ");
    else if (sharedMismatch && !(lhsMod & MODFlags.immutable_))
    {
        buf.writestring("non-shared ");
        mismatches.isNotShared = true;
    }

    if (bothMutable && sharedMismatchOnly)
    {
    }
    else if (lhsMod & MODFlags.immutable_)
        buf.writestring("`immutable` ");
    else if (lhsMod & MODFlags.const_)
        buf.writestring("`const` ");
    else if (lhsMod & MODFlags.wild)
        buf.writestring("`inout` ");
    else
    {
        buf.writestring("mutable ");
        mismatches.isMutable = true;
    }

    return mismatches;
}

///
unittest
{
    OutBuffer buf;
    auto mismatches = MODMatchToBuffer(&buf, MODFlags.shared_, 0);
    assert(buf.peekSlice == "`shared` ");
    assert(!mismatches.isNotShared);

    buf.reset;
    mismatches = MODMatchToBuffer(&buf, 0, MODFlags.shared_);
    assert(buf.peekSlice == "non-shared ");
    assert(mismatches.isNotShared);

    buf.reset;
    mismatches = MODMatchToBuffer(&buf, MODFlags.const_, 0);
    assert(buf.peekSlice == "`const` ");
    assert(!mismatches.isMutable);

    buf.reset;
    mismatches = MODMatchToBuffer(&buf, 0, MODFlags.const_);
    assert(buf.peekSlice == "mutable ");
    assert(mismatches.isMutable);
}

private const(char)* prependSpace(const(char)* str)
{
    if (!str || !*str) return "";

    return (" " ~ str[0 .. strlen(str)] ~ "\0").ptr;
}

/// Flag used by $(LREF resolveFuncCall).
enum FuncResolveFlag : ubyte
{
    stdandard = 0,      /// issue error messages, solve the call.
    quiet = 1,          /// do not issue error message on no match, just return `null`.
    overloadOnly = 2,   /// only resolve overloads.
}

/*******************************************
 * Given a symbol that could be either a FuncDeclaration or
 * a function template, resolve it to a function symbol.
 * Params:
 *      loc =           instantiation location
 *      sc =            instantiation scope
 *      s =             instantiation symbol
 *      tiargs =        initial list of template arguments
 *      tthis =         if !NULL, the `this` argument type
 *      fargs =         arguments to function
 *      flags =         see $(LREF FuncResolveFlag).
 * Returns:
 *      if match is found, then function symbol, else null
 */
FuncDeclaration resolveFuncCall(const ref Loc loc, Scope* sc, Dsymbol s,
    Objects* tiargs, Type tthis, Expressions* fargs, FuncResolveFlag flags)
{
    if (!s)
        return null; // no match

    version (none)
    {
        printf("resolveFuncCall('%s')\n", s.toChars());
        if (tthis)
            printf("\tthis: %s\n", tthis.toChars());
        if (fargs)
        {
            for (size_t i = 0; i < fargs.dim; i++)
            {
                Expression arg = (*fargs)[i];
                assert(arg.type);
                printf("\t%s: ", arg.toChars());
                arg.type.print();
            }
        }
    }

    if (tiargs && arrayObjectIsError(tiargs) ||
        fargs && arrayObjectIsError(cast(Objects*)fargs))
    {
        return null;
    }

    Match m;
    m.last = MATCH.nomatch;
    functionResolve(&m, s, loc, sc, tiargs, tthis, fargs, null);
    auto orig_s = s;

    if (m.last > MATCH.nomatch && m.lastf)
    {
        if (m.count == 1) // exactly one match
        {
            if (!(flags & FuncResolveFlag.quiet))
                m.lastf.functionSemantic();
            return m.lastf;
        }
        if ((flags & FuncResolveFlag.overloadOnly) && !tthis && m.lastf.needThis())
        {
            return m.lastf;
        }
    }

    /* Failed to find a best match.
     * Do nothing or print error.
     */
    if (m.last <= MATCH.nomatch)
    {
        // error was caused on matched function
        if (m.count == 1)
            return m.lastf;

        // if do not print error messages
        if (flags & FuncResolveFlag.quiet)
            return null; // no match
    }

    auto fd = s.isFuncDeclaration();
    auto od = s.isOverDeclaration();
    auto td = s.isTemplateDeclaration();
    if (td && td.funcroot)
        s = fd = td.funcroot;

    OutBuffer tiargsBuf;
    arrayObjectsToBuffer(&tiargsBuf, tiargs);

    OutBuffer fargsBuf;
    fargsBuf.writeByte('(');
    argExpTypesToCBuffer(&fargsBuf, fargs);
    fargsBuf.writeByte(')');
    if (tthis)
        tthis.modToBuffer(&fargsBuf);

    // max num of overloads to print (-v overrides this).
    enum int numOverloadsDisplay = 5;

    if (!m.lastf && !(flags & FuncResolveFlag.quiet)) // no match
    {
        if (!fd && !td && !od)
        {
            /* This case happens when several ctors are mixed in an agregate.
               A (bad) error message is already generated in overloadApply().
               see https://issues.dlang.org/show_bug.cgi?id=19729
            */
        }
        else if (td && !fd) // all of overloads are templates
        {
            .error(loc, "%s `%s.%s` cannot deduce function from argument types `!(%s)%s`, candidates are:",
                td.kind(), td.parent.toPrettyChars(), td.ident.toChars(),
                tiargsBuf.peekString(), fargsBuf.peekString());

            printCandidates(loc, td);
        }
        else if (od)
        {
            .error(loc, "none of the overloads of `%s` are callable using argument types `!(%s)%s`",
                od.ident.toChars(), tiargsBuf.peekString(), fargsBuf.peekString());
        }
        else
        {
            assert(fd);

            // remove when deprecation period of class allocators and deallocators is over
            if (fd.isNewDeclaration() && fd.checkDisabled(loc, sc))
                return null;

            bool hasOverloads = fd.overnext !is null;
            auto tf = fd.type.toTypeFunction();
            if (tthis && !MODimplicitConv(tthis.mod, tf.mod)) // modifier mismatch
            {
                OutBuffer thisBuf, funcBuf;
                MODMatchToBuffer(&thisBuf, tthis.mod, tf.mod);
                auto mismatches = MODMatchToBuffer(&funcBuf, tf.mod, tthis.mod);
                if (hasOverloads)
                {
                    .error(loc, "none of the overloads of `%s` are callable using a %sobject, candidates are:",
                        fd.ident.toChars(), thisBuf.peekString());
                }
                else
                {
                    const(char)* failMessage;
                    functionResolve(&m, orig_s, loc, sc, tiargs, tthis, fargs, &failMessage);
                    if (failMessage)
                    {
                        .error(loc, "%s `%s%s%s` is not callable using argument types `%s`",
                            fd.kind(), fd.toPrettyChars(), parametersTypeToChars(tf.parameterList),
                            tf.modToChars(), fargsBuf.peekString());
                        errorSupplemental(loc, failMessage);
                    }
                    else
                    {
                        auto fullFdPretty = fd.toPrettyChars();
                        .error(loc, "%smethod `%s` is not callable using a %sobject",
                            funcBuf.peekString(), fullFdPretty,
                            thisBuf.peekString());

                        if (mismatches.isNotShared)
                            .errorSupplemental(loc, "Consider adding `shared` to %s", fullFdPretty);
                        else if (mismatches.isMutable)
                            .errorSupplemental(loc, "Consider adding `const` or `inout` to %s", fullFdPretty);
                    }
                }
            }
            else
            {
                //printf("tf = %s, args = %s\n", tf.deco, (*fargs)[0].type.deco);
                if (hasOverloads)
                {
                    .error(loc, "none of the overloads of `%s` are callable using argument types `%s`, candidates are:",
                        fd.toChars(), fargsBuf.peekString());
                }
                else
                {
                    .error(loc, "%s `%s%s%s` is not callable using argument types `%s`",
                        fd.kind(), fd.toPrettyChars(), parametersTypeToChars(tf.parameterList),
                        tf.modToChars(), fargsBuf.peekString());
                    // re-resolve to check for supplemental message
                    const(char)* failMessage;
                    functionResolve(&m, orig_s, loc, sc, tiargs, tthis, fargs, &failMessage);
                    if (failMessage)
                        errorSupplemental(loc, failMessage);
                }
            }

            if (hasOverloads)
                printCandidates(loc, fd);
        }
    }
    else if (m.nextf)
    {
        TypeFunction tf1 = m.lastf.type.toTypeFunction();
        TypeFunction tf2 = m.nextf.type.toTypeFunction();
        const(char)* lastprms = parametersTypeToChars(tf1.parameterList);
        const(char)* nextprms = parametersTypeToChars(tf2.parameterList);

        const(char)* mod1 = prependSpace(MODtoChars(tf1.mod));
        const(char)* mod2 = prependSpace(MODtoChars(tf2.mod));

        .error(loc, "`%s.%s` called with argument types `%s` matches both:\n%s:     `%s%s%s`\nand:\n%s:     `%s%s%s`",
            s.parent.toPrettyChars(), s.ident.toChars(),
            fargsBuf.peekString(),
            m.lastf.loc.toChars(), m.lastf.toPrettyChars(), lastprms, mod1,
            m.nextf.loc.toChars(), m.nextf.toPrettyChars(), nextprms, mod2);
    }
    return null;
}

/*******************************************
 * Prints template and function overload candidates as supplemental errors.
 * Params:
 *      loc =           instantiation location
 *      declaration =   the declaration to print overload candidates for
 */
private void printCandidates(Decl)(const ref Loc loc, Decl declaration)
if (is(Decl == TemplateDeclaration) || is(Decl == FuncDeclaration))
{
    // max num of overloads to print (-v overrides this).
    int numToDisplay = 5;

    overloadApply(declaration, (Dsymbol s)
    {
        Dsymbol nextOverload;

        if (auto fd = s.isFuncDeclaration())
        {
            if (fd.errors || fd.type.ty == Terror)
                return 0;

            auto tf = cast(TypeFunction) fd.type;
            .errorSupplemental(fd.loc, "`%s%s`", fd.toPrettyChars(),
                parametersTypeToChars(tf.parameterList));
            nextOverload = fd.overnext;
        }
        else if (auto td = s.isTemplateDeclaration())
        {
            .errorSupplemental(td.loc, "`%s`", td.toPrettyChars());
            nextOverload = td.overnext;
        }

        if (global.params.verbose || --numToDisplay != 0)
            return 0;

        // Too many overloads to sensibly display.
        // Just show count of remaining overloads.
        int num = 0;
        overloadApply(nextOverload, (s) { ++num; return 0; });

        if (num > 0)
            .errorSupplemental(loc, "... (%d more, -v to show) ...", num);
        return 1;   // stop iterating
    });
}

/**************************************
 * Returns an indirect type one step from t.
 */
Type getIndirection(Type t)
{
    t = t.baseElemOf();
    if (t.ty == Tarray || t.ty == Tpointer)
        return t.nextOf().toBasetype();
    if (t.ty == Taarray || t.ty == Tclass)
        return t;
    if (t.ty == Tstruct)
        return t.hasPointers() ? t : null; // TODO

    // should consider TypeDelegate?
    return null;
}

/**************************************
 * Performs type-based alias analysis between a newly created value and a pre-
 * existing memory reference:
 *
 * Assuming that a reference A to a value of type `ta` was available to the code
 * that created a reference B to a value of type `tb`, it returns whether B
 * might alias memory reachable from A based on the types involved (either
 * directly or via any number of indirections in either A or B).
 *
 * This relation is not symmetric in the two arguments. For example, a
 * a `const(int)` reference can point to a pre-existing `int`, but not the other
 * way round.
 *
 * Examples:
 *
 *      ta,           tb,               result
 *      `const(int)`, `int`,            `false`
 *      `int`,        `const(int)`,     `true`
 *      `int`,        `immutable(int)`, `false`
 *      const(immutable(int)*), immutable(int)*, false   // BUG: returns true
 *
 * Params:
 *      ta = value type being referred to
 *      tb = referred to value type that could be constructed from ta
 *
 * Returns:
 *      true if reference to `tb` is isolated from reference to `ta`
 */
private bool traverseIndirections(Type ta, Type tb)
{
    //printf("traverseIndirections(%s, %s)\n", ta.toChars(), tb.toChars());

    /* Threaded list of aggregate types already examined,
     * used to break cycles.
     * Cycles in type graphs can only occur with aggregates.
     */
    static struct Ctxt
    {
        Ctxt* prev;
        Type type;      // an aggregate type
    }

    static bool traverse(Type ta, Type tb, Ctxt* ctxt, bool reversePass)
    {
        //printf("traverse(%s, %s)\n", ta.toChars(), tb.toChars());
        ta = ta.baseElemOf();
        tb = tb.baseElemOf();

        // First, check if the pointed-to types are convertible to each other such
        // that they might alias directly.
        static bool mayAliasDirect(Type source, Type target)
        {
            return
                // if source is the same as target or can be const-converted to target
                source.constConv(target) != MATCH.nomatch ||
                // if target is void and source can be const-converted to target
                (target.ty == Tvoid && MODimplicitConv(source.mod, target.mod));
        }

        if (mayAliasDirect(reversePass ? tb : ta, reversePass ? ta : tb))
        {
            //printf(" true  mayalias %s %s %d\n", ta.toChars(), tb.toChars(), reversePass);
            return false;
        }
        if (ta.nextOf() && ta.nextOf() == tb.nextOf())
        {
             //printf(" next==next %s %s %d\n", ta.toChars(), tb.toChars(), reversePass);
             return true;
        }

        if (tb.ty == Tclass || tb.ty == Tstruct)
        {
            for (Ctxt* c = ctxt; c; c = c.prev)
                if (tb == c.type)
                    return true;
            Ctxt c;
            c.prev = ctxt;
            c.type = tb;

            /* Traverse the type of each field of the aggregate
             */
            AggregateDeclaration sym = tb.toDsymbol(null).isAggregateDeclaration();
            foreach (v; sym.fields)
            {
                Type tprmi = v.type.addMod(tb.mod);
                //printf("\ttb = %s, tprmi = %s\n", tb.toChars(), tprmi.toChars());
                if (!traverse(ta, tprmi, &c, reversePass))
                    return false;
            }
        }
        else if (tb.ty == Tarray || tb.ty == Taarray || tb.ty == Tpointer)
        {
            Type tind = tb.nextOf();
            if (!traverse(ta, tind, ctxt, reversePass))
                return false;
        }
        else if (tb.hasPointers())
        {
            // BUG: consider the context pointer of delegate types
            return false;
        }

        // Still no match, so try breaking up ta if we have not done so yet.
        if (!reversePass)
            return traverse(tb, ta, ctxt, true);

        return true;
    }

    // To handle arbitrary levels of indirections in both parameters, we
    // recursively descend into aggregate members/levels of indirection in both
    // `ta` and `tb` while avoiding cycles. Start with the original types.
    const result = traverse(ta, tb, null, false);
    //printf("  returns %d\n", result);
    return result;
}

/* For all functions between outerFunc and f, mark them as needing
 * a closure.
 */
private void markAsNeedingClosure(Dsymbol f, FuncDeclaration outerFunc)
{
    for (Dsymbol sx = f; sx && sx != outerFunc; sx = sx.parent)
    {
        FuncDeclaration fy = sx.isFuncDeclaration();
        if (fy && fy.closureVars.dim)
        {
            /* fy needs a closure if it has closureVars[],
             * because the frame pointer in the closure will be accessed.
             */
            fy.requiresClosure = true;
        }
    }
}

/********
 * Given a nested function f inside a function outerFunc, check
 * if any sibling callers of f have escaped. If so, mark
 * all the enclosing functions as needing closures.
 * This is recursive: we need to check the callers of our siblings.
 * Note that nested functions can only call lexically earlier nested
 * functions, so loops are impossible.
 * Params:
 *      f = inner function (nested within outerFunc)
 *      outerFunc = outer function
 *      p = for internal recursion use
 * Returns:
 *      true if any closures were needed
 */
private bool checkEscapingSiblings(FuncDeclaration f, FuncDeclaration outerFunc, void* p = null)
{
    static struct PrevSibling
    {
        PrevSibling* p;
        FuncDeclaration f;
    }

    PrevSibling ps;
    ps.p = cast(PrevSibling*)p;
    ps.f = f;

    //printf("checkEscapingSiblings(f = %s, outerfunc = %s)\n", f.toChars(), outerFunc.toChars());
    bool bAnyClosures = false;
    for (size_t i = 0; i < f.siblingCallers.dim; ++i)
    {
        FuncDeclaration g = f.siblingCallers[i];
        if (g.isThis() || g.tookAddressOf)
        {
            markAsNeedingClosure(g, outerFunc);
            bAnyClosures = true;
        }

        PrevSibling* prev = cast(PrevSibling*)p;
        while (1)
        {
            if (!prev)
            {
                bAnyClosures |= checkEscapingSiblings(g, outerFunc, &ps);
                break;
            }
            if (prev.f == g)
                break;
            prev = prev.p;
        }
    }
    //printf("\t%d\n", bAnyClosures);
    return bAnyClosures;
}

/***********************************************************
 * Used as a way to import a set of functions from another scope into this one.
 */
extern (C++) final class FuncAliasDeclaration : FuncDeclaration
{
    FuncDeclaration funcalias;
    bool hasOverloads;

    extern (D) this(Identifier ident, FuncDeclaration funcalias, bool hasOverloads = true)
    {
        super(funcalias.loc, funcalias.endloc, ident, funcalias.storage_class, funcalias.type);
        assert(funcalias != this);
        this.funcalias = funcalias;

        this.hasOverloads = hasOverloads;
        if (hasOverloads)
        {
            if (FuncAliasDeclaration fad = funcalias.isFuncAliasDeclaration())
                this.hasOverloads = fad.hasOverloads;
        }
        else
        {
            // for internal use
            assert(!funcalias.isFuncAliasDeclaration());
            this.hasOverloads = false;
        }
        userAttribDecl = funcalias.userAttribDecl;
    }

    override inout(FuncAliasDeclaration) isFuncAliasDeclaration() inout
    {
        return this;
    }

    override const(char)* kind() const
    {
        return "function alias";
    }

    override inout(FuncDeclaration) toAliasFunc() inout
    {
        return funcalias.toAliasFunc();
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class FuncLiteralDeclaration : FuncDeclaration
{
    TOK tok;        // TOK.function_ or TOK.delegate_
    Type treq;      // target of return type inference

    // backend
    bool deferToObj;

    extern (D) this(const ref Loc loc, const ref Loc endloc, Type type, TOK tok, ForeachStatement fes, Identifier id = null)
    {
        super(loc, endloc, null, STC.undefined_, type);
        this.ident = id ? id : Id.empty;
        this.tok = tok;
        this.fes = fes;
        //printf("FuncLiteralDeclaration() id = '%s', type = '%s'\n", this.ident.toChars(), type.toChars());
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        //printf("FuncLiteralDeclaration::syntaxCopy('%s')\n", toChars());
        assert(!s);
        auto f = new FuncLiteralDeclaration(loc, endloc, type.syntaxCopy(), tok, fes, ident);
        f.treq = treq; // don't need to copy
        return FuncDeclaration.syntaxCopy(f);
    }

    override bool isNested() const
    {
        //printf("FuncLiteralDeclaration::isNested() '%s'\n", toChars());
        return (tok != TOK.function_) && !isThis();
    }

    override inout(AggregateDeclaration) isThis() inout
    {
        return tok == TOK.delegate_ ? super.isThis() : null;
    }

    override bool isVirtual() const
    {
        return false;
    }

    override bool addPreInvariant()
    {
        return false;
    }

    override bool addPostInvariant()
    {
        return false;
    }

    /*******************************
     * Modify all expression type of return statements to tret.
     *
     * On function literals, return type may be modified based on the context type
     * after its semantic3 is done, in FuncExp::implicitCastTo.
     *
     *  A function() dg = (){ return new B(); } // OK if is(B : A) == true
     *
     * If B to A conversion is convariant that requires offseet adjusting,
     * all return statements should be adjusted to return expressions typed A.
     */
    void modifyReturns(Scope* sc, Type tret)
    {
        import dmd.statement_rewrite_walker;

        extern (C++) final class RetWalker : StatementRewriteWalker
        {
            alias visit = typeof(super).visit;
        public:
            Scope* sc;
            Type tret;
            FuncLiteralDeclaration fld;

            override void visit(ReturnStatement s)
            {
                Expression exp = s.exp;
                if (exp && !exp.type.equals(tret))
                {
                    s.exp = exp.castTo(sc, tret);
                }
            }
        }

        if (semanticRun < PASS.semantic3done)
            return;

        if (fes)
            return;

        scope RetWalker w = new RetWalker();
        w.sc = sc;
        w.tret = tret;
        w.fld = this;
        fbody.accept(w);

        // Also update the inferred function type to match the new return type.
        // This is required so the code generator does not try to cast the
        // modified returns back to the original type.
        if (inferRetType && type.nextOf() != tret)
            type.toTypeFunction().next = tret;
    }

    override inout(FuncLiteralDeclaration) isFuncLiteralDeclaration() inout
    {
        return this;
    }

    override const(char)* kind() const
    {
        // GCC requires the (char*) casts
        return (tok != TOK.function_) ? "delegate" : "function";
    }

    override const(char)* toPrettyChars(bool QualifyTypes = false)
    {
        if (parent)
        {
            TemplateInstance ti = parent.isTemplateInstance();
            if (ti)
                return ti.tempdecl.toPrettyChars(QualifyTypes);
        }
        return Dsymbol.toPrettyChars(QualifyTypes);
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class CtorDeclaration : FuncDeclaration
{
    extern (D) this(const ref Loc loc, const ref Loc endloc, StorageClass stc, Type type)
    {
        super(loc, endloc, Id.ctor, stc, type);
        //printf("CtorDeclaration(loc = %s) %s\n", loc.toChars(), toChars());
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto f = new CtorDeclaration(loc, endloc, storage_class, type.syntaxCopy());
        return FuncDeclaration.syntaxCopy(f);
    }

    override const(char)* kind() const
    {
        return "constructor";
    }

    override const(char)* toChars() const
    {
        return "this";
    }

    override bool isVirtual() const
    {
        return false;
    }

    override bool addPreInvariant()
    {
        return false;
    }

    override bool addPostInvariant()
    {
        return (isThis() && vthis && global.params.useInvariants == CHECKENABLE.on);
    }

    override inout(CtorDeclaration) isCtorDeclaration() inout
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
extern (C++) final class PostBlitDeclaration : FuncDeclaration
{
    extern (D) this(const ref Loc loc, const ref Loc endloc, StorageClass stc, Identifier id)
    {
        super(loc, endloc, id, stc, null);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto dd = new PostBlitDeclaration(loc, endloc, storage_class, ident);
        return FuncDeclaration.syntaxCopy(dd);
    }

    override bool isVirtual() const
    {
        return false;
    }

    override bool addPreInvariant()
    {
        return false;
    }

    override bool addPostInvariant()
    {
        return (isThis() && vthis && global.params.useInvariants == CHECKENABLE.on);
    }

    override bool overloadInsert(Dsymbol s)
    {
        return false; // cannot overload postblits
    }

    override inout(PostBlitDeclaration) isPostBlitDeclaration() inout
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
extern (C++) final class DtorDeclaration : FuncDeclaration
{
    extern (D) this(const ref Loc loc, const ref Loc endloc)
    {
        super(loc, endloc, Id.dtor, STC.undefined_, null);
    }

    extern (D) this(const ref Loc loc, const ref Loc endloc, StorageClass stc, Identifier id)
    {
        super(loc, endloc, id, stc, null);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto dd = new DtorDeclaration(loc, endloc, storage_class, ident);
        return FuncDeclaration.syntaxCopy(dd);
    }

    override const(char)* kind() const
    {
        return "destructor";
    }

    override const(char)* toChars() const
    {
        return "~this";
    }

    override bool isVirtual() const
    {
        // D dtor's don't get put into the vtbl[]
        // this is a hack so that extern(C++) destructors report as virtual, which are manually added to the vtable
        return vtblIndex != -1;
    }

    override bool addPreInvariant()
    {
        return (isThis() && vthis && global.params.useInvariants == CHECKENABLE.on);
    }

    override bool addPostInvariant()
    {
        return false;
    }

    override bool overloadInsert(Dsymbol s)
    {
        return false; // cannot overload destructors
    }

    override inout(DtorDeclaration) isDtorDeclaration() inout
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
extern (C++) class StaticCtorDeclaration : FuncDeclaration
{
    extern (D) this(const ref Loc loc, const ref Loc endloc, StorageClass stc)
    {
        super(loc, endloc, Identifier.generateIdWithLoc("_staticCtor", loc), STC.static_ | stc, null);
    }

    extern (D) this(const ref Loc loc, const ref Loc endloc, string name, StorageClass stc)
    {
        super(loc, endloc, Identifier.generateIdWithLoc(name, loc), STC.static_ | stc, null);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto scd = new StaticCtorDeclaration(loc, endloc, storage_class);
        return FuncDeclaration.syntaxCopy(scd);
    }

    override final inout(AggregateDeclaration) isThis() inout
    {
        return null;
    }

    override final bool isVirtual() const
    {
        return false;
    }

    override final bool addPreInvariant()
    {
        return false;
    }

    override final bool addPostInvariant()
    {
        return false;
    }

    override final bool hasStaticCtorOrDtor()
    {
        return true;
    }

    override final inout(StaticCtorDeclaration) isStaticCtorDeclaration() inout
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
extern (C++) final class SharedStaticCtorDeclaration : StaticCtorDeclaration
{
    extern (D) this(const ref Loc loc, const ref Loc endloc, StorageClass stc)
    {
        super(loc, endloc, "_sharedStaticCtor", stc);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto scd = new SharedStaticCtorDeclaration(loc, endloc, storage_class);
        return FuncDeclaration.syntaxCopy(scd);
    }

    override inout(SharedStaticCtorDeclaration) isSharedStaticCtorDeclaration() inout
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
extern (C++) class StaticDtorDeclaration : FuncDeclaration
{
    VarDeclaration vgate; // 'gate' variable

    extern (D) this(const ref Loc loc, const ref Loc endloc, StorageClass stc)
    {
        super(loc, endloc, Identifier.generateIdWithLoc("_staticDtor", loc), STC.static_ | stc, null);
    }

    extern (D) this(const ref Loc loc, const ref Loc endloc, string name, StorageClass stc)
    {
        super(loc, endloc, Identifier.generateIdWithLoc(name, loc), STC.static_ | stc, null);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto sdd = new StaticDtorDeclaration(loc, endloc, storage_class);
        return FuncDeclaration.syntaxCopy(sdd);
    }

    override final inout(AggregateDeclaration) isThis() inout
    {
        return null;
    }

    override final bool isVirtual() const
    {
        return false;
    }

    override final bool hasStaticCtorOrDtor()
    {
        return true;
    }

    override final bool addPreInvariant()
    {
        return false;
    }

    override final bool addPostInvariant()
    {
        return false;
    }

    override final inout(StaticDtorDeclaration) isStaticDtorDeclaration() inout
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
extern (C++) final class SharedStaticDtorDeclaration : StaticDtorDeclaration
{
    extern (D) this(const ref Loc loc, const ref Loc endloc, StorageClass stc)
    {
        super(loc, endloc, "_sharedStaticDtor", stc);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto sdd = new SharedStaticDtorDeclaration(loc, endloc, storage_class);
        return FuncDeclaration.syntaxCopy(sdd);
    }

    override inout(SharedStaticDtorDeclaration) isSharedStaticDtorDeclaration() inout
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
extern (C++) final class InvariantDeclaration : FuncDeclaration
{
    extern (D) this(const ref Loc loc, const ref Loc endloc, StorageClass stc, Identifier id, Statement fbody)
    {
        super(loc, endloc, id ? id : Identifier.generateId("__invariant"), stc, null);
        this.fbody = fbody;
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto id = new InvariantDeclaration(loc, endloc, storage_class, null, null);
        return FuncDeclaration.syntaxCopy(id);
    }

    override bool isVirtual() const
    {
        return false;
    }

    override bool addPreInvariant()
    {
        return false;
    }

    override bool addPostInvariant()
    {
        return false;
    }

    override inout(InvariantDeclaration) isInvariantDeclaration() inout
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
extern (C++) final class UnitTestDeclaration : FuncDeclaration
{
    char* codedoc;      // for documented unittest

    // toObjFile() these nested functions after this one
    FuncDeclarations deferredNested;

    extern (D) this(const ref Loc loc, const ref Loc endloc, StorageClass stc, char* codedoc)
    {
        super(loc, endloc, Identifier.generateIdWithLoc("__unittest", loc), stc, null);
        this.codedoc = codedoc;
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto utd = new UnitTestDeclaration(loc, endloc, storage_class, codedoc);
        return FuncDeclaration.syntaxCopy(utd);
    }

    override inout(AggregateDeclaration) isThis() inout
    {
        return null;
    }

    override bool isVirtual() const
    {
        return false;
    }

    override bool addPreInvariant()
    {
        return false;
    }

    override bool addPostInvariant()
    {
        return false;
    }

    override inout(UnitTestDeclaration) isUnitTestDeclaration() inout
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
extern (C++) final class NewDeclaration : FuncDeclaration
{
    Parameters* parameters;
    VarArg varargs;

    extern (D) this(const ref Loc loc, const ref Loc endloc, StorageClass stc, Parameters* fparams, VarArg varargs)
    {
        super(loc, endloc, Id.classNew, STC.static_ | stc, null);
        this.parameters = fparams;
        this.varargs = varargs;
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto f = new NewDeclaration(loc, endloc, storage_class, Parameter.arraySyntaxCopy(parameters), varargs);
        return FuncDeclaration.syntaxCopy(f);
    }

    override const(char)* kind() const
    {
        return "allocator";
    }

    override bool isVirtual() const
    {
        return false;
    }

    override bool addPreInvariant()
    {
        return false;
    }

    override bool addPostInvariant()
    {
        return false;
    }

    override inout(NewDeclaration) isNewDeclaration() inout
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
extern (C++) final class DeleteDeclaration : FuncDeclaration
{
    Parameters* parameters;

    extern (D) this(const ref Loc loc, const ref Loc endloc, StorageClass stc, Parameters* fparams)
    {
        super(loc, endloc, Id.classDelete, STC.static_ | stc, null);
        this.parameters = fparams;
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto f = new DeleteDeclaration(loc, endloc, storage_class, Parameter.arraySyntaxCopy(parameters));
        return FuncDeclaration.syntaxCopy(f);
    }

    override const(char)* kind() const
    {
        return "deallocator";
    }

    override bool isDelete()
    {
        return true;
    }

    override bool isVirtual() const
    {
        return false;
    }

    override bool addPreInvariant()
    {
        return false;
    }

    override bool addPostInvariant()
    {
        return false;
    }

    override inout(DeleteDeclaration) isDeleteDeclaration() inout
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

