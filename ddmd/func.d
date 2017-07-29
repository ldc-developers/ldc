/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (c) 1999-2017 by Digital Mars, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(DMDSRC _func.d)
 */

module ddmd.func;

import core.stdc.stdio;
import core.stdc.string;
import ddmd.aggregate;
import ddmd.arraytypes;
import ddmd.blockexit;
import ddmd.gluelayer;
import ddmd.dclass;
import ddmd.declaration;
import ddmd.delegatize;
import ddmd.dinterpret;
import ddmd.dmodule;
import ddmd.dscope;
import ddmd.dstruct;
import ddmd.dsymbol;
import ddmd.dtemplate;
import ddmd.errors;
import ddmd.escape;
import ddmd.expression;
import ddmd.globals;
import ddmd.hdrgen;
import ddmd.id;
import ddmd.identifier;
import ddmd.init;
import ddmd.mars;
import ddmd.mtype;
import ddmd.nogc;
import ddmd.objc;
import ddmd.opover;
import ddmd.root.filename;
import ddmd.root.outbuffer;
import ddmd.root.rootobject;
import ddmd.statement_rewrite_walker;
import ddmd.statementsem;
import ddmd.statement;
import ddmd.target;
import ddmd.tokens;
import ddmd.visitor;

/// Inline Status
enum ILS : int
{
    ILSuninitialized,       /// not computed yet
    ILSno,                  /// cannot inline
    ILSyes,                 /// can inline
}

alias ILSuninitialized = ILS.ILSuninitialized;
alias ILSno = ILS.ILSno;
alias ILSyes = ILS.ILSyes;

enum BUILTIN : int
{
    BUILTINunknown = -1,    /// not known if this is a builtin
    BUILTINno,              /// this is not a builtin
    BUILTINyes,             /// this is a builtin
}

alias BUILTINunknown = BUILTIN.BUILTINunknown;
alias BUILTINno = BUILTIN.BUILTINno;
alias BUILTINyes = BUILTIN.BUILTINyes;


/* Tweak all return statements and dtor call for nrvo_var, for correct NRVO.
 */
extern (C++) final class NrvoWalker : StatementRewriteWalker
{
    alias visit = super.visit;
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
        if (fd.nrvo_can && s.finalbody && (des = s.finalbody.isDtorExpStatement()) !is null && fd.nrvo_var == des.var)
        {
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
            Statement sexception = new DtorExpStatement(Loc(), fd.nrvo_var.edtor, fd.nrvo_var);
            Identifier id = Identifier.generateId("__o");

            Statement handler = new PeelStatement(sexception);
            if (sexception.blockExit(fd, false) & BEfallthru)
            {
                auto ts = new ThrowStatement(Loc(), new IdentifierExp(Loc(), id));
                ts.internalThrow = true;
                handler = new CompoundStatement(Loc(), handler, ts);
            }

            auto catches = new Catches();
            auto ctch = new Catch(Loc(), getThrowable(), id, handler);
            ctch.internalCatch = true;
            ctch.semantic(sc); // Run semantic to resolve identifier '__o'
            catches.push(ctch);

            Statement s2 = new TryCatchStatement(Loc(), s._body, catches);
            replaceCurrent(s2);
            s2.accept(this);
        }
        else
            StatementRewriteWalker.visit(s);
    }
}

enum FUNCFLAGpurityInprocess  = 1;      /// working on determining purity
enum FUNCFLAGsafetyInprocess  = 2;      /// working on determining safety
enum FUNCFLAGnothrowInprocess = 4;      /// working on determining nothrow
enum FUNCFLAGnogcInprocess    = 8;      /// working on determining @nogc
enum FUNCFLAGreturnInprocess  = 0x10;   /// working on inferring 'return' for parameters
enum FUNCFLAGinlineScanned    = 0x20;   /// function has been scanned for inline possibilities
enum FUNCFLAGinferScope       = 0x40;   /// infer 'scope' for parameters


/***********************************************************
 */
extern (C++) class FuncDeclaration : Declaration
{
    Types* fthrows;                     /// Array of Type's of exceptions (not used)
    Statement frequire;                 /// in contract body
    Statement fensure;                  /// out contract body
    Statement fbody;                    /// function body

    FuncDeclarations foverrides;        /// functions this function overrides
    FuncDeclaration fdrequire;          /// function that does the in contract
    FuncDeclaration fdensure;           /// function that does the out contract

    const(char)* mangleString;          /// mangled symbol created from mangleExact()

    version(IN_LLVM)
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

    Identifier outId;                   /// identifier for out statement
    VarDeclaration vresult;             /// variable corresponding to outId
    LabelDsymbol returnLabel;           /// where the return goes

    // used to prevent symbols in different
    // scopes from having the same name
    DsymbolTable localsymtab;
    VarDeclaration vthis;               /// 'this' parameter (member and nested)
    VarDeclaration v_arguments;         /// '_arguments' parameter
    Objc_FuncDeclaration objc;

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
    ILS inlineStatusStmt = ILSuninitialized;
    ILS inlineStatusExp = ILSuninitialized;
    PINLINE inlining = PINLINEdefault;

    CompiledCtfeFunction* ctfeCode;     /// Compiled code for interpreter (not actually)
    int inlineNest;                     /// !=0 if nested inline
    bool isArrayOp;                     /// true if array operation

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
    int hasReturnExp;

    // Support for NRVO (named return value optimization)
    bool nrvo_can = true;               /// true means we can do NRVO
    VarDeclaration nrvo_var;            /// variable to replace with shidden
    Symbol* shidden;                    /// hidden pointer passed to function

    ReturnStatements* returns;

    GotoStatements* gotos;              /// Gotos with forward references

    /// set if this is a known, builtin function we can evaluate at compile time
    BUILTIN builtin = BUILTINunknown;

    /// set if someone took the address of this function
    int tookAddressOf;

    bool requiresClosure;               // this function needs a closure

    /// local variables in this function which are referenced by nested functions
    VarDeclarations closureVars;
    /// Sibling nested functions which called this one
    FuncDeclarations siblingCallers;

    FuncDeclarations *inlinedNestedCallees;

    uint flags;                         /// FUNCFLAGxxxxx

    final extern (D) this(Loc loc, Loc endloc, Identifier id, StorageClass storage_class, Type type)
    {
        super(id);
        objc = Objc_FuncDeclaration(this);
        //printf("FuncDeclaration(id = '%s', type = %p)\n", id.toChars(), type);
        //printf("storage_class = x%x\n", storage_class);
        this.storage_class = storage_class;
        this.type = type;
        if (type)
        {
            // Normalize storage_class, because function-type related attributes
            // are already set in the 'type' in parsing phase.
            this.storage_class &= ~(STC_TYPECTOR | STC_FUNCATTR);
        }
        this.loc = loc;
        this.endloc = endloc;
        /* The type given for "infer the return type" is a TypeFunction with
         * NULL for the return type.
         */
        inferRetType = (type && type.nextOf() is null);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        //printf("FuncDeclaration::syntaxCopy('%s')\n", toChars());
        FuncDeclaration f = s ? cast(FuncDeclaration)s : new FuncDeclaration(loc, endloc, ident, storage_class, type.syntaxCopy());
        f.outId = outId;
        f.frequire = frequire ? frequire.syntaxCopy() : null;
        f.fensure = fensure ? fensure.syntaxCopy() : null;
        f.fbody = fbody ? fbody.syntaxCopy() : null;
        assert(!fthrows); // deprecated
        version(IN_LLVM)
        {
            f.intrinsicName = intrinsicName ? strdup(intrinsicName) : null;
        }
        return f;
    }

    version(IN_LLVM)
    {
        final private Parameters* outToRef(Parameters* params)
        {
            auto result = new Parameters();

            int outToRefDg(size_t n, Parameter p)
            {
                if (p.storageClass & STCout)
                {
                    p = p.syntaxCopy();
                    p.storageClass &= ~STCout;
                    p.storageClass |= STCref;
                }
                result.push(p);
                return 0;
            }

            Parameter._foreach(params, &outToRefDg);
            return result;
        }
    }

    /// Do the semantic analysis on the external interface to the function.
    override void semantic(Scope* sc)
    {
        TypeFunction f;
        AggregateDeclaration ad;
        InterfaceDeclaration id;

        version (none)
        {
            printf("FuncDeclaration::semantic(sc = %p, this = %p, '%s', linkage = %d)\n", sc, this, toPrettyChars(), sc.linkage);
            if (isFuncLiteralDeclaration())
                printf("\tFuncLiteralDeclaration()\n");
            printf("sc.parent = %s, parent = %s\n", sc.parent.toChars(), parent ? parent.toChars() : "");
            printf("type: %p, %s\n", type, type.toChars());
        }

        if (semanticRun != PASSinit && isFuncLiteralDeclaration())
        {
            /* Member functions that have return types that are
             * forward references can have semantic() run more than
             * once on them.
             * See test\interface2.d, test20
             */
            return;
        }

        if (semanticRun >= PASSsemanticdone)
            return;
        assert(semanticRun <= PASSsemantic);
        semanticRun = PASSsemantic;

        if (_scope)
        {
            sc = _scope;
            _scope = null;
        }

        parent = sc.parent;
        Dsymbol parent = toParent();

        foverrides.setDim(0); // reset in case semantic() is being retried for this function

        storage_class |= sc.stc & ~STCref;
        ad = isThis();
        // Don't nest structs b/c of generated methods which should not access the outer scopes.
        // https://issues.dlang.org/show_bug.cgi?id=16627
        if (ad && !generated)
        {
            storage_class |= ad.storage_class & (STC_TYPECTOR | STCsynchronized);
            ad.makeNested();
        }
        if (sc.func)
            storage_class |= sc.func.storage_class & STCdisable;
        // Remove prefix storage classes silently.
        if ((storage_class & STC_TYPECTOR) && !(ad || isNested()))
            storage_class &= ~STC_TYPECTOR;

        //printf("function storage_class = x%llx, sc.stc = x%llx, %x\n", storage_class, sc.stc, Declaration::isFinal());

        FuncLiteralDeclaration fld = isFuncLiteralDeclaration();
        if (fld && fld.treq)
        {
            Type treq = fld.treq;
            assert(treq.nextOf().ty == Tfunction);
            if (treq.ty == Tdelegate)
                fld.tok = TOKdelegate;
            else if (treq.ty == Tpointer && treq.nextOf().ty == Tfunction)
                fld.tok = TOKfunction;
            else
                assert(0);
            linkage = (cast(TypeFunction)treq.nextOf()).linkage;
        }
        else
            linkage = sc.linkage;
        inlining = sc.inlining;
        protection = sc.protection;
        userAttribDecl = sc.userAttribDecl;
        version(IN_LLVM)
        {
            emitInstrumentation = sc.emitInstrumentation;
        }

        if (!originalType)
            originalType = type.syntaxCopy();
        if (!type.deco)
        {
            sc = sc.push();
            sc.stc |= storage_class & (STCdisable | STCdeprecated); // forward to function type

            TypeFunction tf = cast(TypeFunction)type;
            if (sc.func)
            {
                /* If the nesting parent is pure without inference,
                 * then this function defaults to pure too.
                 *
                 *  auto foo() pure {
                 *    auto bar() {}     // become a weak purity funciton
                 *    class C {         // nested class
                 *      auto baz() {}   // become a weak purity funciton
                 *    }
                 *
                 *    static auto boo() {}   // typed as impure
                 *    // Even though, boo cannot call any impure functions.
                 *    // See also Expression::checkPurity().
                 *  }
                 */
                if (tf.purity == PUREimpure && (isNested() || isThis()))
                {
                    FuncDeclaration fd = null;
                    for (Dsymbol p = toParent2(); p; p = p.toParent2())
                    {
                        if (AggregateDeclaration adx = p.isAggregateDeclaration())
                        {
                            if (adx.isNested())
                                continue;
                            break;
                        }
                        if ((fd = p.isFuncDeclaration()) !is null)
                            break;
                    }

                    /* If the parent's purity is inferred, then this function's purity needs
                     * to be inferred first.
                     */
                    if (fd && fd.isPureBypassingInference() >= PUREweak && !isInstantiated())
                    {
                        tf.purity = PUREfwdref; // default to pure
                    }
                }
            }

            if (tf.isref)
                sc.stc |= STCref;
            if (tf.isscope)
                sc.stc |= STCscope;
            if (tf.isnothrow)
                sc.stc |= STCnothrow;
            if (tf.isnogc)
                sc.stc |= STCnogc;
            if (tf.isproperty)
                sc.stc |= STCproperty;
            if (tf.purity == PUREfwdref)
                sc.stc |= STCpure;
            if (tf.trust != TRUSTdefault)
                sc.stc &= ~(STCsafe | STCsystem | STCtrusted);
            if (tf.trust == TRUSTsafe)
                sc.stc |= STCsafe;
            if (tf.trust == TRUSTsystem)
                sc.stc |= STCsystem;
            if (tf.trust == TRUSTtrusted)
                sc.stc |= STCtrusted;

            if (isCtorDeclaration())
            {
                sc.flags |= SCOPEctor;
                Type tret = ad.handleType();
                assert(tret);
                tret = tret.addStorageClass(storage_class | sc.stc);
                tret = tret.addMod(type.mod);
                tf.next = tret;
                if (ad.isStructDeclaration())
                    sc.stc |= STCref;
            }

            // 'return' on a non-static class member function implies 'scope' as well
            if (ad && ad.isClassDeclaration() && (tf.isreturn || sc.stc & STCreturn) && !(sc.stc & STCstatic))
                sc.stc |= STCscope;

            // If 'this' has no pointers, remove 'scope' as it has no meaning
            if (sc.stc & STCscope && ad && ad.isStructDeclaration() && !ad.type.hasPointers())
            {
                sc.stc &= ~STCscope;
                tf.isscope = false;
            }

            sc.linkage = linkage;

            if (!tf.isNaked() && !(isThis() || isNested()))
            {
                OutBuffer buf;
                MODtoBuffer(&buf, tf.mod);
                error("without 'this' cannot be %s", buf.peekString());
                tf.mod = 0; // remove qualifiers
            }

            /* Apply const, immutable, wild and shared storage class
             * to the function type. Do this before type semantic.
             */
            auto stc = storage_class;
            if (type.isImmutable())
                stc |= STCimmutable;
            if (type.isConst())
                stc |= STCconst;
            if (type.isShared() || storage_class & STCsynchronized)
                stc |= STCshared;
            if (type.isWild())
                stc |= STCwild;
            type = type.addSTC(stc);

            type = type.semantic(loc, sc);
            sc = sc.pop();
        }
        if (type.ty != Tfunction)
        {
            if (type.ty != Terror)
            {
                error("%s must be a function instead of %s", toChars(), type.toChars());
                type = Type.terror;
            }
            errors = true;
            return;
        }
        else
        {
            // Merge back function attributes into 'originalType'.
            // It's used for mangling, ddoc, and json output.
            TypeFunction tfo = cast(TypeFunction)originalType;
            TypeFunction tfx = cast(TypeFunction)type;
            tfo.mod = tfx.mod;
            tfo.isscope = tfx.isscope;
            tfo.isref = tfx.isref;
            tfo.isnothrow = tfx.isnothrow;
            tfo.isnogc = tfx.isnogc;
            tfo.isproperty = tfx.isproperty;
            tfo.purity = tfx.purity;
            tfo.trust = tfx.trust;

            storage_class &= ~(STC_TYPECTOR | STC_FUNCATTR);
        }

        f = cast(TypeFunction)type;
        size_t nparams = Parameter.dim(f.parameters);

        if ((storage_class & STCauto) && !f.isref && !inferRetType)
            error("storage class 'auto' has no effect if return type is not inferred");

        /* Functions can only be 'scope' if they have a 'this'
         */
        if (f.isscope && !isNested() && !ad)
        {
            error("functions cannot be scope");
        }

        if (f.isreturn && !needThis() && !isNested())
        {
            /* Non-static nested functions have a hidden 'this' pointer to which
             * the 'return' applies
             */
            error("static member has no 'this' to which 'return' can apply");
        }

        if (isAbstract() && !isVirtual())
        {
            const(char)* sfunc;
            if (isStatic())
                sfunc = "static";
            else if (protection.kind == PROTprivate || protection.kind == PROTpackage)
                sfunc = protectionToChars(protection.kind);
            else
                sfunc = "non-virtual";
            error("%s functions cannot be abstract", sfunc);
        }

        if (isOverride() && !isVirtual())
        {
            PROTKIND kind = prot().kind;
            if ((kind == PROTprivate || kind == PROTpackage) && isMember())
                error("%s method is not virtual and cannot override", protectionToChars(kind));
            else
                error("cannot override a non-virtual function");
        }

        if (isAbstract() && isFinalFunc())
            error("cannot be both final and abstract");
        version (none)
        {
            if (isAbstract() && fbody)
                error("abstract functions cannot have bodies");
        }

        version (none)
        {
            if (isStaticConstructor() || isStaticDestructor())
            {
                if (!isStatic() || type.nextOf().ty != Tvoid)
                    error("static constructors / destructors must be static void");
                if (f.arguments && f.arguments.dim)
                    error("static constructors / destructors must have empty parameter list");
                // BUG: check for invalid storage classes
            }
        }

        id = parent.isInterfaceDeclaration();
        if (id)
        {
            storage_class |= STCabstract;
            if (isCtorDeclaration() || isPostBlitDeclaration() || isDtorDeclaration() || isInvariantDeclaration() || isNewDeclaration() || isDelete())
                error("constructors, destructors, postblits, invariants, new and delete functions are not allowed in interface %s", id.toChars());
            if (fbody && isVirtual())
                error("function body only allowed in final functions in interface %s", id.toChars());
        }
        if (UnionDeclaration ud = parent.isUnionDeclaration())
        {
            if (isPostBlitDeclaration() || isDtorDeclaration() || isInvariantDeclaration())
                error("destructors, postblits and invariants are not allowed in union %s", ud.toChars());
        }

        /* Contracts can only appear without a body when they are virtual interface functions
         */
        if (!fbody && (fensure || frequire) && !(id && isVirtual()))
            error("in and out contracts require function body");

        if (StructDeclaration sd = parent.isStructDeclaration())
        {
            if (isCtorDeclaration())
            {
                goto Ldone;
            }
        }

        if (ClassDeclaration cd = parent.isClassDeclaration())
        {
            if (isCtorDeclaration())
            {
                goto Ldone;
            }

            if (storage_class & STCabstract)
                cd.isabstract = ABSyes;

            // if static function, do not put in vtbl[]
            if (!isVirtual())
            {
                //printf("\tnot virtual\n");
                goto Ldone;
            }
            // Suppress further errors if the return type is an error
            if (type.nextOf() == Type.terror)
                goto Ldone;

            bool may_override = false;
            for (size_t i = 0; i < cd.baseclasses.dim; i++)
            {
                BaseClass* b = (*cd.baseclasses)[i];
                ClassDeclaration cbd = b.type.toBasetype().isClassHandle();
                if (!cbd)
                    continue;
                for (size_t j = 0; j < cbd.vtbl.dim; j++)
                {
                    FuncDeclaration f2 = cbd.vtbl[j].isFuncDeclaration();
                    if (!f2 || f2.ident != ident)
                        continue;
                    if (cbd.parent && cbd.parent.isTemplateInstance())
                    {
                        if (!f2.functionSemantic())
                            goto Ldone;
                    }
                    may_override = true;
                }
            }
            if (may_override && type.nextOf() is null)
            {
                /* If same name function exists in base class but 'this' is auto return,
                 * cannot find index of base class's vtbl[] to override.
                 */
                error("return type inference is not supported if may override base class function");
            }

            /* Find index of existing function in base class's vtbl[] to override
             * (the index will be the same as in cd's current vtbl[])
             */
            int vi = cd.baseClass ? findVtblIndex(&cd.baseClass.vtbl, cast(int)cd.baseClass.vtbl.dim) : -1;

            bool doesoverride = false;
            switch (vi)
            {
            case -1:
            Lintro:
                /* Didn't find one, so
                 * This is an 'introducing' function which gets a new
                 * slot in the vtbl[].
                 */

                // Verify this doesn't override previous final function
                if (cd.baseClass)
                {
                    Dsymbol s = cd.baseClass.search(loc, ident);
                    if (s)
                    {
                        FuncDeclaration f2 = s.isFuncDeclaration();
                        if (f2)
                        {
                            f2 = f2.overloadExactMatch(type);
                            if (f2 && f2.isFinalFunc() && f2.prot().kind != PROTprivate)
                                error("cannot override final function %s", f2.toPrettyChars());
                        }
                    }
                }

                /* These quirky conditions mimic what VC++ appears to do
                 */
                if (global.params.mscoff && cd.cpp &&
                    cd.baseClass && cd.baseClass.vtbl.dim)
                {
                    /* if overriding an interface function, then this is not
                     * introducing and don't put it in the class vtbl[]
                     */
                    interfaceVirtual = overrideInterface();
                    if (interfaceVirtual)
                    {
                        //printf("\tinterface function %s\n", toChars());
                        cd.vtblFinal.push(this);
                        goto Linterfaces;
                    }
                }

                if (isFinalFunc())
                {
                    // Don't check here, as it may override an interface function
                    //if (isOverride())
                    //    error("is marked as override, but does not override any function");
                    cd.vtblFinal.push(this);
                }
                else
                {
                    //printf("\tintroducing function %s\n", toChars());
                    introducing = 1;
                    if (cd.cpp && Target.reverseCppOverloads)
                    {
                        // with dmc, overloaded functions are grouped and in reverse order
                        vtblIndex = cast(int)cd.vtbl.dim;
                        for (size_t i = 0; i < cd.vtbl.dim; i++)
                        {
                            if (cd.vtbl[i].ident == ident && cd.vtbl[i].parent == parent)
                            {
                                vtblIndex = cast(int)i;
                                break;
                            }
                        }
                        // shift all existing functions back
                        for (size_t i = cd.vtbl.dim; i > vtblIndex; i--)
                        {
                            FuncDeclaration fd = cd.vtbl[i - 1].isFuncDeclaration();
                            assert(fd);
                            fd.vtblIndex++;
                        }
                        cd.vtbl.insert(vtblIndex, this);
                    }
                    else
                    {
                        // Append to end of vtbl[]
                        vi = cast(int)cd.vtbl.dim;
                        cd.vtbl.push(this);
                        vtblIndex = vi;
                    }
                }
                break;

            case -2:
                // can't determine because of forward references
                return;

            default:
                {
                    FuncDeclaration fdv = cd.baseClass.vtbl[vi].isFuncDeclaration();
                    FuncDeclaration fdc = cd.vtbl[vi].isFuncDeclaration();
                    // This function is covariant with fdv

                    if (fdc == this)
                    {
                        doesoverride = true;
                        break;
                    }

                    if (fdc.toParent() == parent)
                    {
                        //printf("vi = %d,\tthis = %p %s %s @ [%s]\n\tfdc  = %p %s %s @ [%s]\n\tfdv  = %p %s %s @ [%s]\n",
                        //        vi, this, this.toChars(), this.type.toChars(), this.loc.toChars(),
                        //            fdc,  fdc .toChars(), fdc .type.toChars(), fdc .loc.toChars(),
                        //            fdv,  fdv .toChars(), fdv .type.toChars(), fdv .loc.toChars());

                        // fdc overrides fdv exactly, then this introduces new function.
                        if (fdc.type.mod == fdv.type.mod && this.type.mod != fdv.type.mod)
                            goto Lintro;
                    }

                    // This function overrides fdv
                    if (fdv.isFinalFunc())
                        error("cannot override final function %s", fdv.toPrettyChars());

                    doesoverride = true;
                    if (!isOverride())
                        .error(loc, "cannot implicitly override base class method %s with %s; add 'override' attribute", fdv.toPrettyChars(), toPrettyChars());
                    if (fdc.toParent() == parent)
                    {
                        // If both are mixins, or both are not, then error.
                        // If either is not, the one that is not overrides the other.
                        bool thismixin = this.parent.isClassDeclaration() !is null;
                        bool fdcmixin = fdc.parent.isClassDeclaration() !is null;
                        if (thismixin == fdcmixin)
                        {
                            error("multiple overrides of same function");
                        }
                        else if (!thismixin) // fdc overrides fdv
                        {
                            // this doesn't override any function
                            break;
                        }
                    }
                    cd.vtbl[vi] = this;
                    vtblIndex = vi;

                    /* Remember which functions this overrides
                     */
                    foverrides.push(fdv);

                    /* This works by whenever this function is called,
                     * it actually returns tintro, which gets dynamically
                     * cast to type. But we know that tintro is a base
                     * of type, so we could optimize it by not doing a
                     * dynamic cast, but just subtracting the isBaseOf()
                     * offset if the value is != null.
                     */

                    if (fdv.tintro)
                        tintro = fdv.tintro;
                    else if (!type.equals(fdv.type))
                    {
                        /* Only need to have a tintro if the vptr
                         * offsets differ
                         */
                        int offset;
                        if (fdv.type.nextOf().isBaseOf(type.nextOf(), &offset))
                        {
                            tintro = fdv.type;
                        }
                    }
                    break;
                }
            }

            /* Go through all the interface bases.
             * If this function is covariant with any members of those interface
             * functions, set the tintro.
             */
        Linterfaces:
            foreach (b; cd.interfaces)
            {
                vi = findVtblIndex(&b.sym.vtbl, cast(int)b.sym.vtbl.dim);
                switch (vi)
                {
                case -1:
                    break;

                case -2:
                    // can't determine because of forward references
                    return;

                default:
                    {
                        auto fdv = cast(FuncDeclaration)b.sym.vtbl[vi];
                        Type ti = null;

                        /* Remember which functions this overrides
                         */
                        foverrides.push(fdv);

                        /* Should we really require 'override' when implementing
                         * an interface function?
                         */
                        //if (!isOverride())
                        //    warning(loc, "overrides base class function %s, but is not marked with 'override'", fdv.toPrettyChars());

                        if (fdv.tintro)
                            ti = fdv.tintro;
                        else if (!type.equals(fdv.type))
                        {
                            /* Only need to have a tintro if the vptr
                             * offsets differ
                             */
                            int offset;
                            if (fdv.type.nextOf().isBaseOf(type.nextOf(), &offset))
                            {
                                ti = fdv.type;
                            }
                        }
                        if (ti)
                        {
                            if (tintro)
                            {
                                if (!tintro.nextOf().equals(ti.nextOf()) && !tintro.nextOf().isBaseOf(ti.nextOf(), null) && !ti.nextOf().isBaseOf(tintro.nextOf(), null))
                                {
                                    error("incompatible covariant types %s and %s", tintro.toChars(), ti.toChars());
                                }
                            }
                            tintro = ti;
                        }
                        goto L2;
                    }
                }
            }

            if (!doesoverride && isOverride() && (type.nextOf() || !may_override))
            {
                BaseClass* bc = null;
                Dsymbol s = null;
                for (size_t i = 0; i < cd.baseclasses.dim; i++)
                {
                    bc = (*cd.baseclasses)[i];
                    s = bc.sym.search_correct(ident);
                    if (s)
                        break;
                }

                if (s)
                    error("does not override any function, did you mean to override '%s%s'?",
                        bc.sym.isCPPclass() ? "extern (C++) ".ptr : "".ptr, s.toPrettyChars());
                else
                    error("does not override any function");
            }

        L2:
            /* Go through all the interface bases.
             * Disallow overriding any final functions in the interface(s).
             */
            foreach (b; cd.interfaces)
            {
                if (b.sym)
                {
                    Dsymbol s = search_function(b.sym, ident);
                    if (s)
                    {
                        FuncDeclaration f2 = s.isFuncDeclaration();
                        if (f2)
                        {
                            f2 = f2.overloadExactMatch(type);
                            if (f2 && f2.isFinalFunc() && f2.prot().kind != PROTprivate)
                                error("cannot override final function %s.%s", b.sym.toChars(), f2.toPrettyChars());
                        }
                    }
                }
            }

            if (isOverride)
            {
                if (storage_class & STCdisable)
                    deprecation("overridden functions cannot be annotated @disable");
                if (isDeprecated)
                    deprecation("deprecated functions cannot be annotated @disable");
            }

        }
        else if (isOverride() && !parent.isTemplateInstance())
            error("override only applies to class member functions");

        // Reflect this.type to f because it could be changed by findVtblIndex
        assert(type.ty == Tfunction);
        f = cast(TypeFunction)type;

        /* Do not allow template instances to add virtual functions
         * to a class.
         */
        if (isVirtual())
        {
            TemplateInstance ti = parent.isTemplateInstance();
            if (ti)
            {
                // Take care of nested templates
                while (1)
                {
                    TemplateInstance ti2 = ti.tempdecl.parent.isTemplateInstance();
                    if (!ti2)
                        break;
                    ti = ti2;
                }

                // If it's a member template
                ClassDeclaration cd = ti.tempdecl.isClassMember();
                if (cd)
                {
                    error("cannot use template to add virtual function to class '%s'", cd.toChars());
                }
            }
        }

        if (isMain())
            checkDmain();       // Check main() parameters and return type

        if (isVirtual() && semanticRun != PASSsemanticdone)
        {
            /* Rewrite contracts as nested functions, then call them.
             * Doing it as nested functions means that overriding functions
             * can call them.
             */
            if (frequire)
            {
                version(IN_LLVM)
                {
                    /* In LDC, we can't rely on the codegen hacks DMD has to be able
                     * to just magically call the contract function parameterless with
                     * the parameters being picked up from the outer stack frame.
                     *
                     * Thus, we actually pass all the function parameters to the
                     * __require call, rewriting out parameters to ref ones because
                     * they have already been zeroed in the outer function.
                     *
                     * Also initialize fdrequireParams here - it will get filled in
                     * in semantic3.
                     */
                    fdrequireParams = new Expressions();
                    auto params = outToRef((cast(TypeFunction)type).parameters);
                    auto tf = new TypeFunction(params, Type.tvoid, 0, LINKd);
                }
                else
                {
                    /*   in { ... }
                     * becomes:
                     *   void __require() { ... }
                     *   __require();
                     */
                    auto tf = new TypeFunction(null, Type.tvoid, 0, LINKd);
                }
                Loc loc = frequire.loc;
                tf.isnothrow = f.isnothrow;
                tf.isnogc = f.isnogc;
                tf.purity = f.purity;
                tf.trust = f.trust;
                auto fd = new FuncDeclaration(loc, loc, Id.require, STCundefined, tf);
                fd.fbody = frequire;
                Statement s1 = new ExpStatement(loc, fd);
                version(IN_LLVM)
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

            if (!outId && f.nextOf() && f.nextOf().toBasetype().ty != Tvoid)
                outId = Id.result; // provide a default
            version(IN_LLVM)
            {
                /* We need to initialize fdensureParams here and not in the block below
                 * to have the parameter available when calling a base class ensure(),
                 * even if this functions doesn't have an out contract.
                 */
                fdensureParams = new Expressions();
                if (outId)
                    fdensureParams.push(new IdentifierExp(loc, outId));
            }
            if (fensure)
            {
                version(IN_LLVM)
                {
                    /* Same as for in contracts, see above. */
                    auto fparams = outToRef((cast(TypeFunction)type).parameters);
                }
                else
                {
                    /*   out (result) { ... }
                     * becomes:
                     *   void __ensure(ref tret result) { ... }
                     *   __ensure(result);
                     */
                    auto fparams = new Parameters();
                }
                Loc loc = fensure.loc;
                Parameter p = null;
                if (outId)
                {
                    p = new Parameter(STCref | STCconst, f.nextOf(), outId, null);
                    version(IN_LLVM)
                    {
                        fparams.insert(0, p);
                    }
                    else
                    {
                        fparams.push(p);
                    }
                }
                auto tf = new TypeFunction(fparams, Type.tvoid, 0, LINKd);
                tf.isnothrow = f.isnothrow;
                tf.isnogc = f.isnogc;
                tf.purity = f.purity;
                tf.trust = f.trust;
                auto fd = new FuncDeclaration(loc, loc, Id.ensure, STCundefined, tf);
                fd.fbody = fensure;
                Statement s1 = new ExpStatement(loc, fd);
                version(IN_LLVM)
                {
                    Expression e = new CallExp(loc, new VarExp(loc, fd, false), fdensureParams);
                }
                else
                {
                    Expression eresult = null;
                    if (outId)
                        eresult = new IdentifierExp(loc, outId);
                    Expression e = new CallExp(loc, new VarExp(loc, fd, false), eresult);
                }
                Statement s2 = new ExpStatement(loc, e);
                fensure = new CompoundStatement(loc, s1, s2);
                fdensure = fd;
            }
        }

    Ldone:
        /* Purity and safety can be inferred for some functions by examining
         * the function body.
         */
        if (canInferAttributes(sc))
            initInferAttributes();

        Module.dprogress++;
        version(IN_LLVM)
        {
            //LDC relies on semanticRun variable not being reset here
            if(semanticRun < PASSsemanticdone)
                semanticRun = PASSsemanticdone;
        }
        else
        {
            semanticRun = PASSsemanticdone;
        }

        /* Save scope for possible later use (if we need the
         * function internals)
         */
        _scope = sc.copy();
        _scope.setNoFree();

        static __gshared bool printedMain = false; // semantic might run more than once
        if (global.params.verbose && !printedMain)
        {
            const(char)* type = isMain() ? "main" : isWinMain() ? "winmain" : isDllMain() ? "dllmain" : cast(const(char)*)null;
            Module mod = sc._module;

            if (type && mod)
            {
                printedMain = true;
                const(char)* name = FileName.searchPath(global.path, mod.srcfile.toChars(), true);
                fprintf(global.stdmsg, "entry     %-10s\t%s\n", type, name);
            }
        }

        if (fbody && isMain() && sc._module.isRoot())
            genCmain(sc);

        assert(type.ty != Terror || errors);
    }

    override final void semantic2(Scope* sc)
    {
        if (semanticRun >= PASSsemantic2done)
            return;
        assert(semanticRun <= PASSsemantic2);

        semanticRun = PASSsemantic2;

        objc_FuncDeclaration_semantic_setSelector(this, sc);
        objc_FuncDeclaration_semantic_validateSelector(this);
        if (ClassDeclaration cd = parent.isClassDeclaration())
        {
            objc_FuncDeclaration_semantic_checkLinkage(this);
        }
    }

    // Do the semantic analysis on the internals of the function.
    override final void semantic3(Scope* sc)
    {
        VarDeclaration _arguments = null;

        if (!parent)
        {
            if (global.errors)
                return;
            //printf("FuncDeclaration::semantic3(%s '%s', sc = %p)\n", kind(), toChars(), sc);
            assert(0);
        }
        if (errors || isError(parent))
        {
            errors = true;
            return;
        }
        //printf("FuncDeclaration::semantic3('%s.%s', %p, sc = %p, loc = %s)\n", parent.toChars(), toChars(), this, sc, loc.toChars());
        //fflush(stdout);
        //printf("storage class = x%x %x\n", sc.stc, storage_class);
        //{ static int x; if (++x == 2) *(char*)0=0; }
        //printf("\tlinkage = %d\n", sc.linkage);

        if (ident == Id.assign && !inuse)
        {
            if (storage_class & STCinference)
            {
                /* Bugzilla 15044: For generated opAssign function, any errors
                 * from its body need to be gagged.
                 */
                uint oldErrors = global.startGagging();
                ++inuse;
                semantic3(sc);
                --inuse;
                if (global.endGagging(oldErrors))   // if errors happened
                {
                    // Disable generated opAssign, because some members forbid identity assignment.
                    storage_class |= STCdisable;
                    fbody = null;   // remove fbody which contains the error
                    semantic3Errors = false;
                }
                return;
            }
        }

        //printf(" sc.incontract = %d\n", (sc.flags & SCOPEcontract));
        if (semanticRun >= PASSsemantic3)
            return;
        semanticRun = PASSsemantic3;
        semantic3Errors = false;

        if (!type || type.ty != Tfunction)
            return;
        TypeFunction f = cast(TypeFunction)type;
        if (!inferRetType && f.next.ty == Terror)
            return;

        if (!fbody && inferRetType && !f.next)
        {
            error("has no function body with return type inference");
            return;
        }

        uint oldErrors = global.errors;

        if (frequire)
        {
            for (size_t i = 0; i < foverrides.dim; i++)
            {
                FuncDeclaration fdv = foverrides[i];
                if (fdv.fbody && !fdv.frequire)
                {
                    error("cannot have an in contract when overriden function %s does not have an in contract", fdv.toPrettyChars());
                    break;
                }
            }
        }

        uint fensure_endlin = endloc.linnum;
        if (fensure)
            if (auto s = fensure.isScopeStatement())
                fensure_endlin = s.endloc.linnum;

        frequire = mergeFrequire(frequire);
        fensure = mergeFensure(fensure, outId);
        if (fbody || frequire || fensure)
        {
            /* Symbol table into which we place parameters and nested functions,
             * solely to diagnose name collisions.
             */
            localsymtab = new DsymbolTable();

            // Establish function scope
            auto ss = new ScopeDsymbol();
            // find enclosing scope symbol, might skip symbol-less CTFE and/or FuncExp scopes
            for (auto scx = sc; ; scx = scx.enclosing)
            {
                if (scx.scopesym)
                {
                    ss.parent = scx.scopesym;
                    break;
                }
            }
            ss.loc = loc;
            ss.endlinnum = endloc.linnum;
            Scope* sc2 = sc.push(ss);
            sc2.func = this;
            sc2.parent = this;
            sc2.callSuper = 0;
            sc2.sbreak = null;
            sc2.scontinue = null;
            sc2.sw = null;
            sc2.fes = fes;
            sc2.linkage = LINKd;
            sc2.stc &= ~(STCauto | STCscope | STCstatic | STCabstract | STCdeprecated | STCoverride | STC_TYPECTOR | STCfinal | STCtls | STCgshared | STCref | STCreturn | STCproperty | STCnothrow | STCpure | STCsafe | STCtrusted | STCsystem);
            sc2.protection = Prot(PROTpublic);
            sc2.explicitProtection = 0;
            sc2.aligndecl = null;
            if (this.ident != Id.require && this.ident != Id.ensure)
                sc2.flags = sc.flags & ~SCOPEcontract;
            sc2.flags &= ~SCOPEcompile;
            sc2.tf = null;
            sc2.os = null;
            sc2.noctor = 0;
            sc2.userAttribDecl = null;
            if (sc2.intypeof == 1)
                sc2.intypeof = 2;
            sc2.fieldinit = null;
            sc2.fieldinit_dim = 0;

            /* Note: When a lambda is defined immediately under aggregate member
             * scope, it should be contextless due to prevent interior pointers.
             * e.g.
             *      // dg points 'this' - it's interior pointer
             *      class C { int x; void delegate() dg = (){ this.x = 1; }; }
             *
             * However, lambdas could be used inside typeof, in order to check
             * some expressions varidity at compile time. For such case the lambda
             * body can access aggregate instance members.
             * e.g.
             *      class C { int x; static assert(is(typeof({ this.x = 1; }))); }
             *
             * To properly accept it, mark these lambdas as member functions -
             * isThis() returns true and isNested() returns false.
             */
            if (auto fld = isFuncLiteralDeclaration())
            {
                if (auto ad = isMember2())
                {
                    if (!sc.intypeof)
                    {
                        if (fld.tok == TOKdelegate)
                            error("cannot be %s members", ad.kind());
                        else
                            fld.tok = TOKfunction;
                    }
                    else
                    {
                        if (fld.tok != TOKfunction)
                            fld.tok = TOKdelegate;
                    }
                    assert(!isNested());
                }
            }

            // Declare 'this'
            auto ad = isThis();
            vthis = declareThis(sc2, ad);
            //printf("[%s] ad = %p vthis = %p\n", loc.toChars(), ad, vthis);
            //if (vthis) printf("\tvthis.type = %s\n", vthis.type.toChars());

            // Declare hidden variable _arguments[] and _argptr
            if (f.varargs == 1)
            {
                if (f.linkage == LINKd)
                {
                    // Declare _arguments[]
                    v_arguments = new VarDeclaration(Loc(), Type.typeinfotypelist.type, Id._arguments_typeinfo, null);
                    v_arguments.storage_class |= STCtemp | STCparameter;
                    v_arguments.semantic(sc2);
                    sc2.insert(v_arguments);
                    v_arguments.parent = this;

                    //Type *t = Type::typeinfo.type.constOf().arrayOf();
                    Type t = Type.dtypeinfo.type.arrayOf();
                    _arguments = new VarDeclaration(Loc(), t, Id._arguments, null);
                    _arguments.storage_class |= STCtemp;
                    _arguments.semantic(sc2);
                    sc2.insert(_arguments);
                    _arguments.parent = this;
                }
                if (f.linkage == LINKd || (f.parameters && Parameter.dim(f.parameters)))
                {
                    // Declare _argptr
                    version (IN_LLVM)
                        Type t = Type.tvalist.semantic(loc, sc);
                    else
                        Type t = Type.tvalist;
                    // Init is handled in FuncDeclaration_toObjFile
                    v_argptr = new VarDeclaration(Loc(), t, Id._argptr, new VoidInitializer(loc));
                    v_argptr.storage_class |= STCtemp;
                    v_argptr.semantic(sc2);
                    sc2.insert(v_argptr);
                    v_argptr.parent = this;
                }
            }

            version(IN_LLVM)
            {
                // Make sure semantic analysis has been run on argument types. This is
                // e.g. needed for TypeTuple!(int, int) to be picked up as two int
                // parameters by the Parameter functions.
                if (f.parameters)
                {
                    for (size_t i = 0; i < Parameter.dim(f.parameters); i++)
                    {
                        Parameter arg = Parameter.getNth(f.parameters, i);
                        Type nw = arg.type.semantic(Loc(), sc);
                        if (arg.type != nw)
                        {
                            arg.type = nw;
                            // Examine this index again.
                            // This is important if it turned into a tuple.
                            // In particular, the empty tuple should be handled or the
                            // next parameter will be skipped.
                            // LDC_FIXME: Maybe we only need to do this for tuples,
                            //            and can add tuple.length after decrement?
                            i--;
                        }
                    }
                }
            }

            /* Declare all the function parameters as variables
             * and install them in parameters[]
             */
            size_t nparams = Parameter.dim(f.parameters);
            if (nparams)
            {
                /* parameters[] has all the tuples removed, as the back end
                 * doesn't know about tuples
                 */
                parameters = new VarDeclarations();
                parameters.reserve(nparams);
                for (size_t i = 0; i < nparams; i++)
                {
                    Parameter fparam = Parameter.getNth(f.parameters, i);
                    Identifier id = fparam.ident;
                    StorageClass stc = 0;
                    if (!id)
                    {
                        /* Generate identifier for un-named parameter,
                         * because we need it later on.
                         */
                        fparam.ident = id = Identifier.generateId("_param_", i);
                        stc |= STCtemp;
                    }
                    Type vtype = fparam.type;
                    auto v = new VarDeclaration(loc, vtype, id, null);
                    //printf("declaring parameter %s of type %s\n", v.toChars(), v.type.toChars());
                    stc |= STCparameter;
                    if (f.varargs == 2 && i + 1 == nparams)
                        stc |= STCvariadic;
                    if (flags & FUNCFLAGinferScope)
                        stc |= STCmaybescope;
                    stc |= fparam.storageClass & (STCin | STCout | STCref | STCreturn | STCscope | STClazy | STCfinal | STC_TYPECTOR | STCnodtor);
                    v.storage_class = stc;
                    v.semantic(sc2);
                    if (!sc2.insert(v))
                        error("parameter %s.%s is already defined", toChars(), v.toChars());
                    else
                        parameters.push(v);
                    localsymtab.insert(v);
                    v.parent = this;
                    version(IN_LLVM)
                    {
                        if (fdrequireParams)
                            fdrequireParams.push(new VarExp(loc, v));
                        if (fdensureParams)
                            fdensureParams.push(new VarExp(loc, v));
                    }
                }
            }

            // Declare the tuple symbols and put them in the symbol table,
            // but not in parameters[].
            if (f.parameters)
            {
                for (size_t i = 0; i < f.parameters.dim; i++)
                {
                    Parameter fparam = (*f.parameters)[i];
                    if (!fparam.ident)
                        continue; // never used, so ignore
                    if (fparam.type.ty == Ttuple)
                    {
                        TypeTuple t = cast(TypeTuple)fparam.type;
                        size_t dim = Parameter.dim(t.arguments);
                        auto exps = new Objects();
                        exps.setDim(dim);
                        for (size_t j = 0; j < dim; j++)
                        {
                            Parameter narg = Parameter.getNth(t.arguments, j);
                            assert(narg.ident);
                            VarDeclaration v = sc2.search(Loc(), narg.ident, null).isVarDeclaration();
                            assert(v);
                            Expression e = new VarExp(v.loc, v);
                            (*exps)[j] = e;
                        }
                        assert(fparam.ident);
                        auto v = new TupleDeclaration(loc, fparam.ident, exps);
                        //printf("declaring tuple %s\n", v.toChars());
                        v.isexp = true;
                        if (!sc2.insert(v))
                            error("parameter %s.%s is already defined", toChars(), v.toChars());
                        localsymtab.insert(v);
                        v.parent = this;
                    }
                }
            }

            // Precondition invariant
            Statement fpreinv = null;
            if (addPreInvariant())
            {
                Expression e = addInvariant(loc, sc, ad, vthis);
                if (e)
                    fpreinv = new ExpStatement(Loc(), e);
            }

            // Postcondition invariant
            Statement fpostinv = null;
            if (addPostInvariant())
            {
                Expression e = addInvariant(loc, sc, ad, vthis);
                if (e)
                    fpostinv = new ExpStatement(Loc(), e);
            }

            Scope* scout = null;
            if (fensure || addPostInvariant())
            {
                if ((fensure && global.params.useOut) || fpostinv)
                {
                    returnLabel = new LabelDsymbol(Id.returnLabel);
                }

                // scope of out contract (need for vresult.semantic)
                auto sym = new ScopeDsymbol();
                sym.parent = sc2.scopesym;
                sym.loc = loc;
                sym.endlinnum = fensure_endlin;
                scout = sc2.push(sym);
            }

            if (fbody)
            {
                auto sym = new ScopeDsymbol();
                sym.parent = sc2.scopesym;
                sym.loc = loc;
                sym.endlinnum = endloc.linnum;
                sc2 = sc2.push(sym);

                auto ad2 = isMember2();

                /* If this is a class constructor
                 */
                if (ad2 && isCtorDeclaration())
                {
                    sc2.allocFieldinit(ad2.fields.dim);
                    foreach (v; ad2.fields)
                    {
                        v.ctorinit = 0;
                    }
                }

                if (!inferRetType && retStyle(f) != RETstack)
                    nrvo_can = 0;

                bool inferRef = (f.isref && (storage_class & STCauto));

                fbody = fbody.semantic(sc2);
                if (!fbody)
                    fbody = new CompoundStatement(Loc(), new Statements());

                if (naked)
                {
                    fpreinv = null;         // can't accommodate with no stack frame
                    fpostinv = null;
                }

                assert(type == f || (type.ty == Tfunction && f.purity == PUREimpure && (cast(TypeFunction)type).purity >= PUREfwdref));
                f = cast(TypeFunction)type;

                if (inferRetType)
                {
                    // If no return type inferred yet, then infer a void
                    if (!f.next)
                        f.next = Type.tvoid;
                    if (f.checkRetType(loc))
                        fbody = new ErrorStatement();
                }
                if (global.params.vcomplex && f.next !is null)
                    f.next.checkComplexTransition(loc);

                if (returns && !fbody.isErrorStatement())
                {
                    for (size_t i = 0; i < returns.dim;)
                    {
                        Expression exp = (*returns)[i].exp;
                        if (exp.op == TOKvar && (cast(VarExp)exp).var == vresult)
                        {
                            if (f.next.ty == Tvoid && isMain())
                                exp.type = Type.tint32;
                            else
                                exp.type = f.next;
                            // Remove `return vresult;` from returns
                            returns.remove(i);
                            continue;
                        }
                        if (inferRef && f.isref && !exp.type.constConv(f.next)) // Bugzilla 13336
                            f.isref = false;
                        i++;
                    }
                }
                if (f.isref) // Function returns a reference
                {
                    if (storage_class & STCauto)
                        storage_class &= ~STCauto;
                }
                if (retStyle(f) != RETstack)
                    nrvo_can = 0;

                if (fbody.isErrorStatement())
                {
                }
                else if (isStaticCtorDeclaration())
                {
                    /* It's a static constructor. Ensure that all
                     * ctor consts were initialized.
                     */
                    ScopeDsymbol pd = toParent().isScopeDsymbol();
                    for (size_t i = 0; i < pd.members.dim; i++)
                    {
                        Dsymbol s = (*pd.members)[i];
                        s.checkCtorConstInit();
                    }
                }
                else if (ad2 && isCtorDeclaration())
                {
                    ClassDeclaration cd = ad2.isClassDeclaration();

                    // Verify that all the ctorinit fields got initialized
                    if (!(sc2.callSuper & CSXthis_ctor))
                    {
                        foreach (i, v; ad2.fields)
                        {
                            if (v.isThisDeclaration())
                                continue;
                            if (v.ctorinit == 0)
                            {
                                /* Current bugs in the flow analysis:
                                 * 1. union members should not produce error messages even if
                                 *    not assigned to
                                 * 2. structs should recognize delegating opAssign calls as well
                                 *    as delegating calls to other constructors
                                 */
                                if (v.isCtorinit() && !v.type.isMutable() && cd)
                                    error("missing initializer for %s field %s", MODtoChars(v.type.mod), v.toChars());
                                else if (v.storage_class & STCnodefaultctor)
                                    .error(loc, "field %s must be initialized in constructor", v.toChars());
                                else if (v.type.needsNested())
                                    .error(loc, "field %s must be initialized in constructor, because it is nested struct", v.toChars());
                            }
                            else
                            {
                                bool mustInit = (v.storage_class & STCnodefaultctor || v.type.needsNested());
                                if (mustInit && !(sc2.fieldinit[i] & CSXthis_ctor))
                                {
                                    error("field %s must be initialized but skipped", v.toChars());
                                }
                            }
                        }
                    }
                    sc2.freeFieldinit();

                    if (cd && !(sc2.callSuper & CSXany_ctor) && cd.baseClass && cd.baseClass.ctor)
                    {
                        sc2.callSuper = 0;

                        // Insert implicit super() at start of fbody
                        FuncDeclaration fd = resolveFuncCall(Loc(), sc2, cd.baseClass.ctor, null, vthis.type, null, 1);
                        if (!fd)
                        {
                            error("no match for implicit super() call in constructor");
                        }
                        else if (fd.storage_class & STCdisable)
                        {
                            error("cannot call super() implicitly because it is annotated with @disable");
                        }
                        else
                        {
                            Expression e1 = new SuperExp(Loc());
                            Expression e = new CallExp(Loc(), e1);
                            e = e.semantic(sc2);
                            Statement s = new ExpStatement(Loc(), e);
                            fbody = new CompoundStatement(Loc(), s, fbody);
                        }
                    }
                    //printf("callSuper = x%x\n", sc2.callSuper);
                }

                int blockexit = BEnone;
                if (!fbody.isErrorStatement())
                {
                    // Check for errors related to 'nothrow'.
                    uint nothrowErrors = global.errors;
                    blockexit = fbody.blockExit(this, f.isnothrow);
                    if (f.isnothrow && (global.errors != nothrowErrors))
                        .error(loc, "nothrow %s '%s' may throw", kind(), toPrettyChars());
                    if (flags & FUNCFLAGnothrowInprocess)
                    {
                        if (type == f)
                            f = cast(TypeFunction)f.copy();
                        f.isnothrow = !(blockexit & BEthrow);
                    }
                }

                if (fbody.isErrorStatement())
                {
                }
                else if (ad2 && isCtorDeclaration())
                {
                    /* Append:
                     *  return this;
                     * to function body
                     */
                    if (blockexit & BEfallthru)
                    {
                        Statement s = new ReturnStatement(loc, null);
                        s = s.semantic(sc2);
                        fbody = new CompoundStatement(loc, fbody, s);
                        hasReturnExp |= 1;
                    }
                }
                else if (fes)
                {
                    // For foreach(){} body, append a return 0;
                    if (blockexit & BEfallthru)
                    {
                        Expression e = new IntegerExp(0);
                        Statement s = new ReturnStatement(Loc(), e);
                        fbody = new CompoundStatement(Loc(), fbody, s);
                        hasReturnExp |= 1;
                    }
                    assert(!returnLabel);
                }
                else
                {
                    const(bool) inlineAsm = (hasReturnExp & 8) != 0;
                    if ((blockexit & BEfallthru) && f.next.ty != Tvoid && !inlineAsm)
                    {
                        Expression e;
                        if (!hasReturnExp)
                            error("has no return statement, but is expected to return a value of type %s", f.next.toChars());
                        else
                            error("no return exp; or assert(0); at end of function");
                        if (global.params.useAssert && !global.params.useInline)
                        {
                            /* Add an assert(0, msg); where the missing return
                             * should be.
                             */
                            e = new AssertExp(endloc, new IntegerExp(0), new StringExp(loc, cast(char*)"missing return expression"));
                        }
                        else
                            e = new HaltExp(endloc);
                        e = new CommaExp(Loc(), e, f.next.defaultInit());
                        e = e.semantic(sc2);
                        Statement s = new ExpStatement(Loc(), e);
                        fbody = new CompoundStatement(Loc(), fbody, s);
                    }
                }

                if (returns)
                {
                    bool implicit0 = (f.next.ty == Tvoid && isMain());
                    Type tret = implicit0 ? Type.tint32 : f.next;
                    assert(tret.ty != Tvoid);
                    if (vresult || returnLabel)
                        buildResultVar(scout ? scout : sc2, tret);

                    /* Cannot move this loop into NrvoWalker, because
                     * returns[i] may be in the nested delegate for foreach-body.
                     */
                    for (size_t i = 0; i < returns.dim; i++)
                    {
                        ReturnStatement rs = (*returns)[i];
                        Expression exp = rs.exp;
                        if (exp.op == TOKerror)
                            continue;
                        if (tret.ty == Terror)
                        {
                            // Bugzilla 13702
                            exp = checkGC(sc2, exp);
                            continue;
                        }

                        if (!exp.implicitConvTo(tret) && parametersIntersect(exp.type))
                        {
                            if (exp.type.immutableOf().implicitConvTo(tret))
                                exp = exp.castTo(sc2, exp.type.immutableOf());
                            else if (exp.type.wildOf().implicitConvTo(tret))
                                exp = exp.castTo(sc2, exp.type.wildOf());
                        }
                        exp = exp.implicitCastTo(sc2, tret);

                        if (f.isref)
                        {
                            // Function returns a reference
                            exp = exp.toLvalue(sc2, exp);
                            checkEscapeRef(sc2, exp, false);
                        }
                        else
                        {
                            exp = exp.optimize(WANTvalue);

                            /* Bugzilla 10789:
                             * If NRVO is not possible, all returned lvalues should call their postblits.
                             */
                            if (!nrvo_can)
                                exp = doCopyOrMove(sc2, exp);

                            if (tret.hasPointers())
                                checkEscape(sc2, exp, false);
                        }

                        exp = checkGC(sc2, exp);

                        if (vresult)
                        {
                            // Create: return vresult = exp;
                            exp = new BlitExp(rs.loc, vresult, exp);
                            exp.type = vresult.type;

                            if (rs.caseDim)
                                exp = Expression.combine(exp, new IntegerExp(rs.caseDim));
                        }
                        else if (tintro && !tret.equals(tintro.nextOf()))
                        {
                            exp = exp.implicitCastTo(sc2, tintro.nextOf());
                        }
                        rs.exp = exp;
                    }
                }
                if (nrvo_var || returnLabel)
                {
                    scope NrvoWalker nw = new NrvoWalker();
                    nw.fd = this;
                    nw.sc = sc2;
                    nw.visitStmt(fbody);
                }

                sc2 = sc2.pop();
            }

            Statement freq = frequire;
            Statement fens = fensure;

            /* Do the semantic analysis on the [in] preconditions and
             * [out] postconditions.
             */
            if (freq)
            {
                /* frequire is composed of the [in] contracts
                 */
                auto sym = new ScopeDsymbol();
                sym.parent = sc2.scopesym;
                sym.loc = loc;
                sym.endlinnum = endloc.linnum;
                sc2 = sc2.push(sym);
                sc2.flags = (sc2.flags & ~SCOPEcontract) | SCOPErequire;

                // BUG: need to error if accessing out parameters
                // BUG: need to treat parameters as const
                // BUG: need to disallow returns and throws
                // BUG: verify that all in and ref parameters are read
                freq = freq.semantic(sc2);
                freq.blockExit(this, false);

                sc2 = sc2.pop();

                if (!global.params.useIn)
                    freq = null;
            }
            if (fens)
            {
                /* fensure is composed of the [out] contracts
                 */
                if (f.next.ty == Tvoid && outId)
                    error("void functions have no result");

                if (fensure && f.next.ty != Tvoid)
                    buildResultVar(scout, f.next);

                sc2 = scout; //push
                sc2.flags = (sc2.flags & ~SCOPEcontract) | SCOPEensure;

                // BUG: need to treat parameters as const
                // BUG: need to disallow returns and throws
                if (inferRetType && fdensure && (cast(TypeFunction)fdensure.type).parameters)
                {
                    // Return type was unknown in the first semantic pass
                    Parameter p = (*(cast(TypeFunction)fdensure.type).parameters)[0];
                    p.type = f.next;
                }
                fens = fens.semantic(sc2);
                fens.blockExit(this, false);

                sc2 = sc2.pop();

                if (!global.params.useOut)
                    fens = null;
            }
            if (fbody && fbody.isErrorStatement())
            {
            }
            else
            {
                auto a = new Statements();
                // Merge in initialization of 'out' parameters
                if (parameters)
                {
                    for (size_t i = 0; i < parameters.dim; i++)
                    {
                        VarDeclaration v = (*parameters)[i];
                        if (v.storage_class & STCout)
                        {
                            assert(v._init);
                            ExpInitializer ie = v._init.isExpInitializer();
                            assert(ie);
                            if (ie.exp.op == TOKconstruct)
                                ie.exp.op = TOKassign; // construction occured in parameter processing
                            a.push(new ExpStatement(Loc(), ie.exp));
                        }
                    }
                }

// we'll handle variadics ourselves
static if (!IN_LLVM)
{
                if (_arguments)
                {
                    /* Advance to elements[] member of TypeInfo_Tuple with:
                     *  _arguments = v_arguments.elements;
                     */
                    Expression e = new VarExp(Loc(), v_arguments);
                    e = new DotIdExp(Loc(), e, Id.elements);
                    e = new ConstructExp(Loc(), _arguments, e);
                    e = e.semantic(sc2);

                    _arguments._init = new ExpInitializer(Loc(), e);
                    auto de = new DeclarationExp(Loc(), _arguments);
                    a.push(new ExpStatement(Loc(), de));
                }
}

                // Merge contracts together with body into one compound statement

                if (freq || fpreinv)
                {
                    if (!freq)
                        freq = fpreinv;
                    else if (fpreinv)
                        freq = new CompoundStatement(Loc(), freq, fpreinv);

                    a.push(freq);
                }

                if (fbody)
                    a.push(fbody);

                if (fens || fpostinv)
                {
                    if (!fens)
                        fens = fpostinv;
                    else if (fpostinv)
                        fens = new CompoundStatement(Loc(), fpostinv, fens);

                    auto ls = new LabelStatement(Loc(), Id.returnLabel, fens);
                    returnLabel.statement = ls;
                    a.push(returnLabel.statement);

                    if (f.next.ty != Tvoid && vresult)
                    {
version(IN_LLVM)
{
                        Expression e = null;
                        if (isCtorDeclaration())
                        {
                            ThisExp te = new ThisExp(Loc());
                            te.type = vthis.type;
                            te.var = vthis;
                            e = te;
                        }
                        else
                        {
                            e = new VarExp(Loc(), vresult);
                        }
}
else
{
                        // Create: return vresult;
                        Expression e = new VarExp(Loc(), vresult);
}
                        if (tintro)
                        {
                            e = e.implicitCastTo(sc, tintro.nextOf());
                            e = e.semantic(sc);
                        }
                        auto s = new ReturnStatement(Loc(), e);
                        a.push(s);
                    }
                }
                if (isMain() && f.next.ty == Tvoid)
                {
                    // Add a return 0; statement
                    Statement s = new ReturnStatement(Loc(), new IntegerExp(0));
                    a.push(s);
                }

                Statement sbody = new CompoundStatement(Loc(), a);

                /* Append destructor calls for parameters as finally blocks.
                 */
                if (parameters)
                {
                    foreach (v; *parameters)
                    {
                        if (v.storage_class & (STCref | STCout | STClazy))
                            continue;
                        if (v.needsScopeDtor())
                        {
                            // same with ExpStatement.scopeCode()
                            Statement s = new DtorExpStatement(Loc(), v.edtor, v);
                            v.storage_class |= STCnodtor;

                            s = s.semantic(sc2);

                            bool isnothrow = f.isnothrow & !(flags & FUNCFLAGnothrowInprocess);
                            int blockexit = s.blockExit(this, isnothrow);
                            if (f.isnothrow && isnothrow && blockexit & BEthrow)
                                .error(loc, "nothrow %s '%s' may throw", kind(), toPrettyChars());
                            if (flags & FUNCFLAGnothrowInprocess && blockexit & BEthrow)
                                f.isnothrow = false;

                            if (sbody.blockExit(this, f.isnothrow) == BEfallthru)
                                sbody = new CompoundStatement(Loc(), sbody, s);
                            else
                                sbody = new TryFinallyStatement(Loc(), sbody, s);
                        }
                    }
                }
                // from this point on all possible 'throwers' are checked
                flags &= ~FUNCFLAGnothrowInprocess;

                if (isSynchronized())
                {
                    /* Wrap the entire function body in a synchronized statement
                     */
                    ClassDeclaration cd = isThis() ? isThis().isClassDeclaration() : parent.isClassDeclaration();
                    if (cd)
                    {
                        if (!global.params.is64bit && global.params.isWindows && !isStatic() && !sbody.usesEH() && !global.params.trace)
                        {
                            /* The back end uses the "jmonitor" hack for syncing;
                             * no need to do the sync at this level.
                             */
                        }
                        else
                        {
                            Expression vsync;
                            if (isStatic())
                            {
                                // The monitor is in the ClassInfo
                                vsync = new DotIdExp(loc, DsymbolExp.resolve(loc, sc2, cd, false), Id.classinfo);
                            }
                            else
                            {
                                // 'this' is the monitor
                                vsync = new VarExp(loc, vthis);
                            }
                            sbody = new PeelStatement(sbody); // don't redo semantic()
                            sbody = new SynchronizedStatement(loc, vsync, sbody);
                            sbody = sbody.semantic(sc2);
                        }
                    }
                    else
                    {
                        error("synchronized function %s must be a member of a class", toChars());
                    }
                }

                // If declaration has no body, don't set sbody to prevent incorrect codegen.
                InterfaceDeclaration id = parent.isInterfaceDeclaration();
                if (fbody || id && (fdensure || fdrequire) && isVirtual())
                    fbody = sbody;
            }

            // Fix up forward-referenced gotos
            if (gotos)
            {
                for (size_t i = 0; i < gotos.dim; ++i)
                {
                    (*gotos)[i].checkLabel();
                }
            }

            if (naked && (fensure || frequire))
                error("naked assembly functions with contracts are not supported");

            sc2.callSuper = 0;
            sc2.pop();
        }

        if (checkClosure())
        {
            // We should be setting errors here instead of relying on the global error count.
            //errors = true;
        }

        /* If function survived being marked as impure, then it is pure
         */
        if (flags & FUNCFLAGpurityInprocess)
        {
            flags &= ~FUNCFLAGpurityInprocess;
            if (type == f)
                f = cast(TypeFunction)f.copy();
            f.purity = PUREfwdref;
        }

        if (flags & FUNCFLAGsafetyInprocess)
        {
            flags &= ~FUNCFLAGsafetyInprocess;
            if (type == f)
                f = cast(TypeFunction)f.copy();
            f.trust = TRUSTsafe;
        }

        if (flags & FUNCFLAGnogcInprocess)
        {
            flags &= ~FUNCFLAGnogcInprocess;
            if (type == f)
                f = cast(TypeFunction)f.copy();
            f.isnogc = true;
        }

        if (flags & FUNCFLAGreturnInprocess)
        {
            flags &= ~FUNCFLAGreturnInprocess;
            if (storage_class & STCreturn)
            {
                if (type == f)
                    f = cast(TypeFunction)f.copy();
                f.isreturn = true;
            }
        }

        flags &= ~FUNCFLAGinferScope;

        // Infer STCscope
        if (parameters)
        {
            size_t nfparams = Parameter.dim(f.parameters);
            assert(nfparams == parameters.dim);
            foreach (u, v; *parameters)
            {
                if (v.storage_class & STCmaybescope)
                {
                    //printf("Inferring scope for %s\n", v.toChars());
                    Parameter p = Parameter.getNth(f.parameters, u);
                    v.storage_class &= ~STCmaybescope;
                    v.storage_class |= STCscope;
                    p.storageClass |= STCscope;
                    assert(!(p.storageClass & STCmaybescope));
                }
            }
        }

        if (vthis && vthis.storage_class & STCmaybescope)
        {
            vthis.storage_class &= ~STCmaybescope;
            vthis.storage_class |= STCscope;
            f.isscope = true;
        }

        // reset deco to apply inference result to mangled name
        if (f != type)
            f.deco = null;

        // Do semantic type AFTER pure/nothrow inference.
        if (!f.deco && ident != Id.xopEquals && ident != Id.xopCmp)
        {
            sc = sc.push();
            if (isCtorDeclaration()) // Bugzilla #15665
                sc.flags |= SCOPEctor;
            sc.stc = 0;
            sc.linkage = linkage; // Bugzilla 8496
            type = f.semantic(loc, sc);
            sc = sc.pop();
        }

        /* If this function had instantiated with gagging, error reproduction will be
         * done by TemplateInstance::semantic.
         * Otherwise, error gagging should be temporarily ungagged by functionSemantic3.
         */
        semanticRun = PASSsemantic3done;
        semantic3Errors = (global.errors != oldErrors) || (fbody && fbody.isErrorStatement());
        if (type.ty == Terror)
            errors = true;
        //printf("-FuncDeclaration::semantic3('%s.%s', sc = %p, loc = %s)\n", parent.toChars(), toChars(), sc, loc.toChars());
        //fflush(stdout);
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
            semantic(_scope);
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
            if (ad && ad.sizeok != SIZEOKdone)
            {
                /* Currently dmd cannot resolve forward references per methods,
                 * then setting SIZOKfwd is too conservative and would break existing code.
                 * So, just stop method attributes inference until ad.semantic() done.
                 */
                //ad.sizeok = SIZEOKfwd;
            }
            else
                return functionSemantic3() || !errors;
        }

        if (storage_class & STCinference)
            return functionSemantic3() || !errors;

        return !errors;
    }

    /****************************************************
     * Resolve forward reference of function body.
     * Returns false if any errors exist in the body.
     */
    final bool functionSemantic3()
    {
        if (semanticRun < PASSsemantic3 && _scope)
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
            semantic3(_scope);
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
    final bool checkForwardRef(Loc loc)
    {
        if (!functionSemantic())
            return true;

        /* No deco means the functionSemantic() call could not resolve
         * forward referenes in the type of this function.
         */
        if (!type.deco)
        {
            bool inSemantic3 = (inferRetType && semanticRun >= PASSsemantic3);
            .error(loc, "forward reference to %s'%s'",
                (inSemantic3 ? "inferred return type of function " : "").ptr,
                toChars());
            return true;
        }
        return false;
    }

    // called from semantic3
    final VarDeclaration declareThis(Scope* sc, AggregateDeclaration ad)
    {
        if (ad)
        {
            //printf("declareThis() %s\n", toChars());
            Type thandle = ad.handleType();
            assert(thandle);
            thandle = thandle.addMod(type.mod);
            thandle = thandle.addStorageClass(storage_class);
            VarDeclaration v = new ThisDeclaration(loc, thandle);
            v.storage_class |= STCparameter;
            if (thandle.ty == Tstruct)
            {
                v.storage_class |= STCref;
                // if member function is marked 'inout', then 'this' is 'return ref'
                if (type.ty == Tfunction && (cast(TypeFunction)type).iswild & 2)
                    v.storage_class |= STCreturn;
            }
            if (type.ty == Tfunction)
            {
                TypeFunction tf = cast(TypeFunction)type;
                if (tf.isreturn)
                    v.storage_class |= STCreturn;
                if (tf.isscope)
                    v.storage_class |= STCscope;
            }
            if (flags & FUNCFLAGinferScope)
                v.storage_class |= STCmaybescope;

            v.semantic(sc);
            if (!sc.insert(v))
                assert(0);
            v.parent = this;
            return v;
        }
        if (isNested())
        {
            /* The 'this' for a nested function is the link to the
             * enclosing function's stack frame.
             * Note that nested functions and member functions are disjoint.
             */
            VarDeclaration v = new ThisDeclaration(loc, Type.tvoid.pointerTo());
            v.storage_class |= STCparameter;
            if (type.ty == Tfunction)
            {
                TypeFunction tf = cast(TypeFunction)type;
                if (tf.isreturn)
                    v.storage_class |= STCreturn;
                if (tf.isscope)
                    v.storage_class |= STCscope;
            }
            if (flags & FUNCFLAGinferScope)
                v.storage_class |= STCmaybescope;

            v.semantic(sc);
            if (!sc.insert(v))
                assert(0);
            v.parent = this;
            return v;
        }
        return null;
    }

    override final bool equals(RootObject o)
    {
        if (this == o)
            return true;

        Dsymbol s = isDsymbol(o);
        if (s)
        {
            FuncDeclaration fd1 = this;
            FuncDeclaration fd2 = s.isFuncDeclaration();
            if (!fd2)
                return false;

            FuncAliasDeclaration fa1 = fd1.isFuncAliasDeclaration();
            FuncAliasDeclaration fa2 = fd2.isFuncAliasDeclaration();
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
     * Returns:
     *      -1      didn't find one
     *      -2      can't determine because of forward references
     */
    final int findVtblIndex(Dsymbols* vtbl, int dim)
    {
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
                        return vi; // no need to look further

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
                int cov = type.covariant(fdv.type, &stc);
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
     * Params:
     *  cd = class that function is in
     * Returns:
     *  base class if overriding, null if not
     */
    final BaseClass* overrideInterface()
    {
        ClassDeclaration cd = parent.isClassDeclaration();
        foreach (b; cd.interfaces)
        {
            auto v = findVtblIndex(&b.sym.vtbl, cast(int)b.sym.vtbl.dim);
            if (v >= 0)
                return b;
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
            if (!ad.aliassym && ad.type.ty != Tident && ad.type.ty != Tinstance)
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
                    tf.nextOf().implicitConvTo(t.nextOf()) >= MATCHconst)
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
    final FuncDeclaration overloadModMatch(Loc loc, Type tthis, ref bool hasOverloads)
    {
        //printf("FuncDeclaration::overloadModMatch('%s')\n", toChars());
        Match m;
        m.last = MATCHnomatch;
        overloadApply(this, (Dsymbol s)
        {
            auto f = s.isFuncDeclaration();
            if (!f || f == m.lastf) // skip duplicates
                return 0;
            m.anyf = f;

            auto tf = cast(TypeFunction)f.type;
            //printf("tf = %s\n", tf.toChars());

            MATCH match;
            if (tthis) // non-static functions are preferred than static ones
            {
                if (f.needThis())
                    match = f.isCtorDeclaration() ? MATCHexact : MODmethodConv(tthis.mod, tf.mod);
                else
                    match = MATCHconst; // keep static funciton in overload candidates
            }
            else // static functions are preferred than non-static ones
            {
                if (f.needThis())
                    match = MATCHconvert;
                else
                    match = MATCHexact;
            }
            if (match == MATCHnomatch)
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
            if (m.last <= MATCHconvert)
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
            auto tf = cast(TypeFunction)this.type;
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

        TypeFunction tf = cast(TypeFunction)type;
        TypeFunction tg = cast(TypeFunction)g.type;
        size_t nfparams = Parameter.dim(tf.parameters);

        /* If both functions have a 'this' pointer, and the mods are not
         * the same and g's is not const, then this is less specialized.
         */
        if (needThis() && g.needThis() && tf.mod != tg.mod)
        {
            if (isCtorDeclaration())
            {
                if (!MODimplicitConv(tg.mod, tf.mod))
                    return MATCHnomatch;
            }
            else
            {
                if (!MODimplicitConv(tf.mod, tg.mod))
                    return MATCHnomatch;
            }
        }

        /* Create a dummy array of arguments out of the parameters to f()
         */
        Expressions args;
        args.setDim(nfparams);
        for (size_t u = 0; u < nfparams; u++)
        {
            Parameter p = Parameter.getNth(tf.parameters, u);
            Expression e;
            if (p.storageClass & (STCref | STCout))
            {
                e = new IdentifierExp(Loc(), p.ident);
                e.type = p.type;
            }
            else
                e = p.type.defaultInitLiteral(Loc());
            args[u] = e;
        }

        MATCH m = tg.callMatch(null, &args, 1);
        if (m > MATCHnomatch)
        {
            /* A variadic parameter list is less specialized than a
             * non-variadic one.
             */
            if (tf.varargs && !tg.varargs)
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
        return MATCHnomatch;
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
    final int getLevel(Loc loc, Scope* sc, FuncDeclaration fd)
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
        if (!(sc.flags & SCOPEconstraint))
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
        functionToBufferWithIdent(cast(TypeFunction)type, &buf, toChars());
        return buf.extractString();
    }

    final bool isMain()
    {
        return ident == Id.main && linkage != LINKc && !isMember() && !isNested();
    }

    final bool isCMain()
    {
        return ident == Id.main && linkage == LINKc && !isMember() && !isNested();
    }

    final bool isWinMain()
    {
        //printf("FuncDeclaration::isWinMain() %s\n", toChars());
        version (none)
        {
            bool x = ident == Id.WinMain && linkage != LINKc && !isMember();
            printf("%s\n", x ? "yes" : "no");
            return x;
        }
        else
        {
            return ident == Id.WinMain && linkage != LINKc && !isMember();
        }
    }

    final bool isDllMain()
    {
        return ident == Id.DllMain && linkage != LINKc && !isMember();
    }

    final bool isRtInit()
    {
        return ident == Id.rt_init && linkage == LINKc && !isMember() && !isNested();
    }

    override final bool isExport()
    {
        return protection.kind == PROTexport;
    }

    override final bool isImportedSymbol()
    {
        //printf("isImportedSymbol()\n");
        //printf("protection = %d\n", protection);
        return (protection.kind == PROTexport) && !fbody;
    }

    override final bool isCodeseg()
    {
        return true; // functions are always in the code segment
    }

    override final bool isOverloadable()
    {
        return true; // functions can be overloaded
    }

    /**********************************
     * Decide if attributes for this function can be inferred from examining
     * the function body.
     * Returns:
     *  true if can
     */
    private final bool canInferAttributes(Scope* sc)
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
            (storage_class & STCinference) ||           // do attribute inference
            (inferRetType && !isCtorDeclaration()))
            return true;

        if (isInstantiated())
        {
            TemplateInstance ti = parent.isTemplateInstance();
            if (ti is null || ti.isTemplateMixin() || ti.tempdecl.ident == ident)
                return true;
        }

        return false;
    }

    /*****************************************
     * Initialize for inferring the attributes of this function.
     */
    private final void initInferAttributes()
    {
        assert(type.ty == Tfunction);
        TypeFunction tf = cast(TypeFunction)type;
        if (tf.purity == PUREimpure) // purity not specified
            flags |= FUNCFLAGpurityInprocess;

        if (tf.trust == TRUSTdefault)
            flags |= FUNCFLAGsafetyInprocess;

        if (!tf.isnothrow)
            flags |= FUNCFLAGnothrowInprocess;

        if (!tf.isnogc)
            flags |= FUNCFLAGnogcInprocess;

        if (!isVirtual() || introducing)
            flags |= FUNCFLAGreturnInprocess;

        // Initialize for inferring STCscope
        if (global.params.vsafe)
            flags |= FUNCFLAGinferScope;
    }

    final PURE isPure()
    {
        //printf("FuncDeclaration::isPure() '%s'\n", toChars());
        assert(type.ty == Tfunction);
        TypeFunction tf = cast(TypeFunction)type;
        if (flags & FUNCFLAGpurityInprocess)
            setImpure();
        if (tf.purity == PUREfwdref)
            tf.purityLevel();
        PURE purity = tf.purity;
        if (purity > PUREweak && isNested())
            purity = PUREweak;
        if (purity > PUREweak && needThis())
        {
            // The attribute of the 'this' reference affects purity strength
            if (type.mod & MODimmutable)
            {
            }
            else if (type.mod & (MODconst | MODwild) && purity >= PUREconst)
                purity = PUREconst;
            else
                purity = PUREweak;
        }
        tf.purity = purity;
        // ^ This rely on the current situation that every FuncDeclaration has a
        //   unique TypeFunction.
        return purity;
    }

    final PURE isPureBypassingInference()
    {
        if (flags & FUNCFLAGpurityInprocess)
            return PUREfwdref;
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
        if (flags & FUNCFLAGpurityInprocess)
        {
            flags &= ~FUNCFLAGpurityInprocess;
            if (fes)
                fes.func.setImpure();
        }
        else if (isPure())
            return true;
        return false;
    }

    final bool isSafe()
    {
        assert(type.ty == Tfunction);
        if (flags & FUNCFLAGsafetyInprocess)
            setUnsafe();
        return (cast(TypeFunction)type).trust == TRUSTsafe;
    }

    final bool isSafeBypassingInference()
    {
        return !(flags & FUNCFLAGsafetyInprocess) && isSafe();
    }

    final bool isTrusted()
    {
        assert(type.ty == Tfunction);
        if (flags & FUNCFLAGsafetyInprocess)
            setUnsafe();
        return (cast(TypeFunction)type).trust == TRUSTtrusted;
    }

    /**************************************
     * The function is doing something unsave,
     * so mark it as unsafe.
     * If there's a safe error, return true.
     */
    final bool setUnsafe()
    {
        if (flags & FUNCFLAGsafetyInprocess)
        {
            flags &= ~FUNCFLAGsafetyInprocess;
            (cast(TypeFunction)type).trust = TRUSTsystem;
            if (fes)
                fes.func.setUnsafe();
        }
        else if (isSafe())
            return true;
        return false;
    }

    final bool isNogc()
    {
        assert(type.ty == Tfunction);
        if (flags & FUNCFLAGnogcInprocess)
            setGC();
        return (cast(TypeFunction)type).isnogc;
    }

    final bool isNogcBypassingInference()
    {
        return !(flags & FUNCFLAGnogcInprocess) && isNogc();
    }

    /**************************************
     * The function is doing something that may allocate with the GC,
     * so mark it as not nogc (not no-how).
     * Returns:
     *      true if function is marked as @nogc, meaning a user error occurred
     */
    final bool setGC()
    {
        if (flags & FUNCFLAGnogcInprocess)
        {
            flags &= ~FUNCFLAGnogcInprocess;
            (cast(TypeFunction)type).isnogc = false;
            if (fes)
                fes.func.setGC();
        }
        else if (isNogc())
            return true;
        return false;
    }

    final void printGCUsage(Loc loc, const(char)* warn)
    {
        if (!global.params.vgc)
            return;

        Module m = getModule();
        if (m && m.isRoot() && !inUnittest())
        {
            fprintf(global.stdmsg, "%s: vgc: %s\n", loc.toChars(), warn);
        }
    }

    /********************************************
     * Returns true if the function return value has no indirection
     * which comes from the parameters.
     */
    final bool isolateReturn()
    {
        assert(type.ty == Tfunction);
        TypeFunction tf = cast(TypeFunction)type;
        assert(tf.next);

        Type treti = tf.next;
        treti = tf.isref ? treti : getIndirection(treti);
        if (!treti)
            return true; // target has no mutable indirection
        return parametersIntersect(treti);
    }

    /********************************************
     * Returns true if an object typed t can have indirections
     * which come from the parameters.
     */
    final bool parametersIntersect(Type t)
    {
        assert(t);
        if (!isPureBypassingInference() || isNested())
            return false;

        assert(type.ty == Tfunction);
        TypeFunction tf = cast(TypeFunction)type;

        //printf("parametersIntersect(%s) t = %s\n", tf.toChars(), t.toChars());

        size_t dim = Parameter.dim(tf.parameters);
        for (size_t i = 0; i < dim; i++)
        {
            Parameter fparam = Parameter.getNth(tf.parameters, i);
            if (!fparam.type)
                continue;
            Type tprmi = (fparam.storageClass & (STClazy | STCout | STCref)) ? fparam.type : getIndirection(fparam.type);
            if (!tprmi)
                continue; // there is no mutable indirection

            //printf("\t[%d] tprmi = %d %s\n", i, tprmi.ty, tprmi.toChars());
            if (traverseIndirections(tprmi, t))
                return false;
        }
        if (AggregateDeclaration ad = isCtorDeclaration() ? null : isThis())
        {
            Type tthis = ad.getType().addMod(tf.mod);
            //printf("\ttthis = %s\n", tthis.toChars());
            if (traverseIndirections(tthis, t))
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
    bool isNested()
    {
        auto f = toAliasFunc();
        //printf("\ttoParent2() = '%s'\n", f.toParent2().toChars());
        return ((f.storage_class & STCstatic) == 0) &&
                (f.linkage == LINKd) &&
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
    override AggregateDeclaration isThis()
    {
        //printf("+FuncDeclaration::isThis() '%s'\n", toChars());
        auto ad = (storage_class & STCstatic) ? null : isMember2();
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
    bool isVirtual()
    {
        if (toAliasFunc() != this)
            return toAliasFunc().isVirtual();

        Dsymbol p = toParent();
        version (none)
        {
            printf("FuncDeclaration::isVirtual(%s)\n", toChars());
            printf("isMember:%p isStatic:%d private:%d ctor:%d !Dlinkage:%d\n", isMember(), isStatic(), protection == PROTprivate, isCtorDeclaration(), linkage != LINKd);
            printf("result is %d\n", isMember() && !(isStatic() || protection == PROTprivate || protection == PROTpackage) && p.isClassDeclaration() && !(p.isInterfaceDeclaration() && isFinalFunc()));
        }
        return isMember() && !(isStatic() || protection.kind == PROTprivate || protection.kind == PROTpackage) && p.isClassDeclaration() && !(p.isInterfaceDeclaration() && isFinalFunc());
    }

    bool isFinalFunc()
    {
        if (toAliasFunc() != this)
            return toAliasFunc().isFinalFunc();

        ClassDeclaration cd;
        version (none)
        {
            printf("FuncDeclaration::isFinalFunc(%s), %x\n", toChars(), Declaration.isFinal());
            printf("%p %d %d %d\n", isMember(), isStatic(), Declaration.isFinal(), ((cd = toParent().isClassDeclaration()) !is null && cd.storage_class & STCfinal));
            printf("result is %d\n", isMember() && (Declaration.isFinal() || ((cd = toParent().isClassDeclaration()) !is null && cd.storage_class & STCfinal)));
            if (cd)
                printf("\tmember of %s\n", cd.toChars());
        }
        return isMember() && (Declaration.isFinal() || ((cd = toParent().isClassDeclaration()) !is null && cd.storage_class & STCfinal));
    }

    bool addPreInvariant()
    {
        AggregateDeclaration ad = isThis();
        ClassDeclaration cd = ad ? ad.isClassDeclaration() : null;
        return (ad && !(cd && cd.isCPPclass()) && global.params.useInvariants && (protection.kind == PROTprotected || protection.kind == PROTpublic || protection.kind == PROTexport) && !naked);
    }

    bool addPostInvariant()
    {
        AggregateDeclaration ad = isThis();
        ClassDeclaration cd = ad ? ad.isClassDeclaration() : null;
        return (ad && !(cd && cd.isCPPclass()) && ad.inv && global.params.useInvariants && (protection.kind == PROTprotected || protection.kind == PROTpublic || protection.kind == PROTexport) && !naked);
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
    final bool checkNestedReference(Scope* sc, Loc loc)
    {
        //printf("FuncDeclaration::checkNestedReference() %s\n", toPrettyChars());

        if (auto fld = this.isFuncLiteralDeclaration())
        {
            if (fld.tok == TOKreserved)
            {
                fld.tok = TOKfunction;
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
        checkNestedRef(fdthis, p);

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
                    if (!sc.intypeof && !(sc.flags & SCOPEcompile))
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
         * -or-
         * 5) this function returns a local struct/class
         *
         * Note that since a non-virtual function can be called by
         * a virtual one, if that non-virtual function accesses a closure
         * var, the closure still has to be taken. Hence, we check for isThis()
         * instead of isVirtual(). (thanks to David Friedman)
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

                    /* Bugzilla 12406: Iterate all closureVars to mark all descendant
                     * nested functions that access to the closing context of this funciton.
                     */
                }
            }
        }
        if (requiresClosure)
            goto Lyes;

        /* Look for case (5)
         */
        if (closureVars.dim)
        {
            assert(type.ty == Tfunction);
            Type tret = (cast(TypeFunction)type).next;
            assert(tret);
            tret = tret.toBasetype();
            //printf("\t\treturning %s\n", tret.toChars());
            if (tret.ty == Tclass || tret.ty == Tstruct)
            {
                Dsymbol st = tret.toDsymbol(null);
                //printf("\t\treturning class/struct %s\n", tret.toChars());
                for (Dsymbol s = st.parent; s; s = s.parent)
                {
                    //printf("\t\t\tparent = %s %s\n", s.kind(), s.toChars());
                    if (s == this)
                    {
                        //printf("\t\treturning local %s\n", st.toChars());
                        goto Lyes;
                    }
                }
            }
        }

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
    final bool checkClosure()
    {
        if (!needsClosure())
            return false;

        if (setGC())
        {
            error("is @nogc yet allocates closures with the GC");
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
         * context had better match, or Bugzilla 7335 will bite.
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
     * Declare result variable lazily.
     */
    final void buildResultVar(Scope* sc, Type tret)
    {
        if (!vresult)
        {
            Loc loc = fensure ? fensure.loc : this.loc;

            /* If inferRetType is true, tret may not be a correct return type yet.
             * So, in here it may be a temporary type for vresult, and after
             * fbody.semantic() running, vresult.type might be modified.
             */
            vresult = new VarDeclaration(loc, tret, outId ? outId : Id.result, null);
            vresult.storage_class |= STCnodtor;

            if (outId == Id.result)
                vresult.storage_class |= STCtemp;
            if (!isVirtual())
                vresult.storage_class |= STCconst;
            vresult.storage_class |= STCresult;

            // set before the semantic() for checkNestedReference()
            vresult.parent = this;
        }

        if (sc && vresult.semanticRun == PASSinit)
        {
            assert(type.ty == Tfunction);
            TypeFunction tf = cast(TypeFunction)type;
            if (tf.isref)
                vresult.storage_class |= STCref;
            vresult.type = tret;

            vresult.semantic(sc);

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
        if (params is null)
            params = fdrequireParams;

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

version(IN_LLVM)
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
        for (size_t i = 0; i < foverrides.dim; i++)
        {
            FuncDeclaration fdv = foverrides[i];

            /* The semantic pass on the contracts of the overridden functions must
             * be completed before code generation occurs (bug 3602).
             */
            if (fdv.fdrequire && fdv.fdrequire.semanticRun != PASSsemantic3done)
            {
                assert(fdv._scope);
                Scope* sc = fdv._scope.push();
                sc.stc &= ~STCoverride;
                fdv.semantic3(sc);
                sc.pop();
            }

version(IN_LLVM)
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
version(IN_LLVM)
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
     * Merge into this function the 'out' contracts of all it overrides.
     * 'out's are AND'd together, i.e. all of them need to pass.
     */
    // IN_LLVM replaced: final Statement mergeFensure(Statement sf, Identifier oid)
    final Statement mergeFensure(Statement sf, Identifier oid, Expressions *params = null)
    {
        version(IN_LLVM)
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
        for (size_t i = 0; i < foverrides.dim; i++)
        {
            FuncDeclaration fdv = foverrides[i];

            /* The semantic pass on the contracts of the overridden functions must
             * be completed before code generation occurs (bug 3602 and 5230).
             */
            if (fdv.fdensure && fdv.fdensure.semanticRun != PASSsemantic3done)
            {
                assert(fdv._scope);
                Scope* sc = fdv._scope.push();
                sc.stc &= ~STCoverride;
                fdv.semantic3(sc);
                sc.pop();
            }

version(IN_LLVM)
            sf = fdv.mergeFensure(sf, oid, params);
else
            sf = fdv.mergeFensure(sf, oid);
            if (fdv.fdensure)
            {
                //printf("fdv.fensure: %s\n", fdv.fensure.toChars());
                // Make the call: __ensure(result)
                Expression eresult = null;
                if (outId)
                {
version(IN_LLVM)
                    eresult = (*params)[0];
else
                    eresult = new IdentifierExp(loc, oid);

                    Type t1 = fdv.type.nextOf().toBasetype();
version(IN_LLVM)
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
                         * See bugzilla 5204 and 10479.
                         */
                        auto ei = new ExpInitializer(Loc(), eresult);
                        auto v = new VarDeclaration(Loc(), t1, Identifier.generateId("__covres"), ei);
                        v.storage_class |= STCtemp;
                        auto de = new DeclarationExp(Loc(), v);
                        auto ve = new VarExp(Loc(), v);
                        eresult = new CommaExp(Loc(), de, ve);
                    }
                }
version(IN_LLVM)
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
     * Return the function's parameter list, and whether
     * it is variadic or not.
     */
    final Parameters* getParameters(int* pvarargs)
    {
        Parameters* fparameters = null;
        int fvarargs = 0;

        if (type)
        {
            assert(type.ty == Tfunction);
            TypeFunction fdtype = cast(TypeFunction)type;
            fparameters = fdtype.parameters;
            fvarargs = fdtype.varargs;
        }
        if (pvarargs)
            *pvarargs = fvarargs;

        return fparameters;
    }

    /**********************************
     * Generate a FuncDeclaration for a runtime library function.
     */
    static FuncDeclaration genCfunc(Parameters* fparams, Type treturn, const(char)* name, StorageClass stc = 0)
    {
        return genCfunc(fparams, treturn, Identifier.idPool(name, strlen(name)), stc);
    }

    static FuncDeclaration genCfunc(Parameters* fparams, Type treturn, Identifier id, StorageClass stc = 0)
    {
        FuncDeclaration fd;
        TypeFunction tf;
        Dsymbol s;
        static __gshared DsymbolTable st = null;

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
            tf = new TypeFunction(fparams, treturn, 0, LINKc, stc);
            fd = new FuncDeclaration(Loc(), Loc(), id, STCstatic, tf);
            fd.protection = Prot(PROTpublic);
            fd.linkage = LINKc;

            st.insert(fd);
        }
        return fd;
    }

    /******************
     * Check parameters and return type of D main() function.
     * Issue error messages.
     */
    final void checkDmain()
    {
        TypeFunction tf = cast(TypeFunction)type;
        const nparams = Parameter.dim(tf.parameters);
        bool argerr;
        if (nparams == 1)
        {
            auto fparam0 = Parameter.getNth(tf.parameters, 0);
            auto t = fparam0.type.toBasetype();
            if (t.ty != Tarray ||
                t.nextOf().ty != Tarray ||
                t.nextOf().nextOf().ty != Tchar ||
                fparam0.storageClass & (STCout | STCref | STClazy))
            {
                argerr = true;
            }
        }

        if (!tf.nextOf())
            error("must return int or void");
        else if (tf.nextOf().ty != Tint32 && tf.nextOf().ty != Tvoid)
            error("must return int or void, not %s", tf.nextOf().toChars());
        else if (tf.varargs || nparams >= 2 || argerr)
            error("parameters must be main() or main(string[] args)");
    }

    override final inout(FuncDeclaration) isFuncDeclaration() inout
    {
        return this;
    }

    FuncDeclaration toAliasFunc()
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
 *      direct  call invariant directly
 * Returns:
 *      void expression that calls the invariant
 */
extern (C++) Expression addInvariant(Loc loc, Scope* sc, AggregateDeclaration ad, VarDeclaration vthis)
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
            // Workaround for bugzilla 13394: For the correct mangling,
            // run attribute inference on inv if needed.
            inv.functionSemantic();
        }

        //e = new DsymbolExp(Loc(), inv);
        //e = new CallExp(Loc(), e);
        //e = e.semantic(sc2);

        /* Bugzilla 13113: Currently virtual invariant calls completely
         * bypass attribute enforcement.
         * Change the behavior of pre-invariant call by following it.
         */
        e = new ThisExp(Loc());
        e.type = vthis.type;
        e = new DotVarExp(Loc(), e, inv, false);
        e.type = inv.type;
        e = new CallExp(Loc(), e);
        e.type = Type.tvoid;
    }

    return e;
}

/***************************************************
 * Visit each overloaded function/template in turn, and call dg(s) on it.
 * Exit when no more, or dg(s) returns nonzero.
 * Returns:
 *      ==0     continue
 *      !=0     done
 */
extern (D) int overloadApply(Dsymbol fstart, scope int delegate(Dsymbol) dg)
{
    Dsymbol next;
    for (Dsymbol d = fstart; d; d = next)
    {
        if (auto od = d.isOverDeclaration())
        {
            if (od.hasOverloads)
            {
                if (int r = overloadApply(od.aliassym, dg))
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
                if (int r = overloadApply(fa.funcalias, dg))
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

extern (C++) int overloadApply(Dsymbol fstart, void* param, int function(void*, Dsymbol) fp)
{
    return overloadApply(fstart, s => (*fp)(param, s));
}

extern (C++) static void MODMatchToBuffer(OutBuffer* buf, ubyte lhsMod, ubyte rhsMod)
{
    bool bothMutable = ((lhsMod & rhsMod) == 0);
    bool sharedMismatch = ((lhsMod ^ rhsMod) & MODshared) != 0;
    bool sharedMismatchOnly = ((lhsMod ^ rhsMod) == MODshared);

    if (lhsMod & MODshared)
        buf.writestring("shared ");
    else if (sharedMismatch && !(lhsMod & MODimmutable))
        buf.writestring("non-shared ");

    if (bothMutable && sharedMismatchOnly)
    {
    }
    else if (lhsMod & MODimmutable)
        buf.writestring("immutable ");
    else if (lhsMod & MODconst)
        buf.writestring("const ");
    else if (lhsMod & MODwild)
        buf.writestring("inout ");
    else
        buf.writestring("mutable ");
}

/*******************************************
 * Given a symbol that could be either a FuncDeclaration or
 * a function template, resolve it to a function symbol.
 *      loc             instantiation location
 *      sc              instantiation scope
 *      tiargs          initial list of template arguments
 *      tthis           if !NULL, the 'this' pointer argument
 *      fargs           arguments to function
 *      flags           1: do not issue error message on no match, just return NULL
 *                      2: overloadResolve only
 */
extern (C++) FuncDeclaration resolveFuncCall(Loc loc, Scope* sc, Dsymbol s,
    Objects* tiargs, Type tthis, Expressions* fargs, int flags = 0)
{
    if (!s)
        return null; // no match

    version (none)
    {
        printf("resolveFuncCall('%s')\n", s.toChars());
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
    m.last = MATCHnomatch;

    functionResolve(&m, s, loc, sc, tiargs, tthis, fargs);

    if (m.last > MATCHnomatch && m.lastf)
    {
        if (m.count == 1) // exactly one match
        {
            if (!(flags & 1))
                m.lastf.functionSemantic();
            return m.lastf;
        }
        if ((flags & 2) && !tthis && m.lastf.needThis())
        {
            return m.lastf;
        }
    }

    /* Failed to find a best match.
     * Do nothing or print error.
     */
    if (m.last <= MATCHnomatch)
    {
        // error was caused on matched function
        if (m.count == 1)
            return m.lastf;

        // if do not print error messages
        if (flags & 1)
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

    if (!m.lastf && !(flags & 1)) // no match
    {
        if (td && !fd) // all of overloads are templates
        {
            .error(loc, "%s %s.%s cannot deduce function from argument types !(%s)%s, candidates are:",
                td.kind(), td.parent.toPrettyChars(), td.ident.toChars(),
                tiargsBuf.peekString(), fargsBuf.peekString());

            // Display candidate templates (even if there are no multiple overloads)
            int numToDisplay = numOverloadsDisplay;
            overloadApply(td, (Dsymbol s)
            {
                auto td = s.isTemplateDeclaration();
                if (!td)
                    return 0;
                .errorSupplemental(td.loc, "%s", td.toPrettyChars());
                if (global.params.verbose || --numToDisplay != 0 || !td.overnext)
                    return 0;

                // Too many overloads to sensibly display.
                // Just show count of remaining overloads.
                int num = 0;
                overloadApply(td.overnext, (s) { ++num; return 0; });
                if (num > 0)
                    .errorSupplemental(loc, "... (%d more, -v to show) ...", num);
                return 1;   // stop iterating
            });
        }
        else if (od)
        {
            .error(loc, "none of the overloads of '%s' are callable using argument types !(%s)%s",
                od.ident.toChars(), tiargsBuf.peekString(), fargsBuf.peekString());
        }
        else
        {
            assert(fd);

            bool hasOverloads = fd.overnext !is null;
            auto tf = cast(TypeFunction)fd.type;
            if (tthis && !MODimplicitConv(tthis.mod, tf.mod)) // modifier mismatch
            {
                OutBuffer thisBuf, funcBuf;
                MODMatchToBuffer(&thisBuf, tthis.mod, tf.mod);
                MODMatchToBuffer(&funcBuf, tf.mod, tthis.mod);
                if (hasOverloads)
                {
                    .error(loc, "none of the overloads of '%s' are callable using a %sobject, candidates are:",
                        fd.ident.toChars(), thisBuf.peekString());
                }
                else
                {
                    .error(loc, "%smethod %s is not callable using a %sobject",
                        funcBuf.peekString(), fd.toPrettyChars(),
                        thisBuf.peekString());
                }
            }
            else
            {
                //printf("tf = %s, args = %s\n", tf.deco, (*fargs)[0].type.deco);
                if (hasOverloads)
                {
                    .error(loc, "none of the overloads of '%s' are callable using argument types %s, candidates are:",
                        fd.ident.toChars(), fargsBuf.peekString());
                }
                else
                {
                    fd.error(loc, "%s%s is not callable using argument types %s",
                        parametersTypeToChars(tf.parameters, tf.varargs),
                        tf.modToChars(), fargsBuf.peekString());
                }
            }

            // Display candidate functions
            int numToDisplay = numOverloadsDisplay;
            overloadApply(hasOverloads ? fd : null, (Dsymbol s)
            {
                auto fd = s.isFuncDeclaration();
                auto td = s.isTemplateDeclaration();
                if (fd)
                {
                    if (fd.errors || fd.type.ty == Terror)
                        return 0;

                    auto tf = cast(TypeFunction)fd.type;
                    .errorSupplemental(fd.loc, "%s%s", fd.toPrettyChars(),
                        parametersTypeToChars(tf.parameters, tf.varargs));
                }
                else
                {
                    .errorSupplemental(td.loc, "%s", td.toPrettyChars());
                }

                if (global.params.verbose || --numToDisplay != 0 || !fd)
                    return 0;

                // Too many overloads to sensibly display.
                int num = 0;
                overloadApply(fd.overnext, (s){ ++num; return 0; });
                if (num > 0)
                    .errorSupplemental(loc, "... (%d more, -v to show) ...", num);
                return 1;   // stop iterating
            });
        }
    }
    else if (m.nextf)
    {
        TypeFunction tf1 = cast(TypeFunction)m.lastf.type;
        TypeFunction tf2 = cast(TypeFunction)m.nextf.type;
        const(char)* lastprms = parametersTypeToChars(tf1.parameters, tf1.varargs);
        const(char)* nextprms = parametersTypeToChars(tf2.parameters, tf2.varargs);
        .error(loc, "%s.%s called with argument types %s matches both:\n%s:     %s%s\nand:\n%s:     %s%s",
            s.parent.toPrettyChars(), s.ident.toChars(),
            fargsBuf.peekString(),
            m.lastf.loc.toChars(), m.lastf.toPrettyChars(), lastprms,
            m.nextf.loc.toChars(), m.nextf.toPrettyChars(), nextprms);
    }
    return null;
}

/**************************************
 * Returns an indirect type one step from t.
 */
extern (C++) Type getIndirection(Type t)
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
 * Returns true if memory reachable through a reference B to a value of type tb,
 * which has been constructed with a reference A to a value of type ta
 * available, can alias memory reachable from A based on the types involved
 * (either directly or via any number of indirections).
 *
 * Note that this relation is not symmetric in the two arguments. For example,
 * a const(int) reference can point to a pre-existing int, but not the other
 * way round.
 */
extern (C++) bool traverseIndirections(Type ta, Type tb, void* p = null, bool reversePass = false)
{
    Type source = ta;
    Type target = tb;
    if (reversePass)
    {
        source = tb;
        target = ta;
    }

    if (source.constConv(target))
        return true;
    else if (target.ty == Tvoid && MODimplicitConv(source.mod, target.mod))
        return true;

    // No direct match, so try breaking up one of the types (starting with tb).
    Type tbb = tb.toBasetype().baseElemOf();
    if (tbb != tb)
        return traverseIndirections(ta, tbb, p, reversePass);

    // context date to detect circular look up
    struct Ctxt
    {
        Ctxt* prev;
        Type type;
    }

    Ctxt* ctxt = cast(Ctxt*)p;
    if (tb.ty == Tclass || tb.ty == Tstruct)
    {
        for (Ctxt* c = ctxt; c; c = c.prev)
            if (tb == c.type)
                return false;
        Ctxt c;
        c.prev = ctxt;
        c.type = tb;

        AggregateDeclaration sym = tb.toDsymbol(null).isAggregateDeclaration();
        for (size_t i = 0; i < sym.fields.dim; i++)
        {
            VarDeclaration v = sym.fields[i];
            Type tprmi = v.type.addMod(tb.mod);
            //printf("\ttb = %s, tprmi = %s\n", tb.toChars(), tprmi.toChars());
            if (traverseIndirections(ta, tprmi, &c, reversePass))
                return true;
        }
    }
    else if (tb.ty == Tarray || tb.ty == Taarray || tb.ty == Tpointer)
    {
        Type tind = tb.nextOf();
        if (traverseIndirections(ta, tind, ctxt, reversePass))
            return true;
    }
    else if (tb.hasPointers())
    {
        // FIXME: function pointer/delegate types should be considered.
        return true;
    }

    // Still no match, so try breaking up ta if we have note done so yet.
    if (!reversePass)
        return traverseIndirections(tb, ta, ctxt, true);

    return false;
}

/* For all functions between outerFunc and f, mark them as needing
 * a closure.
 */
extern (C++) void markAsNeedingClosure(Dsymbol f, FuncDeclaration outerFunc)
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

/* Given a nested function f inside a function outerFunc, check
 * if any sibling callers of f have escaped. If so, mark
 * all the enclosing functions as needing closures.
 * Return true if any closures were detected.
 * This is recursive: we need to check the callers of our siblings.
 * Note that nested functions can only call lexically earlier nested
 * functions, so loops are impossible.
 */
extern (C++) bool checkEscapingSiblings(FuncDeclaration f, FuncDeclaration outerFunc, void* p = null)
{
    struct PrevSibling
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

    override FuncDeclaration toAliasFunc()
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
    TOK tok;        // TOKfunction or TOKdelegate
    Type treq;      // target of return type inference

    // backend
    bool deferToObj;

    extern (D) this(Loc loc, Loc endloc, Type type, TOK tok, ForeachStatement fes, Identifier id = null)
    {
        super(loc, endloc, null, STCundefined, type);
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

    override bool isNested()
    {
        //printf("FuncLiteralDeclaration::isNested() '%s'\n", toChars());
        return (tok != TOKfunction) && !isThis();
    }

    override AggregateDeclaration isThis()
    {
        return tok == TOKdelegate ? super.isThis() : null;
    }

    override bool isVirtual()
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
        import ddmd.statement_rewrite_walker;

        extern (C++) final class RetWalker : StatementRewriteWalker
        {
            alias visit = super.visit;
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

        if (semanticRun < PASSsemantic3done)
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
            (cast(TypeFunction)type).next = tret;
    }

    override inout(FuncLiteralDeclaration) isFuncLiteralDeclaration() inout
    {
        return this;
    }

    override const(char)* kind() const
    {
        // GCC requires the (char*) casts
        return (tok != TOKfunction) ? cast(char*)"delegate" : cast(char*)"function";
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
    extern (D) this(Loc loc, Loc endloc, StorageClass stc, Type type)
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

    override void semantic(Scope* sc)
    {
        //printf("CtorDeclaration::semantic() %s\n", toChars());
        if (semanticRun >= PASSsemanticdone)
            return;
        if (_scope)
        {
            sc = _scope;
            _scope = null;
        }

        parent = sc.parent;
        Dsymbol p = toParent2();
        AggregateDeclaration ad = p.isAggregateDeclaration();
        if (!ad)
        {
            .error(loc, "constructor can only be a member of aggregate, not %s %s", p.kind(), p.toChars());
            type = Type.terror;
            errors = true;
            return;
        }

        sc = sc.push();
        sc.stc &= ~STCstatic; // not a static constructor
        sc.flags |= SCOPEctor;

        FuncDeclaration.semantic(sc);

        sc.pop();

        if (errors)
            return;

        TypeFunction tf = cast(TypeFunction)type;
        assert(tf && tf.ty == Tfunction);

        /* See if it's the default constructor
         * But, template constructor should not become a default constructor.
         */
        if (ad && (!parent.isTemplateInstance() || parent.isTemplateMixin()))
        {
            immutable dim = Parameter.dim(tf.parameters);

            if (auto sd = ad.isStructDeclaration())
            {
                if (dim == 0 && tf.varargs == 0) // empty default ctor w/o any varargs
                {
                    if (fbody || !(storage_class & STCdisable))
                    {
                        error("default constructor for structs only allowed " ~
                            "with @disable, no body, and no parameters");
                        storage_class |= STCdisable;
                        fbody = null;
                    }
                    sd.noDefaultCtor = true;
                }
                else if (dim == 0 && tf.varargs) // allow varargs only ctor
                {
                }
                else if (dim && Parameter.getNth(tf.parameters, 0).defaultArg)
                {
                    // if the first parameter has a default argument, then the rest does as well
                    if (storage_class & STCdisable)
                    {
                        deprecation("@disable'd constructor cannot have default "~
                                    "arguments for all parameters.");
                        deprecationSupplemental(loc, "Use @disable this(); if you want to disable default initialization.");
                    }
                    else
                        deprecation("all parameters have default arguments, "~
                                    "but structs cannot have default constructors.");
                }
            }
            else if (dim == 0 && tf.varargs == 0)
            {
                ad.defaultCtor = this;
            }
        }
    }

    override const(char)* kind() const
    {
        return "constructor";
    }

    override const(char)* toChars() const
    {
        return "this";
    }

    override bool isVirtual()
    {
        return false;
    }

    override bool addPreInvariant()
    {
        return false;
    }

    override bool addPostInvariant()
    {
        return (isThis() && vthis && global.params.useInvariants);
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
    extern (D) this(Loc loc, Loc endloc, StorageClass stc, Identifier id)
    {
        super(loc, endloc, id, stc, null);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto dd = new PostBlitDeclaration(loc, endloc, storage_class, ident);
        return FuncDeclaration.syntaxCopy(dd);
    }

    override void semantic(Scope* sc)
    {
        //printf("PostBlitDeclaration::semantic() %s\n", toChars());
        //printf("ident: %s, %s, %p, %p\n", ident.toChars(), Id::dtor.toChars(), ident, Id::dtor);
        //printf("stc = x%llx\n", sc.stc);
        if (semanticRun >= PASSsemanticdone)
            return;
        if (_scope)
        {
            sc = _scope;
            _scope = null;
        }

        parent = sc.parent;
        Dsymbol p = toParent2();
        StructDeclaration ad = p.isStructDeclaration();
        if (!ad)
        {
            .error(loc, "postblit can only be a member of struct/union, not %s %s", p.kind(), p.toChars());
            type = Type.terror;
            errors = true;
            return;
        }
        if (ident == Id.postblit && semanticRun < PASSsemantic)
            ad.postblits.push(this);
        if (!type)
            type = new TypeFunction(null, Type.tvoid, false, LINKd, storage_class);

        sc = sc.push();
        sc.stc &= ~STCstatic; // not static
        sc.linkage = LINKd;

        FuncDeclaration.semantic(sc);

        sc.pop();
    }

    override bool isVirtual()
    {
        return false;
    }

    override bool addPreInvariant()
    {
        return false;
    }

    override bool addPostInvariant()
    {
        return (isThis() && vthis && global.params.useInvariants);
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
    extern (D) this(Loc loc, Loc endloc)
    {
        super(loc, endloc, Id.dtor, STCundefined, null);
    }

    extern (D) this(Loc loc, Loc endloc, StorageClass stc, Identifier id)
    {
        super(loc, endloc, id, stc, null);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto dd = new DtorDeclaration(loc, endloc, storage_class, ident);
        return FuncDeclaration.syntaxCopy(dd);
    }

    override void semantic(Scope* sc)
    {
        //printf("DtorDeclaration::semantic() %s\n", toChars());
        //printf("ident: %s, %s, %p, %p\n", ident.toChars(), Id::dtor.toChars(), ident, Id::dtor);
        if (semanticRun >= PASSsemanticdone)
            return;
        if (_scope)
        {
            sc = _scope;
            _scope = null;
        }

        parent = sc.parent;
        Dsymbol p = toParent2();
        AggregateDeclaration ad = p.isAggregateDeclaration();
        if (!ad)
        {
            .error(loc, "destructor can only be a member of aggregate, not %s %s", p.kind(), p.toChars());
            type = Type.terror;
            errors = true;
            return;
        }
        if (ident == Id.dtor && semanticRun < PASSsemantic)
            ad.dtors.push(this);
        if (!type)
            type = new TypeFunction(null, Type.tvoid, false, LINKd, storage_class);

        sc = sc.push();
        sc.stc &= ~STCstatic; // not a static destructor
        if (sc.linkage != LINKcpp)
            sc.linkage = LINKd;

        FuncDeclaration.semantic(sc);

        sc.pop();
    }

    override const(char)* kind() const
    {
        return "destructor";
    }

    override const(char)* toChars() const
    {
        return "~this";
    }

    override bool isVirtual()
    {
        // false so that dtor's don't get put into the vtbl[]
        return false;
    }

    override bool addPreInvariant()
    {
        return (isThis() && vthis && global.params.useInvariants);
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
    final extern (D) this(Loc loc, Loc endloc, StorageClass stc)
    {
        super(loc, endloc, Identifier.generateId("_staticCtor"), STCstatic | stc, null);
    }

    final extern (D) this(Loc loc, Loc endloc, const(char)* name, StorageClass stc)
    {
        super(loc, endloc, Identifier.generateId(name), STCstatic | stc, null);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto scd = new StaticCtorDeclaration(loc, endloc, storage_class);
        return FuncDeclaration.syntaxCopy(scd);
    }

    override final void semantic(Scope* sc)
    {
        //printf("StaticCtorDeclaration::semantic()\n");
        if (semanticRun >= PASSsemanticdone)
            return;
        if (_scope)
        {
            sc = _scope;
            _scope = null;
        }

        parent = sc.parent;
        Dsymbol p = parent.pastMixin();
        if (!p.isScopeDsymbol())
        {
            const(char)* s = (isSharedStaticCtorDeclaration() ? "shared " : "");
            .error(loc, "%sstatic constructor can only be member of module/aggregate/template, not %s %s", s, p.kind(), p.toChars());
            type = Type.terror;
            errors = true;
            return;
        }
        if (!type)
            type = new TypeFunction(null, Type.tvoid, false, LINKd, storage_class);

        /* If the static ctor appears within a template instantiation,
         * it could get called multiple times by the module constructors
         * for different modules. Thus, protect it with a gate.
         */
        if (isInstantiated() && semanticRun < PASSsemantic)
        {
            /* Add this prefix to the function:
             *      static int gate;
             *      if (++gate != 1) return;
             * Note that this is not thread safe; should not have threads
             * during static construction.
             */
            auto v = new VarDeclaration(Loc(), Type.tint32, Id.gate, null);
            v.storage_class = STCtemp | (isSharedStaticCtorDeclaration() ? STCstatic : STCtls);

            auto sa = new Statements();
            Statement s = new ExpStatement(Loc(), v);
            sa.push(s);

            Expression e = new IdentifierExp(Loc(), v.ident);
            e = new AddAssignExp(Loc(), e, new IntegerExp(1));
            e = new EqualExp(TOKnotequal, Loc(), e, new IntegerExp(1));
            s = new IfStatement(Loc(), null, e, new ReturnStatement(Loc(), null), null, Loc());

            sa.push(s);
            if (fbody)
                sa.push(fbody);

            fbody = new CompoundStatement(Loc(), sa);
        }

        FuncDeclaration.semantic(sc);

        // We're going to need ModuleInfo
        Module m = getModule();
        if (!m)
            m = sc._module;
        if (m)
        {
            m.needmoduleinfo = 1;
            //printf("module1 %s needs moduleinfo\n", m.toChars());
        }
    }

    override final AggregateDeclaration isThis()
    {
        return null;
    }

    override final bool isVirtual()
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
    extern (D) this(Loc loc, Loc endloc, StorageClass stc)
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

    final extern (D) this(Loc loc, Loc endloc, StorageClass stc)
    {
        super(loc, endloc, Identifier.generateId("_staticDtor"), STCstatic | stc, null);
    }

    final extern (D) this(Loc loc, Loc endloc, const(char)* name, StorageClass stc)
    {
        super(loc, endloc, Identifier.generateId(name), STCstatic | stc, null);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto sdd = new StaticDtorDeclaration(loc, endloc, storage_class);
        return FuncDeclaration.syntaxCopy(sdd);
    }

    override final void semantic(Scope* sc)
    {
        if (semanticRun >= PASSsemanticdone)
            return;
        if (_scope)
        {
            sc = _scope;
            _scope = null;
        }

        parent = sc.parent;
        Dsymbol p = parent.pastMixin();
        if (!p.isScopeDsymbol())
        {
            const(char)* s = (isSharedStaticDtorDeclaration() ? "shared " : "");
            .error(loc, "%sstatic destructor can only be member of module/aggregate/template, not %s %s", s, p.kind(), p.toChars());
            type = Type.terror;
            errors = true;
            return;
        }
        if (!type)
            type = new TypeFunction(null, Type.tvoid, false, LINKd, storage_class);

        /* If the static ctor appears within a template instantiation,
         * it could get called multiple times by the module constructors
         * for different modules. Thus, protect it with a gate.
         */
        if (isInstantiated() && semanticRun < PASSsemantic)
        {
            /* Add this prefix to the function:
             *      static int gate;
             *      if (--gate != 0) return;
             * Increment gate during constructor execution.
             * Note that this is not thread safe; should not have threads
             * during static destruction.
             */
            auto v = new VarDeclaration(Loc(), Type.tint32, Id.gate, null);
            v.storage_class = STCtemp | (isSharedStaticDtorDeclaration() ? STCstatic : STCtls);

            auto sa = new Statements();
            Statement s = new ExpStatement(Loc(), v);
            sa.push(s);

            Expression e = new IdentifierExp(Loc(), v.ident);
            e = new AddAssignExp(Loc(), e, new IntegerExp(-1));
            e = new EqualExp(TOKnotequal, Loc(), e, new IntegerExp(0));
            s = new IfStatement(Loc(), null, e, new ReturnStatement(Loc(), null), null, Loc());

            sa.push(s);
            if (fbody)
                sa.push(fbody);

            fbody = new CompoundStatement(Loc(), sa);

            vgate = v;
        }

        FuncDeclaration.semantic(sc);

        // We're going to need ModuleInfo
        Module m = getModule();
        if (!m)
            m = sc._module;
        if (m)
        {
            m.needmoduleinfo = 1;
            //printf("module2 %s needs moduleinfo\n", m.toChars());
        }
    }

    override final AggregateDeclaration isThis()
    {
        return null;
    }

    override final bool isVirtual()
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
    extern (D) this(Loc loc, Loc endloc, StorageClass stc)
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
    extern (D) this(Loc loc, Loc endloc, StorageClass stc, Identifier id, Statement fbody)
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

    override void semantic(Scope* sc)
    {
        if (semanticRun >= PASSsemanticdone)
            return;
        if (_scope)
        {
            sc = _scope;
            _scope = null;
        }

        parent = sc.parent;
        Dsymbol p = parent.pastMixin();
        AggregateDeclaration ad = p.isAggregateDeclaration();
        if (!ad)
        {
            .error(loc, "invariant can only be a member of aggregate, not %s %s", p.kind(), p.toChars());
            type = Type.terror;
            errors = true;
            return;
        }
        if (ident != Id.classInvariant &&
             semanticRun < PASSsemantic &&
             !ad.isUnionDeclaration()           // users are on their own with union fields
           )
            ad.invs.push(this);
        if (!type)
            type = new TypeFunction(null, Type.tvoid, false, LINKd, storage_class);

        sc = sc.push();
        sc.stc &= ~STCstatic; // not a static invariant
        sc.stc |= STCconst; // invariant() is always const
        sc.flags = (sc.flags & ~SCOPEcontract) | SCOPEinvariant;
        sc.linkage = LINKd;

        FuncDeclaration.semantic(sc);

        sc.pop();
    }

    override bool isVirtual()
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
 * Generate unique unittest function Id so we can have multiple
 * instances per module.
 */
extern (C++) static Identifier unitTestId(Loc loc)
{
    OutBuffer buf;
    buf.printf("__unittestL%u_", loc.linnum);
    return Identifier.generateId(buf.peekString());
}

/***********************************************************
 */
extern (C++) final class UnitTestDeclaration : FuncDeclaration
{
    char* codedoc;      // for documented unittest

    // toObjFile() these nested functions after this one
    FuncDeclarations deferredNested;

    extern (D) this(Loc loc, Loc endloc, StorageClass stc, char* codedoc)
    {
        super(loc, endloc, unitTestId(loc), stc, null);
        this.codedoc = codedoc;
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto utd = new UnitTestDeclaration(loc, endloc, storage_class, codedoc);
        return FuncDeclaration.syntaxCopy(utd);
    }

    override void semantic(Scope* sc)
    {
        if (semanticRun >= PASSsemanticdone)
            return;
        if (_scope)
        {
            sc = _scope;
            _scope = null;
        }

        protection = sc.protection;

        parent = sc.parent;
        Dsymbol p = parent.pastMixin();
        if (!p.isScopeDsymbol())
        {
            .error(loc, "unittest can only be a member of module/aggregate/template, not %s %s", p.kind(), p.toChars());
            type = Type.terror;
            errors = true;
            return;
        }

        if (global.params.useUnitTests)
        {
            if (!type)
                type = new TypeFunction(null, Type.tvoid, false, LINKd, storage_class);
            Scope* sc2 = sc.push();
            sc2.linkage = LINKd;
            FuncDeclaration.semantic(sc2);
            sc2.pop();
        }

        version (none)
        {
            // We're going to need ModuleInfo even if the unit tests are not
            // compiled in, because other modules may import this module and refer
            // to this ModuleInfo.
            // (This doesn't make sense to me?)
            Module m = getModule();
            if (!m)
                m = sc._module;
            if (m)
            {
                //printf("module3 %s needs moduleinfo\n", m.toChars());
                m.needmoduleinfo = 1;
            }
        }
    }

    override AggregateDeclaration isThis()
    {
        return null;
    }

    override bool isVirtual()
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
    int varargs;

    extern (D) this(Loc loc, Loc endloc, StorageClass stc, Parameters* fparams, int varargs)
    {
        super(loc, endloc, Id.classNew, STCstatic | stc, null);
        this.parameters = fparams;
        this.varargs = varargs;
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto f = new NewDeclaration(loc, endloc, storage_class, Parameter.arraySyntaxCopy(parameters), varargs);
        return FuncDeclaration.syntaxCopy(f);
    }

    override void semantic(Scope* sc)
    {
        //printf("NewDeclaration::semantic()\n");
        if (semanticRun >= PASSsemanticdone)
            return;
        if (_scope)
        {
            sc = _scope;
            _scope = null;
        }

        parent = sc.parent;
        Dsymbol p = parent.pastMixin();
        if (!p.isAggregateDeclaration())
        {
            .error(loc, "allocator can only be a member of aggregate, not %s %s", p.kind(), p.toChars());
            type = Type.terror;
            errors = true;
            return;
        }
        Type tret = Type.tvoid.pointerTo();
        if (!type)
            type = new TypeFunction(parameters, tret, varargs, LINKd, storage_class);

        type = type.semantic(loc, sc);
        assert(type.ty == Tfunction);

        // Check that there is at least one argument of type size_t
        TypeFunction tf = cast(TypeFunction)type;
        if (Parameter.dim(tf.parameters) < 1)
        {
            error("at least one argument of type size_t expected");
        }
        else
        {
            Parameter fparam = Parameter.getNth(tf.parameters, 0);
            if (!fparam.type.equals(Type.tsize_t))
                error("first argument must be type size_t, not %s", fparam.type.toChars());
        }

        FuncDeclaration.semantic(sc);
    }

    override const(char)* kind() const
    {
        return "allocator";
    }

    override bool isVirtual()
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

    extern (D) this(Loc loc, Loc endloc, StorageClass stc, Parameters* fparams)
    {
        super(loc, endloc, Id.classDelete, STCstatic | stc, null);
        this.parameters = fparams;
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto f = new DeleteDeclaration(loc, endloc, storage_class, Parameter.arraySyntaxCopy(parameters));
        return FuncDeclaration.syntaxCopy(f);
    }

    override void semantic(Scope* sc)
    {
        //printf("DeleteDeclaration::semantic()\n");
        if (semanticRun >= PASSsemanticdone)
            return;
        if (_scope)
        {
            sc = _scope;
            _scope = null;
        }

        parent = sc.parent;
        Dsymbol p = parent.pastMixin();
        if (!p.isAggregateDeclaration())
        {
            .error(loc, "deallocator can only be a member of aggregate, not %s %s", p.kind(), p.toChars());
            type = Type.terror;
            errors = true;
            return;
        }
        if (!type)
            type = new TypeFunction(parameters, Type.tvoid, 0, LINKd, storage_class);

        type = type.semantic(loc, sc);
        assert(type.ty == Tfunction);

        // Check that there is only one argument of type void*
        TypeFunction tf = cast(TypeFunction)type;
        if (Parameter.dim(tf.parameters) != 1)
        {
            error("one argument of type void* expected");
        }
        else
        {
            Parameter fparam = Parameter.getNth(tf.parameters, 0);
            if (!fparam.type.equals(Type.tvoid.pointerTo()))
                error("one argument of type void* expected, not %s", fparam.type.toChars());
        }

        FuncDeclaration.semantic(sc);
    }

    override const(char)* kind() const
    {
        return "deallocator";
    }

    override bool isDelete()
    {
        return true;
    }

    override bool isVirtual()
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
