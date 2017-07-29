/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (c) 1999-2017 by Digital Mars, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(DMDSRC _dsymbol.d)
 */

module ddmd.dsymbol;

import core.stdc.stdarg;
import core.stdc.stdio;
import core.stdc.string;
import core.stdc.stdlib;

import ddmd.aggregate;
import ddmd.aliasthis;
import ddmd.arraytypes;
import ddmd.attrib;
import ddmd.gluelayer;
import ddmd.dclass;
import ddmd.declaration;
import ddmd.denum;
import ddmd.dimport;
import ddmd.dmodule;
import ddmd.dscope;
import ddmd.dstruct;
import ddmd.dtemplate;
import ddmd.errors;
import ddmd.expression;
import ddmd.func;
import ddmd.globals;
import ddmd.id;
import ddmd.identifier;
import ddmd.init;
import ddmd.lexer;
import ddmd.mtype;
import ddmd.nspace;
import ddmd.opover;
import ddmd.root.aav;
import ddmd.root.rmem;
import ddmd.root.rootobject;
import ddmd.root.speller;
import ddmd.statement;
import ddmd.tokens;
import ddmd.visitor;

version(IN_LLVM)
{
    // Functions to construct/destruct Dsymbol.ir
    extern (C++) void* newIrDsymbol();
    extern (C++) void deleteIrDsymbol(void*);
}

struct Ungag
{
    uint oldgag;

    extern (D) this(uint old)
    {
        this.oldgag = old;
    }

    extern (C++) ~this()
    {
        global.gag = oldgag;
    }
}

enum PROTKIND : int
{
    PROTundefined,
    PROTnone,           // no access
    PROTprivate,
    PROTpackage,
    PROTprotected,
    PROTpublic,
    PROTexport,
}

alias PROTundefined = PROTKIND.PROTundefined;
alias PROTnone = PROTKIND.PROTnone;
alias PROTprivate = PROTKIND.PROTprivate;
alias PROTpackage = PROTKIND.PROTpackage;
alias PROTprotected = PROTKIND.PROTprotected;
alias PROTpublic = PROTKIND.PROTpublic;
alias PROTexport = PROTKIND.PROTexport;

struct Prot
{
    PROTKIND kind;
    Package pkg;

    extern (D) this(PROTKIND kind)
    {
        this.kind = kind;
    }

    extern (C++):

    /**
     * Checks if `this` is superset of `other` restrictions.
     * For example, "protected" is more restrictive than "public".
     */
    bool isMoreRestrictiveThan(const Prot other) const
    {
        return this.kind < other.kind;
    }

    /**
     * Checks if `this` is absolutely identical protection attribute to `other`
     */
    bool opEquals(ref const Prot other) const
    {
        if (this.kind == other.kind)
        {
            if (this.kind == PROTpackage)
                return this.pkg == other.pkg;
            return true;
        }
        return false;
    }

    /**
     * Checks if parent defines different access restrictions than this one.
     *
     * Params:
     *  parent = protection attribute for scope that hosts this one
     *
     * Returns:
     *  'true' if parent is already more restrictive than this one and thus
     *  no differentiation is needed.
     */
    bool isSubsetOf(ref const Prot parent) const
    {
        if (this.kind != parent.kind)
            return false;
        if (this.kind == PROTpackage)
        {
            if (!this.pkg)
                return true;
            if (!parent.pkg)
                return false;
            if (parent.pkg.isAncestorPackageOf(this.pkg))
                return true;
        }
        return true;
    }
}

enum PASS : int
{
    PASSinit,           // initial state
    PASSsemantic,       // semantic() started
    PASSsemanticdone,   // semantic() done
    PASSsemantic2,      // semantic2() started
    PASSsemantic2done,  // semantic2() done
    PASSsemantic3,      // semantic3() started
    PASSsemantic3done,  // semantic3() done
    PASSinline,         // inline started
    PASSinlinedone,     // inline done
    PASSobj,            // toObjFile() run
}

alias PASSinit = PASS.PASSinit;
alias PASSsemantic = PASS.PASSsemantic;
alias PASSsemanticdone = PASS.PASSsemanticdone;
alias PASSsemantic2 = PASS.PASSsemantic2;
alias PASSsemantic2done = PASS.PASSsemantic2done;
alias PASSsemantic3 = PASS.PASSsemantic3;
alias PASSsemantic3done = PASS.PASSsemantic3done;
alias PASSinline = PASS.PASSinline;
alias PASSinlinedone = PASS.PASSinlinedone;
alias PASSobj = PASS.PASSobj;

// Search options
enum : int
{
    IgnoreNone              = 0x00, // default
    IgnorePrivateImports    = 0x01, // don't search private imports
    IgnoreErrors            = 0x02, // don't give error messages
    IgnoreAmbiguous         = 0x04, // return NULL if ambiguous
    SearchLocalsOnly        = 0x08, // only look at locals (don't search imports)
    SearchImportsOnly       = 0x10, // only look in imports
    SearchUnqualifiedModule = 0x20, // the module scope search is unqualified,
                                    // meaning don't search imports in that scope,
                                    // because qualified module searches search
                                    // their imports
    IgnoreSymbolVisibility  = 0x80, // also find private and package protected symbols
}

extern (C++) alias Dsymbol_apply_ft_t = int function(Dsymbol, void*);

/***********************************************************
 */
extern (C++) class Dsymbol : RootObject
{
    Identifier ident;
    Dsymbol parent;
    Symbol* csym;           // symbol for code generator
    Symbol* isym;           // import version of csym
    const(char)* comment;   // documentation comment for this Dsymbol
    Loc loc;                // where defined
    Scope* _scope;          // !=null means context to use for semantic()
    const(char)* prettystring;  // cached value of toPrettyChars()
    bool errors;            // this symbol failed to pass semantic()
    PASS semanticRun;

    DeprecatedDeclaration depdecl;           // customized deprecation message
    UserAttributeDeclaration userAttribDecl;    // user defined attributes

    // !=null means there's a ddoc unittest associated with this symbol
    // (only use this with ddoc)
    UnitTestDeclaration ddocUnittest;

    version(IN_LLVM)
    {
        // llvm stuff
        uint llvmInternal;

        void* ir; // IrDsymbol*
    }

    final extern (D) this()
    {
        //printf("Dsymbol::Dsymbol(%p)\n", this);
        this.semanticRun = PASSinit;
        version(IN_LLVM)
        {
            this.ir = newIrDsymbol();
        }
    }

    final extern (D) this(Identifier ident)
    {
        //printf("Dsymbol::Dsymbol(%p, ident)\n", this);
        this.ident = ident;
        this.semanticRun = PASSinit;
        version(IN_LLVM)
        {
            this.ir = newIrDsymbol();
        }
    }

    version(IN_LLVM)
    {
        extern (D) ~this()
        {
            deleteIrDsymbol(this.ir);
            this.ir = null;
        }
    }

    static Dsymbol create(Identifier ident)
    {
        return new Dsymbol(ident);
    }

    override const(char)* toChars()
    {
        return ident ? ident.toChars() : "__anonymous";
    }

    // helper to print fully qualified (template) arguments
    const(char)* toPrettyCharsHelper()
    {
        return toChars();
    }

    final ref Loc getLoc()
    {
        if (!loc.filename) // avoid bug 5861.
        {
            auto m = getModule();
            if (m && m.srcfile)
                loc.filename = m.srcfile.toChars();
        }
        return loc;
    }

    final const(char)* locToChars()
    {
        return getLoc().toChars();
    }

    override bool equals(RootObject o)
    {
        if (this == o)
            return true;
        Dsymbol s = cast(Dsymbol)o;
        // Overload sets don't have an ident
        if (s && ident && s.ident && ident.equals(s.ident))
            return true;
        return false;
    }

    final bool isAnonymous()
    {
        return ident is null;
    }

    final void error(Loc loc, const(char)* format, ...)
    {
        va_list ap;
        va_start(ap, format);
        .verror(loc, format, ap, kind(), toPrettyChars());
        va_end(ap);
    }

    final void error(const(char)* format, ...)
    {
        va_list ap;
        va_start(ap, format);
        .verror(getLoc(), format, ap, kind(), toPrettyChars());
        va_end(ap);
    }

    final void deprecation(Loc loc, const(char)* format, ...)
    {
        va_list ap;
        va_start(ap, format);
        .vdeprecation(loc, format, ap, kind(), toPrettyChars());
        va_end(ap);
    }

    final void deprecation(const(char)* format, ...)
    {
        va_list ap;
        va_start(ap, format);
        .vdeprecation(getLoc(), format, ap, kind(), toPrettyChars());
        va_end(ap);
    }

    final void checkDeprecated(Loc loc, Scope* sc)
    {
        if (global.params.useDeprecated != 1 && isDeprecated())
        {
            // Don't complain if we're inside a deprecated symbol's scope
            for (Dsymbol sp = sc.parent; sp; sp = sp.parent)
            {
                if (sp.isDeprecated())
                    goto L1;
            }
            for (Scope* sc2 = sc; sc2; sc2 = sc2.enclosing)
            {
                if (sc2.scopesym && sc2.scopesym.isDeprecated())
                    goto L1;
                // If inside a StorageClassDeclaration that is deprecated
                if (sc2.stc & STCdeprecated)
                    goto L1;
            }
            const(char)* message = null;
            for (Dsymbol p = this; p; p = p.parent)
            {
                message = p.depdecl ? p.depdecl.getMessage() : null;
                if (message)
                    break;
            }
            if (message)
                deprecation(loc, "is deprecated - %s", message);
            else
                deprecation(loc, "is deprecated");
        }
    L1:
        Declaration d = isDeclaration();
        if (d && d.storage_class & STCdisable)
        {
            if (!(sc.func && sc.func.storage_class & STCdisable))
            {
                if (d.toParent() && d.isPostBlitDeclaration())
                    d.toParent().error(loc, "is not copyable because it is annotated with @disable");
                else
                    error(loc, "is not callable because it is annotated with @disable");
            }
        }
    }

    /**********************************
     * Determine which Module a Dsymbol is in.
     */
    final Module getModule()
    {
        //printf("Dsymbol::getModule()\n");
        if (TemplateInstance ti = isInstantiated())
            return ti.tempdecl.getModule();
        Dsymbol s = this;
        while (s)
        {
            //printf("\ts = %s '%s'\n", s.kind(), s.toPrettyChars());
            Module m = s.isModule();
            if (m)
                return m;
            s = s.parent;
        }
        return null;
    }

    /**********************************
     * Determine which Module a Dsymbol is in, as far as access rights go.
     */
    final Module getAccessModule()
    {
        //printf("Dsymbol::getAccessModule()\n");
        if (TemplateInstance ti = isInstantiated())
            return ti.tempdecl.getAccessModule();
        Dsymbol s = this;
        while (s)
        {
            //printf("\ts = %s '%s'\n", s.kind(), s.toPrettyChars());
            Module m = s.isModule();
            if (m)
                return m;
            TemplateInstance ti = s.isTemplateInstance();
            if (ti && ti.enclosing)
            {
                /* Because of local template instantiation, the parent isn't where the access
                 * rights come from - it's the template declaration
                 */
                s = ti.tempdecl;
            }
            else
                s = s.parent;
        }
        return null;
    }

    final inout(Dsymbol) pastMixin() inout
    {
        //printf("Dsymbol::pastMixin() %s\n", toChars());
        if (!isTemplateMixin())
            return this;
        if (!parent)
            return null;
        return parent.pastMixin();
    }

    /**********************************
     * `parent` field returns a lexically enclosing scope symbol this is a member of.
     *
     * `toParent()` returns a logically enclosing scope symbol this is a member of.
     * It skips over TemplateMixin's.
     *
     * `toParent2()` returns an enclosing scope symbol this is living at runtime.
     * It skips over both TemplateInstance's and TemplateMixin's.
     * It's used when looking for the 'this' pointer of the enclosing function/class.
     *
     * Examples:
     *  module mod;
     *  template Foo(alias a) { mixin Bar!(); }
     *  mixin template Bar() {
     *    public {  // ProtDeclaration
     *      void baz() { a = 2; }
     *    }
     *  }
     *  void test() {
     *    int v = 1;
     *    alias foo = Foo!(v);
     *    foo.baz();
     *    assert(v == 2);
     *  }
     *
     *  // s == FuncDeclaration('mod.test.Foo!().Bar!().baz()')
     *  // s.parent == TemplateMixin('mod.test.Foo!().Bar!()')
     *  // s.toParent() == TemplateInstance('mod.test.Foo!()')
     *  // s.toParent2() == FuncDeclaration('mod.test')
     */
    final inout(Dsymbol) toParent() inout
    {
        return parent ? parent.pastMixin() : null;
    }

    /// ditto
    final inout(Dsymbol) toParent2() inout
    {
        if (!parent || !parent.isTemplateInstance)
            return parent;
        return parent.toParent2;
    }

    final inout(TemplateInstance) isInstantiated() inout
    {
        if (!parent)
            return null;
        auto ti = parent.isTemplateInstance();
        if (ti && !ti.isTemplateMixin())
            return ti;
        return parent.isInstantiated();
    }

    // Check if this function is a member of a template which has only been
    // instantiated speculatively, eg from inside is(typeof()).
    // Return the speculative template instance it is part of,
    // or NULL if not speculative.
    final inout(TemplateInstance) isSpeculative() inout
    {
        if (!parent)
            return null;
        auto ti = parent.isTemplateInstance();
        if (ti && ti.gagged)
            return ti;
        if (!parent.toParent())
            return null;
        return parent.isSpeculative();
    }

    final Ungag ungagSpeculative() const
    {
        uint oldgag = global.gag;
        if (global.gag && !isSpeculative() && !toParent2().isFuncDeclaration())
            global.gag = 0;
        return Ungag(oldgag);
    }

    // kludge for template.isSymbol()
    override final DYNCAST dyncast() const
    {
        return DYNCAST.dsymbol;
    }

    /*************************************
     * Do syntax copy of an array of Dsymbol's.
     */
    static Dsymbols* arraySyntaxCopy(Dsymbols* a)
    {
        Dsymbols* b = null;
        if (a)
        {
            b = a.copy();
            for (size_t i = 0; i < b.dim; i++)
            {
                (*b)[i] = (*b)[i].syntaxCopy(null);
            }
        }
        return b;
    }

    Identifier getIdent()
    {
        return ident;
    }

    const(char)* toPrettyChars(bool QualifyTypes = false)
    {
        if (prettystring && !QualifyTypes)
            return prettystring;

        //printf("Dsymbol::toPrettyChars() '%s'\n", toChars());
        if (!parent)
        {
            auto s = toChars();
            if (!QualifyTypes)
                prettystring = s;
            return s;
        }

        // Computer number of components
        size_t complength = 0;
        for (Dsymbol p = this; p; p = p.parent)
            ++complength;

        // Allocate temporary array comp[]
        alias T = const(char)[];
        auto compptr = cast(T*)malloc(complength * T.sizeof);
        if (!compptr)
            Mem.error();
        auto comp = compptr[0 .. complength];

        // Fill in comp[] and compute length of final result
        size_t length = 0;
        int i;
        for (Dsymbol p = this; p; p = p.parent)
        {
            const s = QualifyTypes ? p.toPrettyCharsHelper() : p.toChars();
            const len = strlen(s);
            comp[i] = s[0 .. len];
            ++i;
            length += len + 1;
        }

        auto s = cast(char*)mem.xmalloc(length);
        auto q = s + length - 1;
        *q = 0;
        foreach (j; 0 .. complength)
        {
            const t = comp[j].ptr;
            const len = comp[j].length;
            q -= len;
            memcpy(q, t, len);
            if (q == s)
                break;
            *--q = '.';
        }
        free(comp.ptr);
        if (!QualifyTypes)
            prettystring = s;
        return s;
    }

    const(char)* kind() const
    {
        return "symbol";
    }

    /*********************************
     * If this symbol is really an alias for another,
     * return that other.
     * If needed, semantic() is invoked due to resolve forward reference.
     */
    Dsymbol toAlias()
    {
        return this;
    }

    /*********************************
     * Resolve recursive tuple expansion in eponymous template.
     */
    Dsymbol toAlias2()
    {
        return toAlias();
    }

    /*********************************
     * Iterate this dsymbol or members of this scoped dsymbol, then
     * call `fp` with the found symbol and `param`.
     * Params:
     *  fp = function pointer to process the iterated symbol.
     *       If it returns nonzero, the iteration will be aborted.
     *  param = a parameter passed to fp.
     * Returns:
     *  nonzero if the iteration is aborted by the return value of fp,
     *  or 0 if it's completed.
     */
    int apply(Dsymbol_apply_ft_t fp, void* param)
    {
        return (*fp)(this, param);
    }

    void addMember(Scope* sc, ScopeDsymbol sds)
    {
        //printf("Dsymbol::addMember('%s')\n", toChars());
        //printf("Dsymbol::addMember(this = %p, '%s' scopesym = '%s')\n", this, toChars(), sds.toChars());
        //printf("Dsymbol::addMember(this = %p, '%s' sds = %p, sds.symtab = %p)\n", this, toChars(), sds, sds.symtab);
        parent = sds;
        if (!isAnonymous()) // no name, so can't add it to symbol table
        {
            if (!sds.symtabInsert(this)) // if name is already defined
            {
                Dsymbol s2 = sds.symtab.lookup(ident);
                if (!s2.overloadInsert(this))
                {
                    sds.multiplyDefined(Loc(), this, s2);
                    errors = true;
                }
            }
            if (sds.isAggregateDeclaration() || sds.isEnumDeclaration())
            {
                if (ident == Id.__sizeof || ident == Id.__xalignof || ident == Id._mangleof)
                {
                    error(".%s property cannot be redefined", ident.toChars());
                    errors = true;
                }
            }
        }
    }

    /*************************************
     * Set scope for future semantic analysis so we can
     * deal better with forward references.
     */
    void setScope(Scope* sc)
    {
        //printf("Dsymbol::setScope() %p %s, %p stc = %llx\n", this, toChars(), sc, sc.stc);
        if (!sc.nofree)
            sc.setNoFree(); // may need it even after semantic() finishes
        _scope = sc;
        if (sc.depdecl)
            depdecl = sc.depdecl;
        if (!userAttribDecl)
            userAttribDecl = sc.userAttribDecl;
    }

    void importAll(Scope* sc)
    {
    }

    /*************************************
     * Does semantic analysis on the public face of declarations.
     */
    void semantic(Scope* sc)
    {
        error("%p has no semantic routine", this);
    }

    /*************************************
     * Does semantic analysis on initializers and members of aggregates.
     */
    void semantic2(Scope* sc)
    {
        // Most Dsymbols have no further semantic analysis needed
    }

    /*************************************
     * Does semantic analysis on function bodies.
     */
    void semantic3(Scope* sc)
    {
        // Most Dsymbols have no further semantic analysis needed
    }

    /*********************************************
     * Search for ident as member of s.
     * Params:
     *  loc = location to print for error messages
     *  ident = identifier to search for
     *  flags = IgnoreXXXX
     * Returns:
     *  null if not found
     */
    Dsymbol search(Loc loc, Identifier ident, int flags = IgnoreNone)
    {
        //printf("Dsymbol::search(this=%p,%s, ident='%s')\n", this, toChars(), ident.toChars());
        return null;
    }

    final Dsymbol search_correct(Identifier ident)
    {
        /***************************************************
         * Search for symbol with correct spelling.
         */
        extern (D) void* symbol_search_fp(const(char)* seed, ref int cost)
        {
            /* If not in the lexer's string table, it certainly isn't in the symbol table.
             * Doing this first is a lot faster.
             */
            size_t len = strlen(seed);
            if (!len)
                return null;
            Identifier id = Identifier.lookup(seed, len);
            if (!id)
                return null;
            cost = 0;
            Dsymbol s = this;
            Module.clearCache();
            return cast(void*)s.search(Loc(), id, IgnoreErrors);
        }

        if (global.gag)
            return null; // don't do it for speculative compiles; too time consuming
        return cast(Dsymbol)speller(ident.toChars(), &symbol_search_fp, idchars);
    }

    /***************************************
     * Search for identifier id as a member of 'this'.
     * id may be a template instance.
     * Returns:
     *      symbol found, NULL if not
     */
    final Dsymbol searchX(Loc loc, Scope* sc, RootObject id)
    {
        //printf("Dsymbol::searchX(this=%p,%s, ident='%s')\n", this, toChars(), ident.toChars());
        Dsymbol s = toAlias();
        Dsymbol sm;
        if (Declaration d = s.isDeclaration())
        {
            if (d.inuse)
            {
                .error(loc, "circular reference to '%s'", d.toPrettyChars());
                return null;
            }
        }
        switch (id.dyncast())
        {
        case DYNCAST.identifier:
            sm = s.search(loc, cast(Identifier)id);
            break;
        case DYNCAST.dsymbol:
            {
                // It's a template instance
                //printf("\ttemplate instance id\n");
                Dsymbol st = cast(Dsymbol)id;
                TemplateInstance ti = st.isTemplateInstance();
                sm = s.search(loc, ti.name);
                if (!sm)
                {
                    sm = s.search_correct(ti.name);
                    if (sm)
                        .error(loc, "template identifier '%s' is not a member of %s '%s', did you mean %s '%s'?", ti.name.toChars(), s.kind(), s.toPrettyChars(), sm.kind(), sm.toChars());
                    else
                        .error(loc, "template identifier '%s' is not a member of %s '%s'", ti.name.toChars(), s.kind(), s.toPrettyChars());
                    return null;
                }
                sm = sm.toAlias();
                TemplateDeclaration td = sm.isTemplateDeclaration();
                if (!td)
                {
                    .error(loc, "%s.%s is not a template, it is a %s", s.toPrettyChars(), ti.name.toChars(), sm.kind());
                    return null;
                }
                ti.tempdecl = td;
                if (!ti.semanticRun)
                    ti.semantic(sc);
                sm = ti.toAlias();
                break;
            }
        case DYNCAST.type:
        case DYNCAST.expression:
        default:
            assert(0);
        }
        return sm;
    }

    bool overloadInsert(Dsymbol s)
    {
        //printf("Dsymbol::overloadInsert('%s')\n", s.toChars());
        return false;
    }

    /*********************************
     * Returns:
     *  SIZE_INVALID when the size cannot be determined
     */
    d_uns64 size(Loc loc)
    {
        error("Dsymbol '%s' has no size", toChars());
        return SIZE_INVALID;
    }

    bool isforwardRef()
    {
        return false;
    }

    // is a 'this' required to access the member
    AggregateDeclaration isThis()
    {
        return null;
    }

    // is Dsymbol exported?
    bool isExport()
    {
        return false;
    }

    // is Dsymbol imported?
    bool isImportedSymbol()
    {
        return false;
    }

    // is Dsymbol deprecated?
    bool isDeprecated()
    {
        return false;
    }

    bool isOverloadable()
    {
        return false;
    }

    // is this a LabelDsymbol()?
    LabelDsymbol isLabel()
    {
        return null;
    }

    /// Returns an AggregateDeclaration when toParent() is that.
    final AggregateDeclaration isMember()
    {
        //printf("Dsymbol::isMember() %s\n", toChars());
        auto p = toParent();
        //printf("parent is %s %s\n", p.kind(), p.toChars());
        return p ? p.isAggregateDeclaration() : null;
    }

    /// Returns an AggregateDeclaration when toParent2() is that.
    final AggregateDeclaration isMember2()
    {
        //printf("Dsymbol::isMember2() '%s'\n", toChars());
        auto p = toParent2();
        //printf("parent is %s %s\n", p.kind(), p.toChars());
        return p ? p.isAggregateDeclaration() : null;
    }

    // is this a member of a ClassDeclaration?
    final ClassDeclaration isClassMember()
    {
        auto ad = isMember();
        return ad ? ad.isClassDeclaration() : null;
    }

    // is this a type?
    Type getType()
    {
        return null;
    }

    // need a 'this' pointer?
    bool needThis()
    {
        return false;
    }

    /*************************************
     */
    Prot prot()
    {
        return Prot(PROTpublic);
    }

    /**************************************
     * Copy the syntax.
     * Used for template instantiations.
     * If s is NULL, allocate the new object, otherwise fill it in.
     */
    Dsymbol syntaxCopy(Dsymbol s)
    {
        print();
        printf("%s %s\n", kind(), toChars());
        assert(0);
    }

    /**************************************
     * Determine if this symbol is only one.
     * Returns:
     *      false, *ps = NULL: There are 2 or more symbols
     *      true,  *ps = NULL: There are zero symbols
     *      true,  *ps = symbol: The one and only one symbol
     */
    bool oneMember(Dsymbol* ps, Identifier ident)
    {
        //printf("Dsymbol::oneMember()\n");
        *ps = this;
        return true;
    }

    /*****************************************
     * Same as Dsymbol::oneMember(), but look at an array of Dsymbols.
     */
    static bool oneMembers(Dsymbols* members, Dsymbol* ps, Identifier ident)
    {
        //printf("Dsymbol::oneMembers() %d\n", members ? members.dim : 0);
        Dsymbol s = null;
        if (members)
        {
            for (size_t i = 0; i < members.dim; i++)
            {
                Dsymbol sx = (*members)[i];
                bool x = sx.oneMember(ps, ident);
                //printf("\t[%d] kind %s = %d, s = %p\n", i, sx.kind(), x, *ps);
                if (!x)
                {
                    //printf("\tfalse 1\n");
                    assert(*ps is null);
                    return false;
                }
                if (*ps)
                {
                    assert(ident);
                    if (!(*ps).ident || !(*ps).ident.equals(ident))
                        continue;
                    if (!s)
                        s = *ps;
                    else if (s.isOverloadable() && (*ps).isOverloadable())
                    {
                        // keep head of overload set
                        FuncDeclaration f1 = s.isFuncDeclaration();
                        FuncDeclaration f2 = (*ps).isFuncDeclaration();
                        if (f1 && f2)
                        {
                            assert(!f1.isFuncAliasDeclaration());
                            assert(!f2.isFuncAliasDeclaration());
                            for (; f1 != f2; f1 = f1.overnext0)
                            {
                                if (f1.overnext0 is null)
                                {
                                    f1.overnext0 = f2;
                                    break;
                                }
                            }
                        }
                    }
                    else // more than one symbol
                    {
                        *ps = null;
                        //printf("\tfalse 2\n");
                        return false;
                    }
                }
            }
        }
        *ps = s; // s is the one symbol, null if none
        //printf("\ttrue\n");
        return true;
    }

    void setFieldOffset(AggregateDeclaration ad, uint* poffset, bool isunion)
    {
    }

    /*****************************************
     * Is Dsymbol a variable that contains pointers?
     */
    bool hasPointers()
    {
        //printf("Dsymbol::hasPointers() %s\n", toChars());
        return false;
    }

    bool hasStaticCtorOrDtor()
    {
        //printf("Dsymbol::hasStaticCtorOrDtor() %s\n", toChars());
        return false;
    }

    void addLocalClass(ClassDeclarations*)
    {
    }

    void checkCtorConstInit()
    {
    }

    /****************************************
     * Add documentation comment to Dsymbol.
     * Ignore NULL comments.
     */
    void addComment(const(char)* comment)
    {
        //if (comment)
        //    printf("adding comment '%s' to symbol %p '%s'\n", comment, this, toChars());
        if (!this.comment)
            this.comment = comment;
        else if (comment && strcmp(cast(char*)comment, cast(char*)this.comment) != 0)
        {
            // Concatenate the two
            this.comment = Lexer.combineComments(this.comment, comment, true);
        }
    }

    /****************************************
     * Returns true if this symbol is defined in a non-root module without instantiation.
     */
    final bool inNonRoot()
    {
        Dsymbol s = parent;
        for (; s; s = s.toParent())
        {
            if (auto ti = s.isTemplateInstance())
            {
                return false;
            }
            if (auto m = s.isModule())
            {
                if (!m.isRoot())
                    return true;
                break;
            }
        }
        return false;
    }

    // Eliminate need for dynamic_cast
    inout(Package) isPackage() inout
    {
        return null;
    }

    inout(Module) isModule() inout
    {
        return null;
    }

    inout(EnumMember) isEnumMember() inout
    {
        return null;
    }

    inout(TemplateDeclaration) isTemplateDeclaration() inout
    {
        return null;
    }

    inout(TemplateInstance) isTemplateInstance() inout
    {
        return null;
    }

    inout(TemplateMixin) isTemplateMixin() inout
    {
        return null;
    }

    inout(Nspace) isNspace() inout
    {
        return null;
    }

    inout(Declaration) isDeclaration() inout
    {
        return null;
    }

    inout(ThisDeclaration) isThisDeclaration() inout
    {
        return null;
    }

    inout(TypeInfoDeclaration) isTypeInfoDeclaration() inout
    {
        return null;
    }

    inout(TupleDeclaration) isTupleDeclaration() inout
    {
        return null;
    }

    inout(AliasDeclaration) isAliasDeclaration() inout
    {
        return null;
    }

    inout(AggregateDeclaration) isAggregateDeclaration() inout
    {
        return null;
    }

    inout(FuncDeclaration) isFuncDeclaration() inout
    {
        return null;
    }

    inout(FuncAliasDeclaration) isFuncAliasDeclaration() inout
    {
        return null;
    }

    inout(OverDeclaration) isOverDeclaration() inout
    {
        return null;
    }

    inout(FuncLiteralDeclaration) isFuncLiteralDeclaration() inout
    {
        return null;
    }

    inout(CtorDeclaration) isCtorDeclaration() inout
    {
        return null;
    }

    inout(PostBlitDeclaration) isPostBlitDeclaration() inout
    {
        return null;
    }

    inout(DtorDeclaration) isDtorDeclaration() inout
    {
        return null;
    }

    inout(StaticCtorDeclaration) isStaticCtorDeclaration() inout
    {
        return null;
    }

    inout(StaticDtorDeclaration) isStaticDtorDeclaration() inout
    {
        return null;
    }

    inout(SharedStaticCtorDeclaration) isSharedStaticCtorDeclaration() inout
    {
        return null;
    }

    inout(SharedStaticDtorDeclaration) isSharedStaticDtorDeclaration() inout
    {
        return null;
    }

    inout(InvariantDeclaration) isInvariantDeclaration() inout
    {
        return null;
    }

    inout(UnitTestDeclaration) isUnitTestDeclaration() inout
    {
        return null;
    }

    inout(NewDeclaration) isNewDeclaration() inout
    {
        return null;
    }

    inout(VarDeclaration) isVarDeclaration() inout
    {
        return null;
    }

    inout(ClassDeclaration) isClassDeclaration() inout
    {
        return null;
    }

    inout(StructDeclaration) isStructDeclaration() inout
    {
        return null;
    }

    inout(UnionDeclaration) isUnionDeclaration() inout
    {
        return null;
    }

    inout(InterfaceDeclaration) isInterfaceDeclaration() inout
    {
        return null;
    }

    inout(ScopeDsymbol) isScopeDsymbol() inout
    {
        return null;
    }

    inout(WithScopeSymbol) isWithScopeSymbol() inout
    {
        return null;
    }

    inout(ArrayScopeSymbol) isArrayScopeSymbol() inout
    {
        return null;
    }

    inout(Import) isImport() inout
    {
        return null;
    }

    inout(EnumDeclaration) isEnumDeclaration() inout
    {
        return null;
    }

    inout(DeleteDeclaration) isDeleteDeclaration() inout
    {
        return null;
    }

    inout(SymbolDeclaration) isSymbolDeclaration() inout
    {
        return null;
    }

    inout(AttribDeclaration) isAttribDeclaration() inout
    {
        return null;
    }

    inout(AnonDeclaration) isAnonDeclaration() inout
    {
        return null;
    }

    inout(OverloadSet) isOverloadSet() inout
    {
        return null;
    }

    /************
     */
    void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Dsymbol that generates a scope
 */
extern (C++) class ScopeDsymbol : Dsymbol
{
    Dsymbols* members;          // all Dsymbol's in this scope
    DsymbolTable symtab;        // members[] sorted into table
    uint endlinnum;             // the linnumber of the statement after the scope (0 if unknown)

private:
    /// symbols whose members have been imported, i.e. imported modules and template mixins
    Dsymbols* importedScopes;
    PROTKIND* prots;            // array of PROTKIND, one for each import

    import ddmd.root.array : BitArray;
    BitArray accessiblePackages, privateAccessiblePackages;// whitelists of accessible (imported) packages

public:
    final extern (D) this()
    {
    }

    final extern (D) this(Identifier id)
    {
        super(id);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        //printf("ScopeDsymbol::syntaxCopy('%s')\n", toChars());
        ScopeDsymbol sds = s ? cast(ScopeDsymbol)s : new ScopeDsymbol(ident);
        sds.members = arraySyntaxCopy(members);
        sds.endlinnum = endlinnum;
        return sds;
    }

    /*****************************************
     * This function is #1 on the list of functions that eat cpu time.
     * Be very, very careful about slowing it down.
     */
    override Dsymbol search(Loc loc, Identifier ident, int flags = SearchLocalsOnly)
    {
        //printf("%s.ScopeDsymbol::search(ident='%s', flags=x%x)\n", toChars(), ident.toChars(), flags);
        //if (strcmp(ident.toChars(),"c") == 0) *(char*)0=0;

        // Look in symbols declared in this module
        if (symtab && !(flags & SearchImportsOnly))
        {
            //printf(" look in locals\n");
            auto s1 = symtab.lookup(ident);
            if (s1)
            {
                //printf("\tfound in locals = '%s.%s'\n",toChars(),s1.toChars());
                return s1;
            }
        }
        //printf(" not found in locals\n");

        // Look in imported scopes
        if (importedScopes)
        {
            //printf(" look in imports\n");
            Dsymbol s = null;
            OverloadSet a = null;
            // Look in imported modules
            for (size_t i = 0; i < importedScopes.dim; i++)
            {
                // If private import, don't search it
                if ((flags & IgnorePrivateImports) && prots[i] == PROTprivate)
                    continue;
                int sflags = flags & (IgnoreErrors | IgnoreAmbiguous | IgnoreSymbolVisibility); // remember these in recursive searches
                Dsymbol ss = (*importedScopes)[i];
                //printf("\tscanning import '%s', prots = %d, isModule = %p, isImport = %p\n", ss.toChars(), prots[i], ss.isModule(), ss.isImport());

                if (ss.isModule())
                {
                    if (flags & SearchLocalsOnly)
                        continue;
                }
                else // mixin template
                {
                    if (flags & SearchImportsOnly)
                        continue;
                    // compatibility with -transition=import (Bugzilla 15925)
                    // SearchLocalsOnly should always get set for new lookup rules
                    sflags |= (flags & SearchLocalsOnly);
                }

                /* Don't find private members if ss is a module
                 */
                Dsymbol s2 = ss.search(loc, ident, sflags | (ss.isModule() ? IgnorePrivateImports : IgnoreNone));
                import ddmd.access : symbolIsVisible;
                if (!s2 || !(flags & IgnoreSymbolVisibility) && !symbolIsVisible(this, s2))
                    continue;
                if (!s)
                {
                    s = s2;
                    if (s && s.isOverloadSet())
                        a = mergeOverloadSet(ident, a, s);
                }
                else if (s2 && s != s2)
                {
                    if (s.toAlias() == s2.toAlias() || s.getType() == s2.getType() && s.getType())
                    {
                        /* After following aliases, we found the same
                         * symbol, so it's not an ambiguity.  But if one
                         * alias is deprecated or less accessible, prefer
                         * the other.
                         */
                        if (s.isDeprecated() || s.prot().isMoreRestrictiveThan(s2.prot()) && s2.prot().kind != PROTnone)
                            s = s2;
                    }
                    else
                    {
                        /* Two imports of the same module should be regarded as
                         * the same.
                         */
                        Import i1 = s.isImport();
                        Import i2 = s2.isImport();
                        if (!(i1 && i2 && (i1.mod == i2.mod || (!i1.parent.isImport() && !i2.parent.isImport() && i1.ident.equals(i2.ident)))))
                        {
                            /* Bugzilla 8668:
                             * Public selective import adds AliasDeclaration in module.
                             * To make an overload set, resolve aliases in here and
                             * get actual overload roots which accessible via s and s2.
                             */
                            s = s.toAlias();
                            s2 = s2.toAlias();
                            /* If both s2 and s are overloadable (though we only
                             * need to check s once)
                             */
                            if ((s2.isOverloadSet() || s2.isOverloadable()) && (a || s.isOverloadable()))
                            {
                                a = mergeOverloadSet(ident, a, s2);
                                continue;
                            }
                            if (flags & IgnoreAmbiguous) // if return NULL on ambiguity
                                return null;
                            if (!(flags & IgnoreErrors))
                                ScopeDsymbol.multiplyDefined(loc, s, s2);
                            break;
                        }
                    }
                }
            }
            if (s)
            {
                /* Build special symbol if we had multiple finds
                 */
                if (a)
                {
                    if (!s.isOverloadSet())
                        a = mergeOverloadSet(ident, a, s);
                    s = a;
                }
                // TODO: remove once private symbol visibility has been deprecated
                if (!(flags & IgnoreErrors) && s.prot().kind == PROTprivate &&
                    !s.isOverloadable() && !s.parent.isTemplateMixin() && !s.parent.isNspace())
                {
                    AliasDeclaration ad = void;
                    // accessing private selective and renamed imports is
                    // deprecated by restricting the symbol visibility
                    if (s.isImport() || (ad = s.isAliasDeclaration()) !is null && ad._import !is null)
                    {}
                    else
                        error(loc, "%s %s is private", s.kind(), s.toPrettyChars());
                }
                //printf("\tfound in imports %s.%s\n", toChars(), s.toChars());
                return s;
            }
            //printf(" not found in imports\n");
        }
        return null;
    }

    final OverloadSet mergeOverloadSet(Identifier ident, OverloadSet os, Dsymbol s)
    {
        if (!os)
        {
            os = new OverloadSet(ident);
            os.parent = this;
        }
        if (OverloadSet os2 = s.isOverloadSet())
        {
            // Merge the cross-module overload set 'os2' into 'os'
            if (os.a.dim == 0)
            {
                os.a.setDim(os2.a.dim);
                memcpy(os.a.tdata(), os2.a.tdata(), (os.a[0]).sizeof * os2.a.dim);
            }
            else
            {
                for (size_t i = 0; i < os2.a.dim; i++)
                {
                    os = mergeOverloadSet(ident, os, os2.a[i]);
                }
            }
        }
        else
        {
            assert(s.isOverloadable());
            /* Don't add to os[] if s is alias of previous sym
             */
            for (size_t j = 0; j < os.a.dim; j++)
            {
                Dsymbol s2 = os.a[j];
                if (s.toAlias() == s2.toAlias())
                {
                    if (s2.isDeprecated() || (s2.prot().isMoreRestrictiveThan(s.prot()) && s.prot().kind != PROTnone))
                    {
                        os.a[j] = s;
                    }
                    goto Lcontinue;
                }
            }
            os.push(s);
        Lcontinue:
        }
        return os;
    }

    final void importScope(Dsymbol s, Prot protection)
    {
        //printf("%s.ScopeDsymbol::importScope(%s, %d)\n", toChars(), s.toChars(), protection);
        // No circular or redundant import's
        if (s != this)
        {
            if (!importedScopes)
                importedScopes = new Dsymbols();
            else
            {
                for (size_t i = 0; i < importedScopes.dim; i++)
                {
                    Dsymbol ss = (*importedScopes)[i];
                    if (ss == s) // if already imported
                    {
                        if (protection.kind > prots[i])
                            prots[i] = protection.kind; // upgrade access
                        return;
                    }
                }
            }
            importedScopes.push(s);
            prots = cast(PROTKIND*)mem.xrealloc(prots, importedScopes.dim * (prots[0]).sizeof);
            prots[importedScopes.dim - 1] = protection.kind;
        }
    }

    final void addAccessiblePackage(Package p, Prot protection)
    {
        auto pary = protection.kind == PROTprivate ? &privateAccessiblePackages : &accessiblePackages;
        if (pary.length <= p.tag)
            pary.length = p.tag + 1;
        (*pary)[p.tag] = true;
    }

    bool isPackageAccessible(Package p, Prot protection, int flags = 0)
    {
        if (p.tag < accessiblePackages.length && accessiblePackages[p.tag] ||
            protection.kind == PROTprivate && p.tag < privateAccessiblePackages.length && privateAccessiblePackages[p.tag])
            return true;
        foreach (i, ss; importedScopes ? (*importedScopes)[] : null)
        {
            // only search visible scopes && imported modules should ignore private imports
            if (protection.kind <= prots[i] &&
                ss.isScopeDsymbol.isPackageAccessible(p, protection, IgnorePrivateImports))
                return true;
        }
        return false;
    }

    override final bool isforwardRef()
    {
        return (members is null);
    }

    static void multiplyDefined(Loc loc, Dsymbol s1, Dsymbol s2)
    {
        version (none)
        {
            printf("ScopeDsymbol::multiplyDefined()\n");
            printf("s1 = %p, '%s' kind = '%s', parent = %s\n", s1, s1.toChars(), s1.kind(), s1.parent ? s1.parent.toChars() : "");
            printf("s2 = %p, '%s' kind = '%s', parent = %s\n", s2, s2.toChars(), s2.kind(), s2.parent ? s2.parent.toChars() : "");
        }
        if (loc.filename)
        {
            .error(loc, "%s at %s conflicts with %s at %s", s1.toPrettyChars(), s1.locToChars(), s2.toPrettyChars(), s2.locToChars());
        }
        else
        {
            s1.error(s1.loc, "conflicts with %s %s at %s", s2.kind(), s2.toPrettyChars(), s2.locToChars());
        }
    }

    override const(char)* kind() const
    {
        return "ScopeDsymbol";
    }

    /*******************************************
     * Look for member of the form:
     *      const(MemberInfo)[] getMembers(string);
     * Returns NULL if not found
     */
    final FuncDeclaration findGetMembers()
    {
        Dsymbol s = search_function(this, Id.getmembers);
        FuncDeclaration fdx = s ? s.isFuncDeclaration() : null;
        version (none)
        {
            // Finish
            static __gshared TypeFunction tfgetmembers;
            if (!tfgetmembers)
            {
                Scope sc;
                auto parameters = new Parameters();
                Parameters* p = new Parameter(STCin, Type.tchar.constOf().arrayOf(), null, null);
                parameters.push(p);
                Type tret = null;
                tfgetmembers = new TypeFunction(parameters, tret, 0, LINKd);
                tfgetmembers = cast(TypeFunction)tfgetmembers.semantic(Loc(), &sc);
            }
            if (fdx)
                fdx = fdx.overloadExactMatch(tfgetmembers);
        }
        if (fdx && fdx.isVirtual())
            fdx = null;
        return fdx;
    }

    Dsymbol symtabInsert(Dsymbol s)
    {
        return symtab.insert(s);
    }

    /****************************************
     * Return true if any of the members are static ctors or static dtors, or if
     * any members have members that are.
     */
    override bool hasStaticCtorOrDtor()
    {
        if (members)
        {
            for (size_t i = 0; i < members.dim; i++)
            {
                Dsymbol member = (*members)[i];
                if (member.hasStaticCtorOrDtor())
                    return true;
            }
        }
        return false;
    }

    /***************************************
     * Determine number of Dsymbols, folding in AttribDeclaration members.
     */
    static size_t dim(Dsymbols* members)
    {
        size_t n = 0;
        int dimDg(size_t idx, Dsymbol s)
        {
            ++n;
            return 0;
        }

        _foreach(null, members, &dimDg, &n);
        return n;
    }

    /***************************************
     * Get nth Dsymbol, folding in AttribDeclaration members.
     * Returns:
     *      Dsymbol*        nth Dsymbol
     *      NULL            not found, *pn gets incremented by the number
     *                      of Dsymbols
     */
    static Dsymbol getNth(Dsymbols* members, size_t nth, size_t* pn = null)
    {
        Dsymbol sym = null;

        int getNthSymbolDg(size_t n, Dsymbol s)
        {
            if (n == nth)
            {
                sym = s;
                return 1;
            }
            return 0;
        }

        int res = _foreach(null, members, &getNthSymbolDg);
        return res ? sym : null;
    }

    extern (D) alias ForeachDg = int delegate(size_t idx, Dsymbol s);

    /***************************************
     * Expands attribute declarations in members in depth first
     * order. Calls dg(size_t symidx, Dsymbol *sym) for each
     * member.
     * If dg returns !=0, stops and returns that value else returns 0.
     * Use this function to avoid the O(N + N^2/2) complexity of
     * calculating dim and calling N times getNth.
     * Returns:
     *  last value returned by dg()
     */
    extern (D) static int _foreach(Scope* sc, Dsymbols* members, scope ForeachDg dg, size_t* pn = null)
    {
        assert(dg);
        if (!members)
            return 0;
        size_t n = pn ? *pn : 0; // take over index
        int result = 0;
        foreach (size_t i; 0 .. members.dim)
        {
            Dsymbol s = (*members)[i];
            if (AttribDeclaration a = s.isAttribDeclaration())
                result = _foreach(sc, a.include(sc, null), dg, &n);
            else if (TemplateMixin tm = s.isTemplateMixin())
                result = _foreach(sc, tm.members, dg, &n);
            else if (s.isTemplateInstance())
            {
            }
            else if (s.isUnitTestDeclaration())
            {
            }
            else
                result = dg(n++, s);
            if (result)
                break;
        }
        if (pn)
            *pn = n; // update index
        return result;
    }

    override final inout(ScopeDsymbol) isScopeDsymbol() inout
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * With statement scope
 */
extern (C++) final class WithScopeSymbol : ScopeDsymbol
{
    WithStatement withstate;

    extern (D) this(WithStatement withstate)
    {
        this.withstate = withstate;
    }

    override Dsymbol search(Loc loc, Identifier ident, int flags = SearchLocalsOnly)
    {
        //printf("WithScopeSymbol.search(%s)\n", ident.toChars());
        if (flags & SearchImportsOnly)
            return null;
        // Acts as proxy to the with class declaration
        Dsymbol s = null;
        Expression eold = null;
        for (Expression e = withstate.exp; e != eold; e = resolveAliasThis(_scope, e))
        {
            if (e.op == TOKscope)
            {
                s = (cast(ScopeExp)e).sds;
            }
            else if (e.op == TOKtype)
            {
                s = e.type.toDsymbol(null);
            }
            else
            {
                Type t = e.type.toBasetype();
                s = t.toDsymbol(null);
            }
            if (s)
            {
                s = s.search(loc, ident, flags);
                if (s)
                    return s;
            }
            eold = e;
        }
        return null;
    }

    override inout(WithScopeSymbol) isWithScopeSymbol() inout
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Array Index/Slice scope
 */
extern (C++) final class ArrayScopeSymbol : ScopeDsymbol
{
    Expression exp;         // IndexExp or SliceExp
    TypeTuple type;         // for tuple[length]
    TupleDeclaration td;    // for tuples of objects
    Scope* sc;

    extern (D) this(Scope* sc, Expression e)
    {
        assert(e.op == TOKindex || e.op == TOKslice || e.op == TOKarray);
        exp = e;
        this.sc = sc;
    }

    extern (D) this(Scope* sc, TypeTuple t)
    {
        type = t;
        this.sc = sc;
    }

    extern (D) this(Scope* sc, TupleDeclaration s)
    {
        td = s;
        this.sc = sc;
    }

    override Dsymbol search(Loc loc, Identifier ident, int flags = IgnoreNone)
    {
        //printf("ArrayScopeSymbol::search('%s', flags = %d)\n", ident.toChars(), flags);
        if (ident == Id.dollar)
        {
            VarDeclaration* pvar;
            Expression ce;
        L1:
            if (td)
            {
                /* $ gives the number of elements in the tuple
                 */
                auto v = new VarDeclaration(loc, Type.tsize_t, Id.dollar, null);
                Expression e = new IntegerExp(Loc(), td.objects.dim, Type.tsize_t);
                v._init = new ExpInitializer(Loc(), e);
                v.storage_class |= STCtemp | STCstatic | STCconst;
                v.semantic(sc);
                return v;
            }
            if (type)
            {
                /* $ gives the number of type entries in the type tuple
                 */
                auto v = new VarDeclaration(loc, Type.tsize_t, Id.dollar, null);
                Expression e = new IntegerExp(Loc(), type.arguments.dim, Type.tsize_t);
                v._init = new ExpInitializer(Loc(), e);
                v.storage_class |= STCtemp | STCstatic | STCconst;
                v.semantic(sc);
                return v;
            }
            if (exp.op == TOKindex)
            {
                /* array[index] where index is some function of $
                 */
                IndexExp ie = cast(IndexExp)exp;
                pvar = &ie.lengthVar;
                ce = ie.e1;
            }
            else if (exp.op == TOKslice)
            {
                /* array[lwr .. upr] where lwr or upr is some function of $
                 */
                SliceExp se = cast(SliceExp)exp;
                pvar = &se.lengthVar;
                ce = se.e1;
            }
            else if (exp.op == TOKarray)
            {
                /* array[e0, e1, e2, e3] where e0, e1, e2 are some function of $
                 * $ is a opDollar!(dim)() where dim is the dimension(0,1,2,...)
                 */
                ArrayExp ae = cast(ArrayExp)exp;
                pvar = &ae.lengthVar;
                ce = ae.e1;
            }
            else
            {
                /* Didn't find $, look in enclosing scope(s).
                 */
                return null;
            }
            while (ce.op == TOKcomma)
                ce = (cast(CommaExp)ce).e2;
            /* If we are indexing into an array that is really a type
             * tuple, rewrite this as an index into a type tuple and
             * try again.
             */
            if (ce.op == TOKtype)
            {
                Type t = (cast(TypeExp)ce).type;
                if (t.ty == Ttuple)
                {
                    type = cast(TypeTuple)t;
                    goto L1;
                }
            }
            /* *pvar is lazily initialized, so if we refer to $
             * multiple times, it gets set only once.
             */
            if (!*pvar) // if not already initialized
            {
                /* Create variable v and set it to the value of $
                 */
                VarDeclaration v;
                Type t;
                if (ce.op == TOKtuple)
                {
                    /* It is for an expression tuple, so the
                     * length will be a const.
                     */
                    Expression e = new IntegerExp(Loc(), (cast(TupleExp)ce).exps.dim, Type.tsize_t);
                    v = new VarDeclaration(loc, Type.tsize_t, Id.dollar, new ExpInitializer(Loc(), e));
                    v.storage_class |= STCtemp | STCstatic | STCconst;
                }
                else if (ce.type && (t = ce.type.toBasetype()) !is null && (t.ty == Tstruct || t.ty == Tclass))
                {
                    // Look for opDollar
                    assert(exp.op == TOKarray || exp.op == TOKslice);
                    AggregateDeclaration ad = isAggregate(t);
                    assert(ad);
                    Dsymbol s = ad.search(loc, Id.opDollar);
                    if (!s) // no dollar exists -- search in higher scope
                        return null;
                    s = s.toAlias();
                    Expression e = null;
                    // Check for multi-dimensional opDollar(dim) template.
                    if (TemplateDeclaration td = s.isTemplateDeclaration())
                    {
                        dinteger_t dim = 0;
                        if (exp.op == TOKarray)
                        {
                            dim = (cast(ArrayExp)exp).currentDimension;
                        }
                        else if (exp.op == TOKslice)
                        {
                            dim = 0; // slices are currently always one-dimensional
                        }
                        else
                        {
                            assert(0);
                        }
                        auto tiargs = new Objects();
                        Expression edim = new IntegerExp(Loc(), dim, Type.tsize_t);
                        edim = edim.semantic(sc);
                        tiargs.push(edim);
                        e = new DotTemplateInstanceExp(loc, ce, td.ident, tiargs);
                    }
                    else
                    {
                        /* opDollar exists, but it's not a template.
                         * This is acceptable ONLY for single-dimension indexing.
                         * Note that it's impossible to have both template & function opDollar,
                         * because both take no arguments.
                         */
                        if (exp.op == TOKarray && (cast(ArrayExp)exp).arguments.dim != 1)
                        {
                            exp.error("%s only defines opDollar for one dimension", ad.toChars());
                            return null;
                        }
                        Declaration d = s.isDeclaration();
                        assert(d);
                        e = new DotVarExp(loc, ce, d);
                    }
                    e = e.semantic(sc);
                    if (!e.type)
                        exp.error("%s has no value", e.toChars());
                    t = e.type.toBasetype();
                    if (t && t.ty == Tfunction)
                        e = new CallExp(e.loc, e);
                    v = new VarDeclaration(loc, null, Id.dollar, new ExpInitializer(Loc(), e));
                    v.storage_class |= STCtemp | STCctfe | STCrvalue;
                }
                else
                {
                    /* For arrays, $ will either be a compile-time constant
                     * (in which case its value in set during constant-folding),
                     * or a variable (in which case an expression is created in
                     * toir.c).
                     */
                    auto e = new VoidInitializer(Loc());
                    e.type = Type.tsize_t;
                    v = new VarDeclaration(loc, Type.tsize_t, Id.dollar, e);
                    v.storage_class |= STCtemp | STCctfe; // it's never a true static variable
                }
                *pvar = v;
            }
            (*pvar).semantic(sc);
            return (*pvar);
        }
        return null;
    }

    override inout(ArrayScopeSymbol) isArrayScopeSymbol() inout
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Overload Sets
 */
extern (C++) final class OverloadSet : Dsymbol
{
    Dsymbols a;     // array of Dsymbols

    extern (D) this(Identifier ident, OverloadSet os = null)
    {
        super(ident);
        if (os)
        {
            for (size_t i = 0; i < os.a.dim; i++)
                a.push(os.a[i]);
        }
    }

    void push(Dsymbol s)
    {
        a.push(s);
    }

    override inout(OverloadSet) isOverloadSet() inout
    {
        return this;
    }

    override const(char)* kind() const
    {
        return "overloadset";
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 * Table of Dsymbol's
 */
extern (C++) final class DsymbolTable : RootObject
{
    AA* tab;

    // Look up Identifier. Return Dsymbol if found, NULL if not.
    Dsymbol lookup(const Identifier ident)
    {
        //printf("DsymbolTable::lookup(%s)\n", (char*)ident.string);
        return cast(Dsymbol)dmd_aaGetRvalue(tab, cast(void*)ident);
    }

    // Insert Dsymbol in table. Return NULL if already there.
    Dsymbol insert(Dsymbol s)
    {
        //printf("DsymbolTable::insert(this = %p, '%s')\n", this, s.ident.toChars());
        const ident = s.ident;
        Dsymbol* ps = cast(Dsymbol*)dmd_aaGet(&tab, cast(void*)ident);
        if (*ps)
            return null; // already in table
        *ps = s;
        return s;
    }

    // Look for Dsymbol in table. If there, return it. If not, insert s and return that.
    Dsymbol update(Dsymbol s)
    {
        const ident = s.ident;
        Dsymbol* ps = cast(Dsymbol*)dmd_aaGet(&tab, cast(void*)ident);
        *ps = s;
        return s;
    }

    // when ident and s are not the same
    Dsymbol insert(const Identifier ident, Dsymbol s)
    {
        //printf("DsymbolTable::insert()\n");
        Dsymbol* ps = cast(Dsymbol*)dmd_aaGet(&tab, cast(void*)ident);
        if (*ps)
            return null; // already in table
        *ps = s;
        return s;
    }

    /*****
     * Returns:
     *  number of symbols in symbol table
     */
    uint len() const pure
    {
        return cast(uint)dmd_aaLen(tab);
    }
}
