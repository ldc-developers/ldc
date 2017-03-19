/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (c) 1999-2016 by Digital Mars, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(DMDSRC _denum.d)
 */

module ddmd.denum;

import core.stdc.stdio;
import ddmd.gluelayer;
import ddmd.declaration;
import ddmd.dmodule;
import ddmd.dscope;
import ddmd.dsymbol;
import ddmd.errors;
import ddmd.expression;
import ddmd.globals;
import ddmd.id;
import ddmd.identifier;
import ddmd.init;
import ddmd.mtype;
import ddmd.tokens;
import ddmd.visitor;

/***********************************************************
 */
extern (C++) final class EnumDeclaration : ScopeDsymbol
{
    /* The separate, and distinct, cases are:
     *  1. enum { ... }
     *  2. enum : memtype { ... }
     *  3. enum id { ... }
     *  4. enum id : memtype { ... }
     *  5. enum id : memtype;
     *  6. enum id;
     */
    Type type;              // the TypeEnum
    Type memtype;           // type of the members
    Prot protection;
    Expression maxval;
    Expression minval;
    Expression defaultval;  // default initializer
    bool isdeprecated;
    bool added;
    int inuse;

    extern (D) this(Loc loc, Identifier id, Type memtype)
    {
        super(id);
        //printf("EnumDeclaration() %s\n", toChars());
        this.loc = loc;
        type = new TypeEnum(this);
        this.memtype = memtype;
        protection = Prot(PROTundefined);
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        auto ed = new EnumDeclaration(loc, ident, memtype ? memtype.syntaxCopy() : null);
        return ScopeDsymbol.syntaxCopy(ed);
    }

    override void addMember(Scope* sc, ScopeDsymbol sds)
    {
        version (none)
        {
            printf("EnumDeclaration::addMember() %s\n", toChars());
            for (size_t i = 0; i < members.dim; i++)
            {
                EnumMember em = (*members)[i].isEnumMember();
                printf("    member %s\n", em.toChars());
            }
        }

        /* Anonymous enum members get added to enclosing scope.
         */
        ScopeDsymbol scopesym = isAnonymous() ? sds : this;

        if (!isAnonymous())
        {
            ScopeDsymbol.addMember(sc, sds);
            if (!symtab)
                symtab = new DsymbolTable();
        }

        if (members)
        {
            for (size_t i = 0; i < members.dim; i++)
            {
                EnumMember em = (*members)[i].isEnumMember();
                em.ed = this;
                //printf("add %s to scope %s\n", em.toChars(), scopesym.toChars());
                em.addMember(sc, isAnonymous() ? scopesym : this);
            }
        }
        added = true;
    }

    override void setScope(Scope* sc)
    {
        if (semanticRun > PASSinit)
            return;
        ScopeDsymbol.setScope(sc);
    }

    override void semantic(Scope* sc)
    {
        //printf("EnumDeclaration::semantic(sd = %p, '%s') %s\n", sc.scopesym, sc.scopesym.toChars(), toChars());
        //printf("EnumDeclaration::semantic() %p %s\n", this, toChars());
        if (semanticRun >= PASSsemanticdone)
            return; // semantic() already completed
        if (semanticRun == PASSsemantic)
        {
            assert(memtype);
            .error(loc, "circular reference to enum base type %s", memtype.toChars());
            errors = true;
            semanticRun = PASSsemanticdone;
            return;
        }
        uint dprogress_save = Module.dprogress;

        Scope* scx = null;
        if (_scope)
        {
            sc = _scope;
            scx = _scope; // save so we don't make redundant copies
            _scope = null;
        }

        parent = sc.parent;
        type = type.semantic(loc, sc);

        protection = sc.protection;
        if (sc.stc & STCdeprecated)
            isdeprecated = true;
        userAttribDecl = sc.userAttribDecl;

        semanticRun = PASSsemantic;

        if (!members && !memtype) // enum ident;
        {
            semanticRun = PASSsemanticdone;
            return;
        }

        if (!symtab)
            symtab = new DsymbolTable();

        /* The separate, and distinct, cases are:
         *  1. enum { ... }
         *  2. enum : memtype { ... }
         *  3. enum ident { ... }
         *  4. enum ident : memtype { ... }
         *  5. enum ident : memtype;
         *  6. enum ident;
         */

        if (memtype)
        {
            memtype = memtype.semantic(loc, sc);

            /* Check to see if memtype is forward referenced
             */
            if (memtype.ty == Tenum)
            {
                EnumDeclaration sym = cast(EnumDeclaration)memtype.toDsymbol(sc);
                if (!sym.memtype || !sym.members || !sym.symtab || sym._scope)
                {
                    // memtype is forward referenced, so try again later
                    _scope = scx ? scx : sc.copy();
                    _scope.setNoFree();
                    _scope._module.addDeferredSemantic(this);
                    Module.dprogress = dprogress_save;
                    //printf("\tdeferring %s\n", toChars());
                    semanticRun = PASSinit;
                    return;
                }
            }
            if (memtype.ty == Tvoid)
            {
                error("base type must not be void");
                memtype = Type.terror;
            }
            if (memtype.ty == Terror)
            {
                errors = true;
                if (members)
                {
                    for (size_t i = 0; i < members.dim; i++)
                    {
                        Dsymbol s = (*members)[i];
                        s.errors = true; // poison all the members
                    }
                }
                semanticRun = PASSsemanticdone;
                return;
            }
        }

        semanticRun = PASSsemanticdone;

        if (!members) // enum ident : memtype;
            return;

        if (members.dim == 0)
        {
            error("enum %s must have at least one member", toChars());
            errors = true;
            return;
        }

        Module.dprogress++;

        Scope* sce;
        if (isAnonymous())
            sce = sc;
        else
        {
            sce = sc.push(this);
            sce.parent = this;
        }
        sce = sce.startCTFE();
        sce.setNoFree(); // needed for getMaxMinValue()

        /* Each enum member gets the sce scope
         */
        for (size_t i = 0; i < members.dim; i++)
        {
            EnumMember em = (*members)[i].isEnumMember();
            if (em)
                em._scope = sce;
        }

        if (!added)
        {
            /* addMember() is not called when the EnumDeclaration appears as a function statement,
             * so we have to do what addMember() does and install the enum members in the right symbol
             * table
             */
            ScopeDsymbol scopesym = null;
            if (isAnonymous())
            {
                /* Anonymous enum members get added to enclosing scope.
                 */
                for (Scope* sct = sce; 1; sct = sct.enclosing)
                {
                    assert(sct);
                    if (sct.scopesym)
                    {
                        scopesym = sct.scopesym;
                        if (!sct.scopesym.symtab)
                            sct.scopesym.symtab = new DsymbolTable();
                        break;
                    }
                }
            }
            else
            {
                // Otherwise enum members are in the EnumDeclaration's symbol table
                scopesym = this;
            }

            for (size_t i = 0; i < members.dim; i++)
            {
                EnumMember em = (*members)[i].isEnumMember();
                if (em)
                {
                    em.ed = this;
                    em.addMember(sc, scopesym);
                }
            }
        }

        for (size_t i = 0; i < members.dim; i++)
        {
            EnumMember em = (*members)[i].isEnumMember();
            if (em)
                em.semantic(em._scope);
        }
        //printf("defaultval = %lld\n", defaultval);

        //if (defaultval) printf("defaultval: %s %s\n", defaultval.toChars(), defaultval.type.toChars());
        //printf("members = %s\n", members.toChars());
    }

    override bool oneMember(Dsymbol* ps, Identifier ident)
    {
        if (isAnonymous())
            return Dsymbol.oneMembers(members, ps, ident);
        return Dsymbol.oneMember(ps, ident);
    }

    override Type getType()
    {
        return type;
    }

    override const(char)* kind() const
    {
        return "enum";
    }

    override Dsymbol search(Loc loc, Identifier ident, int flags = SearchLocalsOnly)
    {
        //printf("%s.EnumDeclaration::search('%s')\n", toChars(), ident.toChars());
        if (_scope)
        {
            // Try one last time to resolve this enum
            semantic(_scope);
        }

        if (!members || !symtab || _scope)
        {
            error("is forward referenced when looking for '%s'", ident.toChars());
            //*(char*)0=0;
            return null;
        }

        Dsymbol s = ScopeDsymbol.search(loc, ident, flags);
        return s;
    }

    // is Dsymbol deprecated?
    override bool isDeprecated()
    {
        return isdeprecated;
    }

    override Prot prot()
    {
        return protection;
    }

    /******************************
     * Get the value of the .max/.min property as an Expression.
     * Lazily computes the value and caches it in maxval/minval.
     * Reports any errors.
     * Params:
     *      loc = location to use for error messages
     *      id = Id::max or Id::min
     * Returns:
     *      corresponding value of .max/.min
     */
    Expression getMaxMinValue(Loc loc, Identifier id)
    {
        //printf("EnumDeclaration::getMaxValue()\n");
        bool first = true;

        Expression* pval = (id == Id.max) ? &maxval : &minval;

        Expression errorReturn()
        {
            *pval = new ErrorExp();
            return *pval;
        }

        if (inuse)
        {
            error(loc, "recursive definition of .%s property", id.toChars());
            return errorReturn();
        }
        if (*pval)
            goto Ldone;

        if (_scope)
            semantic(_scope);
        if (errors)
            return errorReturn();
        if (semanticRun == PASSinit || !members)
        {
            error("is forward referenced looking for .%s", id.toChars());
            return errorReturn();
        }
        if (!(memtype && memtype.isintegral()))
        {
            error(loc, "has no .%s property because base type %s is not an integral type", id.toChars(), memtype ? memtype.toChars() : "");
            return errorReturn();
        }

        for (size_t i = 0; i < members.dim; i++)
        {
            EnumMember em = (*members)[i].isEnumMember();
            if (!em)
                continue;
            if (em.errors)
                return errorReturn();

            Expression e = em.value;
            if (first)
            {
                *pval = e;
                first = false;
            }
            else
            {
                /* In order to work successfully with UDTs,
                 * build expressions to do the comparisons,
                 * and let the semantic analyzer and constant
                 * folder give us the result.
                 */

                /* Compute:
                 *   if (e > maxval)
                 *      maxval = e;
                 */
                Expression ec = new CmpExp(id == Id.max ? TOKgt : TOKlt, em.loc, e, *pval);
                inuse++;
                ec = ec.semantic(em._scope);
                inuse--;
                ec = ec.ctfeInterpret();
                if (ec.toInteger())
                    *pval = e;
            }
        }
    Ldone:
        Expression e = *pval;
        if (e.op != TOKerror)
        {
            e = e.copy();
            e.loc = loc;
        }
        return e;
    }

    Expression getDefaultValue(Loc loc)
    {
        //printf("EnumDeclaration::getDefaultValue() %p %s\n", this, toChars());
        if (defaultval)
            return defaultval;

        if (_scope)
            semantic(_scope);
        if (errors)
            goto Lerrors;
        if (semanticRun == PASSinit || !members)
        {
            error(loc, "forward reference of %s.init", toChars());
            goto Lerrors;
        }

        for (size_t i = 0; i < members.dim; i++)
        {
            EnumMember em = (*members)[i].isEnumMember();
            if (!em)
                continue;
            defaultval = em.value;
            return defaultval;
        }

    Lerrors:
        defaultval = new ErrorExp();
        return defaultval;
    }

    Type getMemtype(Loc loc)
    {
        if (loc.linnum == 0)
            loc = this.loc;
        if (_scope)
        {
            /* Enum is forward referenced. We don't need to resolve the whole thing,
             * just the base type
             */
            if (memtype)
                memtype = memtype.semantic(loc, _scope);
            else
            {
                if (!isAnonymous() && members)
                    memtype = Type.tint32;
            }
        }
        if (!memtype)
        {
            if (!isAnonymous() && members)
                memtype = Type.tint32;
            else
            {
                error(loc, "is forward referenced looking for base type");
                return Type.terror;
            }
        }
        return memtype;
    }

    override inout(EnumDeclaration) isEnumDeclaration() inout
    {
        return this;
    }

    Symbol* sinit;

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}

/***********************************************************
 */
extern (C++) final class EnumMember : VarDeclaration
{
    /* Can take the following forms:
     *  1. id
     *  2. id = value
     *  3. type id = value
     */
    @property ref value() { return (cast(ExpInitializer)_init).exp; }

    // A cast() is injected to 'value' after semantic(),
    // but 'origValue' will preserve the original value,
    // or previous value + 1 if none was specified.
    Expression origValue;

    Type origType;

    EnumDeclaration ed;

    extern (D) this(Loc loc, Identifier id, Expression value, Type origType)
    {
        super(loc, null, id ? id : Id.empty, new ExpInitializer(loc, value));
        this.origValue = value;
        this.origType = origType;
    }

    override Dsymbol syntaxCopy(Dsymbol s)
    {
        assert(!s);
        return new EnumMember(loc, ident, value ? value.syntaxCopy() : null, origType ? origType.syntaxCopy() : null);
    }

    override const(char)* kind() const
    {
        return "enum member";
    }

    override void semantic(Scope* sc)
    {
        //printf("EnumMember::semantic() %s\n", toChars());

        void errorReturn()
        {
            errors = true;
            semanticRun = PASSsemanticdone;
        }

        if (errors || semanticRun >= PASSsemanticdone)
            return;
        if (semanticRun == PASSsemantic)
        {
            error("circular reference to enum member");
            return errorReturn();
        }
        assert(ed);
        ed.semantic(sc);
        if (ed.errors)
            return errorReturn();
        if (errors || semanticRun >= PASSsemanticdone)
            return;

        if (_scope)
            sc = _scope;

        protection = ed.isAnonymous() ? ed.protection : Prot(PROTpublic);
        linkage = LINKd;
        storage_class = STCmanifest;
        userAttribDecl = ed.isAnonymous() ? ed.userAttribDecl : null;

        semanticRun = PASSsemantic;

        // The first enum member is special
        bool first = (this == (*ed.members)[0]);

        if (origType)
        {
            origType = origType.semantic(loc, sc);
            type = origType;
            assert(value); // "type id;" is not a valid enum member declaration
        }

        if (value)
        {
            Expression e = value;
            assert(e.dyncast() == DYNCAST_EXPRESSION);
            e = e.semantic(sc);
            e = resolveProperties(sc, e);
            e = e.ctfeInterpret();
            if (e.op == TOKerror)
                return errorReturn();
            if (first && !ed.memtype && !ed.isAnonymous())
            {
                ed.memtype = e.type;
                if (ed.memtype.ty == Terror)
                {
                    ed.errors = true;
                    return errorReturn();
                }
                if (ed.memtype.ty != Terror)
                {
                    /* Bugzilla 11746: All of named enum members should have same type
                     * with the first member. If the following members were referenced
                     * during the first member semantic, their types should be unified.
                     */
                    for (size_t i = 0; i < ed.members.dim; i++)
                    {
                        EnumMember em = (*ed.members)[i].isEnumMember();
                        if (!em || em == this || em.semanticRun < PASSsemanticdone || em.origType)
                            continue;

                        //printf("[%d] em = %s, em.semanticRun = %d\n", i, toChars(), em.semanticRun);
                        Expression ev = em.value;
                        ev = ev.implicitCastTo(sc, ed.memtype);
                        ev = ev.ctfeInterpret();
                        ev = ev.castTo(sc, ed.type);
                        if (ev.op == TOKerror)
                            ed.errors = true;
                        em.value = ev;
                    }
                    if (ed.errors)
                    {
                        ed.memtype = Type.terror;
                        return errorReturn();
                    }
                }
            }

            if (ed.memtype && !origType)
            {
                e = e.implicitCastTo(sc, ed.memtype);
                e = e.ctfeInterpret();

                // save origValue for better json output
                origValue = e;

                if (!ed.isAnonymous())
                {
                    e = e.castTo(sc, ed.type);
                    e = e.ctfeInterpret();
                }
            }
            else if (origType)
            {
                e = e.implicitCastTo(sc, origType);
                e = e.ctfeInterpret();
                assert(ed.isAnonymous());

                // save origValue for better json output
                origValue = e;
            }
            value = e;
        }
        else if (first)
        {
            Type t;
            if (ed.memtype)
                t = ed.memtype;
            else
            {
                t = Type.tint32;
                if (!ed.isAnonymous())
                    ed.memtype = t;
            }
            Expression e = new IntegerExp(loc, 0, Type.tint32);
            e = e.implicitCastTo(sc, t);
            e = e.ctfeInterpret();

            // save origValue for better json output
            origValue = e;

            if (!ed.isAnonymous())
            {
                e = e.castTo(sc, ed.type);
                e = e.ctfeInterpret();
            }
            value = e;
        }
        else
        {
            /* Find the previous enum member,
             * and set this to be the previous value + 1
             */
            EnumMember emprev = null;
            for (size_t i = 0; i < ed.members.dim; i++)
            {
                EnumMember em = (*ed.members)[i].isEnumMember();
                if (em)
                {
                    if (em == this)
                        break;
                    emprev = em;
                }
            }
            assert(emprev);
            if (emprev.semanticRun < PASSsemanticdone) // if forward reference
                emprev.semantic(emprev._scope); // resolve it
            if (emprev.errors)
                return errorReturn();

            Expression eprev = emprev.value;
            Type tprev = eprev.type.equals(ed.type) ? ed.memtype : eprev.type;

            Expression emax = tprev.getProperty(ed.loc, Id.max, 0);
            emax = emax.semantic(sc);
            emax = emax.ctfeInterpret();

            // Set value to (eprev + 1).
            // But first check that (eprev != emax)
            assert(eprev);
            Expression e = new EqualExp(TOKequal, loc, eprev, emax);
            e = e.semantic(sc);
            e = e.ctfeInterpret();
            if (e.toInteger())
            {
                error("initialization with (%s.%s + 1) causes overflow for type '%s'",
                    emprev.ed.toChars(), emprev.toChars(), ed.memtype.toChars());
                return errorReturn();
            }

            // Now set e to (eprev + 1)
            e = new AddExp(loc, eprev, new IntegerExp(loc, 1, Type.tint32));
            e = e.semantic(sc);
            e = e.castTo(sc, eprev.type);
            e = e.ctfeInterpret();

            // save origValue (without cast) for better json output
            if (e.op != TOKerror) // avoid duplicate diagnostics
            {
                assert(emprev.origValue);
                origValue = new AddExp(loc, emprev.origValue, new IntegerExp(loc, 1, Type.tint32));
                origValue = origValue.semantic(sc);
                origValue = origValue.ctfeInterpret();
            }

            if (e.op == TOKerror)
                return errorReturn();
            if (e.type.isfloating())
            {
                // Check that e != eprev (not always true for floats)
                Expression etest = new EqualExp(TOKequal, loc, e, eprev);
                etest = etest.semantic(sc);
                etest = etest.ctfeInterpret();
                if (etest.toInteger())
                {
                    error("has inexact value, due to loss of precision");
                    return errorReturn();
                }
            }
            value = e;
        }
        if (!origType)
            type = value.type;

        assert(origValue);
        semanticRun = PASSsemanticdone;
    }

    Expression getVarExp(Loc loc, Scope* sc)
    {
        semantic(sc);
        if (errors)
            return new ErrorExp();
        Expression e = new VarExp(loc, this);
        return e.semantic(sc);
    }

    override inout(EnumMember) isEnumMember() inout
    {
        return this;
    }

    override void accept(Visitor v)
    {
        v.visit(this);
    }
}
