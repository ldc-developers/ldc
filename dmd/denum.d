/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/denum.d, _denum.d)
 * Documentation:  https://dlang.org/phobos/dmd_denum.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/denum.d
 */

module dmd.denum;

import dmd.gluelayer;
import dmd.declaration;
import dmd.dscope;
import dmd.dsymbol;
import dmd.dsymbolsem;
import dmd.expression;
import dmd.expressionsem;
import dmd.globals;
import dmd.id;
import dmd.identifier;
import dmd.init;
import dmd.mtype;
import dmd.tokens;
import dmd.typesem;
import dmd.visitor;

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

    extern (D) this(const ref Loc loc, Identifier id, Type memtype)
    {
        super(id);
        //printf("EnumDeclaration() %s\n", toChars());
        this.loc = loc;
        type = new TypeEnum(this);
        this.memtype = memtype;
        protection = Prot(Prot.Kind.undefined);
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
        if (semanticRun > PASS.init)
            return;
        ScopeDsymbol.setScope(sc);
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

    override Dsymbol search(const ref Loc loc, Identifier ident, int flags = SearchLocalsOnly)
    {
        //printf("%s.EnumDeclaration::search('%s')\n", toChars(), ident.toChars());
        if (_scope)
        {
            // Try one last time to resolve this enum
            dsymbolSemantic(this, _scope);
        }

        if (!members || !symtab || _scope)
        {
            error("is forward referenced when looking for `%s`", ident.toChars());
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
    Expression getMaxMinValue(const ref Loc loc, Identifier id)
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
            error(loc, "recursive definition of `.%s` property", id.toChars());
            return errorReturn();
        }
        if (*pval)
            goto Ldone;

        if (_scope)
            dsymbolSemantic(this, _scope);
        if (errors)
            return errorReturn();
        if (semanticRun == PASS.init || !members)
        {
            error("is forward referenced looking for `.%s`", id.toChars());
            return errorReturn();
        }
        if (!(memtype && memtype.isintegral()))
        {
            error(loc, "has no `.%s` property because base type `%s` is not an integral type", id.toChars(), memtype ? memtype.toChars() : "");
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
                Expression ec = new CmpExp(id == Id.max ? TOK.greaterThan : TOK.lessThan, em.loc, e, *pval);
                inuse++;
                ec = ec.expressionSemantic(em._scope);
                inuse--;
                ec = ec.ctfeInterpret();
                if (ec.toInteger())
                    *pval = e;
            }
        }
    Ldone:
        Expression e = *pval;
        if (e.op != TOK.error)
        {
            e = e.copy();
            e.loc = loc;
        }
        return e;
    }

    Expression getDefaultValue(const ref Loc loc)
    {
        //printf("EnumDeclaration::getDefaultValue() %p %s\n", this, toChars());
        if (defaultval)
            return defaultval;

        if (_scope)
            dsymbolSemantic(this, _scope);
        if (errors)
            goto Lerrors;
        if (semanticRun == PASS.init || !members)
        {
            error(loc, "forward reference of `%s.init`", toChars());
            goto Lerrors;
        }

        foreach (const i; 0 .. members.dim)
        {
            EnumMember em = (*members)[i].isEnumMember();
            if (em)
            {
                defaultval = em.value;
                return defaultval;
            }
        }

    Lerrors:
        defaultval = new ErrorExp();
        return defaultval;
    }

    Type getMemtype(const ref Loc loc)
    {
        if (_scope)
        {
            /* Enum is forward referenced. We don't need to resolve the whole thing,
             * just the base type
             */
            if (memtype)
            {
                Loc locx = loc.isValid() ? loc : this.loc;
                memtype = memtype.typeSemantic(locx, _scope);
            }
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
                Loc locx = loc.isValid() ? loc : this.loc;
                error(locx, "is forward referenced looking for base type");
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

    // A cast() is injected to 'value' after dsymbolSemantic(),
    // but 'origValue' will preserve the original value,
    // or previous value + 1 if none was specified.
    Expression origValue;

    Type origType;

    EnumDeclaration ed;

    extern (D) this(const ref Loc loc, Identifier id, Expression value, Type origType)
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

    Expression getVarExp(const ref Loc loc, Scope* sc)
    {
        dsymbolSemantic(this, sc);
        if (errors)
            return new ErrorExp();
        Expression e = new VarExp(loc, this);
        return e.expressionSemantic(sc);
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
