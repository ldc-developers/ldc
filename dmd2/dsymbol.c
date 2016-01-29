
/* Compiler implementation of the D programming language
 * Copyright (c) 1999-2014 by Digital Mars
 * All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/D-Programming-Language/dmd/blob/master/src/dsymbol.c
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "rmem.h"
#include "speller.h"
#include "aav.h"

#include "mars.h"
#include "dsymbol.h"
#include "aggregate.h"
#include "identifier.h"
#include "module.h"
#include "mtype.h"
#include "expression.h"
#include "statement.h"
#include "declaration.h"
#include "id.h"
#include "scope.h"
#include "init.h"
#include "import.h"
#include "template.h"
#include "attrib.h"
#include "enum.h"
#include "lexer.h"

#if IN_LLVM
#include "../gen/pragma.h"
#endif
const char* Pprotectionnames[] = {NULL, "none", "private", "package", "protected", "public", "export"};

/****************************** Dsymbol ******************************/

Dsymbol::Dsymbol()
{
    //printf("Dsymbol::Dsymbol(%p)\n", this);
    this->ident = NULL;
    this->parent = NULL;
#if IN_DMD
    this->csym = NULL;
    this->isym = NULL;
#endif
    this->loc = Loc();
    this->comment = NULL;
    this->scope = NULL;
    this->semanticRun = PASSinit;
    this->errors = false;
    this->depmsg = NULL;
    this->userAttribDecl = NULL;
    this->ddocUnittest = NULL;
#if IN_LLVM
    this->llvmInternal = LLVMnone;
#endif
}

Dsymbol::Dsymbol(Identifier *ident)
{
    //printf("Dsymbol::Dsymbol(%p, ident)\n", this);
    this->ident = ident;
    this->parent = NULL;
#if IN_DMD
    this->csym = NULL;
    this->isym = NULL;
#endif
    this->loc = Loc();
    this->comment = NULL;
    this->scope = NULL;
    this->semanticRun = PASSinit;
    this->errors = false;
    this->depmsg = NULL;
    this->userAttribDecl = NULL;
    this->ddocUnittest = NULL;
#if IN_LLVM
    this->llvmInternal = LLVMnone;
#endif
}

Dsymbol *Dsymbol::create(Identifier *ident)
{
    return new Dsymbol(ident);
}

bool Dsymbol::equals(RootObject *o)
{
    if (this == o)
        return true;
    Dsymbol *s = (Dsymbol *)(o);
    // Overload sets don't have an ident
    if (s && ident && s->ident && ident->equals(s->ident))
        return true;
    return false;
}

/**************************************
 * Copy the syntax.
 * Used for template instantiations.
 * If s is NULL, allocate the new object, otherwise fill it in.
 */

Dsymbol *Dsymbol::syntaxCopy(Dsymbol *s)
{
    print();
    printf("%s %s\n", kind(), toChars());
    assert(0);
    return NULL;
}

/**************************************
 * Determine if this symbol is only one.
 * Returns:
 *      false, *ps = NULL: There are 2 or more symbols
 *      true,  *ps = NULL: There are zero symbols
 *      true,  *ps = symbol: The one and only one symbol
 */

bool Dsymbol::oneMember(Dsymbol **ps, Identifier *ident)
{
    //printf("Dsymbol::oneMember()\n");
    *ps = this;
    return true;
}

/*****************************************
 * Same as Dsymbol::oneMember(), but look at an array of Dsymbols.
 */

bool Dsymbol::oneMembers(Dsymbols *members, Dsymbol **ps, Identifier *ident)
{
    //printf("Dsymbol::oneMembers() %d\n", members ? members->dim : 0);
    Dsymbol *s = NULL;

    if (members)
    {
        for (size_t i = 0; i < members->dim; i++)
        {
            Dsymbol *sx = (*members)[i];
            bool x = sx->oneMember(ps, ident);
            //printf("\t[%d] kind %s = %d, s = %p\n", i, sx->kind(), x, *ps);
            if (!x)
            {
                //printf("\tfalse 1\n");
                assert(*ps == NULL);
                return false;
            }
            if (*ps)
            {
                assert(ident);
                if (!(*ps)->ident || !(*ps)->ident->equals(ident))
                    continue;
                if (!s)
                    s = *ps;
                else if (s->isOverloadable() && (*ps)->isOverloadable())
                {
                    // keep head of overload set
                    FuncDeclaration *f1 = s->isFuncDeclaration();
                    FuncDeclaration *f2 = (*ps)->isFuncDeclaration();
                    if (f1 && f2)
                    {
                        assert(!f1->isFuncAliasDeclaration());
                        assert(!f2->isFuncAliasDeclaration());
                        for (; f1 != f2; f1 = f1->overnext0)
                        {
                            if (f1->overnext0 == NULL)
                            {
                                f1->overnext0 = f2;
                                break;
                            }
                        }
                    }
                }
                else                    // more than one symbol
                {
                    *ps = NULL;
                    //printf("\tfalse 2\n");
                    return false;
                }
            }
        }
    }
    *ps = s;            // s is the one symbol, NULL if none
    //printf("\ttrue\n");
    return true;
}

/*****************************************
 * Is Dsymbol a variable that contains pointers?
 */

bool Dsymbol::hasPointers()
{
    //printf("Dsymbol::hasPointers() %s\n", toChars());
    return false;
}

bool Dsymbol::hasStaticCtorOrDtor()
{
    //printf("Dsymbol::hasStaticCtorOrDtor() %s\n", toChars());
    return false;
}

void Dsymbol::setFieldOffset(AggregateDeclaration *ad, unsigned *poffset, bool isunion)
{
}

Identifier *Dsymbol::getIdent()
{
    return ident;
}

char *Dsymbol::toChars()
{
    return ident ? ident->toChars() : (char *)"__anonymous";
}

char *Dsymbol::toPrettyCharsHelper()
{
    return toChars();
}

const char *Dsymbol::toPrettyChars(bool QualifyTypes)
{   Dsymbol *p;
    char *s;
    char *q;
    size_t len;

    //printf("Dsymbol::toPrettyChars() '%s'\n", toChars());
    if (!parent)
        return toChars();

    len = 0;
    for (p = this; p; p = p->parent)
        len += strlen(QualifyTypes ? p->toPrettyCharsHelper() : p->toChars()) + 1;

    s = (char *)mem.xmalloc(len);
    q = s + len - 1;
    *q = 0;
    for (p = this; p; p = p->parent)
    {
        char *t = QualifyTypes ? p->toPrettyCharsHelper() : p->toChars();
        len = strlen(t);
        q -= len;
        memcpy(q, t, len);
        if (q == s)
            break;
        q--;
        *q = '.';
    }
    return s;
}

Loc& Dsymbol::getLoc()
{
    if (!loc.filename)  // avoid bug 5861.
    {
        Module *m = getModule();

        if (m && m->srcfile)
            loc.filename = m->srcfile->toChars();
    }
    return loc;
}

char *Dsymbol::locToChars()
{
    return getLoc().toChars();
}

const char *Dsymbol::kind()
{
    return "symbol";
}

/*********************************
 * If this symbol is really an alias for another,
 * return that other.
 */
Dsymbol *Dsymbol::toAlias()
{
    return this;
}

/*********************************
 * Resolve recursive tuple expansion in eponymous template.
 */
Dsymbol *Dsymbol::toAlias2()
{
    return toAlias();
}

Dsymbol *Dsymbol::toParent()
{
    return parent ? parent->pastMixin() : NULL;
}

Dsymbol *Dsymbol::pastMixin()
{
    Dsymbol *s = this;

    //printf("Dsymbol::pastMixin() %s\n", toChars());
    while (s && s->isTemplateMixin())
        s = s->parent;
    return s;
}

/**********************************
 * Use this instead of toParent() when looking for the
 * 'this' pointer of the enclosing function/class.
 * This skips over both TemplateInstance's and TemplateMixin's.
 */

Dsymbol *Dsymbol::toParent2()
{
    Dsymbol *s = parent;
    while (s && s->isTemplateInstance())
        s = s->parent;
    return s;
}

TemplateInstance *Dsymbol::isInstantiated()
{
    for (Dsymbol *s = parent; s; s = s->parent)
    {
        TemplateInstance *ti = s->isTemplateInstance();
        if (ti && !ti->isTemplateMixin())
            return ti;
    }
    return NULL;
}

// Check if this function is a member of a template which has only been
// instantiated speculatively, eg from inside is(typeof()).
// Return the speculative template instance it is part of,
// or NULL if not speculative.
TemplateInstance *Dsymbol::isSpeculative()
{
    Dsymbol *par = parent;
    while (par)
    {
        TemplateInstance *ti = par->isTemplateInstance();
        if (ti && ti->gagged)
            return ti;
        par = par->toParent();
    }
    return NULL;
}

Ungag Dsymbol::ungagSpeculative()
{
    unsigned oldgag = global.gag;

    if (global.gag && !isSpeculative() && !toParent2()->isFuncDeclaration())
        global.gag = 0;

    return Ungag(oldgag);
}

bool Dsymbol::isAnonymous()
{
    return ident == NULL;
}

/*************************************
 * Set scope for future semantic analysis so we can
 * deal better with forward references.
 */

void Dsymbol::setScope(Scope *sc)
{
    //printf("Dsymbol::setScope() %p %s, %p stc = %llx\n", this, toChars(), sc, sc->stc);
    if (!sc->nofree)
        sc->setNoFree();                // may need it even after semantic() finishes
    scope = sc;
    if (sc->depmsg)
        depmsg = sc->depmsg;

    if (!userAttribDecl)
        userAttribDecl = sc->userAttribDecl;
}

void Dsymbol::importAll(Scope *sc)
{
}

/*************************************
 * Does semantic analysis on the public face of declarations.
 */

void Dsymbol::semantic(Scope *sc)
{
    error("%p has no semantic routine", this);
}

/*************************************
 * Does semantic analysis on initializers and members of aggregates.
 */

void Dsymbol::semantic2(Scope *sc)
{
    // Most Dsymbols have no further semantic analysis needed
}

/*************************************
 * Does semantic analysis on function bodies.
 */

void Dsymbol::semantic3(Scope *sc)
{
    // Most Dsymbols have no further semantic analysis needed
}

/*********************************************
 * Search for ident as member of s.
 * Input:
 *      flags:  (see IgnoreXXX declared in dsymbol.h)
 * Returns:
 *      NULL if not found
 */

Dsymbol *Dsymbol::search(Loc loc, Identifier *ident, int flags)
{
    //printf("Dsymbol::search(this=%p,%s, ident='%s')\n", this, toChars(), ident->toChars());
    return NULL;
}

/***************************************************
 * Search for symbol with correct spelling.
 */

void *symbol_search_fp(void *arg, const char *seed, int *cost)
{
    /* If not in the lexer's string table, it certainly isn't in the symbol table.
     * Doing this first is a lot faster.
     */
    size_t len = strlen(seed);
    if (!len)
        return NULL;
    Identifier *id = Identifier::lookup(seed, len);
    if (!id)
        return NULL;

    *cost = 0;
    Dsymbol *s = (Dsymbol *)arg;
    Module::clearCache();
    return (void *)s->search(Loc(), id, IgnoreErrors);
}

Dsymbol *Dsymbol::search_correct(Identifier *ident)
{
    if (global.gag)
        return NULL;            // don't do it for speculative compiles; too time consuming

    return (Dsymbol *)speller(ident->toChars(), &symbol_search_fp, (void *)this, idchars);
}

/***************************************
 * Search for identifier id as a member of 'this'.
 * id may be a template instance.
 * Returns:
 *      symbol found, NULL if not
 */
Dsymbol *Dsymbol::searchX(Loc loc, Scope *sc, RootObject *id)
{
    //printf("Dsymbol::searchX(this=%p,%s, ident='%s')\n", this, toChars(), ident->toChars());
    Dsymbol *s = toAlias();
    Dsymbol *sm;

    if (Declaration *d = s->isDeclaration())
    {
        if (d->inuse)
        {
            ::error(loc, "circular reference to '%s'", d->toPrettyChars());
            return NULL;
        }
    }

    switch (id->dyncast())
    {
        case DYNCAST_IDENTIFIER:
            sm = s->search(loc, (Identifier *)id);
            break;

        case DYNCAST_DSYMBOL:
        {
            // It's a template instance
            //printf("\ttemplate instance id\n");
            Dsymbol *st = (Dsymbol *)id;
            TemplateInstance *ti = st->isTemplateInstance();
            sm = s->search(loc, ti->name);
            if (!sm)
            {
                sm = s->search_correct(ti->name);
                if (sm)
                    ::error(loc, "template identifier '%s' is not a member of %s '%s', did you mean %s '%s'?",
                          ti->name->toChars(), s->kind(), s->toPrettyChars(), sm->kind(), sm->toChars());
                else
                    ::error(loc, "template identifier '%s' is not a member of %s '%s'",
                          ti->name->toChars(), s->kind(), s->toPrettyChars());
                return NULL;
            }
            sm = sm->toAlias();
            TemplateDeclaration *td = sm->isTemplateDeclaration();
            if (!td)
            {
                ::error(loc, "%s.%s is not a template, it is a %s", s->toPrettyChars(), ti->name->toChars(), sm->kind());
                return NULL;
            }
            ti->tempdecl = td;
            if (!ti->semanticRun)
                ti->semantic(sc);
            sm = ti->toAlias();
            break;
        }

        case DYNCAST_TYPE:
        case DYNCAST_EXPRESSION:
        default:
            assert(0);
    }
    return sm;
}

bool Dsymbol::overloadInsert(Dsymbol *s)
{
    //printf("Dsymbol::overloadInsert('%s')\n", s->toChars());
    return false;
}

unsigned Dsymbol::size(Loc loc)
{
    error("Dsymbol '%s' has no size", toChars());
    return 0;
}

bool Dsymbol::isforwardRef()
{
    return false;
}

AggregateDeclaration *Dsymbol::isThis()
{
    return NULL;
}

AggregateDeclaration *Dsymbol::isAggregateMember()      // are we a member of an aggregate?
{
    Dsymbol *parent = toParent();
    if (parent && parent->isAggregateDeclaration())
        return (AggregateDeclaration *)parent;
    return NULL;
}

AggregateDeclaration *Dsymbol::isAggregateMember2()     // are we a member of an aggregate?
{
    Dsymbol *parent = toParent2();
    if (parent && parent->isAggregateDeclaration())
        return (AggregateDeclaration *)parent;
    return NULL;
}

ClassDeclaration *Dsymbol::isClassMember()      // are we a member of a class?
{
    AggregateDeclaration *ad = isAggregateMember();
    return ad ? ad->isClassDeclaration() : NULL;
}

bool Dsymbol::isExport()
{
    return false;
}

bool Dsymbol::isImportedSymbol()
{
    return false;
}

bool Dsymbol::isDeprecated()
{
    return false;
}

bool Dsymbol::muteDeprecationMessage()
{
    return false;
}

bool Dsymbol::isOverloadable()
{
    return false;
}

bool Dsymbol::hasOverloads()
{
    return false;
}

LabelDsymbol *Dsymbol::isLabel()                // is this a LabelDsymbol()?
{
    return NULL;
}

AggregateDeclaration *Dsymbol::isMember()       // is this a member of an AggregateDeclaration?
{
    //printf("Dsymbol::isMember() %s\n", toChars());
    Dsymbol *parent = toParent();
    //printf("parent is %s %s\n", parent->kind(), parent->toChars());
    return parent ? parent->isAggregateDeclaration() : NULL;
}

Type *Dsymbol::getType()
{
    return NULL;
}

bool Dsymbol::needThis()
{
    return false;
}

int Dsymbol::apply(Dsymbol_apply_ft_t fp, void *param)
{
    return (*fp)(this, param);
}

void Dsymbol::addMember(Scope *sc, ScopeDsymbol *sds)
{
    //printf("Dsymbol::addMember('%s')\n", toChars());
    //printf("Dsymbol::addMember(this = %p, '%s' scopesym = '%s')\n", this, toChars(), sds->toChars());
    //printf("Dsymbol::addMember(this = %p, '%s' sds = %p, sds->symtab = %p)\n", this, toChars(), sds, sds->symtab);
    parent = sds;
    if (!isAnonymous())         // no name, so can't add it to symbol table
    {
        if (!sds->symtabInsert(this))    // if name is already defined
        {
            Dsymbol *s2 = sds->symtab->lookup(ident);
            if (!s2->overloadInsert(this))
            {
                sds->multiplyDefined(Loc(), this, s2);
            }
        }
        if (sds->isAggregateDeclaration() || sds->isEnumDeclaration())
        {
            if (ident == Id::__sizeof || ident == Id::__xalignof || ident == Id::mangleof)
                error(".%s property cannot be redefined", ident->toChars());
        }
    }
}

void Dsymbol::error(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    ::verror(getLoc(), format, ap, kind(), toPrettyChars());
    va_end(ap);
}

void Dsymbol::error(Loc loc, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    ::verror(loc, format, ap, kind(), toPrettyChars());
    va_end(ap);
}

void Dsymbol::deprecation(Loc loc, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    ::vdeprecation(loc, format, ap, kind(), toPrettyChars());
    va_end(ap);
}

void Dsymbol::deprecation(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    ::vdeprecation(getLoc(), format, ap, kind(), toPrettyChars());
    va_end(ap);
}

void Dsymbol::checkDeprecated(Loc loc, Scope *sc)
{
    if (global.params.useDeprecated != 1 && isDeprecated() && !muteDeprecationMessage())
    {
        // Don't complain if we're inside a deprecated symbol's scope
        for (Dsymbol *sp = sc->parent; sp; sp = sp->parent)
        {   if (sp->isDeprecated())
                goto L1;
        }

        for (Scope *sc2 = sc; sc2; sc2 = sc2->enclosing)
        {
            if (sc2->scopesym && sc2->scopesym->isDeprecated())
                goto L1;

            // If inside a StorageClassDeclaration that is deprecated
            if (sc2->stc & STCdeprecated)
                goto L1;
        }

        char *message = NULL;
        for (Dsymbol *p = this; p; p = p->parent)
        {
            message = p->depmsg;
            if (message)
                break;
        }

        if (message)
            deprecation(loc, "is deprecated - %s", message);
        else
            deprecation(loc, "is deprecated");
    }

  L1:
    Declaration *d = isDeclaration();
    if (d && d->storage_class & STCdisable)
    {
        if (!(sc->func && sc->func->storage_class & STCdisable))
        {
            if (d->toParent() && d->isPostBlitDeclaration())
                d->toParent()->error(loc, "is not copyable because it is annotated with @disable");
            else
                error(loc, "is not callable because it is annotated with @disable");
        }
    }
}

/**********************************
 * Determine which Module a Dsymbol is in.
 */

Module *Dsymbol::getModule()
{
    //printf("Dsymbol::getModule()\n");
    if (TemplateInstance *ti = isInstantiated())
        return ti->tempdecl->getModule();

    Dsymbol *s = this;
    while (s)
    {
        //printf("\ts = %s '%s'\n", s->kind(), s->toPrettyChars());
        Module *m = s->isModule();
        if (m)
            return m;
        s = s->parent;
    }
    return NULL;
}

/**********************************
 * Determine which Module a Dsymbol is in, as far as access rights go.
 */

Module *Dsymbol::getAccessModule()
{
    //printf("Dsymbol::getAccessModule()\n");
    if (TemplateInstance *ti = isInstantiated())
        return ti->tempdecl->getAccessModule();

    Dsymbol *s = this;
    while (s)
    {
        //printf("\ts = %s '%s'\n", s->kind(), s->toPrettyChars());
        Module *m = s->isModule();
        if (m)
            return m;
        TemplateInstance *ti = s->isTemplateInstance();
        if (ti && ti->enclosing)
        {
            /* Because of local template instantiation, the parent isn't where the access
             * rights come from - it's the template declaration
             */
            s = ti->tempdecl;
        }
        else
            s = s->parent;
    }
    return NULL;
}

/*************************************
 */

Prot Dsymbol::prot()
{
    return Prot(PROTpublic);
}

/*************************************
 * Do syntax copy of an array of Dsymbol's.
 */

Dsymbols *Dsymbol::arraySyntaxCopy(Dsymbols *a)
{

    Dsymbols *b = NULL;
    if (a)
    {
        b = a->copy();
        for (size_t i = 0; i < b->dim; i++)
        {
            (*b)[i] = (*b)[i]->syntaxCopy(NULL);
        }
    }
    return b;
}

/****************************************
 * Add documentation comment to Dsymbol.
 * Ignore NULL comments.
 */

void Dsymbol::addComment(const utf8_t *comment)
{
    //if (comment)
        //printf("adding comment '%s' to symbol %p '%s'\n", comment, this, toChars());

    if (!this->comment)
        this->comment = comment;
    else if (comment && strcmp((char *)comment, (char *)this->comment) != 0)
    {   // Concatenate the two
        this->comment = Lexer::combineComments(this->comment, comment);
    }
}

/****************************************
 * Returns true if this symbol is defined in a non-root module without instantiation.
 */
bool Dsymbol::inNonRoot()
{
    Dsymbol *s = parent;
    for (; s; s = s->toParent())
    {
        if (TemplateInstance *ti = s->isTemplateInstance())
        {
            return false;
        }
        if (Module *m = s->isModule())
        {
            if (!m->isRoot())
                return true;
            break;
        }
    }
    return false;
}

/********************************* OverloadSet ****************************/

OverloadSet::OverloadSet(Identifier *ident)
    : Dsymbol(ident)
{
}

void OverloadSet::push(Dsymbol *s)
{
    a.push(s);
}

const char *OverloadSet::kind()
{
    return "overloadset";
}


/********************************* ScopeDsymbol ****************************/

ScopeDsymbol::ScopeDsymbol()
    : Dsymbol()
{
    members = NULL;
    symtab = NULL;
    imports = NULL;
    prots = NULL;
}

ScopeDsymbol::ScopeDsymbol(Identifier *id)
    : Dsymbol(id)
{
    members = NULL;
    symtab = NULL;
    imports = NULL;
    prots = NULL;
}

Dsymbol *ScopeDsymbol::syntaxCopy(Dsymbol *s)
{
    //printf("ScopeDsymbol::syntaxCopy('%s')\n", toChars());
    ScopeDsymbol *sds = s ? (ScopeDsymbol *)s : new ScopeDsymbol(ident);
    sds->members = arraySyntaxCopy(members);
    return sds;
}

/*****************************************
 * This function is #1 on the list of functions that eat cpu time.
 * Be very, very careful about slowing it down.
 */

Dsymbol *ScopeDsymbol::search(Loc loc, Identifier *ident, int flags)
{
    //printf("%s->ScopeDsymbol::search(ident='%s', flags=x%x)\n", toChars(), ident->toChars(), flags);
    //if (strcmp(ident->toChars(),"c") == 0) *(char*)0=0;

    // Look in symbols declared in this module
    Dsymbol *s1 = symtab ? symtab->lookup(ident) : NULL;
    //printf("\ts1 = %p, imports = %p, %d\n", s1, imports, imports ? imports->dim : 0);
    if (s1)
    {
        //printf("\ts = '%s.%s'\n",toChars(),s1->toChars());
        return s1;
    }

    if (imports)
    {
        Dsymbol *s = NULL;
        OverloadSet *a = NULL;
        int sflags = flags & (IgnoreErrors | IgnoreAmbiguous); // remember these in recursive searches

        // Look in imported modules
        for (size_t i = 0; i < imports->dim; i++)
        {
            // If private import, don't search it
            if ((flags & IgnorePrivateMembers) && prots[i] == PROTprivate)
                continue;

            Dsymbol *ss = (*imports)[i];

            //printf("\tscanning import '%s', prots = %d, isModule = %p, isImport = %p\n", ss->toChars(), prots[i], ss->isModule(), ss->isImport());
            /* Don't find private members if ss is a module
             */
            Dsymbol *s2 = ss->search(loc, ident, sflags | (ss->isModule() ? IgnorePrivateMembers : IgnoreNone));
            if (!s)
            {
                s = s2;
                if (s && s->isOverloadSet())
                    a = mergeOverloadSet(a, s);
            }
            else if (s2 && s != s2)
            {
                if (s->toAlias() == s2->toAlias() ||
                    s->getType() == s2->getType() && s->getType())
                {
                    /* After following aliases, we found the same
                     * symbol, so it's not an ambiguity.  But if one
                     * alias is deprecated or less accessible, prefer
                     * the other.
                     */
                    if (s->isDeprecated() ||
                        s->prot().isMoreRestrictiveThan(s2->prot()) && s2->prot().kind != PROTnone)
                        s = s2;
                }
                else
                {
                    /* Two imports of the same module should be regarded as
                     * the same.
                     */
                    Import *i1 = s->isImport();
                    Import *i2 = s2->isImport();
                    if (!(i1 && i2 &&
                          (i1->mod == i2->mod ||
                           (!i1->parent->isImport() && !i2->parent->isImport() &&
                            i1->ident->equals(i2->ident))
                          )
                         )
                       )
                    {
                        /* Bugzilla 8668:
                         * Public selective import adds AliasDeclaration in module.
                         * To make an overload set, resolve aliases in here and
                         * get actual overload roots which accessible via s and s2.
                         */
                        s = s->toAlias();
                        s2 = s2->toAlias();

                        /* If both s2 and s are overloadable (though we only
                         * need to check s once)
                         */
                        if ((s2->isOverloadSet() || s2->isOverloadable()) &&
                            (a || s->isOverloadable()))
                        {
                            a = mergeOverloadSet(a, s2);
                            continue;
                        }
                        if (flags & IgnoreAmbiguous)    // if return NULL on ambiguity
                            return NULL;
                        if (!(flags & IgnoreErrors))
                            ScopeDsymbol::multiplyDefined(loc, s, s2);
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
                if (!s->isOverloadSet())
                    a = mergeOverloadSet(a, s);
                s = a;
            }

            if (!(flags & IgnoreErrors) && s->prot().kind == PROTprivate && !s->parent->isTemplateMixin())
            {
                if (!s->isImport())
                    error(loc, "%s %s is private", s->kind(), s->toPrettyChars());
            }
            return s;
        }
    }

    return s1;
}

OverloadSet *ScopeDsymbol::mergeOverloadSet(OverloadSet *os, Dsymbol *s)
{
    if (!os)
    {
        os = new OverloadSet(s->ident);
        os->parent = this;
    }
    if (OverloadSet *os2 = s->isOverloadSet())
    {
        // Merge the cross-module overload set 'os2' into 'os'
        if (os->a.dim == 0)
        {
            os->a.setDim(os2->a.dim);
            memcpy(os->a.tdata(), os2->a.tdata(), sizeof(os->a[0]) * os2->a.dim);
        }
        else
        {
            for (size_t i = 0; i < os2->a.dim; i++)
            {
                os = mergeOverloadSet(os, os2->a[i]);
            }
        }
    }
    else
    {
        assert(s->isOverloadable());

        /* Don't add to os[] if s is alias of previous sym
         */
        for (size_t j = 0; j < os->a.dim; j++)
        {
            Dsymbol *s2 = os->a[j];
            if (s->toAlias() == s2->toAlias())
            {
                if (s2->isDeprecated() ||
                    (s2->prot().isMoreRestrictiveThan(s->prot()) &&
                     s->prot().kind != PROTnone))
                {
                    os->a[j] = s;
                }
                goto Lcontinue;
            }
        }
        os->push(s);
    Lcontinue:
        ;
    }
    return os;
}

void ScopeDsymbol::importScope(Dsymbol *s, Prot protection)
{
    //printf("%s->ScopeDsymbol::importScope(%s, %d)\n", toChars(), s->toChars(), protection);

    // No circular or redundant import's
    if (s != this)
    {
        if (!imports)
            imports = new Dsymbols();
        else
        {
            for (size_t i = 0; i < imports->dim; i++)
            {
                Dsymbol *ss = (*imports)[i];
                if (ss == s)                    // if already imported
                {
                    if (protection.kind > prots[i])
                        prots[i] = protection.kind;  // upgrade access
                    return;
                }
            }
        }
        imports->push(s);
        prots = (PROTKIND *)mem.xrealloc(prots, imports->dim * sizeof(prots[0]));
        prots[imports->dim - 1] = protection.kind;
    }
}

bool ScopeDsymbol::isforwardRef()
{
    return (members == NULL);
}

void ScopeDsymbol::multiplyDefined(Loc loc, Dsymbol *s1, Dsymbol *s2)
{
#if 0
    printf("ScopeDsymbol::multiplyDefined()\n");
    printf("s1 = %p, '%s' kind = '%s', parent = %s\n", s1, s1->toChars(), s1->kind(), s1->parent ? s1->parent->toChars() : "");
    printf("s2 = %p, '%s' kind = '%s', parent = %s\n", s2, s2->toChars(), s2->kind(), s2->parent ? s2->parent->toChars() : "");
#endif
    if (loc.filename)
    {   ::error(loc, "%s at %s conflicts with %s at %s",
            s1->toPrettyChars(),
            s1->locToChars(),
            s2->toPrettyChars(),
            s2->locToChars());
    }
    else
    {
        s1->error(s1->loc, "conflicts with %s %s at %s",
            s2->kind(),
            s2->toPrettyChars(),
            s2->locToChars());
    }
}

const char *ScopeDsymbol::kind()
{
    return "ScopeDsymbol";
}

Dsymbol *ScopeDsymbol::symtabInsert(Dsymbol *s)
{
    return symtab->insert(s);
}

/****************************************
 * Return true if any of the members are static ctors or static dtors, or if
 * any members have members that are.
 */

bool ScopeDsymbol::hasStaticCtorOrDtor()
{
    if (members)
    {
        for (size_t i = 0; i < members->dim; i++)
        {   Dsymbol *member = (*members)[i];

            if (member->hasStaticCtorOrDtor())
                return true;
        }
    }
    return false;
}

/***************************************
 * Determine number of Dsymbols, folding in AttribDeclaration members.
 */

static int dimDg(void *ctx, size_t n, Dsymbol *)
{
    ++*(size_t *)ctx;
    return 0;
}

size_t ScopeDsymbol::dim(Dsymbols *members)
{
    size_t n = 0;
    foreach(NULL, members, &dimDg, &n);
    return n;
}

/***************************************
 * Get nth Dsymbol, folding in AttribDeclaration members.
 * Returns:
 *      Dsymbol*        nth Dsymbol
 *      NULL            not found, *pn gets incremented by the number
 *                      of Dsymbols
 */

struct GetNthSymbolCtx
{
    size_t nth;
    Dsymbol *sym;
};

static int getNthSymbolDg(void *ctx, size_t n, Dsymbol *sym)
{
    GetNthSymbolCtx *p = (GetNthSymbolCtx *)ctx;
    if (n == p->nth)
    {   p->sym = sym;
        return 1;
    }
    return 0;
}

Dsymbol *ScopeDsymbol::getNth(Dsymbols *members, size_t nth, size_t *pn)
{
    GetNthSymbolCtx ctx = { nth, NULL };
    int res = foreach(NULL, members, &getNthSymbolDg, &ctx);
    return res ? ctx.sym : NULL;
}

/***************************************
 * Expands attribute declarations in members in depth first
 * order. Calls dg(void *ctx, size_t symidx, Dsymbol *sym) for each
 * member.
 * If dg returns !=0, stops and returns that value else returns 0.
 * Use this function to avoid the O(N + N^2/2) complexity of
 * calculating dim and calling N times getNth.
 */

int ScopeDsymbol::foreach(Scope *sc, Dsymbols *members, ScopeDsymbol::ForeachDg dg, void *ctx, size_t *pn)
{
    assert(dg);
    if (!members)
        return 0;

    size_t n = pn ? *pn : 0; // take over index
    int result = 0;
    for (size_t i = 0; i < members->dim; i++)
    {   Dsymbol *s = (*members)[i];

        if (AttribDeclaration *a = s->isAttribDeclaration())
            result = foreach(sc, a->include(sc, NULL), dg, ctx, &n);
        else if (TemplateMixin *tm = s->isTemplateMixin())
            result = foreach(sc, tm->members, dg, ctx, &n);
        else if (s->isTemplateInstance())
            ;
        else if (s->isUnitTestDeclaration())
            ;
        else
            result = dg(ctx, n++, s);

        if (result)
            break;
    }

    if (pn)
        *pn = n; // update index
    return result;
}

/*******************************************
 * Look for member of the form:
 *      const(MemberInfo)[] getMembers(string);
 * Returns NULL if not found
 */

FuncDeclaration *ScopeDsymbol::findGetMembers()
{
    Dsymbol *s = search_function(this, Id::getmembers);
    FuncDeclaration *fdx = s ? s->isFuncDeclaration() : NULL;

#if 0  // Finish
    static TypeFunction *tfgetmembers;

    if (!tfgetmembers)
    {
        Scope sc;
        Parameters *parameters = new Parameters;
        Parameters *p = new Parameter(STCin, Type::tchar->constOf()->arrayOf(), NULL, NULL);
        parameters->push(p);

        Type *tret = NULL;
        tfgetmembers = new TypeFunction(parameters, tret, 0, LINKd);
        tfgetmembers = (TypeFunction *)tfgetmembers->semantic(Loc(), &sc);
    }
    if (fdx)
        fdx = fdx->overloadExactMatch(tfgetmembers);
#endif
    if (fdx && fdx->isVirtual())
        fdx = NULL;

    return fdx;
}


/****************************** WithScopeSymbol ******************************/

WithScopeSymbol::WithScopeSymbol(WithStatement *withstate)
    : ScopeDsymbol()
{
    this->withstate = withstate;
}

Dsymbol *WithScopeSymbol::search(Loc loc, Identifier *ident, int flags)
{
    // Acts as proxy to the with class declaration
    Dsymbol *s = NULL;
    Expression *eold = NULL;
    for (Expression *e = withstate->exp; e != eold; e = resolveAliasThis(scope, e))
    {
        if (e->op == TOKimport)
        {
            s = ((ScopeExp *)e)->sds;
        }
        else if (e->op == TOKtype)
        {
            s = e->type->toDsymbol(NULL);
        }
        else
        {
            Type *t = e->type->toBasetype();
            s = t->toDsymbol(NULL);
        }
        if (s)
        {
            s = s->search(loc, ident);
            if (s)
                return s;
        }
        eold = e;
    }
    return NULL;
}

/****************************** ArrayScopeSymbol ******************************/

ArrayScopeSymbol::ArrayScopeSymbol(Scope *sc, Expression *e)
    : ScopeDsymbol()
{
    assert(e->op == TOKindex || e->op == TOKslice || e->op == TOKarray);
    exp = e;
    type = NULL;
    td = NULL;
    this->sc = sc;
}

ArrayScopeSymbol::ArrayScopeSymbol(Scope *sc, TypeTuple *t)
    : ScopeDsymbol()
{
    exp = NULL;
    type = t;
    td = NULL;
    this->sc = sc;
}

ArrayScopeSymbol::ArrayScopeSymbol(Scope *sc, TupleDeclaration *s)
    : ScopeDsymbol()
{
    exp = NULL;
    type = NULL;
    td = s;
    this->sc = sc;
}

Dsymbol *ArrayScopeSymbol::search(Loc loc, Identifier *ident, int flags)
{
    //printf("ArrayScopeSymbol::search('%s', flags = %d)\n", ident->toChars(), flags);
    if (ident == Id::dollar)
    {
        VarDeclaration **pvar;
        Expression *ce;

    L1:
        if (td)
        {
            /* $ gives the number of elements in the tuple
             */
            VarDeclaration *v = new VarDeclaration(loc, Type::tsize_t, Id::dollar, NULL);
            Expression *e = new IntegerExp(Loc(), td->objects->dim, Type::tsize_t);
            v->init = new ExpInitializer(Loc(), e);
            v->storage_class |= STCtemp | STCstatic | STCconst;
            v->semantic(sc);
            return v;
        }

        if (type)
        {
            /* $ gives the number of type entries in the type tuple
             */
            VarDeclaration *v = new VarDeclaration(loc, Type::tsize_t, Id::dollar, NULL);
            Expression *e = new IntegerExp(Loc(), type->arguments->dim, Type::tsize_t);
            v->init = new ExpInitializer(Loc(), e);
            v->storage_class |= STCtemp | STCstatic | STCconst;
            v->semantic(sc);
            return v;
        }

        if (exp->op == TOKindex)
        {
            /* array[index] where index is some function of $
             */
            IndexExp *ie = (IndexExp *)exp;
            pvar = &ie->lengthVar;
            ce = ie->e1;
        }
        else if (exp->op == TOKslice)
        {
            /* array[lwr .. upr] where lwr or upr is some function of $
             */
            SliceExp *se = (SliceExp *)exp;
            pvar = &se->lengthVar;
            ce = se->e1;
        }
        else if (exp->op == TOKarray)
        {
            /* array[e0, e1, e2, e3] where e0, e1, e2 are some function of $
             * $ is a opDollar!(dim)() where dim is the dimension(0,1,2,...)
             */
            ArrayExp *ae = (ArrayExp *)exp;
            pvar = &ae->lengthVar;
            ce = ae->e1;
        }
        else
        {
            /* Didn't find $, look in enclosing scope(s).
             */
            return NULL;
        }

        while (ce->op == TOKcomma)
            ce = ((CommaExp *)ce)->e2;

        /* If we are indexing into an array that is really a type
         * tuple, rewrite this as an index into a type tuple and
         * try again.
         */
        if (ce->op == TOKtype)
        {
            Type *t = ((TypeExp *)ce)->type;
            if (t->ty == Ttuple)
            {
                type = (TypeTuple *)t;
                goto L1;
            }
        }

        /* *pvar is lazily initialized, so if we refer to $
         * multiple times, it gets set only once.
         */
        if (!*pvar)             // if not already initialized
        {
            /* Create variable v and set it to the value of $
             */
            VarDeclaration *v;
            Type *t;
            if (ce->op == TOKtuple)
            {
                /* It is for an expression tuple, so the
                 * length will be a const.
                 */
                Expression *e = new IntegerExp(Loc(), ((TupleExp *)ce)->exps->dim, Type::tsize_t);
                v = new VarDeclaration(loc, Type::tsize_t, Id::dollar, new ExpInitializer(Loc(), e));
                v->storage_class |= STCtemp | STCstatic | STCconst;
            }
            else if (ce->type && (t = ce->type->toBasetype()) != NULL &&
                     (t->ty == Tstruct || t->ty == Tclass))
            {
                // Look for opDollar
                assert(exp->op == TOKarray || exp->op == TOKslice);
                AggregateDeclaration *ad = isAggregate(t);
                assert(ad);

                Dsymbol *s = ad->search(loc, Id::opDollar);
                if (!s)  // no dollar exists -- search in higher scope
                    return NULL;
                s = s->toAlias();

                Expression *e = NULL;
                // Check for multi-dimensional opDollar(dim) template.
                if (TemplateDeclaration *td = s->isTemplateDeclaration())
                {
                    dinteger_t dim = 0;
                    if (exp->op == TOKarray)
                    {
                        dim = ((ArrayExp *)exp)->currentDimension;
                    }
                    else if (exp->op == TOKslice)
                    {
                        dim = 0; // slices are currently always one-dimensional
                    }
                    else
                    {
                        assert(0);
                    }

                    Objects *tiargs = new Objects();
                    Expression *edim = new IntegerExp(Loc(), dim, Type::tsize_t);
                    edim = edim->semantic(sc);
                    tiargs->push(edim);
                    e = new DotTemplateInstanceExp(loc, ce, td->ident, tiargs);
                }
                else
                {
                    /* opDollar exists, but it's not a template.
                     * This is acceptable ONLY for single-dimension indexing.
                     * Note that it's impossible to have both template & function opDollar,
                     * because both take no arguments.
                     */
                    if (exp->op == TOKarray && ((ArrayExp *)exp)->arguments->dim != 1)
                    {
                        exp->error("%s only defines opDollar for one dimension", ad->toChars());
                        return NULL;
                    }
                    Declaration *d = s->isDeclaration();
                    assert(d);
                    e = new DotVarExp(loc, ce, d);
                }
                e = e->semantic(sc);
                if (!e->type)
                    exp->error("%s has no value", e->toChars());
                t = e->type->toBasetype();
                if (t && t->ty == Tfunction)
                    e = new CallExp(e->loc, e);
                v = new VarDeclaration(loc, NULL, Id::dollar, new ExpInitializer(Loc(), e));
                v->storage_class |= STCtemp | STCctfe | STCrvalue;
            }
            else
            {
                /* For arrays, $ will either be a compile-time constant
                 * (in which case its value in set during constant-folding),
                 * or a variable (in which case an expression is created in
                 * toir.c).
                 */
                VoidInitializer *e = new VoidInitializer(Loc());
                e->type = Type::tsize_t;
                v = new VarDeclaration(loc, Type::tsize_t, Id::dollar, e);
                v->storage_class |= STCtemp | STCctfe; // it's never a true static variable
            }
            *pvar = v;
        }
        (*pvar)->semantic(sc);
        return (*pvar);
    }
    return NULL;
}


/****************************** DsymbolTable ******************************/

DsymbolTable::DsymbolTable()
{
    tab = NULL;
}

Dsymbol *DsymbolTable::lookup(Identifier *ident)
{
    //printf("DsymbolTable::lookup(%s)\n", (char*)ident->string);
    return (Dsymbol *)dmd_aaGetRvalue(tab, (void *)ident);
}

Dsymbol *DsymbolTable::insert(Dsymbol *s)
{
    //printf("DsymbolTable::insert(this = %p, '%s')\n", this, s->ident->toChars());
    Identifier *ident = s->ident;
    Dsymbol **ps = (Dsymbol **)dmd_aaGet(&tab, (void *)ident);
    if (*ps)
        return NULL;            // already in table
    *ps = s;
    return s;
}

Dsymbol *DsymbolTable::insert(Identifier *ident, Dsymbol *s)
{
    //printf("DsymbolTable::insert()\n");
    Dsymbol **ps = (Dsymbol **)dmd_aaGet(&tab, (void *)ident);
    if (*ps)
        return NULL;            // already in table
    *ps = s;
    return s;
}

Dsymbol *DsymbolTable::update(Dsymbol *s)
{
    Identifier *ident = s->ident;
    Dsymbol **ps = (Dsymbol **)dmd_aaGet(&tab, (void *)ident);
    *ps = s;
    return s;
}

/****************************** Prot ******************************/

Prot::Prot()
{
    this->kind = PROTundefined;
    this->pkg = NULL;
}

Prot::Prot(PROTKIND kind)
{
    this->kind = kind;
    this->pkg = NULL;
}

/**
 * Checks if `this` is superset of `other` restrictions.
 * For example, "protected" is more restrictive than "public".
 */
bool Prot::isMoreRestrictiveThan(Prot other)
{
    return this->kind < other.kind;
}

/**
 * Checks if `this` is absolutely identical protection attribute to `other`
 */
bool Prot::operator==(Prot other)
{
    if (this->kind == other.kind)
    {
        if (this->kind == PROTpackage)
            return this->pkg == other.pkg;
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
bool Prot::isSubsetOf(Prot parent)
{
    if (this->kind != parent.kind)
        return false;

    if (this->kind == PROTpackage)
    {
        if (!this->pkg)
            return true;
        if (!parent.pkg)
            return false;
        if (parent.pkg->isAncestorPackageOf(this->pkg))
            return true;
    }

    return true;
}
