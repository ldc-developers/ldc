
// Compiler implementation of the D programming language
// Copyright (c) 1999-2010 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "rmem.h"

#include "init.h"
#include "declaration.h"
#include "attrib.h"
#include "cond.h"
#include "scope.h"
#include "id.h"
#include "expression.h"
#include "dsymbol.h"
#include "aggregate.h"
#include "module.h"
#include "parse.h"
#include "template.h"
#if TARGET_NET
 #include "frontend.net/pragma.h"
#endif

#if IN_LLVM
#include "../gen/enums.h"

#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<bool> ignoreUnsupportedPragmas("ignore",
    llvm::cl::desc("Ignore unsupported pragmas"),
    llvm::cl::ZeroOrMore);

#endif


extern void obj_includelib(const char *name);

#if IN_DMD
void obj_startaddress(Symbol *s);
#endif


/********************************* AttribDeclaration ****************************/

AttribDeclaration::AttribDeclaration(Array *decl)
        : Dsymbol()
{
    this->decl = decl;
}

Array *AttribDeclaration::include(Scope *sc, ScopeDsymbol *sd)
{
    return decl;
}

int AttribDeclaration::addMember(Scope *sc, ScopeDsymbol *sd, int memnum)
{
    int m = 0;
    Array *d = include(sc, sd);

    if (d)
    {
        for (unsigned i = 0; i < d->dim; i++)
        {   Dsymbol *s = (Dsymbol *)d->data[i];
            m |= s->addMember(sc, sd, m | memnum);
        }
    }
    return m;
}

void AttribDeclaration::setScopeNewSc(Scope *sc,
        StorageClass stc, enum LINK linkage, enum PROT protection, int explicitProtection,
        unsigned structalign)
{
    if (decl)
    {
        Scope *newsc = sc;
        if (stc != sc->stc ||
            linkage != sc->linkage ||
            protection != sc->protection ||
            explicitProtection != sc->explicitProtection ||
            structalign != sc->structalign)
        {
            // create new one for changes
            newsc = new Scope(*sc);
            newsc->flags &= ~SCOPEfree;
            newsc->stc = stc;
            newsc->linkage = linkage;
            newsc->protection = protection;
            newsc->explicitProtection = explicitProtection;
            newsc->structalign = structalign;
        }
        for (unsigned i = 0; i < decl->dim; i++)
        {   Dsymbol *s = (Dsymbol *)decl->data[i];

            s->setScope(newsc); // yes, the only difference from semanticNewSc()
        }
        if (newsc != sc)
        {
            sc->offset = newsc->offset;
            newsc->pop();
        }
    }
}

void AttribDeclaration::semanticNewSc(Scope *sc,
        StorageClass stc, enum LINK linkage, enum PROT protection, int explicitProtection,
        unsigned structalign)
{
    if (decl)
    {
        Scope *newsc = sc;
        if (stc != sc->stc ||
            linkage != sc->linkage ||
            protection != sc->protection ||
            explicitProtection != sc->explicitProtection ||
            structalign != sc->structalign)
        {
            // create new one for changes
            newsc = new Scope(*sc);
            newsc->flags &= ~SCOPEfree;
            newsc->stc = stc;
            newsc->linkage = linkage;
            newsc->protection = protection;
            newsc->explicitProtection = explicitProtection;
            newsc->structalign = structalign;
        }
        for (unsigned i = 0; i < decl->dim; i++)
        {   Dsymbol *s = (Dsymbol *)decl->data[i];

            s->semantic(newsc);
        }
        if (newsc != sc)
        {
            sc->offset = newsc->offset;
            newsc->pop();
        }
    }
}

void AttribDeclaration::semantic(Scope *sc)
{
    Array *d = include(sc, NULL);

    //printf("\tAttribDeclaration::semantic '%s', d = %p\n",toChars(), d);
    if (d)
    {
        for (unsigned i = 0; i < d->dim; i++)
        {
            Dsymbol *s = (Dsymbol *)d->data[i];

            s->semantic(sc);
        }
    }
}

void AttribDeclaration::semantic2(Scope *sc)
{
    Array *d = include(sc, NULL);

    if (d)
    {
        for (unsigned i = 0; i < d->dim; i++)
        {   Dsymbol *s = (Dsymbol *)d->data[i];
            s->semantic2(sc);
        }
    }
}

void AttribDeclaration::semantic3(Scope *sc)
{
    Array *d = include(sc, NULL);

    if (d)
    {
        for (unsigned i = 0; i < d->dim; i++)
        {   Dsymbol *s = (Dsymbol *)d->data[i];
            s->semantic3(sc);
        }
    }
}

void AttribDeclaration::inlineScan()
{
    Array *d = include(NULL, NULL);

    if (d)
    {
        for (unsigned i = 0; i < d->dim; i++)
        {   Dsymbol *s = (Dsymbol *)d->data[i];
            //printf("AttribDeclaration::inlineScan %s\n", s->toChars());
            s->inlineScan();
        }
    }
}

void AttribDeclaration::addComment(unsigned char *comment)
{
    if (comment)
    {
        Array *d = include(NULL, NULL);

        if (d)
        {
            for (unsigned i = 0; i < d->dim; i++)
            {   Dsymbol *s = (Dsymbol *)d->data[i];
                //printf("AttribDeclaration::addComment %s\n", s->toChars());
                s->addComment(comment);
            }
        }
    }
}

void AttribDeclaration::emitComment(Scope *sc)
{
    //printf("AttribDeclaration::emitComment(sc = %p)\n", sc);

    /* A general problem with this, illustrated by BUGZILLA 2516,
     * is that attributes are not transmitted through to the underlying
     * member declarations for template bodies, because semantic analysis
     * is not done for template declaration bodies
     * (only template instantiations).
     * Hence, Ddoc omits attributes from template members.
     */

    Array *d = include(NULL, NULL);

    if (d)
    {
        for (unsigned i = 0; i < d->dim; i++)
        {   Dsymbol *s = (Dsymbol *)d->data[i];
            //printf("AttribDeclaration::emitComment %s\n", s->toChars());
            s->emitComment(sc);
        }
    }
}

#if IN_DMD

void AttribDeclaration::toObjFile(int multiobj)
{
    Array *d = include(NULL, NULL);

    if (d)
    {
        for (unsigned i = 0; i < d->dim; i++)
        {   Dsymbol *s = (Dsymbol *)d->data[i];
            s->toObjFile(multiobj);
        }
    }
}

int AttribDeclaration::cvMember(unsigned char *p)
{
    int nwritten = 0;
    int n;
    Array *d = include(NULL, NULL);

    if (d)
    {
        for (unsigned i = 0; i < d->dim; i++)
        {   Dsymbol *s = (Dsymbol *)d->data[i];
            n = s->cvMember(p);
            if (p)
                p += n;
            nwritten += n;
        }
    }
    return nwritten;
}
#endif

int AttribDeclaration::hasPointers()
{
    Array *d = include(NULL, NULL);

    if (d)
    {
        for (size_t i = 0; i < d->dim; i++)
        {
            Dsymbol *s = (Dsymbol *)d->data[i];
            if (s->hasPointers())
                return 1;
        }
    }
    return 0;
}

const char *AttribDeclaration::kind()
{
    return "attribute";
}

int AttribDeclaration::oneMember(Dsymbol **ps)
{
    Array *d = include(NULL, NULL);

    return Dsymbol::oneMembers(d, ps);
}

void AttribDeclaration::checkCtorConstInit()
{
    Array *d = include(NULL, NULL);

    if (d)
    {
        for (unsigned i = 0; i < d->dim; i++)
        {   Dsymbol *s = (Dsymbol *)d->data[i];
            s->checkCtorConstInit();
        }
    }
}

/****************************************
 */

void AttribDeclaration::addLocalClass(ClassDeclarations *aclasses)
{
    Array *d = include(NULL, NULL);

    if (d)
    {
        for (unsigned i = 0; i < d->dim; i++)
        {   Dsymbol *s = (Dsymbol *)d->data[i];
            s->addLocalClass(aclasses);
        }
    }
}


void AttribDeclaration::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    if (decl)
    {
        buf->writenl();
        buf->writeByte('{');
        buf->writenl();
        for (unsigned i = 0; i < decl->dim; i++)
        {
            Dsymbol *s = (Dsymbol *)decl->data[i];

            buf->writestring("    ");
            s->toCBuffer(buf, hgs);
        }
        buf->writeByte('}');
    }
    else
        buf->writeByte(';');
    buf->writenl();
}

/************************* StorageClassDeclaration ****************************/

StorageClassDeclaration::StorageClassDeclaration(StorageClass stc, Array *decl)
        : AttribDeclaration(decl)
{
    this->stc = stc;
}

Dsymbol *StorageClassDeclaration::syntaxCopy(Dsymbol *s)
{
    StorageClassDeclaration *scd;

    assert(!s);
    scd = new StorageClassDeclaration(stc, Dsymbol::arraySyntaxCopy(decl));
    return scd;
}

void StorageClassDeclaration::setScope(Scope *sc)
{
    if (decl)
    {
        StorageClass scstc = sc->stc;

        /* These sets of storage classes are mutually exclusive,
         * so choose the innermost or most recent one.
         */
        if (stc & (STCauto | STCscope | STCstatic | STCextern | STCmanifest))
            scstc &= ~(STCauto | STCscope | STCstatic | STCextern | STCmanifest);
        if (stc & (STCauto | STCscope | STCstatic | STCtls | STCmanifest | STCgshared))
            scstc &= ~(STCauto | STCscope | STCstatic | STCtls | STCmanifest | STCgshared);
        if (stc & (STCconst | STCimmutable | STCmanifest))
            scstc &= ~(STCconst | STCimmutable | STCmanifest);
        if (stc & (STCgshared | STCshared | STCtls))
            scstc &= ~(STCgshared | STCshared | STCtls);
        if (stc & (STCsafe | STCtrusted | STCsystem))
            scstc &= ~(STCsafe | STCtrusted | STCsystem);
        scstc |= stc;

        setScopeNewSc(sc, scstc, sc->linkage, sc->protection, sc->explicitProtection, sc->structalign);
    }
}

void StorageClassDeclaration::semantic(Scope *sc)
{
    if (decl)
    {
        StorageClass scstc = sc->stc;

        /* These sets of storage classes are mutually exclusive,
         * so choose the innermost or most recent one.
         */
        if (stc & (STCauto | STCscope | STCstatic | STCextern | STCmanifest))
            scstc &= ~(STCauto | STCscope | STCstatic | STCextern | STCmanifest);
        if (stc & (STCauto | STCscope | STCstatic | STCtls | STCmanifest | STCgshared))
            scstc &= ~(STCauto | STCscope | STCstatic | STCtls | STCmanifest | STCgshared);
        if (stc & (STCconst | STCimmutable | STCmanifest))
            scstc &= ~(STCconst | STCimmutable | STCmanifest);
        if (stc & (STCgshared | STCshared | STCtls))
            scstc &= ~(STCgshared | STCshared | STCtls);
        if (stc & (STCsafe | STCtrusted | STCsystem))
            scstc &= ~(STCsafe | STCtrusted | STCsystem);
        scstc |= stc;

        semanticNewSc(sc, scstc, sc->linkage, sc->protection, sc->explicitProtection, sc->structalign);
    }
}

void StorageClassDeclaration::stcToCBuffer(OutBuffer *buf, StorageClass stc)
{
    struct SCstring
    {
        StorageClass stc;
        enum TOK tok;
    };

    static SCstring table[] =
    {
        { STCauto,         TOKauto },
        { STCscope,        TOKscope },
        { STCstatic,       TOKstatic },
        { STCextern,       TOKextern },
        { STCconst,        TOKconst },
        { STCfinal,        TOKfinal },
        { STCabstract,     TOKabstract },
        { STCsynchronized, TOKsynchronized },
        { STCdeprecated,   TOKdeprecated },
        { STCoverride,     TOKoverride },
        { STClazy,         TOKlazy },
        { STCalias,        TOKalias },
        { STCout,          TOKout },
        { STCin,           TOKin },
#if DMDV2
        { STCimmutable,    TOKimmutable },
        { STCshared,       TOKshared },
        { STCnothrow,      TOKnothrow },
        { STCpure,         TOKpure },
        { STCref,          TOKref },
        { STCtls,          TOKtls },
        { STCgshared,      TOKgshared },
        { STCproperty,     TOKat },
        { STCsafe,         TOKat },
        { STCtrusted,      TOKat },
        { STCdisable,       TOKat },
#endif
    };

    for (int i = 0; i < sizeof(table)/sizeof(table[0]); i++)
    {
        if (stc & table[i].stc)
        {
            enum TOK tok = table[i].tok;
#if DMDV2
            if (tok == TOKat)
            {   Identifier *id;

                if (stc & STCproperty)
                    id = Id::property;
                else if (stc & STCsafe)
                    id = Id::safe;
                else if (stc & STCtrusted)
                    id = Id::trusted;
                else if (stc & STCdisable)
                    id = Id::disable;
                else
                    assert(0);
                buf->writeByte('@');
                buf->writestring(id->toChars());
            }
            else
#endif
                buf->writestring(Token::toChars(tok));
            buf->writeByte(' ');
        }
    }
}

void StorageClassDeclaration::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    stcToCBuffer(buf, stc);
    AttribDeclaration::toCBuffer(buf, hgs);
}

/********************************* LinkDeclaration ****************************/

LinkDeclaration::LinkDeclaration(enum LINK p, Array *decl)
        : AttribDeclaration(decl)
{
    //printf("LinkDeclaration(linkage = %d, decl = %p)\n", p, decl);
    linkage = p;
}

Dsymbol *LinkDeclaration::syntaxCopy(Dsymbol *s)
{
    LinkDeclaration *ld;

    assert(!s);
    ld = new LinkDeclaration(linkage, Dsymbol::arraySyntaxCopy(decl));
    return ld;
}

void LinkDeclaration::setScope(Scope *sc)
{
    //printf("LinkDeclaration::setScope(linkage = %d, decl = %p)\n", linkage, decl);
    if (decl)
    {
        setScopeNewSc(sc, sc->stc, linkage, sc->protection, sc->explicitProtection, sc->structalign);
    }
}

void LinkDeclaration::semantic(Scope *sc)
{
    //printf("LinkDeclaration::semantic(linkage = %d, decl = %p)\n", linkage, decl);
    if (decl)
    {
        semanticNewSc(sc, sc->stc, linkage, sc->protection, sc->explicitProtection, sc->structalign);
    }
}

void LinkDeclaration::semantic3(Scope *sc)
{
    //printf("LinkDeclaration::semantic3(linkage = %d, decl = %p)\n", linkage, decl);
    if (decl)
    {   enum LINK linkage_save = sc->linkage;

        sc->linkage = linkage;
        for (unsigned i = 0; i < decl->dim; i++)
        {
            Dsymbol *s = (Dsymbol *)decl->data[i];

            s->semantic3(sc);
        }
        sc->linkage = linkage_save;
    }
    else
    {
        sc->linkage = linkage;
    }
}

void LinkDeclaration::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{   const char *p;

    switch (linkage)
    {
        case LINKd:             p = "D";                break;
        case LINKc:             p = "C";                break;
        case LINKcpp:           p = "C++";              break;
        case LINKwindows:       p = "Windows";          break;
        case LINKpascal:        p = "Pascal";           break;

    // LDC
    case LINKintrinsic: p = "Intrinsic"; break;

        default:
            assert(0);
            break;
    }
    buf->writestring("extern (");
    buf->writestring(p);
    buf->writestring(") ");
    AttribDeclaration::toCBuffer(buf, hgs);
}

char *LinkDeclaration::toChars()
{
    return (char *)"extern ()";
}

/********************************* ProtDeclaration ****************************/

ProtDeclaration::ProtDeclaration(enum PROT p, Array *decl)
        : AttribDeclaration(decl)
{
    protection = p;
    //printf("decl = %p\n", decl);
}

Dsymbol *ProtDeclaration::syntaxCopy(Dsymbol *s)
{
    ProtDeclaration *pd;

    assert(!s);
    pd = new ProtDeclaration(protection, Dsymbol::arraySyntaxCopy(decl));
    return pd;
}

void ProtDeclaration::setScope(Scope *sc)
{
    if (decl)
    {
        setScopeNewSc(sc, sc->stc, sc->linkage, protection, 1, sc->structalign);
    }
}

void ProtDeclaration::importAll(Scope *sc)
{
    Scope *newsc = sc;
    if (sc->protection != protection ||
       sc->explicitProtection != 1)
    {
       // create new one for changes
       newsc = new Scope(*sc);
       newsc->flags &= ~SCOPEfree;
       newsc->protection = protection;
       newsc->explicitProtection = 1;
    }

    for (int i = 0; i < decl->dim; i++)
    {
       Dsymbol *s = (Dsymbol *)decl->data[i];
       s->importAll(newsc);
    }

    if (newsc != sc)
       newsc->pop();
}

void ProtDeclaration::semantic(Scope *sc)
{
    if (decl)
    {
        semanticNewSc(sc, sc->stc, sc->linkage, protection, 1, sc->structalign);
    }
}

void ProtDeclaration::protectionToCBuffer(OutBuffer *buf, enum PROT protection)
{
    const char *p;

    switch (protection)
    {
        case PROTprivate:       p = "private";          break;
        case PROTpackage:       p = "package";          break;
        case PROTprotected:     p = "protected";        break;
        case PROTpublic:        p = "public";           break;
        case PROTexport:        p = "export";           break;
        default:
            assert(0);
            break;
    }
    buf->writestring(p);
    buf->writeByte(' ');
}

void ProtDeclaration::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    protectionToCBuffer(buf, protection);
    AttribDeclaration::toCBuffer(buf, hgs);
}

/********************************* AlignDeclaration ****************************/

AlignDeclaration::AlignDeclaration(Loc loc, unsigned sa, Array *decl)
        : AttribDeclaration(decl)
{
    this->loc = loc;
    salign = sa;
}

Dsymbol *AlignDeclaration::syntaxCopy(Dsymbol *s)
{
    AlignDeclaration *ad;

    assert(!s);
    ad = new AlignDeclaration(loc, salign, Dsymbol::arraySyntaxCopy(decl));
    return ad;
}

void AlignDeclaration::setScope(Scope *sc)
{
    //printf("\tAlignDeclaration::setScope '%s'\n",toChars());
    if (decl)
    {
        setScopeNewSc(sc, sc->stc, sc->linkage, sc->protection, sc->explicitProtection, salign);
    }
}

void AlignDeclaration::semantic(Scope *sc)
{
// LDC
// we only support packed structs, as from the spec: align(1) struct Packed { ... }
// other alignments are simply ignored. my tests show this is what llvm-gcc does too ...
    {
        semanticNewSc(sc, sc->stc, sc->linkage, sc->protection, sc->explicitProtection, salign);
    }
}


void AlignDeclaration::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->printf("align (%d)", salign);
    AttribDeclaration::toCBuffer(buf, hgs);
}

/********************************* AnonDeclaration ****************************/

AnonDeclaration::AnonDeclaration(Loc loc, int isunion, Array *decl)
        : AttribDeclaration(decl)
{
    this->loc = loc;
    this->isunion = isunion;
    this->sem = 0;
}

Dsymbol *AnonDeclaration::syntaxCopy(Dsymbol *s)
{
    AnonDeclaration *ad;

    assert(!s);
    ad = new AnonDeclaration(loc, isunion, Dsymbol::arraySyntaxCopy(decl));
    return ad;
}

void AnonDeclaration::semantic(Scope *sc)
{
    //printf("\tAnonDeclaration::semantic %s %p\n", isunion ? "union" : "struct", this);

    Scope *scx = NULL;
    if (scope)
    {   sc = scope;
        scx = scope;
        scope = NULL;
    }

    unsigned dprogress_save = Module::dprogress;

    assert(sc->parent);

    Dsymbol *parent = sc->parent->pastMixin();
    AggregateDeclaration *ad = parent->isAggregateDeclaration();

    if (!ad || (!ad->isStructDeclaration() && !ad->isClassDeclaration()))
    {
        error("can only be a part of an aggregate");
        return;
    }

    if (decl)
    {
        AnonymousAggregateDeclaration aad;
        int adisunion;

        if (sc->anonAgg)
        {   ad = sc->anonAgg;
            adisunion = sc->inunion;
        }
        else
            adisunion = ad->isUnionDeclaration() != NULL;

//      printf("\tsc->anonAgg = %p\n", sc->anonAgg);
//      printf("\tad  = %p\n", ad);
//      printf("\taad = %p\n", &aad);

        sc = sc->push();
        sc->anonAgg = &aad;
        sc->stc &= ~(STCauto | STCscope | STCstatic | STCtls | STCgshared);
        sc->inunion = isunion;
        sc->offset = 0;
        sc->flags = 0;
        aad.structalign = sc->structalign;
        aad.parent = ad;

        for (unsigned i = 0; i < decl->dim; i++)
        {
            Dsymbol *s = (Dsymbol *)decl->data[i];

            s->semantic(sc);
            if (isunion)
                sc->offset = 0;
            if (aad.sizeok == 2)
            {
                break;
            }
        }
        sc = sc->pop();

        // If failed due to forward references, unwind and try again later
        if (aad.sizeok == 2)
        {
            ad->sizeok = 2;
            //printf("\tsetting ad->sizeok %p to 2\n", ad);
            if (!sc->anonAgg)
            {
                scope = scx ? scx : new Scope(*sc);
                scope->setNoFree();
                scope->module->addDeferredSemantic(this);
            }
            Module::dprogress = dprogress_save;
            //printf("\tforward reference %p\n", this);
            return;
        }
        if (sem == 0)
        {   Module::dprogress++;
            sem = 1;
            //printf("\tcompleted %p\n", this);
        }
        else
            ;//printf("\talready completed %p\n", this);

        // 0 sized structs are set to 1 byte
        if (aad.structsize == 0)
        {
            aad.structsize = 1;
            aad.alignsize = 1;
        }

        // Align size of anonymous aggregate
//printf("aad.structalign = %d, aad.alignsize = %d, sc->offset = %d\n", aad.structalign, aad.alignsize, sc->offset);
        ad->alignmember(aad.structalign, aad.alignsize, &sc->offset);
        //ad->structsize = sc->offset;
//printf("sc->offset = %d\n", sc->offset);

        // Add members of aad to ad
        //printf("\tadding members of aad (%p) to '%s'\n", &aad, ad->toChars());
        for (unsigned i = 0; i < aad.fields.dim; i++)
        {
            VarDeclaration *v = (VarDeclaration *)aad.fields.data[i];

#if IN_LLVM
        v->offset2 = sc->offset;
#endif
            v->offset += sc->offset;

#if IN_LLVM
        if (!v->anonDecl)
            v->anonDecl = this;
#endif
            ad->fields.push(v);
        }

        // Add size of aad to ad
        if (adisunion)
        {
            if (aad.structsize > ad->structsize)
                ad->structsize = aad.structsize;
            sc->offset = 0;
        }
        else
        {
            ad->structsize = sc->offset + aad.structsize;
            sc->offset = ad->structsize;
        }

        if (ad->alignsize < aad.alignsize)
            ad->alignsize = aad.alignsize;
    }
}


void AnonDeclaration::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->printf(isunion ? "union" : "struct");
    buf->writestring("\n{\n");
    if (decl)
    {
        for (unsigned i = 0; i < decl->dim; i++)
        {
            Dsymbol *s = (Dsymbol *)decl->data[i];

            //buf->writestring("    ");
            s->toCBuffer(buf, hgs);
        }
    }
    buf->writestring("}\n");
}

const char *AnonDeclaration::kind()
{
    return (isunion ? "anonymous union" : "anonymous struct");
}

/********************************* PragmaDeclaration ****************************/

static bool parseStringExp(Expression* e, std::string& res)
{
    StringExp *s = NULL;

    e = e->optimize(WANTvalue);
    if (e->op == TOKstring && (s = (StringExp *)e))
    {
        char* str = (char*)s->string;
        res = str;
        return true;
    }
    return false;
}

PragmaDeclaration::PragmaDeclaration(Loc loc, Identifier *ident, Expressions *args, Array *decl)
        : AttribDeclaration(decl)
{
    this->loc = loc;
    this->ident = ident;
    this->args = args;
}

Dsymbol *PragmaDeclaration::syntaxCopy(Dsymbol *s)
{
    //printf("PragmaDeclaration::syntaxCopy(%s)\n", toChars());
    PragmaDeclaration *pd;

    assert(!s);
    pd = new PragmaDeclaration(loc, ident,
        Expression::arraySyntaxCopy(args), Dsymbol::arraySyntaxCopy(decl));
    return pd;
}

void PragmaDeclaration::setScope(Scope *sc)
{
#if TARGET_NET
    if (ident == Lexer::idPool("assembly"))
    {
        if (!args || args->dim != 1)
        {
            error("pragma has invalid number of arguments");
        }
        else
        {
            Expression *e = (Expression *)args->data[0];
            e = e->semantic(sc);
            e = e->optimize(WANTvalue | WANTinterpret);
            args->data[0] = (void *)e;
            if (e->op != TOKstring)
            {
                error("string expected, not '%s'", e->toChars());
            }
            PragmaScope* pragma = new PragmaScope(this, sc->parent, static_cast<StringExp*>(e));

            assert(sc);
            pragma->setScope(sc);

            //add to module members
            assert(sc->module);
            assert(sc->module->members);
            sc->module->members->push(pragma);
        }
    }
#endif // TARGET_NET
}

void PragmaDeclaration::semantic(Scope *sc)
{   // Should be merged with PragmaStatement

#if IN_LLVM
    int llvm_internal = 0;
    std::string arg1str;

#endif

    //printf("\tPragmaDeclaration::semantic '%s'\n",toChars());
    if (ident == Id::msg)
    {
        if (args)
        {
            for (size_t i = 0; i < args->dim; i++)
            {
                Expression *e = (Expression *)args->data[i];

                e = e->semantic(sc);
                e = e->optimize(WANTvalue | WANTinterpret);
                if (e->op == TOKstring)
                {
                    StringExp *se = (StringExp *)e;
                    fprintf(stdmsg, "%.*s", (int)se->len, (char *)se->string);
                }
                else
                    fprintf(stdmsg, "%s", e->toChars());
            }
            fprintf(stdmsg, "\n");
        }
        goto Lnodecl;
    }
    else if (ident == Id::lib)
    {
        if (!args || args->dim != 1)
            error("string expected for library name");
        else
        {
            Expression *e = (Expression *)args->data[0];

            e = e->semantic(sc);
            e = e->optimize(WANTvalue | WANTinterpret);
            args->data[0] = (void *)e;
            if (e->op != TOKstring)
                error("string expected for library name, not '%s'", e->toChars());
            else if (global.params.verbose)
            {
                StringExp *se = (StringExp *)e;
                char *name = (char *)mem.malloc(se->len + 1);
                memcpy(name, se->string, se->len);
                name[se->len] = 0;
                printf("library   %s\n", name);
                mem.free(name);
            }
        }
        goto Lnodecl;
    }
#if IN_GCC
    else if (ident == Id::GNU_asm)
    {
        if (! args || args->dim != 2)
            error("identifier and string expected for asm name");
        else
        {
            Expression *e;
            Declaration *d = NULL;
            StringExp *s = NULL;

            e = (Expression *)args->data[0];
            e = e->semantic(sc);
            if (e->op == TOKvar)
            {
                d = ((VarExp *)e)->var;
                if (! d->isFuncDeclaration() && ! d->isVarDeclaration())
                    d = NULL;
            }
            if (!d)
                error("first argument of GNU_asm must be a function or variable declaration");

            e = (Expression *)args->data[1];
            e = e->semantic(sc);
            e = e->optimize(WANTvalue);
            if (e->op == TOKstring && ((StringExp *)e)->sz == 1)
                s = ((StringExp *)e);
            else
                error("second argument of GNU_asm must be a char string");

            if (d && s)
                d->c_ident = Lexer::idPool((char*) s->string);
        }
        goto Lnodecl;
    }
#endif
#if DMDV2
    else if (ident == Id::startaddress)
    {
        if (!args || args->dim != 1)
            error("function name expected for start address");
        else
        {
            Expression *e = (Expression *)args->data[0];
            e = e->semantic(sc);
            e = e->optimize(WANTvalue | WANTinterpret);
            args->data[0] = (void *)e;
            Dsymbol *sa = getDsymbol(e);
            if (!sa || !sa->isFuncDeclaration())
                error("function name expected for start address, not '%s'", e->toChars());
        }
        goto Lnodecl;
    }
#endif
#if TARGET_NET
    else if (ident == Lexer::idPool("assembly"))
    {
    }
#endif // TARGET_NET

// LDC
#if IN_LLVM

    // pragma(intrinsic, "string") { funcdecl(s) }
    else if (ident == Id::intrinsic)
    {
        Expression* expr = (Expression *)args->data[0];
        expr = expr->semantic(sc);
        if (!args || args->dim != 1 || !parseStringExp(expr, arg1str))
        {
             error("requires exactly 1 string literal parameter");
             fatal();
        }
        llvm_internal = LLVMintrinsic;
    }

    // pragma(notypeinfo) { typedecl(s) }
    else if (ident == Id::no_typeinfo)
    {
        if (args && args->dim > 0)
        {
             error("takes no parameters");
             fatal();
        }
        llvm_internal = LLVMno_typeinfo;
    }

    // pragma(nomoduleinfo) ;
    else if (ident == Id::no_moduleinfo)
    {
        if (args && args->dim > 0)
        {
             error("takes no parameters");
             fatal();
        }
        llvm_internal = LLVMno_moduleinfo;
    }

    // pragma(alloca) { funcdecl(s) }
    else if (ident == Id::Alloca)
    {
        if (args && args->dim > 0)
        {
             error("takes no parameters");
             fatal();
        }
        llvm_internal = LLVMalloca;
    }

    // pragma(va_start) { templdecl(s) }
    else if (ident == Id::vastart)
    {
        if (args && args->dim > 0)
        {
             error("takes no parameters");
             fatal();
        }
        llvm_internal = LLVMva_start;
    }

    // pragma(va_copy) { funcdecl(s) }
    else if (ident == Id::vacopy)
    {
        if (args && args->dim > 0)
        {
             error("takes no parameters");
             fatal();
        }
        llvm_internal = LLVMva_copy;
    }

    // pragma(va_end) { funcdecl(s) }
    else if (ident == Id::vaend)
    {
        if (args && args->dim > 0)
        {
             error("takes no parameters");
             fatal();
        }
        llvm_internal = LLVMva_end;
    }

    // pragma(va_arg) { templdecl(s) }
    else if (ident == Id::vaarg)
    {
        if (args && args->dim > 0)
        {
             error("takes no parameters");
             fatal();
        }
        llvm_internal = LLVMva_arg;
    }

    // pragma(ldc, "string") { templdecl(s) }
    else if (ident == Id::ldc)
    {
        Expression* expr = (Expression *)args->data[0];
        expr = expr->semantic(sc);
        if (!args || args->dim != 1 || !parseStringExp(expr, arg1str))
        {
             error("requires exactly 1 string literal parameter");
             fatal();
        }
        else if (arg1str == "verbose")
        {
            sc->module->llvmForceLogging = true;
        }
        else
        {
            error("command '%s' invalid", expr->toChars());
            fatal();
        }
    }

    // pragma(llvm_inline_asm) { templdecl(s) }
    else if (ident == Id::llvm_inline_asm)
    {
        if (args && args->dim > 0)
        {
             error("takes no parameters");
             fatal();
        }
        llvm_internal = LLVMinline_asm;
    }

#endif // LDC


    else if (ignoreUnsupportedPragmas)
    {
        if (global.params.verbose)
        {
            /* Print unrecognized pragmas
             */
            printf("pragma    %s", ident->toChars());
            if (args)
            {
                for (size_t i = 0; i < args->dim; i++)
                {
                    // ignore errors in ignored pragmas.
                    global.gag++;
                    unsigned errors_save = global.errors;

                    Expression *e = (Expression *)args->data[i];
                    e = e->semantic(sc);
                    e = e->optimize(WANTvalue | WANTinterpret);
                    if (i == 0)
                        printf(" (");
                    else
                        printf(",");
                    printf("%s", e->toChars());

                    // restore error state.
                    global.gag--;
                    global.errors = errors_save;
                }
                if (args->dim)
                    printf(")");
            }
            printf("\n");
        }
    }
    else
        error("unrecognized pragma(%s)", ident->toChars());

    if (decl)
    {
        for (unsigned i = 0; i < decl->dim; i++)
        {
            Dsymbol *s = (Dsymbol *)decl->data[i];

            s->semantic(sc);

// LDC
#if IN_LLVM

        if (llvm_internal)
        {
        if (s->llvmInternal)
        {
            error("multiple LDC specific pragmas not allowed not affect the same declaration ('%s' at '%s')", s->toChars(), s->loc.toChars());
            fatal();
        }
        switch(llvm_internal)
        {
        case LLVMintrinsic:
            if (FuncDeclaration* fd = s->isFuncDeclaration())
            {
                fd->llvmInternal = llvm_internal;
                fd->intrinsicName = arg1str;
                fd->linkage = LINKintrinsic;
                ((TypeFunction*)fd->type)->linkage = LINKintrinsic;
            }
            else if (TemplateDeclaration* td = s->isTemplateDeclaration())
            {
                td->llvmInternal = llvm_internal;
                td->intrinsicName = arg1str;
            }
            else
            {
                error("only allowed on function declarations");
                fatal();
            }
            break;

        case LLVMva_start:
        case LLVMva_arg:
            if (TemplateDeclaration* td = s->isTemplateDeclaration())
            {
                if (td->parameters->dim != 1)
                {
                    error("the '%s' pragma template must have exactly one template parameter", ident->toChars());
                    fatal();
                }
                else if (!td->onemember)
                {
                    error("the '%s' pragma template must have exactly one member", ident->toChars());
                    fatal();
                }
                else if (td->overnext || td->overroot)
                {
                    error("the '%s' pragma template must not be overloaded", ident->toChars());
                    fatal();
                }
                td->llvmInternal = llvm_internal;
            }
            else
            {
                error("the '%s' pragma is only allowed on template declarations", ident->toChars());
                fatal();
            }
            break;

        case LLVMva_copy:
        case LLVMva_end:
            if (FuncDeclaration* fd = s->isFuncDeclaration())
            {
                fd->llvmInternal = llvm_internal;
            }
            else
            {
                error("the '%s' pragma is only allowed on function declarations", ident->toChars());
                fatal();
            }
            break;

        case LLVMno_typeinfo:
            s->llvmInternal = llvm_internal;
            break;

        case LLVMalloca:
            if (FuncDeclaration* fd = s->isFuncDeclaration())
            {
                fd->llvmInternal = llvm_internal;
            }
            else
            {
                error("the '%s' pragma must only be used on function declarations of type 'void* function(uint nbytes)'", ident->toChars());
                fatal();
            }
            break;

        case LLVMinline_asm:
            if (TemplateDeclaration* td = s->isTemplateDeclaration())
            {
                if (td->parameters->dim > 1)
                {
                    error("the '%s' pragma template must have exactly zero or one template parameters", ident->toChars());
                    fatal();
                }
                else if (!td->onemember)
                {
                    error("the '%s' pragma template must have exactly one member", ident->toChars());
                    fatal();
                }
                td->llvmInternal = llvm_internal;
            }
            else
            {
                error("the '%s' pragma is only allowed on template declarations", ident->toChars());
                fatal();
            }
            break;

        default:
            warning("the LDC specific pragma '%s' is not yet implemented, ignoring", ident->toChars());
        }
        }

#endif // LDC

    }
    }
    return;

Lnodecl:
    if (decl)
        error("pragma is missing closing ';'");
}

int PragmaDeclaration::oneMember(Dsymbol **ps)
{
    *ps = NULL;
    return TRUE;
}

const char *PragmaDeclaration::kind()
{
    return "pragma";
}

#if IN_DMD
void PragmaDeclaration::toObjFile(int multiobj)
{
    if (ident == Id::lib)
    {
        assert(args && args->dim == 1);

        Expression *e = (Expression *)args->data[0];

        assert(e->op == TOKstring);

        StringExp *se = (StringExp *)e;
        char *name = (char *)mem.malloc(se->len + 1);
        memcpy(name, se->string, se->len);
        name[se->len] = 0;
#if OMFOBJ
        /* The OMF format allows library names to be inserted
         * into the object file. The linker will then automatically
         * search that library, too.
         */
        obj_includelib(name);
#elif ELFOBJ || MACHOBJ
        /* The format does not allow embedded library names,
         * so instead append the library name to the list to be passed
         * to the linker.
         */
        global.params.libfiles->push((void *) name);
#else
        error("pragma lib not supported");
#endif
    }
#if DMDV2
    else if (ident == Id::startaddress)
    {
        assert(args && args->dim == 1);
        Expression *e = (Expression *)args->data[0];
        Dsymbol *sa = getDsymbol(e);
        FuncDeclaration *f = sa->isFuncDeclaration();
        assert(f);
        Symbol *s = f->toSymbol();
        obj_startaddress(s);
    }
#endif
    AttribDeclaration::toObjFile(multiobj);
}
#endif

void PragmaDeclaration::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->printf("pragma(%s", ident->toChars());
    if (args)
    {
        for (size_t i = 0; i < args->dim; i++)
        {
            Expression *e = (Expression *)args->data[i];

            buf->writestring(", ");
            e->toCBuffer(buf, hgs);
        }
    }
    buf->writeByte(')');
    AttribDeclaration::toCBuffer(buf, hgs);
}


/********************************* ConditionalDeclaration ****************************/

ConditionalDeclaration::ConditionalDeclaration(Condition *condition, Array *decl, Array *elsedecl)
        : AttribDeclaration(decl)
{
    //printf("ConditionalDeclaration::ConditionalDeclaration()\n");
    this->condition = condition;
    this->elsedecl = elsedecl;
}

Dsymbol *ConditionalDeclaration::syntaxCopy(Dsymbol *s)
{
    ConditionalDeclaration *dd;

    assert(!s);
    dd = new ConditionalDeclaration(condition->syntaxCopy(),
        Dsymbol::arraySyntaxCopy(decl),
        Dsymbol::arraySyntaxCopy(elsedecl));
    return dd;
}


int ConditionalDeclaration::oneMember(Dsymbol **ps)
{
    //printf("ConditionalDeclaration::oneMember(), inc = %d\n", condition->inc);
    if (condition->inc)
    {
        Array *d = condition->include(NULL, NULL) ? decl : elsedecl;
        return Dsymbol::oneMembers(d, ps);
    }
    *ps = NULL;
    return TRUE;
}

void ConditionalDeclaration::emitComment(Scope *sc)
{
    //printf("ConditionalDeclaration::emitComment(sc = %p)\n", sc);
    if (condition->inc)
    {
        AttribDeclaration::emitComment(sc);
    }
    else if (sc->docbuf)
    {
        /* If generating doc comment, be careful because if we're inside
         * a template, then include(NULL, NULL) will fail.
         */
        Array *d = decl ? decl : elsedecl;
        for (unsigned i = 0; i < d->dim; i++)
        {   Dsymbol *s = (Dsymbol *)d->data[i];
            s->emitComment(sc);
        }
    }
}

// Decide if 'then' or 'else' code should be included

Array *ConditionalDeclaration::include(Scope *sc, ScopeDsymbol *sd)
{
    //printf("ConditionalDeclaration::include()\n");
    assert(condition);
    return condition->include(sc, sd) ? decl : elsedecl;
}

void ConditionalDeclaration::setScope(Scope *sc)
{
    Array *d = include(sc, NULL);

    //printf("\tConditionalDeclaration::setScope '%s', d = %p\n",toChars(), d);
    if (d)
    {
       for (unsigned i = 0; i < d->dim; i++)
       {
           Dsymbol *s = (Dsymbol *)d->data[i];

           s->setScope(sc);
       }
    }
}

void ConditionalDeclaration::importAll(Scope *sc)
{
    Array *d = include(sc, NULL);

    //printf("\tConditionalDeclaration::importAll '%s', d = %p\n",toChars(), d);
    if (d)
    {
       for (unsigned i = 0; i < d->dim; i++)
       {
           Dsymbol *s = (Dsymbol *)d->data[i];

           s->importAll(sc);
       }
    }
}

void ConditionalDeclaration::addComment(unsigned char *comment)
{
    /* Because addComment is called by the parser, if we called
     * include() it would define a version before it was used.
     * But it's no problem to drill down to both decl and elsedecl,
     * so that's the workaround.
     */

    if (comment)
    {
        Array *d = decl;

        for (int j = 0; j < 2; j++)
        {
            if (d)
            {
                for (unsigned i = 0; i < d->dim; i++)
                {   Dsymbol *s;

                    s = (Dsymbol *)d->data[i];
                    //printf("ConditionalDeclaration::addComment %s\n", s->toChars());
                    s->addComment(comment);
                }
            }
            d = elsedecl;
        }
    }
}

void ConditionalDeclaration::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    condition->toCBuffer(buf, hgs);
    if (decl || elsedecl)
    {
        buf->writenl();
        buf->writeByte('{');
        buf->writenl();
        if (decl)
        {
            for (unsigned i = 0; i < decl->dim; i++)
            {
                Dsymbol *s = (Dsymbol *)decl->data[i];

                buf->writestring("    ");
                s->toCBuffer(buf, hgs);
            }
        }
        buf->writeByte('}');
        if (elsedecl)
        {
            buf->writenl();
            buf->writestring("else");
            buf->writenl();
            buf->writeByte('{');
            buf->writenl();
            for (unsigned i = 0; i < elsedecl->dim; i++)
            {
                Dsymbol *s = (Dsymbol *)elsedecl->data[i];

                buf->writestring("    ");
                s->toCBuffer(buf, hgs);
            }
            buf->writeByte('}');
        }
    }
    else
        buf->writeByte(':');
    buf->writenl();
}

/***************************** StaticIfDeclaration ****************************/

StaticIfDeclaration::StaticIfDeclaration(Condition *condition,
        Array *decl, Array *elsedecl)
        : ConditionalDeclaration(condition, decl, elsedecl)
{
    //printf("StaticIfDeclaration::StaticIfDeclaration()\n");
    sd = NULL;
    addisdone = 0;
}


Dsymbol *StaticIfDeclaration::syntaxCopy(Dsymbol *s)
{
    StaticIfDeclaration *dd;

    assert(!s);
    dd = new StaticIfDeclaration(condition->syntaxCopy(),
        Dsymbol::arraySyntaxCopy(decl),
        Dsymbol::arraySyntaxCopy(elsedecl));
    return dd;
}


int StaticIfDeclaration::addMember(Scope *sc, ScopeDsymbol *sd, int memnum)
{
    //printf("StaticIfDeclaration::addMember() '%s'\n",toChars());
    /* This is deferred until semantic(), so that
     * expressions in the condition can refer to declarations
     * in the same scope, such as:
     *
     * template Foo(int i)
     * {
     *     const int j = i + 1;
     *     static if (j == 3)
     *         const int k;
     * }
     */
    this->sd = sd;
    int m = 0;

    if (memnum == 0)
    {   m = AttribDeclaration::addMember(sc, sd, memnum);
        addisdone = 1;
    }
    return m;
}


void StaticIfDeclaration::importAll(Scope *sc)
{
    // do not evaluate condition before semantic pass
}

void StaticIfDeclaration::setScope(Scope *sc)
{
    // do not evaluate condition before semantic pass
}

void StaticIfDeclaration::semantic(Scope *sc)
{
    Array *d = include(sc, sd);

    //printf("\tStaticIfDeclaration::semantic '%s', d = %p\n",toChars(), d);
    if (d)
    {
        if (!addisdone)
        {   AttribDeclaration::addMember(sc, sd, 1);
            addisdone = 1;
        }

        for (unsigned i = 0; i < d->dim; i++)
        {
            Dsymbol *s = (Dsymbol *)d->data[i];

            s->semantic(sc);
        }
    }
}

const char *StaticIfDeclaration::kind()
{
    return "static if";
}


/***************************** CompileDeclaration *****************************/

CompileDeclaration::CompileDeclaration(Loc loc, Expression *exp)
    : AttribDeclaration(NULL)
{
    //printf("CompileDeclaration(loc = %d)\n", loc.linnum);
    this->loc = loc;
    this->exp = exp;
    this->sd = NULL;
    this->compiled = 0;
}

Dsymbol *CompileDeclaration::syntaxCopy(Dsymbol *s)
{
    //printf("CompileDeclaration::syntaxCopy('%s')\n", toChars());
    CompileDeclaration *sc = new CompileDeclaration(loc, exp->syntaxCopy());
    return sc;
}

int CompileDeclaration::addMember(Scope *sc, ScopeDsymbol *sd, int memnum)
{
    //printf("CompileDeclaration::addMember(sc = %p, memnum = %d)\n", sc, memnum);
    this->sd = sd;
    if (memnum == 0)
    {   /* No members yet, so parse the mixin now
         */
        compileIt(sc);
        memnum |= AttribDeclaration::addMember(sc, sd, memnum);
        compiled = 1;
    }
    return memnum;
}

void CompileDeclaration::compileIt(Scope *sc)
{
    //printf("CompileDeclaration::compileIt(loc = %d)\n", loc.linnum);
    exp = exp->semantic(sc);
    exp = resolveProperties(sc, exp);
    exp = exp->optimize(WANTvalue | WANTinterpret);
    if (exp->op != TOKstring)
    {   exp->error("argument to mixin must be a string, not (%s)", exp->toChars());
    }
    else
    {
        StringExp *se = (StringExp *)exp;
        se = se->toUTF8(sc);
        Parser p(sc->module, (unsigned char *)se->string, se->len, 0);
        p.loc = loc;
        p.nextToken();
        decl = p.parseDeclDefs(0);
        if (p.token.value != TOKeof)
            exp->error("incomplete mixin declaration (%s)", se->toChars());
    }
}

void CompileDeclaration::semantic(Scope *sc)
{
    //printf("CompileDeclaration::semantic()\n");

    if (!compiled)
    {
        compileIt(sc);
        AttribDeclaration::addMember(sc, sd, 0);
        compiled = 1;
    }
    AttribDeclaration::semantic(sc);
}

void CompileDeclaration::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("mixin(");
    exp->toCBuffer(buf, hgs);
    buf->writestring(");");
    buf->writenl();
}
