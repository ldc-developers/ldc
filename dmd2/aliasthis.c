
// Compiler implementation of the D programming language
// Copyright (c) 2009-2009 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <stdio.h>
#include <assert.h>

#include "mars.h"
#include "identifier.h"
#include "aliasthis.h"
#include "scope.h"
#include "aggregate.h"
#include "dsymbol.h"
#include "mtype.h"
#include "declaration.h"

#if DMDV2

Expression *resolveAliasThis(Scope *sc, Expression *e)
{
    Type *t = e->type->toBasetype();
    AggregateDeclaration *ad;

    if (t->ty == Tclass)
    {   ad = ((TypeClass *)t)->sym;
        goto L1;
    }
    else if (t->ty == Tstruct)
    {   ad = ((TypeStruct *)t)->sym;
    L1:
        if (ad && ad->aliasthis)
        {
            bool isstatic = (e->op == TOKtype);
            e = new DotIdExp(e->loc, e, ad->aliasthis->ident);
            e = e->semantic(sc);
            if (isstatic && ad->aliasthis->needThis())
            {
                /* non-@property function is not called inside typeof(),
                 * so resolve it ahead.
                 */
                int save = sc->intypeof;
                sc->intypeof = 1;   // bypass "need this" error check
                e = resolveProperties(sc, e);
                sc->intypeof = save;

                e = new TypeExp(e->loc, new TypeTypeof(e->loc, e));
                e = e->semantic(sc);
            }
            e = resolveProperties(sc, e);
        }
    }

    return e;
}

AliasThis::AliasThis(Loc loc, Identifier *ident)
    : Dsymbol(NULL)             // it's anonymous (no identifier)
{
    this->loc = loc;
    this->ident = ident;
}

Dsymbol *AliasThis::syntaxCopy(Dsymbol *s)
{
    assert(!s);
    /* Since there is no semantic information stored here,
     * we don't need to copy it.
     */
    return this;
}

void AliasThis::semantic(Scope *sc)
{
    Dsymbol *parent = sc->parent;
    if (parent)
        parent = parent->pastMixin();
    AggregateDeclaration *ad = NULL;
    if (parent)
        ad = parent->isAggregateDeclaration();
    if (ad)
    {
        assert(ad->members);
        Dsymbol *s = ad->search(loc, ident, 0);
        if (!s)
        {   s = sc->search(loc, ident, NULL);
            if (s)
                ::error(loc, "%s is not a member of %s", s->toChars(), ad->toChars());
            else
                ::error(loc, "undefined identifier %s", ident->toChars());
            return;
        }
        else if (ad->aliasthis && s != ad->aliasthis)
            error("there can be only one alias this");

        /* disable the alias this conversion so the implicit conversion check
         * doesn't use it.
         */
        /* This should use ad->aliasthis directly, but with static foreach and templates
         * ad->type->sym might be different to ad.
         */
        AggregateDeclaration *ad2 = ad->type->toDsymbol(NULL)->isAggregateDeclaration();
        Dsymbol *save = ad2->aliasthis;
        ad2->aliasthis = NULL;

        if (Declaration *d = s->isDeclaration())
        {
            Type *t = d->type;
            assert(t);
            if (ad->type->implicitConvTo(t))
            {
                ::error(loc, "alias this is not reachable as %s already converts to %s", ad->toChars(), t->toChars());
            }
        }

        ad2->aliasthis = save;
        ad->aliasthis = s;
    }
    else
        error("alias this can only appear in struct or class declaration, not %s", parent ? parent->toChars() : "nowhere");
}

const char *AliasThis::kind()
{
    return "alias this";
}

void AliasThis::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("alias ");
    buf->writestring(ident->toChars());
    buf->writestring(" this;\n");
}

#endif
