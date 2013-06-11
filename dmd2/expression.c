// Compiler implementation of the D programming language
// Copyright (c) 1999-2013 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>
#if _MSC_VER
#include <complex>
#else
#if IN_DMD
#include <complex.h>
#endif
#endif

#if _WIN32 && __DMC__
extern "C" char * __cdecl __locale_decpoint;
#endif

#include "rmem.h"
#include "port.h"
#if IN_DMD
#include "root.h"
#endif

#include "mtype.h"
#include "init.h"
#include "expression.h"
#include "template.h"
#include "utf.h"
#include "enum.h"
#include "scope.h"
#include "statement.h"
#include "declaration.h"
#include "aggregate.h"
#include "import.h"
#include "id.h"
#include "dsymbol.h"
#include "module.h"
#include "attrib.h"
#include "hdrgen.h"
#include "parse.h"
#include "doc.h"

#if IN_DMD
Expression *createTypeInfoArray(Scope *sc, Expression *args[], size_t dim);
#endif
Expression *expandVar(int result, VarDeclaration *v);

#define LOGSEMANTIC     0

/*************************************************************
 * Given var, we need to get the
 * right 'this' pointer if var is in an outer class, but our
 * existing 'this' pointer is in an inner class.
 * Input:
 *      e1      existing 'this'
 *      ad      struct or class we need the correct 'this' for
 *      var     the specific member of ad we're accessing
 */

Expression *getRightThis(Loc loc, Scope *sc, AggregateDeclaration *ad,
        Expression *e1, Declaration *var)
{
    //printf("\ngetRightThis(e1 = %s, ad = %s, var = %s)\n", e1->toChars(), ad->toChars(), var->toChars());
 L1:
    Type *t = e1->type->toBasetype();
    //printf("e1->type = %s, var->type = %s\n", e1->type->toChars(), var->type->toChars());

    /* If e1 is not the 'this' pointer for ad
     */
    if (ad &&
        !(t->ty == Tpointer && t->nextOf()->ty == Tstruct &&
          ((TypeStruct *)t->nextOf())->sym == ad)
        &&
        !(t->ty == Tstruct &&
          ((TypeStruct *)t)->sym == ad)
       )
    {
        ClassDeclaration *cd = ad->isClassDeclaration();
        ClassDeclaration *tcd = t->isClassHandle();

        /* e1 is the right this if ad is a base class of e1
         */
        if (!cd || !tcd ||
            !(tcd == cd || cd->isBaseOf(tcd, NULL))
           )
        {
            /* Only classes can be inner classes with an 'outer'
             * member pointing to the enclosing class instance
             */
            if (tcd && tcd->isNested())
            {   /* e1 is the 'this' pointer for an inner class: tcd.
                 * Rewrite it as the 'this' pointer for the outer class.
                 */

                e1 = new DotVarExp(loc, e1, tcd->vthis);
                e1->type = tcd->vthis->type;
                e1->type = e1->type->addMod(t->mod);
                // Do not call checkNestedRef()
                //e1 = e1->semantic(sc);

                // Skip up over nested functions, and get the enclosing
                // class type.
                int n = 0;
                Dsymbol *s;
                for (s = tcd->toParent();
                     s && s->isFuncDeclaration();
                     s = s->toParent())
                {   FuncDeclaration *f = s->isFuncDeclaration();
                    if (f->vthis)
                    {
                        //printf("rewriting e1 to %s's this\n", f->toChars());
                        n++;

                        // LDC seems dmd misses it sometimes here :/
                        if (f->isMember2()) {
                            f->vthis->nestedrefs.push(sc->parent->isFuncDeclaration());
                            f->closureVars.push(f->vthis);
                        }

                        e1 = new VarExp(loc, f->vthis);
                    }
                    else
                    {
                        e1->error("need 'this' of type %s to access member %s"
                                  " from static function %s",
                            ad->toChars(), var->toChars(), f->toChars());
                        e1 = new ErrorExp();
                        return e1;
                    }
                }
                if (s && s->isClassDeclaration())
                {   e1->type = s->isClassDeclaration()->type;
                    e1->type = e1->type->addMod(t->mod);
                    if (n > 1)
                        e1 = e1->semantic(sc);
                }
                else
                    e1 = e1->semantic(sc);
                goto L1;
            }
            /* Can't find a path from e1 to ad
             */
            e1->error("this for %s needs to be type %s not type %s",
                var->toChars(), ad->toChars(), t->toChars());
            e1 = new ErrorExp();
        }
    }
    return e1;
}

/*****************************************
 * Determine if 'this' is available.
 * If it is, return the FuncDeclaration that has it.
 */

FuncDeclaration *hasThis(Scope *sc)
{   FuncDeclaration *fd;
    FuncDeclaration *fdthis;

    //printf("hasThis()\n");
    fdthis = sc->parent->isFuncDeclaration();
    //printf("fdthis = %p, '%s'\n", fdthis, fdthis ? fdthis->toChars() : "");

    /* Special case for inside template constraint
     */
    if (fdthis && (sc->flags & SCOPEstaticif) && fdthis->parent->isTemplateDeclaration())
    {
        //TemplateDeclaration *td = fdthis->parent->isTemplateDeclaration();
        //printf("[%s] td = %s, fdthis->vthis = %p\n", td->loc.toChars(), td->toChars(), fdthis->vthis);
        return fdthis->vthis ? fdthis : NULL;
    }

    // Go upwards until we find the enclosing member function
    fd = fdthis;
    while (1)
    {
        if (!fd)
        {
            goto Lno;
        }
        if (!fd->isNested())
            break;

        Dsymbol *parent = fd->parent;
        while (1)
        {
            if (!parent)
                goto Lno;
            TemplateInstance *ti = parent->isTemplateInstance();
            if (ti)
                parent = ti->parent;
            else
                break;
        }
        fd = parent->isFuncDeclaration();
    }

    if (!fd->isThis())
    {   //printf("test '%s'\n", fd->toChars());
        goto Lno;
    }

    assert(fd->vthis);
    return fd;

Lno:
    return NULL;                // don't have 'this' available
}


/***************************************
 * Pull out any properties.
 */

Expression *resolveProperties(Scope *sc, Expression *e)
{
    //printf("resolveProperties(%s)\n", e->toChars());

    TemplateDeclaration *td;
    Objects *targsi;
    Expression *ethis;
    if (e->op == TOKdotti)
    {
        DotTemplateInstanceExp* dti = (DotTemplateInstanceExp *)e;
        td     = dti->getTempdecl(sc);
                 dti->ti->semanticTiargs(sc);
        targsi = dti->ti->tiargs;
        ethis  = dti->e1;
        goto L1;
    }
    else if (e->op == TOKdottd)
    {
        DotTemplateExp *dte = (DotTemplateExp *)e;
        td     = dte->td;
        targsi = NULL;
        ethis  = dte->e1;
        goto L1;
    }
    else if (e->op == TOKimport)
    {
        Dsymbol *s = ((ScopeExp *)e)->sds;
        td = s->isTemplateDeclaration();
        if (td)
        {
            targsi = NULL;
            ethis  = NULL;
            goto L1;
        }
    }
    else if (e->op == TOKtemplate)
    {
        td     = ((TemplateExp *)e)->td;
        targsi = NULL;
        ethis  = NULL;
    L1:
        assert(td);
        unsigned errors = global.startGagging();
        FuncDeclaration *fd = td->deduceFunctionTemplate(sc, e->loc, targsi, ethis, NULL, 1);
        if (global.endGagging(errors))
            fd = NULL;  // eat "is not a function template" error
        if (fd && fd->type)
        {   assert(fd->type->ty == Tfunction);
            TypeFunction *tf = (TypeFunction *)fd->type;
            if (!tf->isproperty && global.params.enforcePropertySyntax)
            {   error(e->loc, "not a property %s", e->toChars());
                return new ErrorExp();
            }
            e = new CallExp(e->loc, e);
            e = e->semantic(sc);
        }
        goto return_expr;
    }

    if (e->type &&
        e->op != TOKtype)       // function type is not a property
    {
        Type *t = e->type->toBasetype();

        if (t->ty == Tfunction || e->op == TOKoverloadset)
        {
            if (t->ty == Tfunction && !((TypeFunction *)t)->isproperty &&
                global.params.enforcePropertySyntax)
            {
                error(e->loc, "not a property %s", e->toChars());
                return new ErrorExp();
            }
            e = new CallExp(e->loc, e);
            e = e->semantic(sc);
        }

        /* Look for e being a lazy parameter; rewrite as delegate call
         */
        else if (e->op == TOKvar)
        {   VarExp *ve = (VarExp *)e;

            if (ve->var->storage_class & STClazy)
            {
                e = new CallExp(e->loc, e);
                e = e->semantic(sc);
            }
        }

        else if (e->op == TOKdotexp)
        {
            e->error("expression has no value");
            return new ErrorExp();
        }

    }

return_expr:
    if (!e->type)
    {
        error(e->loc, "cannot resolve type for %s", e->toChars());
        e->type = new TypeError();
    }
    return e;
}

/******************************
 * Check the tail CallExp is really property function call.
 */

void checkPropertyCall(Expression *e, Expression *emsg)
{
    while (e->op == TOKcomma)
        e = ((CommaExp *)e)->e2;

    if (e->op == TOKcall)
    {   CallExp *ce = (CallExp *)e;
        TypeFunction *tf;
        if (ce->f)
        {
            tf = (TypeFunction *)ce->f->type;
            /* If a forward reference to ce->f, try to resolve it
             */
            if (!tf->deco && ce->f->scope)
            {   ce->f->semantic(ce->f->scope);
                tf = (TypeFunction *)ce->f->type;
            }
        }
        else if (ce->e1->type->ty == Tfunction)
            tf = (TypeFunction *)ce->e1->type;
        else if (ce->e1->type->ty == Tdelegate)
            tf = (TypeFunction *)ce->e1->type->nextOf();
        else if (ce->e1->type->ty == Tpointer && ce->e1->type->nextOf()->ty == Tfunction)
            tf = (TypeFunction *)ce->e1->type->nextOf();
        else
            assert(0);

        if (!tf->isproperty && global.params.enforcePropertySyntax)
            ce->e1->error("not a property %s", emsg->toChars());
    }
}

/******************************
 * Pull out property with UFCS.
 */

Expression *resolveUFCSProperties(Scope *sc, Expression *e1, Expression *e2 = NULL)
{
    Expression *e = NULL;
    Expression *eleft;
    Identifier *ident;
    Objects* tiargs;
    Loc loc = e1->loc;

    if (e1->op == TOKdot)
    {
        DotIdExp *die = (DotIdExp *)e1;
        eleft  = die->e1;
        ident  = die->ident;
        tiargs = NULL;
        goto L1;
    }
    else if (e1->op == TOKdotti)
    {
        DotTemplateInstanceExp *dti;
        dti = (DotTemplateInstanceExp *)e1;
        eleft  = dti->e1;
        ident  = dti->ti->name;
        tiargs = dti->ti->tiargs;
    L1:
        /* .ident
         * .ident!tiargs
         */
        e = new IdentifierExp(loc, Id::empty);
        if (tiargs)
            e = new DotTemplateInstanceExp(loc, e, ident, tiargs);
        else
            e = new DotIdExp(loc, e, ident);

        if (e2)
        {
            // run semantic without gagging
            e2 = e2->semantic(sc);

            /* .f(e1) = e2
             */
            Expression *ex = e->syntaxCopy();
            Expressions *a1 = new Expressions();
            a1->setDim(1);
            (*a1)[0] = eleft;
            ex = new CallExp(loc, ex, a1);
            ex = ex->trySemantic(sc);

            /* .f(e1, e2)
             */
            Expressions *a2 = new Expressions();
            a2->setDim(2);
            (*a2)[0] = eleft;
            (*a2)[1] = e2;
            e = new CallExp(loc, e, a2);
            if (ex)
            {   // if fallback setter exists, gag errors
                e = e->trySemantic(sc);
                if (!e)
                {   checkPropertyCall(ex, e1);
                    ex = new AssignExp(loc, ex, e2);
                    return ex->semantic(sc);
                }
            }
            else
            {   // strict setter prints errors if fails
                e = e->semantic(sc);
            }
            checkPropertyCall(e, e1);
            return e;
        }
        else
        {
            /* .f(e1)
             */
            Expressions *arguments = new Expressions();
            arguments->setDim(1);
            (*arguments)[0] = eleft;
            e = new CallExp(loc, e, arguments);
            e = e->semantic(sc);
            checkPropertyCall(e, e1);
            return e->semantic(sc);
        }
    }
    return e;
}

/*********************************
 * Attempt to find a type property. If failed, attempt to find
 * UFCS property. If UFCS found, return expression. Otherwise
 * show type property error message.
 * Returns non-NULL only if UFCS property found.
 */
Expression * resolveProperty(Scope *sc, Expression **e1, Expression *e2)
{
    enum TOK op = (*e1)->op;
    UnaExp *una = (UnaExp *)(*e1);
    Type *t = una->e1->type;
    int olderrors = global.errors;
    una->e1 = una->e1->semantic(sc);
    if (global.errors == olderrors && una->e1->type)
    {
        unsigned errors = global.startGagging();
        // try property gagged
        if (op == TOKdotti)
            *e1 = ((DotTemplateInstanceExp *)una)->semantic(sc, 1);
        else if (op == TOKdot)
            *e1 = ((DotIdExp *)una)->semantic(sc, 1);

        if (global.endGagging(errors) || (*e1)->op == TOKerror)
        {
            (*e1)->op = op;
            errors = global.startGagging();  // try UFCS gagged
            Expression *e = resolveUFCSProperties(sc, una, e2);
            if (!global.endGagging(errors) && (*e1)->op != TOKerror)
                return e;  // found UFCS

            // try property non-gagged
            una->type = t;  // restore type
            if (op == TOKdotti)
                *e1 = ((DotTemplateInstanceExp *)una)->semantic(sc, 1);
            else if (op == TOKdot)
                *e1 = ((DotIdExp *)una)->semantic(sc, 1);
        }
    }

    return NULL;
}

/******************************
 * Perform semantic() on an array of Expressions.
 */

Expressions *arrayExpressionSemantic(Expressions *exps, Scope *sc)
{
    if (exps)
    {
        for (size_t i = 0; i < exps->dim; i++)
        {   Expression *e = (*exps)[i];
            if (e)
            {   e = e->semantic(sc);
                (*exps)[i] = e;
            }
        }
    }
    return exps;
}


/******************************
 * Perform canThrow() on an array of Expressions.
 */

#if DMDV2
int arrayExpressionCanThrow(Expressions *exps, bool mustNotThrow)
{
    if (exps)
    {
        for (size_t i = 0; i < exps->dim; i++)
        {   Expression *e = (*exps)[i];
            if (e && e->canThrow(mustNotThrow))
                return 1;
        }
    }
    return 0;
}
#endif

/****************************************
 * Expand tuples.
 */

void expandTuples(Expressions *exps)
{
    //printf("expandTuples()\n");
    if (exps)
    {
        for (size_t i = 0; i < exps->dim; i++)
        {   Expression *arg = (*exps)[i];
            if (!arg)
                continue;

            // Look for tuple with 0 members
            if (arg->op == TOKtype)
            {   TypeExp *e = (TypeExp *)arg;
                if (e->type->toBasetype()->ty == Ttuple)
                {   TypeTuple *tt = (TypeTuple *)e->type->toBasetype();

                    if (!tt->arguments || tt->arguments->dim == 0)
                    {
                        exps->remove(i);
                        if (i == exps->dim)
                            return;
                        i--;
                        continue;
                    }
                }
            }

            // Inline expand all the tuples
            while (arg->op == TOKtuple)
            {   TupleExp *te = (TupleExp *)arg;

                exps->remove(i);                // remove arg
                exps->insert(i, te->exps);      // replace with tuple contents
                if (i == exps->dim)
                    return;             // empty tuple, no more arguments
                arg = (*exps)[i];
            }
        }
    }
}

/****************************************
 * Expand alias this tuples.
 */

TupleDeclaration *isAliasThisTuple(Expression *e)
{
    if (e->type)
    {
        Type *t = e->type->toBasetype();
        AggregateDeclaration *ad;
        if (t->ty == Tstruct)
        {
            ad = ((TypeStruct *)t)->sym;
            goto L1;
        }
        else if (t->ty == Tclass)
        {
            ad = ((TypeClass *)t)->sym;
          L1:
            Dsymbol *s = ad->aliasthis;
            if (s && s->isVarDeclaration())
            {
                TupleDeclaration *td = s->isVarDeclaration()->toAlias()->isTupleDeclaration();
                if (td && td->isexp)
                    return td;
            }
        }
    }
    return NULL;
}

int expandAliasThisTuples(Expressions *exps, size_t starti)
{
    if (!exps || exps->dim == 0)
        return -1;

    for (size_t u = starti; u < exps->dim; u++)
    {
        Expression *exp = (*exps)[u];
        TupleDeclaration *td = isAliasThisTuple(exp);
        if (td)
        {
            exps->remove(u);
            for (size_t i = 0; i<td->objects->dim; ++i)
            {
                Expression *e = isExpression((*td->objects)[i]);
                assert(e);
                assert(e->op == TOKdsymbol);
                DsymbolExp *se = (DsymbolExp *)e;
                Declaration *d = se->s->isDeclaration();
                assert(d);
                e = new DotVarExp(exp->loc, exp, d);
                assert(d->type);
                e->type = d->type;
                exps->insert(u + i, e);
            }
    #if 0
            printf("expansion ->\n");
            for (size_t i = 0; i<exps->dim; ++i)
            {
                Expression *e = (*exps)[i];
                printf("\texps[%d] e = %s %s\n", i, Token::tochars[e->op], e->toChars());
            }
    #endif
            return (int)u;
        }
    }

    return -1;
}

Expressions *arrayExpressionToCommonType(Scope *sc, Expressions *exps, Type **pt)
{
#if DMDV1
    /* The first element sets the type
     */
    Type *t0 = NULL;
    for (size_t i = 0; i < exps->dim; i++)
    {   Expression *e = (*exps)[i];

        if (!e->type)
        {   error("%s has no value", e->toChars());
            e = new ErrorExp();
        }
        e = resolveProperties(sc, e);

        if (!t0)
            t0 = e->type;
        else
            e = e->implicitCastTo(sc, t0);
        (*exps)[i] = e;
    }

    if (!t0)
        t0 = Type::tvoid;
    if (pt)
        *pt = t0;

    // Eventually, we want to make this copy-on-write
    return exps;
#endif
#if DMDV2
    /* The type is determined by applying ?: to each pair.
     */
    /* Still have a problem with:
     *  ubyte[][] = [ cast(ubyte[])"hello", [1]];
     * which works if the array literal is initialized top down with the ubyte[][]
     * type, but fails with this function doing bottom up typing.
     */
    //printf("arrayExpressionToCommonType()\n");
    IntegerExp integerexp(0);
    CondExp condexp(0, &integerexp, NULL, NULL);

    Type *t0 = NULL;
    Expression *e0;
    size_t j0;
    for (size_t i = 0; i < exps->dim; i++)
    {   Expression *e = (*exps)[i];

        e = resolveProperties(sc, e);
        if (!e->type)
        {   e->error("%s has no value", e->toChars());
            e = new ErrorExp();
        }

        e = callCpCtor(e->loc, sc, e, 1);

        if (t0)
        {   if (t0 != e->type)
            {
                /* This applies ?: to merge the types. It's backwards;
                 * ?: should call this function to merge types.
                 */
                condexp.type = NULL;
                condexp.e1 = e0;
                condexp.e2 = e;
                condexp.loc = e->loc;
                condexp.semantic(sc);
                (*exps)[j0] = condexp.e1;
                e = condexp.e2;
                j0 = i;
                e0 = e;
                t0 = e0->type;
            }
        }
        else
        {   j0 = i;
            e0 = e;
            t0 = e->type;
        }
        (*exps)[i] = e;
    }

    if (t0)
    {
        for (size_t i = 0; i < exps->dim; i++)
        {   Expression *e = (*exps)[i];
            e = e->implicitCastTo(sc, t0);
            (*exps)[i] = e;
        }
    }
    else
        t0 = Type::tvoid;               // [] is typed as void[]
    if (pt)
        *pt = t0;

    // Eventually, we want to make this copy-on-write
    return exps;
#endif
}

/****************************************
 * Get TemplateDeclaration enclosing FuncDeclaration.
 */

TemplateDeclaration *getFuncTemplateDecl(Dsymbol *s)
{
    FuncDeclaration *f = s->isFuncDeclaration();
    if (f && f->parent)
    {   TemplateInstance *ti = f->parent->isTemplateInstance();

        if (ti &&
            !ti->isTemplateMixin() &&
            (ti->name == f->ident ||
             ti->toAlias()->ident == f->ident)
            &&
            ti->tempdecl && ti->tempdecl->onemember)
        {
            return ti->tempdecl;
        }
    }
    return NULL;
}

/****************************************
 * Preprocess arguments to function.
 */

void preFunctionParameters(Loc loc, Scope *sc, Expressions *exps)
{
    if (exps)
    {
        expandTuples(exps);

        for (size_t i = 0; i < exps->dim; i++)
        {   Expression *arg = (*exps)[i];

            arg = resolveProperties(sc, arg);
            (*exps)[i] =  arg;

            //arg->rvalue();
        }
    }
}

/************************************************
 * If we want the value of this expression, but do not want to call
 * the destructor on it.
 */

void valueNoDtor(Expression *e)
{
    if (e->op == TOKcall)
    {
        /* The struct value returned from the function is transferred
         * so do not call the destructor on it.
         * Recognize:
         *       ((S _ctmp = S.init), _ctmp).this(...)
         * and make sure the destructor is not called on _ctmp
         * BUG: if e is a CommaExp, we should go down the right side.
         */
        CallExp *ce = (CallExp *)e;
        if (ce->e1->op == TOKdotvar)
        {   DotVarExp *dve = (DotVarExp *)ce->e1;
            if (dve->var->isCtorDeclaration())
            {   // It's a constructor call
                if (dve->e1->op == TOKcomma)
                {   CommaExp *comma = (CommaExp *)dve->e1;
                    if (comma->e2->op == TOKvar)
                    {   VarExp *ve = (VarExp *)comma->e2;
                        VarDeclaration *ctmp = ve->var->isVarDeclaration();
                        if (ctmp)
                            ctmp->noscope = 1;
                    }
                }
            }
        }
    }
}

/********************************************
 * Determine if t is an array of structs that need a postblit.
 */
#if DMDV2
int checkPostblit(Loc loc, Type *t)
{
    t = t->toBasetype();
    while (t->ty == Tsarray)
        t = t->nextOf()->toBasetype();
    if (t->ty == Tstruct)
    {   FuncDeclaration *fd = ((TypeStruct *)t)->sym->postblit;
        if (fd)
        {   if (fd->storage_class & STCdisable)
                fd->toParent()->error(loc, "is not copyable because it is annotated with @disable");
            return 1;
        }
    }
    return 0;
}
#endif

/*********************************************
 * Call copy constructor for struct value argument.
 */
#if DMDV2
Expression *callCpCtor(Loc loc, Scope *sc, Expression *e, int noscope)
{
    if (e->op == TOKarrayliteral)
    {
        ArrayLiteralExp *ae = (ArrayLiteralExp *)e;
        for (size_t i = 0; i < ae->elements->dim; i++)
        {
            ae->elements->tdata()[i] =
                callCpCtor(loc, sc, ae->elements->tdata()[i], noscope);
        }
        e = ae->semantic(sc);
        return e;
    }

    Type *tb = e->type->toBasetype();
    Type *tv = tb;
    while (tv->ty == Tsarray)
        tv = tv->nextOf()->toBasetype();
    if (tv->ty == Tstruct)
    {
        StructDeclaration *sd = ((TypeStruct *)tv)->sym;
        if (sd->cpctor && e->isLvalue())
        {
            /* Create a variable tmp, and replace the argument e with:
             *      (tmp = e),tmp
             * and let AssignExp() handle the construction.
             * This is not the most efficent, ideally tmp would be constructed
             * directly onto the stack.
             */
            Identifier *idtmp = Lexer::uniqueId("__cpcttmp");
            VarDeclaration *tmp = new VarDeclaration(loc, tb, idtmp, new ExpInitializer(0, e));
            tmp->storage_class |= STCctfe;
            tmp->noscope = noscope;
            Expression *ae = new DeclarationExp(loc, tmp);
            e = new CommaExp(loc, ae, new VarExp(loc, tmp));
            e = e->semantic(sc);
        }
    }
    return e;
}
#endif

/****************************************
 * Now that we know the exact type of the function we're calling,
 * the arguments[] need to be adjusted:
 *      1. implicitly convert argument to the corresponding parameter type
 *      2. add default arguments for any missing arguments
 *      3. do default promotions on arguments corresponding to ...
 *      4. add hidden _arguments[] argument
 *      5. call copy constructor for struct value arguments
 * Input:
 *      fd      the function being called, NULL if called indirectly
 * Returns:
 *      return type from function
 */

Type *functionParameters(Loc loc, Scope *sc, TypeFunction *tf,
        Expression *ethis, Expressions *arguments, FuncDeclaration *fd)
{
    //printf("functionParameters()\n");
    assert(arguments);
    assert(fd || tf->next);
    size_t nargs = arguments ? arguments->dim : 0;
    size_t nparams = Parameter::dim(tf->parameters);

    if (nargs > nparams && tf->varargs == 0)
    {   error(loc, "expected %llu arguments, not %llu for non-variadic function type %s", (ulonglong)nparams, (ulonglong)nargs, tf->toChars());
        return Type::terror;
    }

    // If inferring return type, and semantic3() needs to be run if not already run
    if (!tf->next && fd->inferRetType)
    {
        fd->functionSemantic();
    }
    else if (fd && fd->parent)
    {
        TemplateInstance *ti = fd->parent->isTemplateInstance();
        if (ti && ti->tempdecl)
        {
            fd->functionSemantic3();
        }
    }

    size_t n = (nargs > nparams) ? nargs : nparams;   // n = max(nargs, nparams)

    unsigned wildmatch = 0;
    if (ethis && tf->isWild())
    {
        Type *t = ethis->type;
        if (t->isWild())
            wildmatch |= MODwild;
        else if (t->isConst())
            wildmatch |= MODconst;
        else if (t->isImmutable())
            wildmatch |= MODimmutable;
        else
            wildmatch |= MODmutable;
    }

    int done = 0;
    for (size_t i = 0; i < n; i++)
    {
        Expression *arg;

        if (i < nargs)
            arg = (*arguments)[i];
        else
            arg = NULL;

        if (i < nparams)
        {
            Parameter *p = Parameter::getNth(tf->parameters, i);

            if (!arg)
            {
                if (!p->defaultArg || !fd)
                {
                    if (tf->varargs == 2 && i + 1 == nparams)
                        goto L2;
                    error(loc, "expected %llu function arguments, not %llu", (ulonglong)nparams, (ulonglong)nargs);
                    return Type::terror;
                }
                arg = p->defaultArg;
                arg = arg->inlineCopy(sc);
#if DMDV2
                arg = arg->resolveLoc(loc, sc);         // __FILE__ and __LINE__
#endif
                arguments->push(arg);
                nargs++;
            }
            else
            {
                Type *pt = p->type;
                if (tf->varargs == 2 && i + 1 == nparams && pt->nextOf())
                    pt = pt->nextOf();
                arg = arg->inferType(pt);
                (*arguments)[i] = arg;
            }

            if (tf->varargs == 2 && i + 1 == nparams)
            {
                //printf("\t\tvarargs == 2, p->type = '%s'\n", p->type->toChars());
                MATCH m;
                if ((m = arg->implicitConvTo(p->type)) != MATCHnomatch)
                {
                    if (p->type->nextOf() && arg->implicitConvTo(p->type->nextOf()) >= m)
                        goto L2;
                    else if (nargs != nparams)
                    {   error(loc, "expected %llu function arguments, not %llu", (ulonglong)nparams, (ulonglong)nargs);
                        return Type::terror;
                    }
                    goto L1;
                }
             L2:
                Type *tb = p->type->toBasetype();
                Type *tret = p->isLazyArray();
                switch (tb->ty)
                {
                    case Tsarray:
                    case Tarray:
                    {   // Create a static array variable v of type arg->type
#ifdef IN_GCC
                        /* GCC 4.0 does not like zero length arrays used like
                           this; pass a null array value instead. Could also
                           just make a one-element array. */
                        if (nargs - i == 0)
                        {
                            arg = new NullExp(loc);
                            break;
                        }
#endif
                        Identifier *id = Lexer::uniqueId("__arrayArg");
                        Type *t = new TypeSArray(((TypeArray *)tb)->next, new IntegerExp(nargs - i));
                        t = t->semantic(loc, sc);
                        bool isSafe = fd ? fd->isSafe() : tf->trust == TRUSTsafe;
                        VarDeclaration *v = new VarDeclaration(loc, t, id,
                            (isSafe && sc->func) ? NULL : new VoidInitializer(loc));
                        v->storage_class |= STCctfe;
                        v->semantic(sc);
                        v->parent = sc->parent;
                        //sc->insert(v);

                        Expression *c = new DeclarationExp(0, v);
                        c->type = v->type;

                        for (size_t u = i; u < nargs; u++)
                        {   Expression *a = (*arguments)[u];
                            TypeArray *ta = (TypeArray *)tb;
                            if (tret && !ta->next->equals(a->type))
                            {   if (tret->toBasetype()->ty == Tvoid ||
                                    a->implicitConvTo(tret))
                                {
                                    a = a->toDelegate(sc, tret);
                                }
                            }

                            Expression *e = new VarExp(loc, v);
                            e = new IndexExp(loc, e, new IntegerExp(u + 1 - nparams));
                            ConstructExp *ae = new ConstructExp(loc, e, a);
                            if (c)
                                c = new CommaExp(loc, c, ae);
                            else
                                c = ae;
                        }
                        arg = new VarExp(loc, v);
                        if (c)
                            arg = new CommaExp(loc, c, arg);
                        break;
                    }
                    case Tclass:
                    {   /* Set arg to be:
                         *      new Tclass(arg0, arg1, ..., argn)
                         */
                        Expressions *args = new Expressions();
                        args->setDim(nargs - i);
                        for (size_t u = i; u < nargs; u++)
                            (*args)[u - i] = (*arguments)[u];
                        arg = new NewExp(loc, NULL, NULL, p->type, args);
                        break;
                    }
                    default:
                        if (!arg)
                        {   error(loc, "not enough arguments");
                            return Type::terror;
                        }
                        break;
                }
                arg = arg->semantic(sc);
                //printf("\targ = '%s'\n", arg->toChars());
                arguments->setDim(i + 1);
                (*arguments)[i] =  arg;
                nargs = i + 1;
                done = 1;
            }

        L1:
            if (!(p->storageClass & STClazy && p->type->ty == Tvoid))
            {
                unsigned mod = arg->type->wildConvTo(p->type);
                if (mod)
                {
                    wildmatch |= mod;
                }
            }
        }
        if (done)
            break;
    }
    if (wildmatch)
    {   /* Calculate wild matching modifier
         */
        if (wildmatch & MODconst || wildmatch & (wildmatch - 1))
            wildmatch = MODconst;
        else if (wildmatch & MODimmutable)
            wildmatch = MODimmutable;
        else if (wildmatch & MODwild)
            wildmatch = MODwild;
        else
        {   assert(wildmatch & MODmutable);
            wildmatch = MODmutable;
        }
    }

    assert(nargs >= nparams);
    for (size_t i = 0; i < nargs; i++)
    {
        Expression *arg = (*arguments)[i];
        assert(arg);

        if (i < nparams)
        {
            Parameter *p = Parameter::getNth(tf->parameters, i);

            if (!(p->storageClass & STClazy && p->type->ty == Tvoid))
            {
                if (p->type->hasWild())
                {
                    arg = arg->implicitCastTo(sc, p->type->substWildTo(wildmatch));
                    arg = arg->optimize(WANTvalue, p->storageClass & STCref);
                }
                else if (p->type != arg->type)
                {
                    //printf("arg->type = %s, p->type = %s\n", arg->type->toChars(), p->type->toChars());
                    if (arg->op == TOKtype)
                    {   arg->error("cannot pass type %s as function argument", arg->toChars());
                        arg = new ErrorExp();
                        goto L3;
                    }
                    else
                        arg = arg->implicitCastTo(sc, p->type);
                    arg = arg->optimize(WANTvalue, p->storageClass & STCref);
                }
            }
            if (p->storageClass & STCref)
            {
                arg = arg->toLvalue(sc, arg);
            }
            else if (p->storageClass & STCout)
            {
                Type *t = arg->type;
                if (!t->isMutable() || !t->isAssignable())  // check blit assignable
                    arg->error("cannot modify struct %s with immutable members", arg->toChars());
                arg = arg->toLvalue(sc, arg);
            }
            else if (p->storageClass & STClazy)
            {   // Convert lazy argument to a delegate
                arg = arg->toDelegate(sc, p->type);
            }
            else
            {
                Type *tb = arg->type->toBasetype();
                if (tb->ty == Tsarray)
                {
                    // call copy constructor of each element
                    arg = callCpCtor(loc, sc, arg, 1);
                }
#if DMDV2
                else if (tb->ty == Tstruct)
                {
                    if (arg->op == TOKcall && !arg->isLvalue())
                    {
                        /* The struct value returned from the function is transferred
                         * to the function, so the callee should not call the destructor
                         * on it.
                         */
                        valueNoDtor(arg);
                    }
                    else
                    {   /* Not transferring it, so call the copy constructor
                         */
                        arg = callCpCtor(loc, sc, arg, 1);
                    }
                }
#endif
            }

            //printf("arg: %s\n", arg->toChars());
            //printf("type: %s\n", arg->type->toChars());

#if DMDV2
            /* Look for arguments that cannot 'escape' from the called
             * function.
             */
            if (!tf->parameterEscapes(p))
            {
                Expression *a = arg;
                if (a->op == TOKcast)
                    a = ((CastExp *)a)->e1;

                /* Function literals can only appear once, so if this
                 * appearance was scoped, there cannot be any others.
                 */
                if (a->op == TOKfunction)
                {   FuncExp *fe = (FuncExp *)a;
                    fe->fd->tookAddressOf = 0;
                }

                /* For passing a delegate to a scoped parameter,
                 * this doesn't count as taking the address of it.
                 * We only worry about 'escaping' references to the function.
                 */
                else if (a->op == TOKdelegate)
                {   DelegateExp *de = (DelegateExp *)a;
                    if (de->e1->op == TOKvar)
                    {   VarExp *ve = (VarExp *)de->e1;
                        FuncDeclaration *f = ve->var->isFuncDeclaration();
                        if (f)
                        {   f->tookAddressOf--;
                            //printf("tookAddressOf = %d\n", f->tookAddressOf);
                        }
                    }
                }
            }
#endif
            arg = arg->optimize(WANTvalue, (p->storageClass & (STCref | STCout)) != 0);
        }
        else
        {

            // If not D linkage, do promotions
            // LDC: don't do promotions on intrinsics
            if (tf->linkage != LINKd && tf->linkage != LINKintrinsic)
            {
                // Promote bytes, words, etc., to ints
                arg = arg->integralPromotions(sc);

                // Promote floats to doubles
                switch (arg->type->ty)
                {
                    case Tfloat32:
                        arg = arg->castTo(sc, Type::tfloat64);
                        break;

                    case Timaginary32:
                        arg = arg->castTo(sc, Type::timaginary64);
                        break;
                }
            }

            // Do not allow types that need destructors
            if (arg->type->needsDestruction())
            {   arg->error("cannot pass types that need destruction as variadic arguments");
                arg = new ErrorExp();
            }

            // Convert static arrays to dynamic arrays
            // BUG: I don't think this is right for D2
            Type *tb = arg->type->toBasetype();
            if (tb->ty == Tsarray)
            {   TypeSArray *ts = (TypeSArray *)tb;
                Type *ta = ts->next->arrayOf();
                if (ts->size(arg->loc) == 0)
                    arg = new NullExp(arg->loc, ta);
                else
                    arg = arg->castTo(sc, ta);
            }
#if DMDV2
            if (tb->ty == Tstruct)
            {
                arg = callCpCtor(loc, sc, arg, 1);
            }
#endif

            // Give error for overloaded function addresses
            if (arg->op == TOKsymoff)
            {   SymOffExp *se = (SymOffExp *)arg;
                if (
#if DMDV2
                    se->hasOverloads &&
#endif
                    !se->var->isFuncDeclaration()->isUnique())
                {   arg->error("function %s is overloaded", arg->toChars());
                    arg = new ErrorExp();
                }
            }
            arg->rvalue();
            arg = arg->optimize(WANTvalue);
        }
    L3:
        (*arguments)[i] =  arg;
    }

#if !IN_LLVM
    // If D linkage and variadic, add _arguments[] as first argument
    if (tf->linkage == LINKd && tf->varargs == 1)
    {
        assert(arguments->dim >= nparams);
        Expression *e = createTypeInfoArray(sc, (Expression **)&arguments->tdata()[nparams],
                arguments->dim - nparams);
        arguments->insert(0, e);
    }
#endif
    Type *tret = tf->next;
    if (wildmatch)
    {   /* Adjust function return type based on wildmatch
         */
        //printf("wildmatch = x%x, tret = %s\n", wildmatch, tret->toChars());
        tret = tret->substWildTo(wildmatch);
    }
    return tret;
}

/**************************************************
 * Write expression out to buf, but wrap it
 * in ( ) if its precedence is less than pr.
 */

void expToCBuffer(OutBuffer *buf, HdrGenState *hgs, Expression *e, enum PREC pr)
{
#if !IN_LLVM
#ifdef DEBUG
    if (precedence[e->op] == PREC_zero)
        printf("precedence not defined for token '%s'\n",Token::tochars[e->op]);
#endif
    assert(precedence[e->op] != PREC_zero);
    assert(pr != PREC_zero);
#endif

    //if (precedence[e->op] == 0) e->dump(0);
    if (precedence[e->op] < pr ||
        /* Despite precedence, we don't allow a<b<c expressions.
         * They must be parenthesized.
         */
        (pr == PREC_rel && precedence[e->op] == pr))
    {
        buf->writeByte('(');
        e->toCBuffer(buf, hgs);
        buf->writeByte(')');
    }
    else
        e->toCBuffer(buf, hgs);
}

/**************************************************
 * Write out argument list to buf.
 */

void argsToCBuffer(OutBuffer *buf, Expressions *expressions, HdrGenState *hgs)
{
    if (expressions)
    {
        for (size_t i = 0; i < expressions->dim; i++)
        {   Expression *e = (*expressions)[i];

            if (i)
                buf->writestring(", ");
            if (e)
                expToCBuffer(buf, hgs, e, PREC_assign);
        }
    }
}

/**************************************************
 * Write out argument types to buf.
 */

void argExpTypesToCBuffer(OutBuffer *buf, Expressions *arguments, HdrGenState *hgs)
{
    if (arguments)
    {   OutBuffer argbuf;

        for (size_t i = 0; i < arguments->dim; i++)
        {   Expression *e = (*arguments)[i];

            if (i)
                buf->writeByte(',');
            argbuf.reset();
            e->type->toCBuffer2(&argbuf, hgs, 0);
            buf->write(&argbuf);
        }
    }
}

/******************************** Expression **************************/

Expression::Expression(Loc loc, enum TOK op, int size)
    : loc(loc)
{
    //printf("Expression::Expression(op = %d) this = %p\n", op, this);
    this->loc = loc;
    this->op = op;
    this->size = size;
    this->parens = 0;
    type = NULL;

#if IN_LLVM
    cachedLvalue = NULL;
#endif
}

Expression *Expression::syntaxCopy()
{
    //printf("Expression::syntaxCopy()\n");
    //dump(0);
    return copy();
}

/*********************************
 * Does *not* do a deep copy.
 */

Expression *Expression::copy()
{
    Expression *e;
    if (!size)
    {
#ifdef DEBUG
        fprintf(stdmsg, "No expression copy for: %s\n", toChars());
        printf("op = %d\n", op);
        dump(0);
#endif
        assert(0);
    }
    e = (Expression *)mem.malloc(size);
    //printf("Expression::copy(op = %d) e = %p\n", op, e);
    return (Expression *)memcpy(e, this, size);
}

/**************************
 * Semantically analyze Expression.
 * Determine types, fold constants, etc.
 */

Expression *Expression::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("Expression::semantic() %s\n", toChars());
#endif
    if (type)
        type = type->semantic(loc, sc);
    else
        type = Type::tvoid;
    return this;
}

/**********************************
 * Try to run semantic routines.
 * If they fail, return NULL.
 */

Expression *Expression::trySemantic(Scope *sc)
{
    //printf("+trySemantic(%s)\n", toChars());
    unsigned errors = global.startGagging();
    Expression *e = semantic(sc);
    if (global.endGagging(errors))
    {
        e = NULL;
    }
    //printf("-trySemantic(%s)\n", toChars());
    return e;
}

void Expression::print()
{
    fprintf(stdmsg, "%s\n", toChars());
    fflush(stdmsg);
}

char *Expression::toChars()
{
    HdrGenState hgs;
    memset(&hgs, 0, sizeof(hgs));

    OutBuffer buf;
    toCBuffer(&buf, &hgs);
    buf.writeByte(0);
    char *p = (char *)buf.data;
    buf.data = NULL;
    return p;
}

void Expression::error(const char *format, ...)
{
    if (type != Type::terror)
    {
        va_list ap;
        va_start(ap, format);
        ::verror(loc, format, ap);
        va_end( ap );
    }
}

void Expression::warning(const char *format, ...)
{
    if (type != Type::terror)
    {
        va_list ap;
        va_start(ap, format);
        ::vwarning(loc, format, ap);
        va_end( ap );
    }
}

void Expression::deprecation(const char *format, ...)
{
    if (type != Type::terror)
    {
        va_list ap;
        va_start(ap, format);
        ::vdeprecation(loc, format, ap);
        va_end( ap );
    }
}

int Expression::rvalue()
{
    if (type && type->toBasetype()->ty == Tvoid)
    {   error("expression %s is void and has no value", toChars());
#if 0
        dump(0);
        halt();
#endif
        if (!global.gag)
            type = Type::terror;
        return 0;
    }
    return 1;
}

Expression *Expression::combine(Expression *e1, Expression *e2)
{
    if (e1)
    {
        if (e2)
        {
            e1 = new CommaExp(e1->loc, e1, e2);
            e1->type = e2->type;
        }
    }
    else
        e1 = e2;
    return e1;
}

dinteger_t Expression::toInteger()
{
    //printf("Expression %s\n", Token::toChars(op));
    error("Integer constant expression expected instead of %s", toChars());
    return 0;
}

uinteger_t Expression::toUInteger()
{
    //printf("Expression %s\n", Token::toChars(op));
    return (uinteger_t)toInteger();
}

real_t Expression::toReal()
{
    error("Floating point constant expression expected instead of %s", toChars());
    return ldouble(0);
}

real_t Expression::toImaginary()
{
    error("Floating point constant expression expected instead of %s", toChars());
    return ldouble(0);
}

complex_t Expression::toComplex()
{
    error("Floating point constant expression expected instead of %s", toChars());
#ifdef IN_GCC
    return complex_t(real_t(0)); // %% nicer
#else
    return 0.0;
#endif
}

StringExp *Expression::toString()
{
    return NULL;
}

void Expression::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring(Token::toChars(op));
}

void Expression::toMangleBuffer(OutBuffer *buf)
{
    error("expression %s is not a valid template value argument", toChars());
}

/***************************************
 * Return !=0 if expression is an lvalue.
 */

int Expression::isLvalue()
{
    return 0;
}

/*******************************
 * Give error if we're not an lvalue.
 * If we can, convert expression to be an lvalue.
 */

Expression *Expression::toLvalue(Scope *sc, Expression *e)
{
    if (!e)
        e = this;
    else if (!loc.filename)
        loc = e->loc;
    error("%s is not an lvalue", e->toChars());
    return new ErrorExp();
}

/***************************************
 * Parameters:
 *      sc:     scope
 *      flag:   1: do not issue error message for invalid modification
 * Returns:
 *      0:      is not modifiable
 *      1:      is modifiable in default == being related to type->isMutable()
 *      2:      is modifiable, because this is a part of initializing.
 */

int Expression::checkModifiable(Scope *sc, int flag)
{
    return type ? 1 : 0;    // default modifiable
}

Expression *Expression::modifiableLvalue(Scope *sc, Expression *e)
{
    //printf("Expression::modifiableLvalue() %s, type = %s\n", toChars(), type->toChars());

    // See if this expression is a modifiable lvalue (i.e. not const)
    if (checkModifiable(sc) == 1)
    {
        assert(type);
        if (type->isMutable())
        {
            if (!type->isAssignable())
                error("cannot modify struct %s %s with immutable members", toChars(), type->toChars());
        }
        else
        {
            Declaration *var = NULL;
            if (op == TOKvar)
                var = ((VarExp *)this)->var;
            else if (op == TOKdotvar)
                var = ((DotVarExp *)this)->var;
            if (var && var->storage_class & STCctorinit)
            {
                const char *p = var->isStatic() ? "static " : "";
                error("can only initialize %sconst member %s inside %sconstructor",
                    p, var->toChars(), p);
            }
            else
            {
                error("cannot modify %s expression %s", MODtoChars(type->mod), toChars());
            }
        }
    }
    return toLvalue(sc, e);
}


/************************************
 * Detect cases where pointers to the stack can 'escape' the
 * lifetime of the stack frame.
 */

void Expression::checkEscape()
{
}

void Expression::checkEscapeRef()
{
}

void Expression::checkScalar()
{
    if (!type->isscalar() && type->toBasetype() != Type::terror)
        error("'%s' is not a scalar, it is a %s", toChars(), type->toChars());
    rvalue();
}

void Expression::checkNoBool()
{
    if (type->toBasetype()->ty == Tbool)
        error("operation not allowed on bool '%s'", toChars());
}

Expression *Expression::checkIntegral()
{
    if (!type->isintegral())
    {   if (type->toBasetype() != Type::terror)
            error("'%s' is not of integral type, it is a %s", toChars(), type->toChars());
        return new ErrorExp();
    }
    if (!rvalue())
        return new ErrorExp();
    return this;
}

Expression *Expression::checkArithmetic()
{
    if (!type->isintegral() && !type->isfloating())
    {   if (type->toBasetype() != Type::terror)
            error("'%s' is not of arithmetic type, it is a %s", toChars(), type->toChars());
        return new ErrorExp();
    }
    if (!rvalue())
        return new ErrorExp();
    return this;
}

void Expression::checkDeprecated(Scope *sc, Dsymbol *s)
{
    s->checkDeprecated(loc, sc);
}

#if DMDV2
/*********************************************
 * Calling function f.
 * Check the purity, i.e. if we're in a pure function
 * we can only call other pure functions.
 */
void Expression::checkPurity(Scope *sc, FuncDeclaration *f)
{
#if 1
    if (sc->func && !sc->intypeof && !(sc->flags & SCOPEdebug))
    {
        /* Given:
         * void f()
         * { pure void g()
         *   {
         *      void h()
         *      {
         *         void i() { }
         *      }
         *   }
         * }
         * g() can call h() but not f()
         * i() can call h() and g() but not f()
         */

        // Find the closest pure parent of the calling function
        FuncDeclaration *outerfunc = sc->func;
        while ( outerfunc->toParent2() &&
               !outerfunc->isPureBypassingInference() &&
                outerfunc->toParent2()->isFuncDeclaration())
        {
            outerfunc = outerfunc->toParent2()->isFuncDeclaration();
        }

        // Find the closest pure parent of the called function
        if (getFuncTemplateDecl(f))
        {   // The closest pure parent of instantiated template function is
            // always itself.
            if (!f->isPure() && outerfunc->setImpure())
                error("pure function '%s' cannot call impure function '%s'",
                    outerfunc->toChars(), f->toChars());
            return;
        }
        FuncDeclaration *calledparent = f;
        while ( calledparent->toParent2() &&
               !calledparent->isPureBypassingInference() &&
                calledparent->toParent2()->isFuncDeclaration())
        {
            calledparent = calledparent->toParent2()->isFuncDeclaration();
        }

        /* Both escape!allocator and escapeImpl!allocator are impure at [a],
         * but they are nested template function that instantiated in test().
         * Then calling them from [a] doesn't break purity.
         * It's similar to normal impure nested function inside pure function.
         *
         *   auto escapeImpl(alias fun)() {
         *     return fun();
         *   }
         *   auto escape(alias fun)() {
         *     return escape!fun();
         *   }
         *   pure string test() {
         *     char[] allocator() { return new char[1]; }  // impure
         *     return escape!allocator();       // [a]
         *   }
         */
        if (getFuncTemplateDecl(outerfunc) &&
            outerfunc->toParent2() == calledparent &&
            f != calledparent)
        {
            return;
        }

        // If the caller has a pure parent, then either the called func must be pure,
        // OR, they must have the same pure parent.
        if (/*outerfunc->isPure() &&*/    // comment out because we deduce purity now
            !f->isPure() && calledparent != outerfunc)
        {
            if (outerfunc->setImpure())
                error("pure function '%s' cannot call impure function '%s'",
                    outerfunc->toPrettyChars(), f->toPrettyChars());
        }
    }
#else
    if (sc->func && sc->func->isPure() && !sc->intypeof && !f->isPure())
        error("pure function '%s' cannot call impure function '%s'",
            sc->func->toPrettyChars(), f->toPrettyChars());
#endif
}

/*******************************************
 * Accessing variable v.
 * Check for purity and safety violations.
 * If ethis is not NULL, then ethis is the 'this' pointer as in ethis.v
 */

void Expression::checkPurity(Scope *sc, VarDeclaration *v, Expression *ethis)
{
    /* Look for purity and safety violations when accessing variable v
     * from current function.
     */
    if (sc->func &&
        !sc->intypeof &&             // allow violations inside typeof(expression)
        !(sc->flags & SCOPEdebug) && // allow violations inside debug conditionals
        v->ident != Id::ctfe &&      // magic variable never violates pure and safe
        !v->isImmutable() &&         // always safe and pure to access immutables...
        !(v->isConst() && !v->isRef() && (v->isDataseg() || v->isParameter()) &&
          v->type->implicitConvTo(v->type->invariantOf())) &&
            // or const global/parameter values which have no mutable indirections
        !(v->storage_class & STCmanifest) // ...or manifest constants
       )
    {
        if (v->isDataseg())
        {
            /* Accessing global mutable state.
             * Therefore, this function and all its immediately enclosing
             * functions must be pure.
             */
            bool msg = FALSE;
            for (Dsymbol *s = sc->func; s; s = s->toParent2())
            {
                FuncDeclaration *ff = s->isFuncDeclaration();
                if (!ff)
                    break;
                if (ff->setImpure() && !msg)
                {   error("pure function '%s' cannot access mutable static data '%s'",
                        sc->func->toPrettyChars(), v->toChars());
                    msg = TRUE;                     // only need the innermost message
                }
            }
        }
        else
        {
            /* Given:
             * void f()
             * { int fx;
             *   pure void g()
             *   {  int gx;
             *      void h()
             *      {  int hx;
             *         void i() { }
             *      }
             *   }
             * }
             * i() can modify hx and gx but not fx
             */

            Dsymbol *vparent = v->toParent2();
            for (Dsymbol *s = sc->func; s; s = s->toParent2())
            {
                if (s == vparent)
                        break;
                FuncDeclaration *ff = s->isFuncDeclaration();
                if (!ff)
                    break;
                if (ff->setImpure())
                {   error("pure nested function '%s' cannot access mutable data '%s'",
                        ff->toChars(), v->toChars());
                    break;
                }
            }
        }

        /* Do not allow safe functions to access __gshared data
         */
        if (v->storage_class & STCgshared)
        {
            if (sc->func->setUnsafe())
                error("safe function '%s' cannot access __gshared data '%s'",
                    sc->func->toChars(), v->toChars());
        }
    }
}

void Expression::checkSafety(Scope *sc, FuncDeclaration *f)
{
    if (sc->func && !sc->intypeof &&
        !f->isSafe() && !f->isTrusted())
    {
        if (sc->func->setUnsafe())
        {
            if (loc.linnum == 0)  // e.g. implicitly generated dtor
                loc = sc->func->loc;

            error("safe function '%s' cannot call system function '%s'",
                sc->func->toPrettyChars(), f->toPrettyChars());
        }
    }
}
#endif

/*****************************
 * Check that expression can be tested for true or false.
 */

Expression *Expression::checkToBoolean(Scope *sc)
{
    // Default is 'yes' - do nothing

#ifdef DEBUG
    if (!type)
        dump(0);
    assert(type);
#endif

    // Structs can be converted to bool using opCast(bool)()
    Type *tb = type->toBasetype();
    if (tb->ty == Tstruct)
    {   AggregateDeclaration *ad = ((TypeStruct *)tb)->sym;
        /* Don't really need to check for opCast first, but by doing so we
         * get better error messages if it isn't there.
         */
        Dsymbol *fd = search_function(ad, Id::cast);
        if (fd)
        {
            Expression *e = new CastExp(loc, this, Type::tbool);
            e = e->semantic(sc);
            return e;
        }

        // Forward to aliasthis.
        if (ad->aliasthis)
        {
            Expression *e = resolveAliasThis(sc, this);
            e = e->checkToBoolean(sc);
            return e;
        }
    }

    if (!type->checkBoolean())
    {   if (type->toBasetype() != Type::terror)
            error("expression %s of type %s does not have a boolean value", toChars(), type->toChars());
        return new ErrorExp();
    }
    return this;
}

/****************************
 */

Expression *Expression::checkToPointer()
{
    //printf("Expression::checkToPointer()\n");
    Expression *e = this;

    return e;
}

/******************************
 * Take address of expression.
 */

Expression *Expression::addressOf(Scope *sc)
{
    Expression *e;
    Type *t = type;

    //printf("Expression::addressOf()\n");
    e = toLvalue(sc, NULL);
    e = new AddrExp(loc, e);
    e->type = t->pointerTo();
    return e;
}

/******************************
 * If this is a reference, dereference it.
 */

Expression *Expression::deref()
{
    //printf("Expression::deref()\n");
    // type could be null if forward referencing an 'auto' variable
    if (type && type->ty == Treference)
    {
        Expression *e = new PtrExp(loc, this);
        e->type = ((TypeReference *)type)->next;
        return e;
    }
    return this;
}

/********************************
 * Does this expression statically evaluate to a boolean TRUE or FALSE?
 */

int Expression::isBool(int result)
{
    return FALSE;
}

/********************************
 * Does this expression result in either a 1 or a 0?
 */

int Expression::isBit()
{
    return FALSE;
}

/****************************************
 * Resolve __LINE__ and __FILE__ to loc.
 */

Expression *Expression::resolveLoc(Loc loc, Scope *sc)
{
    return this;
}

Expressions *Expression::arraySyntaxCopy(Expressions *exps)
{   Expressions *a = NULL;

    if (exps)
    {
        a = new Expressions();
        a->setDim(exps->dim);
        for (size_t i = 0; i < a->dim; i++)
        {   Expression *e = (*exps)[i];

            if (e)
                e = e->syntaxCopy();
            (*a)[i] = e;
        }
    }
    return a;
}

/***************************************************
 * Recognize expressions of the form:
 *      ((T v = init), v)
 * where v is a temp.
 * This is used in optimizing out unnecessary temporary generation.
 * Returns initializer expression of v if so, NULL if not.
 */

Expression *Expression::isTemp()
{
    //printf("isTemp() %s\n", toChars());
    if (op == TOKcomma)
    {   CommaExp *ec = (CommaExp *)this;
        if (ec->e1->op == TOKdeclaration &&
            ec->e2->op == TOKvar)
        {   DeclarationExp *de = (DeclarationExp *)ec->e1;
            VarExp *ve = (VarExp *)ec->e2;
            if (ve->var == de->declaration && ve->var->storage_class & STCctfe)
            {   VarDeclaration *v = ve->var->isVarDeclaration();
                if (v && v->init)
                {
                    ExpInitializer *ei = v->init->isExpInitializer();
                    if (ei)
                    {   Expression *e = ei->exp;
                        if (e->op == TOKconstruct)
                        {   ConstructExp *ce = (ConstructExp *)e;
                            if (ce->e1->op == TOKvar && ((VarExp *)ce->e1)->var == ve->var)
                                e = ce->e2;
                        }
                        return e;
                    }
                }
            }
        }
    }
    return NULL;
}

/************************************************
 * Destructors are attached to VarDeclarations.
 * Hence, if expression returns a temp that needs a destructor,
 * make sure and create a VarDeclaration for that temp.
 */

Expression *Expression::addDtorHook(Scope *sc)
{
    return this;
}

/******************************** IntegerExp **************************/

IntegerExp::IntegerExp(Loc loc, dinteger_t value, Type *type)
        : Expression(loc, TOKint64, sizeof(IntegerExp))
{
    //printf("IntegerExp(value = %lld, type = '%s')\n", value, type ? type->toChars() : "");
    if (type && !type->isscalar())
    {
        //printf("%s, loc = %d\n", toChars(), loc.linnum);
        if (type->ty != Terror)
            error("integral constant must be scalar type, not %s", type->toChars());
        type = Type::terror;
    }
    this->type = type;
    this->value = value;
}

IntegerExp::IntegerExp(dinteger_t value)
        : Expression(0, TOKint64, sizeof(IntegerExp))
{
    this->type = Type::tint32;
    this->value = value;
}

int IntegerExp::equals(Object *o)
{   IntegerExp *ne;

    if (this == o ||
        (((Expression *)o)->op == TOKint64 &&
         ((ne = (IntegerExp *)o), type->toHeadMutable()->equals(ne->type->toHeadMutable())) &&
         value == ne->value))
        return 1;
    return 0;
}

char *IntegerExp::toChars()
{
    return Expression::toChars();
}

dinteger_t IntegerExp::toInteger()
{   Type *t;

    t = type;
    while (t)
    {
        switch (t->ty)
        {
            case Tbool:         value = (value != 0);           break;
            case Tint8:         value = (d_int8)  value;        break;
            case Tchar:
            case Tuns8:         value = (d_uns8)  value;        break;
            case Tint16:        value = (d_int16) value;        break;
            case Twchar:
            case Tuns16:        value = (d_uns16) value;        break;
            case Tint32:        value = (d_int32) value;        break;
            case Tdchar:
            case Tuns32:        value = (d_uns32) value;        break;
            case Tint64:        value = (d_int64) value;        break;
            case Tuns64:        value = (d_uns64) value;        break;
            case Tpointer:
                if (PTRSIZE == 4)
                    value = (d_uns32) value;
                else if (PTRSIZE == 8)
                    value = (d_uns64) value;
                else
                    assert(0);
                break;

            case Tenum:
            {
                TypeEnum *te = (TypeEnum *)t;
                t = te->sym->memtype;
                continue;
            }

            case Ttypedef:
            {
                TypeTypedef *tt = (TypeTypedef *)t;
                t = tt->sym->basetype;
                continue;
            }

            default:
                /* This can happen if errors, such as
                 * the type is painted on like in fromConstInitializer().
                 */
                if (!global.errors)
                {
                    printf("e = %p, ty = %d\n", this, type->ty);
                    type->print();
                    assert(0);
                }
                break;
        }
        break;
    }
    return value;
}

real_t IntegerExp::toReal()
{
    Type *t;

    toInteger();
    t = type->toBasetype();
    if (t->ty == Tuns64)
        return ldouble((d_uns64)value);
    else
        return ldouble((d_int64)value);
}

real_t IntegerExp::toImaginary()
{
    return ldouble(0);
}

complex_t IntegerExp::toComplex()
{
    return toReal();
}

int IntegerExp::isBool(int result)
{
    int r = toInteger() != 0;
    return result ? r : !r;
}

Expression *IntegerExp::semantic(Scope *sc)
{
    if (!type)
    {
        // Determine what the type of this number is
        dinteger_t number = value;

        if (number & 0x8000000000000000LL)
            type = Type::tuns64;
        else if (number & 0xFFFFFFFF80000000LL)
            type = Type::tint64;
        else
            type = Type::tint32;
    }
    else
    {   if (!type->deco)
            type = type->semantic(loc, sc);
    }
    return this;
}

Expression *IntegerExp::toLvalue(Scope *sc, Expression *e)
{
    if (!e)
        e = this;
    else if (!loc.filename)
        loc = e->loc;
    e->error("constant %s is not an lvalue", e->toChars());
    return new ErrorExp();
}

void IntegerExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    dinteger_t v = toInteger();

    if (type)
    {   Type *t = type;

      L1:
        switch (t->ty)
        {
            case Tenum:
            {   TypeEnum *te = (TypeEnum *)t;
                buf->printf("cast(%s)", te->sym->toChars());
                t = te->sym->memtype;
                goto L1;
            }

            case Ttypedef:
            {   TypeTypedef *tt = (TypeTypedef *)t;
                buf->printf("cast(%s)", tt->sym->toChars());
                t = tt->sym->basetype;
                goto L1;
            }

            case Twchar:        // BUG: need to cast(wchar)
            case Tdchar:        // BUG: need to cast(dchar)
                if ((uinteger_t)v > 0xFF)
                {
                    buf->printf("'\\U%08x'", (unsigned)v);
                    break;
                }
            case Tchar:
            {
                unsigned o = buf->offset;
                if (v == '\'')
                    buf->writestring("'\\''");
                else if (isprint(v) && v != '\\')
                    buf->printf("'%c'", (int)v);
                else
                    buf->printf("'\\x%02x'", (int)v);
                if (hgs->ddoc)
                    escapeDdocString(buf, o);
                break;
            }

            case Tint8:
                buf->writestring("cast(byte)");
                goto L2;

            case Tint16:
                buf->writestring("cast(short)");
                goto L2;

            case Tint32:
            L2:
                buf->printf("%d", (int)v);
                break;

            case Tuns8:
                buf->writestring("cast(ubyte)");
                goto L3;

            case Tuns16:
                buf->writestring("cast(ushort)");
                goto L3;

            case Tuns32:
            L3:
                buf->printf("%uu", (unsigned)v);
                break;

            case Tint64:
                buf->printf("%lldL", v);
                break;

            case Tuns64:
            L4:
                buf->printf("%lluLU", v);
                break;

            case Tbool:
                buf->writestring((char *)(v ? "true" : "false"));
                break;

            case Tpointer:
                buf->writestring("cast(");
                buf->writestring(t->toChars());
                buf->writeByte(')');
                if (PTRSIZE == 4)
                    goto L3;
                else if (PTRSIZE == 8)
                    goto L4;
                else
                    assert(0);

            default:
                /* This can happen if errors, such as
                 * the type is painted on like in fromConstInitializer().
                 */
                if (!global.errors)
                {
#ifdef DEBUG
                    t->print();
#endif
                    assert(0);
                }
                break;
        }
    }
    else if (v & 0x8000000000000000LL)
        buf->printf("0x%llx", v);
    else
        buf->printf("%lld", v);
}

void IntegerExp::toMangleBuffer(OutBuffer *buf)
{
    if ((sinteger_t)value < 0)
        buf->printf("N%lld", -value);
    else
    {
        /* This is an awful hack to maintain backwards compatibility.
         * There really always should be an 'i' before a number, but
         * there wasn't in earlier implementations, so to maintain
         * backwards compatibility it is only done if necessary to disambiguate.
         * See bugzilla 3029
         */
        if (buf->offset > 0 && isdigit(buf->data[buf->offset - 1]))
            buf->writeByte('i');

        buf->printf("%lld", value);
    }
}

/******************************** ErrorExp **************************/

/* Use this expression for error recovery.
 * It should behave as a 'sink' to prevent further cascaded error messages.
 */

ErrorExp::ErrorExp()
    : IntegerExp(0, 0, Type::terror)
{
    op = TOKerror;
}

Expression *ErrorExp::toLvalue(Scope *sc, Expression *e)
{
    return this;
}

void ErrorExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("__error");
}

/******************************** RealExp **************************/

RealExp::RealExp(Loc loc, real_t value, Type *type)
        : Expression(loc, TOKfloat64, sizeof(RealExp))
{
    //printf("RealExp::RealExp(%Lg)\n", value);
    this->value = value;
    this->type = type;
}

char *RealExp::toChars()
{
    char buffer[sizeof(value) * 3 + 8 + 1 + 1];

#ifdef IN_GCC
    value.format(buffer, sizeof(buffer));
#else
    ld_sprint(buffer, 'g', value);
#endif
    if (type->isimaginary())
        strcat(buffer, "i");

    assert(strlen(buffer) < sizeof(buffer));
    return mem.strdup(buffer);
}

dinteger_t RealExp::toInteger()
{
#ifdef IN_GCC
    return (sinteger_t) toReal().toInt();
#else
    return (sinteger_t) toReal();
#endif
}

uinteger_t RealExp::toUInteger()
{
#ifdef IN_GCC
    return (uinteger_t) toReal().toInt();
#else
    return (uinteger_t) toReal();
#endif
}

real_t RealExp::toReal()
{
    return type->isreal() ? value : ldouble(0);
}

real_t RealExp::toImaginary()
{
    return type->isreal() ? ldouble(0) : value;
}

complex_t RealExp::toComplex()
{
#ifdef __DMC__
    return toReal() + toImaginary() * I;
#else
    return complex_t(toReal(), toImaginary());
#endif
}

/********************************
 * Test to see if two reals are the same.
 * Regard NaN's as equivalent.
 * Regard +0 and -0 as different.
 */

int RealEquals(real_t x1, real_t x2)
{
    return (Port::isNan(x1) && Port::isNan(x2)) ||
          // and zero, in order to distinguish +0 from -0
          (x1 == 0 && x2 == 0 && 1./x1 == 1./x2) ||
          // otherwise just compare
          (x1 != 0. && x1 == x2);
}

int RealExp::equals(Object *o)
{   RealExp *ne;

    if (this == o ||
        (((Expression *)o)->op == TOKfloat64 &&
         ((ne = (RealExp *)o), type->toHeadMutable()->equals(ne->type->toHeadMutable())) &&
         RealEquals(value, ne->value)
        )
       )
        return 1;
    return 0;
}

Expression *RealExp::semantic(Scope *sc)
{
    if (!type)
        type = Type::tfloat64;
    else
        type = type->semantic(loc, sc);
    return this;
}

int RealExp::isBool(int result)
{
#ifdef IN_GCC
    return result ? (! value.isZero()) : (value.isZero());
#else
    return result ? (value != 0)
                  : (value == 0);
#endif
}

void floatToBuffer(OutBuffer *buf, Type *type, real_t value)
{
    /* In order to get an exact representation, try converting it
     * to decimal then back again. If it matches, use it.
     * If it doesn't, fall back to hex, which is
     * always exact.
     * Longest string is for -real.max:
     * "-1.18973e+4932\0".length == 17
     * "-0xf.fffffffffffffffp+16380\0".length == 28
     */
    char buffer[32];
    ld_sprint(buffer, 'g', value);
    assert(strlen(buffer) < sizeof(buffer));
#if _WIN32 && __DMC__
    char *save = __locale_decpoint;
    __locale_decpoint = ".";
    real_t r = strtold(buffer, NULL);
    __locale_decpoint = save;
#else
    real_t r = Port::strtold(buffer, NULL);
#endif
#if IN_LLVM
    if (r == value)                     // if exact duplication
        buf->writestring(buffer);
    else
    {
#ifdef __HAIKU__       // broken printf workaround
        char buffer2[25];
        char *ptr = (char *)&value;
        for(int i = 0; i < sizeof(value); i++)
            snprintf(buffer2, sizeof(char), "%x", ptr[i]);

        buf->writestring(buffer2);
#else
        buf->printf("%La", value);     // ensure exact duplication
#endif
    }
#else // if IN_DMD
    if (r != value)                     // if exact duplication
        ld_sprint(buffer, 'a', value);
    buf->writestring(buffer);
#endif



    if (type)
    {
        Type *t = type->toBasetype();
        switch (t->ty)
        {
            case Tfloat32:
            case Timaginary32:
            case Tcomplex32:
                buf->writeByte('F');
                break;

            case Tfloat80:
            case Timaginary80:
            case Tcomplex80:
                buf->writeByte('L');
                break;

            default:
                break;
        }
        if (t->isimaginary())
            buf->writeByte('i');
    }
}

void RealExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    floatToBuffer(buf, type, value);
}

void realToMangleBuffer(OutBuffer *buf, real_t value)
{
    /* Rely on %A to get portable mangling.
     * Must munge result to get only identifier characters.
     *
     * Possible values from %A  => mangled result
     * NAN                      => NAN
     * -INF                     => NINF
     * INF                      => INF
     * -0X1.1BC18BA997B95P+79   => N11BC18BA997B95P79
     * 0X1.9P+2                 => 19P2
     */

    if (Port::isNan(value))
        buf->writestring("NAN");        // no -NAN bugs
    else
    {
#if IN_LLVM
        // The buffer is too small for a longdouble on PPC.
        char buffer[36];
#else
        char buffer[32];
#endif
        int n = ld_sprint(buffer, 'A', value);
        assert(n > 0 && n < sizeof(buffer));
        for (int i = 0; i < n; i++)
        {   char c = buffer[i];

            switch (c)
            {
                case '-':
                    buf->writeByte('N');
                    break;

                case '+':
                case 'X':
                case '.':
                    break;

                case '0':
                    if (i < 2)
                        break;          // skip leading 0X
                default:
                    buf->writeByte(c);
                    break;
            }
        }
    }
}

void RealExp::toMangleBuffer(OutBuffer *buf)
{
    buf->writeByte('e');
    realToMangleBuffer(buf, value);
}


/******************************** ComplexExp **************************/

ComplexExp::ComplexExp(Loc loc, complex_t value, Type *type)
        : Expression(loc, TOKcomplex80, sizeof(ComplexExp))
{
    this->value = value;
    this->type = type;
    //printf("ComplexExp::ComplexExp(%s)\n", toChars());
}

char *ComplexExp::toChars()
{
    char buffer[sizeof(value) * 3 + 8 + 1];

    char buf1[sizeof(value) * 3 + 8 + 1];
    char buf2[sizeof(value) * 3 + 8 + 1];
#ifdef IN_GCC
    creall(value).format(buf1, sizeof(buf1));
    cimagl(value).format(buf2, sizeof(buf2));
#else
    ld_sprint(buf1, 'g', creall(value));
    ld_sprint(buf2, 'g', cimagl(value));
#endif
    sprintf(buffer, "(%s+%si)", buf1, buf2);
    assert(strlen(buffer) < sizeof(buffer));
    return mem.strdup(buffer);
}

dinteger_t ComplexExp::toInteger()
{
#ifdef IN_GCC
    return (sinteger_t) toReal().toInt();
#else
    return (sinteger_t) toReal();
#endif
}

uinteger_t ComplexExp::toUInteger()
{
#ifdef IN_GCC
    return (uinteger_t) toReal().toInt();
#else
    return (uinteger_t) toReal();
#endif
}

real_t ComplexExp::toReal()
{
    return creall(value);
}

real_t ComplexExp::toImaginary()
{
    return cimagl(value);
}

complex_t ComplexExp::toComplex()
{
    return value;
}

int ComplexExp::equals(Object *o)
{   ComplexExp *ne;

    if (this == o ||
        (((Expression *)o)->op == TOKcomplex80 &&
         ((ne = (ComplexExp *)o), type->toHeadMutable()->equals(ne->type->toHeadMutable())) &&
         RealEquals(creall(value), creall(ne->value)) &&
         RealEquals(cimagl(value), cimagl(ne->value))
        )
       )
        return 1;
    return 0;
}

Expression *ComplexExp::semantic(Scope *sc)
{
    if (!type)
        type = Type::tcomplex80;
    else
        type = type->semantic(loc, sc);
    return this;
}

int ComplexExp::isBool(int result)
{
    if (result)
        return (bool)(value);
    else
        return !value;
}

void ComplexExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    /* Print as:
     *  (re+imi)
     */
#ifdef IN_GCC
    char buf1[sizeof(value) * 3 + 8 + 1];
    char buf2[sizeof(value) * 3 + 8 + 1];
    creall(value).format(buf1, sizeof(buf1));
    cimagl(value).format(buf2, sizeof(buf2));
    buf->printf("(%s+%si)", buf1, buf2);
#else
    buf->writeByte('(');
    floatToBuffer(buf, type, creall(value));
    buf->writeByte('+');
    floatToBuffer(buf, type, cimagl(value));
    buf->writestring("i)");
#endif
}

void ComplexExp::toMangleBuffer(OutBuffer *buf)
{
    buf->writeByte('c');
    real_t r = toReal();
    realToMangleBuffer(buf, r);
    buf->writeByte('c');        // separate the two
    r = toImaginary();
    realToMangleBuffer(buf, r);
}

/******************************** IdentifierExp **************************/

IdentifierExp::IdentifierExp(Loc loc, Identifier *ident)
        : Expression(loc, TOKidentifier, sizeof(IdentifierExp))
{
    this->ident = ident;
}

Expression *IdentifierExp::semantic(Scope *sc)
{
    Dsymbol *s;
    Dsymbol *scopesym;

#if LOGSEMANTIC
    printf("IdentifierExp::semantic('%s')\n", ident->toChars());
#endif
    s = sc->search(loc, ident, &scopesym);
    if (s)
    {   Expression *e;
        WithScopeSymbol *withsym;

        /* See if the symbol was a member of an enclosing 'with'
         */
        withsym = scopesym->isWithScopeSymbol();
        if (withsym)
        {
#if DMDV2
            /* Disallow shadowing
             */
            // First find the scope of the with
            Scope *scwith = sc;
            while (scwith->scopesym != scopesym)
            {   scwith = scwith->enclosing;
                assert(scwith);
            }
            // Look at enclosing scopes for symbols with the same name,
            // in the same function
            for (Scope *scx = scwith; scx && scx->func == scwith->func; scx = scx->enclosing)
            {   Dsymbol *s2;

                if (scx->scopesym && scx->scopesym->symtab &&
                    (s2 = scx->scopesym->symtab->lookup(s->ident)) != NULL &&
                    s != s2)
                {
                    error("with symbol %s is shadowing local symbol %s", s->toPrettyChars(), s2->toPrettyChars());
                    return new ErrorExp();
                }
            }
#endif
            s = s->toAlias();

            // Same as wthis.ident
            if (s->needThis() || s->isTemplateDeclaration())
            {
                e = new VarExp(loc, withsym->withstate->wthis);
                e = new DotIdExp(loc, e, ident);
            }
            else
            {   Type *t = withsym->withstate->wthis->type;
                if (t->ty == Tpointer)
                    t = ((TypePointer *)t)->next;
                e = typeDotIdExp(loc, t, ident);
            }
        }
        else
        {
            /* If f is really a function template,
             * then replace f with the function template declaration.
             */
            FuncDeclaration *f = s->isFuncDeclaration();
            if (f)
            {   TemplateDeclaration *tempdecl = getFuncTemplateDecl(f);
                if (tempdecl)
                {
                    if (tempdecl->overroot)         // if not start of overloaded list of TemplateDeclaration's
                        tempdecl = tempdecl->overroot; // then get the start
                    e = new TemplateExp(loc, tempdecl);
                    e = e->semantic(sc);
                    return e;
                }
            }
            // Haven't done overload resolution yet, so pass 1
            e = new DsymbolExp(loc, s, 1);
        }
        return e->semantic(sc);
    }
#if DMDV2
    if (hasThis(sc))
    {
        AggregateDeclaration *ad = sc->getStructClassScope();
        if (ad && ad->aliasthis)
        {
            Expression *e;
            e = new IdentifierExp(loc, Id::This);
            e = new DotIdExp(loc, e, ad->aliasthis->ident);
            e = new DotIdExp(loc, e, ident);
            e = e->trySemantic(sc);
            if (e)
                return e;
        }
    }
    if (ident == Id::ctfe)
    {  // Create the magic __ctfe bool variable
       VarDeclaration *vd = new VarDeclaration(loc, Type::tbool, Id::ctfe, NULL);
       Expression *e = new VarExp(loc, vd);
       e = e->semantic(sc);
       return e;
    }
#endif
    const char *n = importHint(ident->toChars());
    if (n)
        error("'%s' is not defined, perhaps you need to import %s; ?", ident->toChars(), n);
    else
    {
        s = sc->search_correct(ident);
        if (s)
            error("undefined identifier %s, did you mean %s %s?", ident->toChars(), s->kind(), s->toChars());
        else
            error("undefined identifier %s", ident->toChars());
    }
    return new ErrorExp();
}

char *IdentifierExp::toChars()
{
    return ident->toChars();
}

void IdentifierExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    if (hgs->hdrgen)
        buf->writestring(ident->toHChars2());
    else
        buf->writestring(ident->toChars());
}


int IdentifierExp::isLvalue()
{
    return 1;
}

Expression *IdentifierExp::toLvalue(Scope *sc, Expression *e)
{
    return this;
}

/******************************** DollarExp **************************/

DollarExp::DollarExp(Loc loc)
        : IdentifierExp(loc, Id::dollar)
{
}

/******************************** DsymbolExp **************************/

DsymbolExp::DsymbolExp(Loc loc, Dsymbol *s, int hasOverloads)
        : Expression(loc, TOKdsymbol, sizeof(DsymbolExp))
{
    this->s = s;
    this->hasOverloads = hasOverloads;
}

AggregateDeclaration *isAggregate(Type *t);

Expression *DsymbolExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("DsymbolExp::semantic(%s %s)\n", s->kind(), s->toChars());
#endif

Lagain:
    EnumMember *em;
    Expression *e;
    VarDeclaration *v;
    FuncDeclaration *f;
    FuncLiteralDeclaration *fld;
    OverloadSet *o;
    Import *imp;
    Package *pkg;
    Type *t;

    //printf("DsymbolExp:: %p '%s' is a symbol\n", this, toChars());
    //printf("s = '%s', s->kind = '%s'\n", s->toChars(), s->kind());
    if (type && !s->needThis())
        return this;
    if (!s->isFuncDeclaration())        // functions are checked after overloading
        checkDeprecated(sc, s);
    Dsymbol *olds = s;
    s = s->toAlias();
    //printf("s = '%s', s->kind = '%s', s->needThis() = %p\n", s->toChars(), s->kind(), s->needThis());
    if (s != olds && !s->isFuncDeclaration())
        checkDeprecated(sc, s);

    // BUG: This should happen after overload resolution for functions, not before
    if (s->needThis())
    {
        if (hasThis(sc)
#if DMDV2
                && !s->isFuncDeclaration()
#endif
            )
        {
            // Supply an implicit 'this', as in
            //    this.ident

            DotVarExp *de;

            de = new DotVarExp(loc, new ThisExp(loc), s->isDeclaration());
            return de->semantic(sc);
        }
    }

    em = s->isEnumMember();
    if (em)
    {
        e = em->value;
        e->loc = loc;
        e = e->semantic(sc);
        return e;
    }
    v = s->isVarDeclaration();
    if (v)
    {
        //printf("Identifier '%s' is a variable, type '%s'\n", toChars(), v->type->toChars());
        if (!type)
        {   if ((!v->type || !v->type->deco) && v->scope)
                v->semantic(v->scope);
            type = v->type;
            if (!v->type)
            {   error("forward reference of %s %s", v->kind(), v->toChars());
                return new ErrorExp();
            }
        }

        if ((v->storage_class & STCmanifest) && v->init)
        {
            e = v->init->toExpression();
            if (!e)
            {   error("cannot make expression out of initializer for %s", v->toChars());
                return new ErrorExp();
            }
            e = e->copy();
            e->loc = loc;   // for better error message
            e = e->semantic(sc);
            return e;
        }

        e = new VarExp(loc, v);
        e->type = type;
        e = e->semantic(sc);
        return e->deref();
    }
    fld = s->isFuncLiteralDeclaration();
    if (fld)
    {   //printf("'%s' is a function literal\n", fld->toChars());
        e = new FuncExp(loc, fld);
        return e->semantic(sc);
    }
    f = s->isFuncDeclaration();
    if (f)
    {
        f = f->toAliasFunc();
        if (!f->functionSemantic())
            return new ErrorExp();

        if (f->isUnitTestDeclaration())
        {
            error("cannot call unittest function %s", toChars());
            return new ErrorExp();
        }
        if (!f->type->deco)
        {
            error("forward reference to %s", toChars());
            return new ErrorExp();
        }
        FuncDeclaration *fd = s->isFuncDeclaration();
        fd->type = f->type;
        return new VarExp(loc, fd, hasOverloads);
    }
    o = s->isOverloadSet();
    if (o)
    {   //printf("'%s' is an overload set\n", o->toChars());
        return new OverExp(o);
    }
    imp = s->isImport();
    if (imp)
    {
        if (!imp->pkg)
        {   error("forward reference of import %s", imp->toChars());
            return new ErrorExp();
        }
        ScopeExp *ie = new ScopeExp(loc, imp->pkg);
        return ie->semantic(sc);
    }
    pkg = s->isPackage();
    if (pkg)
    {
        ScopeExp *ie;

        ie = new ScopeExp(loc, pkg);
        return ie->semantic(sc);
    }
    Module *mod = s->isModule();
    if (mod)
    {
        ScopeExp *ie;

        ie = new ScopeExp(loc, mod);
        return ie->semantic(sc);
    }

    t = s->getType();
    if (t)
    {
        TypeExp *te = new TypeExp(loc, t);
        return te->semantic(sc);
    }

    TupleDeclaration *tup = s->isTupleDeclaration();
    if (tup)
    {
        for (size_t i = 0; i < tup->objects->dim; i++)
        {
            Dsymbol *sa = getDsymbol((*tup->objects)[i]);
            if (sa && sa->needThis())
            {
                if (hasThis(sc)
#if DMDV2
                        && !sa->isFuncDeclaration()
#endif
                    )
                {
                    // Supply an implicit 'this', as in
                    //    this.ident
                    (*tup->objects)[i] = new DotVarExp(loc, new ThisExp(loc), sa->isDeclaration());
                }
            }
        }

        e = new TupleExp(loc, tup);
        e = e->semantic(sc);
        return e;
    }

    TemplateInstance *ti = s->isTemplateInstance();
    if (ti)
    {   if (!ti->semanticRun)
            ti->semantic(sc);
        s = ti->toAlias();
        if (!s->isTemplateInstance())
            goto Lagain;
        if (ti->errors)
            return new ErrorExp();
        e = new ScopeExp(loc, ti);
        e = e->semantic(sc);
        return e;
    }

    TemplateDeclaration *td = s->isTemplateDeclaration();
    if (td)
    {
        Dsymbol *p = td->toParent2();
        FuncDeclaration *fdthis = hasThis(sc);
        AggregateDeclaration *ad = p ? p->isAggregateDeclaration() : NULL;
        if (fdthis && ad && isAggregate(fdthis->vthis->type) == ad)
        {
            e = new DotTemplateExp(loc, new ThisExp(loc), td);
        }
        else
            e = new TemplateExp(loc, td);
        e = e->semantic(sc);
        return e;
    }

    error("%s '%s' is not a variable", s->kind(), s->toChars());
    return new ErrorExp();
}

char *DsymbolExp::toChars()
{
    return s->toChars();
}

void DsymbolExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring(s->toChars());
}


int DsymbolExp::isLvalue()
{
    return 1;
}

Expression *DsymbolExp::toLvalue(Scope *sc, Expression *e)
{
    return this;
}

/******************************** ThisExp **************************/

ThisExp::ThisExp(Loc loc)
        : Expression(loc, TOKthis, sizeof(ThisExp))
{
    //printf("ThisExp::ThisExp() loc = %d\n", loc.linnum);
    var = NULL;
}

Expression *ThisExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("ThisExp::semantic()\n");
#endif
    if (type && var)
    {   //assert(global.errors || var);
#if IN_LLVM
        var->isVarDeclaration()->checkNestedReference(sc, loc);
#endif
        return this;
    }

    FuncDeclaration *fd = hasThis(sc);  // fd is the uplevel function with the 'this' variable

    /* Special case for typeof(this) and typeof(super) since both
     * should work even if they are not inside a non-static member function
     */
    if (!fd && sc->intypeof)
    {
        // Find enclosing struct or class
        for (Dsymbol *s = sc->getStructClassScope(); 1; s = s->parent)
        {
            if (!s)
            {
                error("%s is not in a class or struct scope", toChars());
                goto Lerr;
            }
            ClassDeclaration *cd = s->isClassDeclaration();
            if (cd)
            {
                type = cd->type;
                return this;
            }
            StructDeclaration *sd = s->isStructDeclaration();
            if (sd)
            {
                type = sd->type;
                return this;
            }
        }
    }
    if (!fd)
        goto Lerr;

    assert(fd->vthis);
    var = fd->vthis;
    assert(var->parent);
    if (!type)
        type = var->type;
    var->isVarDeclaration()->checkNestedReference(sc, loc);
    if (!sc->intypeof)
        sc->callSuper |= CSXthis;
    return this;

Lerr:
    error("'this' is only defined in non-static member functions, not %s", sc->parent->toChars());
    return new ErrorExp();
}

int ThisExp::isBool(int result)
{
    return result ? TRUE : FALSE;
}

void ThisExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("this");
}


int ThisExp::isLvalue()
{
    return 1;
}

Expression *ThisExp::toLvalue(Scope *sc, Expression *e)
{
    return this;
}

Expression *ThisExp::modifiableLvalue(Scope *sc, Expression *e)
{
    if (type->toBasetype()->ty == Tclass)
    {
        error("Cannot modify '%s'", toChars());
        return toLvalue(sc, e);
    }
    return Expression::modifiableLvalue(sc, e);
}

/******************************** SuperExp **************************/

SuperExp::SuperExp(Loc loc)
        : ThisExp(loc)
{
    op = TOKsuper;
}

Expression *SuperExp::semantic(Scope *sc)
{
    ClassDeclaration *cd;
    Dsymbol *s;

#if LOGSEMANTIC
    printf("SuperExp::semantic('%s')\n", toChars());
#endif
    if (type)
        return this;

    FuncDeclaration *fd = hasThis(sc);

    /* Special case for typeof(this) and typeof(super) since both
     * should work even if they are not inside a non-static member function
     */
    if (!fd && sc->intypeof)
    {
        // Find enclosing class
        for (Dsymbol *s = sc->getStructClassScope(); 1; s = s->parent)
        {
            if (!s)
            {
                error("%s is not in a class scope", toChars());
                goto Lerr;
            }
            ClassDeclaration *cd = s->isClassDeclaration();
            if (cd)
            {
                cd = cd->baseClass;
                if (!cd)
                {   error("class %s has no 'super'", s->toChars());
                    goto Lerr;
                }
                type = cd->type;
                return this;
            }
        }
    }
    if (!fd)
        goto Lerr;

    assert(fd->vthis);
    var = fd->vthis;
    assert(var->parent);

    s = fd->toParent();
    while (s && s->isTemplateInstance())
        s = s->toParent();
    if (s->isTemplateDeclaration()) // allow inside template constraint
        s = s->toParent();
    assert(s);
    cd = s->isClassDeclaration();
//printf("parent is %s %s\n", fd->toParent()->kind(), fd->toParent()->toChars());
    if (!cd)
        goto Lerr;
    if (!cd->baseClass)
    {
        error("no base class for %s", cd->toChars());
        type = fd->vthis->type;
    }
    else
    {
        type = cd->baseClass->type;
        type = type->castMod(var->type->mod);
    }

    var->isVarDeclaration()->checkNestedReference(sc, loc);

    if (!sc->intypeof)
        sc->callSuper |= CSXsuper;
    return this;


Lerr:
    error("'super' is only allowed in non-static class member functions");
    return new ErrorExp();
}

void SuperExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("super");
}


/******************************** NullExp **************************/

NullExp::NullExp(Loc loc, Type *type)
        : Expression(loc, TOKnull, sizeof(NullExp))
{
    committed = 0;
    this->type = type;
}

int NullExp::equals(Object *o)
{
    if (o && o->dyncast() == DYNCAST_EXPRESSION)
    {   Expression *e = (Expression *)o;

        if (e->op == TOKnull)
            return TRUE;
    }
    return FALSE;
}

Expression *NullExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("NullExp::semantic('%s')\n", toChars());
#endif
    // NULL is the same as (void *)0
    if (!type)
        type = Type::tnull;
    return this;
}

int NullExp::isBool(int result)
{
    return result ? FALSE : TRUE;
}

StringExp *NullExp::toString()
{
    if (implicitConvTo(Type::tstring))
    {
        StringExp *se = new StringExp(loc, (char*)mem.calloc(1, 1), 0);
        se->type = Type::tstring;
        return se;
    }
    return NULL;
}

void NullExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("null");
}

void NullExp::toMangleBuffer(OutBuffer *buf)
{
    buf->writeByte('n');
}

/******************************** StringExp **************************/

StringExp::StringExp(Loc loc, char *string)
        : Expression(loc, TOKstring, sizeof(StringExp))
{
    this->string = string;
    this->len = strlen(string);
    this->sz = 1;
    this->committed = 0;
    this->postfix = 0;
    this->ownedByCtfe = false;
}

StringExp::StringExp(Loc loc, void *string, size_t len)
        : Expression(loc, TOKstring, sizeof(StringExp))
{
    this->string = string;
    this->len = len;
    this->sz = 1;
    this->committed = 0;
    this->postfix = 0;
    this->ownedByCtfe = false;
}

StringExp::StringExp(Loc loc, void *string, size_t len, unsigned char postfix)
        : Expression(loc, TOKstring, sizeof(StringExp))
{
    this->string = string;
    this->len = len;
    this->sz = 1;
    this->committed = 0;
    this->postfix = postfix;
    this->ownedByCtfe = false;
}

#if 0
Expression *StringExp::syntaxCopy()
{
    printf("StringExp::syntaxCopy() %s\n", toChars());
    return copy();
}
#endif

int StringExp::equals(Object *o)
{
    //printf("StringExp::equals('%s') %s\n", o->toChars(), toChars());
    if (o && o->dyncast() == DYNCAST_EXPRESSION)
    {   Expression *e = (Expression *)o;

        if (e->op == TOKstring)
        {
            return compare(o) == 0;
        }
    }
    return FALSE;
}

Expression *StringExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("StringExp::semantic() %s\n", toChars());
#endif
    if (!type)
    {   OutBuffer buffer;
        size_t newlen = 0;
        const char *p;
        size_t u;
        unsigned c;

        switch (postfix)
        {
            case 'd':
                for (u = 0; u < len;)
                {
                    p = utf_decodeChar((unsigned char *)string, len, &u, &c);
                    if (p)
                    {   error("%s", p);
                        return new ErrorExp();
                    }
                    else
                    {   buffer.write4(c);
                        newlen++;
                    }
                }
                buffer.write4(0);
                string = buffer.extractData();
                len = newlen;
                sz = 4;
                //type = new TypeSArray(Type::tdchar, new IntegerExp(loc, len, Type::tindex));
                type = new TypeDArray(Type::tdchar->invariantOf());
                committed = 1;
                break;

            case 'w':
                for (u = 0; u < len;)
                {
                    p = utf_decodeChar((unsigned char *)string, len, &u, &c);
                    if (p)
                    {   error("%s", p);
                        return new ErrorExp();
                    }
                    else
                    {   buffer.writeUTF16(c);
                        newlen++;
                        if (c >= 0x10000)
                            newlen++;
                    }
                }
                buffer.writeUTF16(0);
                string = buffer.extractData();
                len = newlen;
                sz = 2;
                //type = new TypeSArray(Type::twchar, new IntegerExp(loc, len, Type::tindex));
                type = new TypeDArray(Type::twchar->invariantOf());
                committed = 1;
                break;

            case 'c':
                committed = 1;
            default:
                //type = new TypeSArray(Type::tchar, new IntegerExp(loc, len, Type::tindex));
                type = new TypeDArray(Type::tchar->invariantOf());
                break;
        }
        type = type->semantic(loc, sc);
        //type = type->invariantOf();
        //printf("type = %s\n", type->toChars());
    }
    return this;
}

/**********************************
 * Return length of string.
 */

size_t StringExp::length()
{
    size_t result = 0;
    dchar_t c;
    const char *p;

    switch (sz)
    {
        case 1:
            for (size_t u = 0; u < len;)
            {
                p = utf_decodeChar((unsigned char *)string, len, &u, &c);
                if (p)
                {   error("%s", p);
                    return 0;
                }
                else
                    result++;
            }
            break;

        case 2:
            for (size_t u = 0; u < len;)
            {
                p = utf_decodeWchar((unsigned short *)string, len, &u, &c);
                if (p)
                {   error("%s", p);
                    return 0;
                }
                else
                    result++;
            }
            break;

        case 4:
            result = len;
            break;

        default:
            assert(0);
    }
    return result;
}

StringExp *StringExp::toString()
{
    return this;
}

/****************************************
 * Convert string to char[].
 */

StringExp *StringExp::toUTF8(Scope *sc)
{
    if (sz != 1)
    {   // Convert to UTF-8 string
        committed = 0;
        Expression *e = castTo(sc, Type::tchar->arrayOf());
        e = e->optimize(WANTvalue);
        assert(e->op == TOKstring);
        StringExp *se = (StringExp *)e;
        assert(se->sz == 1);
        return se;
    }
    return this;
}

int StringExp::compare(Object *obj)
{
    //printf("StringExp::compare()\n");
    // Used to sort case statement expressions so we can do an efficient lookup
    StringExp *se2 = (StringExp *)(obj);

    // This is a kludge so isExpression() in template.c will return 5
    // for StringExp's.
    if (!se2)
        return 5;

    assert(se2->op == TOKstring);

    size_t len1 = len;
    size_t len2 = se2->len;

    //printf("sz = %d, len1 = %d, len2 = %d\n", sz, (int)len1, (int)len2);
    if (len1 == len2)
    {
        switch (sz)
        {
            case 1:
                return memcmp((char *)string, (char *)se2->string, len1);

            case 2:
            {
                d_wchar *s1 = (d_wchar *)string;
                d_wchar *s2 = (d_wchar *)se2->string;

                for (size_t u = 0; u < len; u++)
                {
                    if (s1[u] != s2[u])
                        return s1[u] - s2[u];
                }
            }

            case 4:
            {
                d_dchar *s1 = (d_dchar *)string;
                d_dchar *s2 = (d_dchar *)se2->string;

                for (size_t u = 0; u < len; u++)
                {
                    if (s1[u] != s2[u])
                        return s1[u] - s2[u];
                }
            }
            break;

            default:
                assert(0);
        }
    }
    return (int)(len1 - len2);
}

int StringExp::isBool(int result)
{
    return result ? TRUE : FALSE;
}


int StringExp::isLvalue()
{
    /* string literal is rvalue in default, but
     * conversion to reference of static array is only allowed.
     */
    return 0;
}

Expression *StringExp::toLvalue(Scope *sc, Expression *e)
{
    //printf("StringExp::toLvalue(%s)\n", toChars());
    return this;
}

Expression *StringExp::modifiableLvalue(Scope *sc, Expression *e)
{
    e->error("Cannot modify '%s'", toChars());
    return new ErrorExp();
}

unsigned StringExp::charAt(size_t i)
{   unsigned value;

    switch (sz)
    {
        case 1:
            value = ((unsigned char *)string)[i];
            break;

        case 2:
            value = ((unsigned short *)string)[i];
            break;

        case 4:
            value = ((unsigned int *)string)[i];
            break;

        default:
            assert(0);
            break;
    }
    return value;
}

void StringExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writeByte('"');
    unsigned o = buf->offset;
    for (size_t i = 0; i < len; i++)
    {   unsigned c = charAt(i);

        switch (c)
        {
            case '"':
            case '\\':
                if (!hgs->console)
                    buf->writeByte('\\');
            default:
                if (c <= 0xFF)
                {   if (c <= 0x7F && (isprint(c) || hgs->console))
                        buf->writeByte(c);
                    else
                        buf->printf("\\x%02x", c);
                }
                else if (c <= 0xFFFF)
                    buf->printf("\\x%02x\\x%02x", c & 0xFF, c >> 8);
                else
                    buf->printf("\\x%02x\\x%02x\\x%02x\\x%02x",
                        c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF, c >> 24);
                break;
        }
    }
    if (hgs->ddoc)
        escapeDdocString(buf, o);
    buf->writeByte('"');
    if (postfix)
        buf->writeByte(postfix);
}

void StringExp::toMangleBuffer(OutBuffer *buf)
{   char m;
    OutBuffer tmp;
    const char *p;
    unsigned c;
    size_t u;
    unsigned char *q;
    size_t qlen;

    /* Write string in UTF-8 format
     */
    switch (sz)
    {   case 1:
            m = 'a';
            q = (unsigned char *)string;
            qlen = len;
            break;
        case 2:
            m = 'w';
            for (u = 0; u < len; )
            {
                p = utf_decodeWchar((unsigned short *)string, len, &u, &c);
                if (p)
                    error("%s", p);
                else
                    tmp.writeUTF8(c);
            }
            q = tmp.data;
            qlen = tmp.offset;
            break;
        case 4:
            m = 'd';
            for (u = 0; u < len; u++)
            {
                c = ((unsigned *)string)[u];
                if (!utf_isValidDchar(c))
                    error("invalid UCS-32 char \\U%08x", c);
                else
                    tmp.writeUTF8(c);
            }
            q = tmp.data;
            qlen = tmp.offset;
            break;
        default:
            assert(0);
    }
    buf->reserve(1 + 11 + 2 * qlen);
    buf->writeByte(m);
    buf->printf("%d_", (int)qlen); // nbytes <= 11

    for (unsigned char *p = buf->data + buf->offset, *pend = p + 2 * qlen;
         p < pend; p += 2, ++q)
    {
        unsigned char hi = *q >> 4 & 0xF;
        p[0] = (hi < 10 ? hi + '0' : hi - 10 + 'a');
        unsigned char lo = *q & 0xF;
        p[1] = (lo < 10 ? lo + '0' : lo - 10 + 'a');
    }
    buf->offset += 2 * qlen;
}

/************************ ArrayLiteralExp ************************************/

// [ e1, e2, e3, ... ]

ArrayLiteralExp::ArrayLiteralExp(Loc loc, Expressions *elements)
    : Expression(loc, TOKarrayliteral, sizeof(ArrayLiteralExp))
{
    this->elements = elements;
    this->ownedByCtfe = false;
}

ArrayLiteralExp::ArrayLiteralExp(Loc loc, Expression *e)
    : Expression(loc, TOKarrayliteral, sizeof(ArrayLiteralExp))
{
    elements = new Expressions;
    elements->push(e);
    this->ownedByCtfe = false;
}

Expression *ArrayLiteralExp::syntaxCopy()
{
    return new ArrayLiteralExp(loc, arraySyntaxCopy(elements));
}

Expression *ArrayLiteralExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("ArrayLiteralExp::semantic('%s')\n", toChars());
#endif
    if (type)
        return this;

    /* Perhaps an empty array literal [ ] should be rewritten as null?
     */

    arrayExpressionSemantic(elements, sc);    // run semantic() on each element
    expandTuples(elements);

    Type *t0;
    elements = arrayExpressionToCommonType(sc, elements, &t0);

    type = t0->arrayOf();
    //type = new TypeSArray(t0, new IntegerExp(elements->dim));
    type = type->semantic(loc, sc);

    /* Disallow array literals of type void being used.
     */
    if (elements->dim > 0 && t0->ty == Tvoid)
    {   error("%s of type %s has no value", toChars(), type->toChars());
        return new ErrorExp();
    }

    return this;
}

int ArrayLiteralExp::isBool(int result)
{
    size_t dim = elements ? elements->dim : 0;
    return result ? (dim != 0) : (dim == 0);
}

StringExp *ArrayLiteralExp::toString()
{
    TY telem = type->nextOf()->toBasetype()->ty;

    if (telem == Tchar || telem == Twchar || telem == Tdchar ||
        (telem == Tvoid && (!elements || elements->dim == 0)))
    {
        OutBuffer buf;
        if (elements)
            for (int i = 0; i < elements->dim; ++i)
            {
                Expression *ch = (*elements)[i];
                if (ch->op != TOKint64)
                    return NULL;
                buf.writeUTF8(ch->toInteger());
            }
        buf.writebyte(0);

        char prefix = 'c';
        if (telem == Twchar) prefix = 'w';
        else if (telem == Tdchar) prefix = 'd';

        const size_t len = buf.offset - 1;
        StringExp *se = new StringExp(loc, buf.extractData(), len, prefix);
        se->type = type;
        return se;
    }
    return NULL;
}

void ArrayLiteralExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writeByte('[');
    argsToCBuffer(buf, elements, hgs);
    buf->writeByte(']');
}

void ArrayLiteralExp::toMangleBuffer(OutBuffer *buf)
{
    size_t dim = elements ? elements->dim : 0;
    buf->printf("A%zu", dim);
    for (size_t i = 0; i < dim; i++)
    {   Expression *e = (*elements)[i];
        e->toMangleBuffer(buf);
    }
}

/************************ AssocArrayLiteralExp ************************************/

// [ key0 : value0, key1 : value1, ... ]

AssocArrayLiteralExp::AssocArrayLiteralExp(Loc loc,
                Expressions *keys, Expressions *values)
    : Expression(loc, TOKassocarrayliteral, sizeof(AssocArrayLiteralExp))
{
    assert(keys->dim == values->dim);
    this->keys = keys;
    this->values = values;
    this->ownedByCtfe = false;
}

Expression *AssocArrayLiteralExp::syntaxCopy()
{
    return new AssocArrayLiteralExp(loc,
        arraySyntaxCopy(keys), arraySyntaxCopy(values));
}

Expression *AssocArrayLiteralExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("AssocArrayLiteralExp::semantic('%s')\n", toChars());
#endif

    if (type)
        return this;

    // Run semantic() on each element
    arrayExpressionSemantic(keys, sc);
    arrayExpressionSemantic(values, sc);
    expandTuples(keys);
    expandTuples(values);
    if (keys->dim != values->dim)
    {
        error("number of keys is %u, must match number of values %u", keys->dim, values->dim);
        return new ErrorExp();
    }

    Type *tkey = NULL;
    Type *tvalue = NULL;
    keys = arrayExpressionToCommonType(sc, keys, &tkey);
    values = arrayExpressionToCommonType(sc, values, &tvalue);

    if (tkey == Type::terror || tvalue == Type::terror)
        return new ErrorExp;

    type = new TypeAArray(tvalue, tkey);
    type = type->semantic(loc, sc);
    return this;
}


int AssocArrayLiteralExp::isBool(int result)
{
    size_t dim = keys->dim;
    return result ? (dim != 0) : (dim == 0);
}

void AssocArrayLiteralExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writeByte('[');
    for (size_t i = 0; i < keys->dim; i++)
    {   Expression *key = (*keys)[i];
        Expression *value = (*values)[i];

        if (i)
            buf->writeByte(',');
        expToCBuffer(buf, hgs, key, PREC_assign);
        buf->writeByte(':');
        expToCBuffer(buf, hgs, value, PREC_assign);
    }
    buf->writeByte(']');
}

void AssocArrayLiteralExp::toMangleBuffer(OutBuffer *buf)
{
    size_t dim = keys->dim;
    buf->printf("A%zu", dim);
    for (size_t i = 0; i < dim; i++)
    {   Expression *key = (*keys)[i];
        Expression *value = (*values)[i];

        key->toMangleBuffer(buf);
        value->toMangleBuffer(buf);
    }
}

/************************ StructLiteralExp ************************************/

// sd( e1, e2, e3, ... )

StructLiteralExp::StructLiteralExp(Loc loc, StructDeclaration *sd, Expressions *elements, Type *stype)
    : Expression(loc, TOKstructliteral, sizeof(StructLiteralExp))
{
    this->sd = sd;
    if (!elements)
        elements = new Expressions();
    this->elements = elements;
    this->stype = stype;
    this->sinit = NULL;
#if IN_DMD
    this->sym = NULL;
#endif
    this->soffset = 0;
    this->fillHoles = 1;
    this->ownedByCtfe = false;
    this->ctorinit = 0;
#if IN_LLVM
    constType = NULL;
#endif
    //printf("StructLiteralExp::StructLiteralExp(%s)\n", toChars());
}

Expression *StructLiteralExp::syntaxCopy()
{
    return new StructLiteralExp(loc, sd, arraySyntaxCopy(elements), stype);
}

Expression *StructLiteralExp::semantic(Scope *sc)
{   Expression *e;

#if LOGSEMANTIC
    printf("StructLiteralExp::semantic('%s')\n", toChars());
#endif
    if (type)
        return this;

    sd->size(loc);
    if (sd->sizeok != SIZEOKdone)
        return new ErrorExp();
    size_t nfields = sd->fields.dim - sd->isnested;

    elements = arrayExpressionSemantic(elements, sc);   // run semantic() on each element
    expandTuples(elements);
    size_t offset = 0;
    for (size_t i = 0; i < elements->dim; i++)
    {   e = (*elements)[i];
        if (!e)
            continue;

        e = resolveProperties(sc, e);
        if (i >= nfields)
        {
#if 0
            for (size_t i = 0; i < sd->fields.dim; i++)
                printf("[%d] = %s\n", i, sd->fields[i]->toChars());
#endif
            error("more initializers than fields (%d) of %s", nfields, sd->toChars());
            return new ErrorExp();
        }
        Dsymbol *s = sd->fields[i];
        VarDeclaration *v = s->isVarDeclaration();
        assert(v);
        if (v->offset < offset)
        {   error("overlapping initialization for %s", v->toChars());
            return new ErrorExp();
        }
        offset = v->offset + v->type->size();

        Type *telem = v->type;
        if (stype)
            telem = telem->addMod(stype->mod);
        Type *origType = telem;
        while (!e->implicitConvTo(telem) && telem->toBasetype()->ty == Tsarray)
        {   /* Static array initialization, as in:
             *  T[3][5] = e;
             */
            telem = telem->toBasetype()->nextOf();
        }

        if (!e->implicitConvTo(telem))
            telem = origType;  // restore type for better diagnostic

        e = e->implicitCastTo(sc, telem);
        if (e->op == TOKerror)
            return e;

        (*elements)[i] = e;
    }

    /* Fill out remainder of elements[] with default initializers for fields[]
     */
    for (size_t i = elements->dim; i < nfields; i++)
    {   Dsymbol *s = sd->fields[i];
        VarDeclaration *v = s->isVarDeclaration();
        assert(v);
        assert(!v->isThisDeclaration());

        if (v->offset < offset)
        {   e = NULL;
            sd->hasUnions = 1;
        }
        else
        {
            if (v->init)
            {   if (v->init->isVoidInitializer())
                    e = NULL;
                else
                {   e = v->init->toExpression();
                    if (!e)
                    {   error("cannot make expression out of initializer for %s", v->toChars());
                        return new ErrorExp();
                    }
                    else if (v->scope)
                    {   // Do deferred semantic analysis
                        Initializer *i2 = v->init->syntaxCopy();
                        i2 = i2->semantic(v->scope, v->type, INITinterpret);
                        e = i2->toExpression();
                        // remove v->scope (see bug 3426)
                        // but not if gagged, for we might be called again.
                        if (!global.gag)
                        {   v->scope = NULL;
                            v->init = i2;   // save result
                        }
                    }
                }
            }
            else if (v->type->needsNested() && ctorinit)
                e = v->type->defaultInit(loc);
            else
                e = v->type->defaultInitLiteral(loc);
            offset = v->offset + v->type->size();
        }
        elements->push(e);
    }

    type = stype ? stype : sd->type;

    /* If struct requires a destructor, rewrite as:
     *    (S tmp = S()),tmp
     * so that the destructor can be hung on tmp.
     */
    if (sd->dtor && sc->func)
    {
        Identifier *idtmp = Lexer::uniqueId("__sl");
        VarDeclaration *tmp = new VarDeclaration(loc, type, idtmp, new ExpInitializer(0, this));
        tmp->storage_class |= STCctfe;
        Expression *ae = new DeclarationExp(loc, tmp);
        Expression *e = new CommaExp(loc, ae, new VarExp(loc, tmp));
        e = e->semantic(sc);
        return e;
    }

    return this;
}

/**************************************
 * Gets expression at offset of type.
 * Returns NULL if not found.
 */

Expression *StructLiteralExp::getField(Type *type, unsigned offset)
{
    //printf("StructLiteralExp::getField(this = %s, type = %s, offset = %u)\n",
    //  /*toChars()*/"", type->toChars(), offset);
    Expression *e = NULL;
    int i = getFieldIndex(type, offset);

    if (i != -1)
    {
        //printf("\ti = %d\n", i);
        if (i == sd->fields.dim - 1 && sd->isNested())
            return NULL;

        assert(i < elements->dim);
        e = (*elements)[i];
        if (e)
        {
            //printf("e = %s, e->type = %s\n", e->toChars(), e->type->toChars());

            /* If type is a static array, and e is an initializer for that array,
             * then the field initializer should be an array literal of e.
             */
            if (e->type->castMod(0) != type->castMod(0) && type->ty == Tsarray)
            {   TypeSArray *tsa = (TypeSArray *)type;
                uinteger_t length = tsa->dim->toInteger();
                Expressions *z = new Expressions;
                z->setDim(length);
                for (size_t q = 0; q < length; ++q)
                    (*z)[q] = e->copy();
                e = new ArrayLiteralExp(loc, z);
                e->type = type;
            }
            else
            {
                e = e->copy();
                e->type = type;
            }
#if !IN_LLVM
            if (sinit && e->op == TOKstructliteral &&
                e->type->needsNested())
            {
                StructLiteralExp *se = (StructLiteralExp *)e;
                se->sinit = se->sd->toInitializer();
            }
#endif
        }
    }
    return e;
}

/************************************
 * Get index of field.
 * Returns -1 if not found.
 */

int StructLiteralExp::getFieldIndex(Type *type, unsigned offset)
{
    /* Find which field offset is by looking at the field offsets
     */
    if (elements->dim)
    {
        for (size_t i = 0; i < sd->fields.dim; i++)
        {
            Dsymbol *s = sd->fields[i];
            VarDeclaration *v = s->isVarDeclaration();
            assert(v);

            if (offset == v->offset &&
                type->size() == v->type->size())
            {
                /* context field might not be filled. */
                if (i == sd->fields.dim - 1 && sd->isNested())
                    return (int)i;
                Expression *e = (*elements)[i];
                if (e)
                {
                    return (int)i;
                }
                break;
            }
        }
    }
    return -1;
}

void StructLiteralExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring(sd->toChars());
    buf->writeByte('(');
    argsToCBuffer(buf, elements, hgs);
    buf->writeByte(')');
}

void StructLiteralExp::toMangleBuffer(OutBuffer *buf)
{
    size_t dim = elements ? elements->dim : 0;
    buf->printf("S%zu", dim);
    for (size_t i = 0; i < dim; i++)
    {   Expression *e = (*elements)[i];
        if (e)
            e->toMangleBuffer(buf);
        else
            buf->writeByte('v');        // 'v' for void
    }
}

/************************ TypeDotIdExp ************************************/

/* Things like:
 *      int.size
 *      foo.size
 *      (foo).size
 *      cast(foo).size
 */

Expression *typeDotIdExp(Loc loc, Type *type, Identifier *ident)
{
    return new DotIdExp(loc, new TypeExp(loc, type), ident);
}


/************************************************************/

// Mainly just a placeholder

TypeExp::TypeExp(Loc loc, Type *type)
    : Expression(loc, TOKtype, sizeof(TypeExp))
{
    //printf("TypeExp::TypeExp(%s)\n", type->toChars());
    this->type = type;
}

Expression *TypeExp::syntaxCopy()
{
    //printf("TypeExp::syntaxCopy()\n");
    return new TypeExp(loc, type->syntaxCopy());
}

Expression *TypeExp::semantic(Scope *sc)
{
    //printf("TypeExp::semantic(%s)\n", type->toChars());
    Expression *e;
    Type *t;
    Dsymbol *s;

    type->resolve(loc, sc, &e, &t, &s);
    if (e)
    {
        //printf("e = %s %s\n", Token::toChars(e->op), e->toChars());
        e = e->semantic(sc);
    }
    else if (t)
    {
        //printf("t = %d %s\n", t->ty, t->toChars());
        type = t->semantic(loc, sc);
        e = this;
    }
    else if (s)
    {
        //printf("s = %s %s\n", s->kind(), s->toChars());
        e = new DsymbolExp(loc, s, s->hasOverloads());
        e = e->semantic(sc);
    }
    else
        assert(0);

    return e;
}

int TypeExp::rvalue()
{
    error("type %s has no value", toChars());
    return 0;
}

void TypeExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    type->toCBuffer(buf, NULL, hgs);
}

/************************************************************/

// Mainly just a placeholder

ScopeExp::ScopeExp(Loc loc, ScopeDsymbol *pkg)
    : Expression(loc, TOKimport, sizeof(ScopeExp))
{
    //printf("ScopeExp::ScopeExp(pkg = '%s')\n", pkg->toChars());
    //static int count; if (++count == 38) *(char*)0=0;
    this->sds = pkg;
}

Expression *ScopeExp::syntaxCopy()
{
    ScopeExp *se = new ScopeExp(loc, (ScopeDsymbol *)sds->syntaxCopy(NULL));
    return se;
}

Expression *ScopeExp::semantic(Scope *sc)
{
    TemplateInstance *ti;
    ScopeDsymbol *sds2;

#if LOGSEMANTIC
    printf("+ScopeExp::semantic('%s')\n", toChars());
#endif
Lagain:
    ti = sds->isTemplateInstance();
    if (ti && !ti->errors)
    {
        unsigned olderrs = global.errors;
        if (!ti->semanticRun)
            ti->semantic(sc);
        if (ti->inst)
        {
            if (ti->inst->errors)
                return new ErrorExp();
            Dsymbol *s = ti->inst->toAlias();
            sds2 = s->isScopeDsymbol();
            if (!sds2)
            {   Expression *e;

                //printf("s = %s, '%s'\n", s->kind(), s->toChars());
                if (ti->withsym)
                {
                    // Same as wthis.s
                    e = new VarExp(loc, ti->withsym->withstate->wthis);
                    e = new DotVarExp(loc, e, s->isDeclaration());
                }
                else
                    e = new DsymbolExp(loc, s, s->hasOverloads());
                e = e->semantic(sc);
                //printf("-1ScopeExp::semantic()\n");
                return e;
            }
            if (sds2 != sds)
            {
                sds = sds2;
                goto Lagain;
            }
            //printf("sds = %s, '%s'\n", sds->kind(), sds->toChars());
        }
        if (olderrs != global.errors)
            return new ErrorExp();
    }
    else
    {
        //printf("sds = %s, '%s'\n", sds->kind(), sds->toChars());
        //printf("\tparent = '%s'\n", sds->parent->toChars());
        sds->semantic(sc);

        AggregateDeclaration *ad = sds->isAggregateDeclaration();
        if (ad)
            return (new TypeExp(loc, ad->type))->semantic(sc);
    }
    type = Type::tvoid;
    //printf("-2ScopeExp::semantic() %s\n", toChars());
    return this;
}

void ScopeExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    if (sds->isTemplateInstance())
    {
        sds->toCBuffer(buf, hgs);
    }
    else if (hgs != NULL && hgs->ddoc)
    {   // fixes bug 6491
        Module *module = sds->isModule();
        if (module)
            buf->writestring(module->md->toChars());
        else
            buf->writestring(sds->toChars());
    }
    else
    {
        buf->writestring(sds->kind());
        buf->writestring(" ");
        buf->writestring(sds->toChars());
    }
}

/********************** TemplateExp **************************************/

// Mainly just a placeholder

TemplateExp::TemplateExp(Loc loc, TemplateDeclaration *td)
    : Expression(loc, TOKtemplate, sizeof(TemplateExp))
{
    //printf("TemplateExp(): %s\n", td->toChars());
    this->td = td;
}

void TemplateExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring(td->toChars());
}

int TemplateExp::rvalue()
{
    error("template %s has no value", toChars());
    return 0;
}

/********************** NewExp **************************************/

/* thisexp.new(newargs) newtype(arguments) */

NewExp::NewExp(Loc loc, Expression *thisexp, Expressions *newargs,
        Type *newtype, Expressions *arguments)
    : Expression(loc, TOKnew, sizeof(NewExp))
{
    this->thisexp = thisexp;
    this->newargs = newargs;
    this->newtype = newtype;
    this->arguments = arguments;
    member = NULL;
    allocator = NULL;
    onstack = 0;
}

Expression *NewExp::syntaxCopy()
{
    return new NewExp(loc,
        thisexp ? thisexp->syntaxCopy() : NULL,
        arraySyntaxCopy(newargs),
        newtype->syntaxCopy(), arraySyntaxCopy(arguments));
}


Expression *NewExp::semantic(Scope *sc)
{
    Type *tb;
    ClassDeclaration *cdthis = NULL;

#if LOGSEMANTIC
    printf("NewExp::semantic() %s\n", toChars());
    if (thisexp)
        printf("\tthisexp = %s\n", thisexp->toChars());
    printf("\tnewtype: %s\n", newtype->toChars());
#endif
    if (type)                   // if semantic() already run
        return this;

Lagain:
    if (thisexp)
    {   thisexp = thisexp->semantic(sc);
        cdthis = thisexp->type->isClassHandle();
        if (cdthis)
        {
            sc = sc->push(cdthis);
            type = newtype->semantic(loc, sc);
            sc = sc->pop();

            if (!MODimplicitConv(thisexp->type->mod, newtype->mod))
            {
                error("nested type %s should have the same or weaker constancy as enclosing type %s",
                    newtype->toChars(), thisexp->type->toChars());
                goto Lerr;
            }
        }
        else
        {
            error("'this' for nested class must be a class type, not %s", thisexp->type->toChars());
            goto Lerr;
        }
    }
    else
        type = newtype->semantic(loc, sc);
    newtype = type;             // in case type gets cast to something else
    tb = type->toBasetype();
    //printf("tb: %s, deco = %s\n", tb->toChars(), tb->deco);

    arrayExpressionSemantic(newargs, sc);
    preFunctionParameters(loc, sc, newargs);
    arrayExpressionSemantic(arguments, sc);
    preFunctionParameters(loc, sc, arguments);

    if (thisexp && tb->ty != Tclass)
    {   error("e.new is only for allocating nested classes, not %s", tb->toChars());
        goto Lerr;
    }

    if (tb->ty == Tclass)
    {
        TypeClass *tc = (TypeClass *)(tb);
        ClassDeclaration *cd = tc->sym->isClassDeclaration();
        if (cd->scope)
            cd->semantic(NULL);
        if (cd->isInterfaceDeclaration())
        {   error("cannot create instance of interface %s", cd->toChars());
            goto Lerr;
        }
        else if (cd->isAbstract())
        {   error("cannot create instance of abstract class %s", cd->toChars());
            for (size_t i = 0; i < cd->vtbl.dim; i++)
            {   FuncDeclaration *fd = cd->vtbl[i]->isFuncDeclaration();
                if (fd && fd->isAbstract())
                    error("function %s is abstract", fd->toChars());
            }
            goto Lerr;
        }

        if (cd->noDefaultCtor && (!arguments || !arguments->dim))
        {   error("default construction is disabled for type %s", cd->toChars());
            goto Lerr;
        }
        checkDeprecated(sc, cd);
        if (cd->isNested())
        {   /* We need a 'this' pointer for the nested class.
             * Ensure we have the right one.
             */
            Dsymbol *s = cd->toParent2();
            ClassDeclaration *cdn = s->isClassDeclaration();
            FuncDeclaration *fdn = s->isFuncDeclaration();

            //printf("cd isNested, cdn = %s\n", cdn ? cdn->toChars() : "null");
            if (cdn)
            {
                if (!cdthis)
                {
                    // Supply an implicit 'this' and try again
                    thisexp = new ThisExp(loc);
                    for (Dsymbol *sp = sc->parent; 1; sp = sp->parent)
                    {   if (!sp)
                        {
                            error("outer class %s 'this' needed to 'new' nested class %s", cdn->toChars(), cd->toChars());
                            goto Lerr;
                        }
                        ClassDeclaration *cdp = sp->isClassDeclaration();
                        if (!cdp)
                            continue;
                        if (cdp == cdn || cdn->isBaseOf(cdp, NULL))
                            break;
                        // Add a '.outer' and try again
                        thisexp = new DotIdExp(loc, thisexp, Id::outer);
                    }
                    if (!global.errors)
                        goto Lagain;
                }
                if (cdthis)
                {
                    //printf("cdthis = %s\n", cdthis->toChars());
                    if (cdthis != cdn && !cdn->isBaseOf(cdthis, NULL))
                    {   error("'this' for nested class must be of type %s, not %s", cdn->toChars(), thisexp->type->toChars());
                        goto Lerr;
                    }
                }
                        }
            else if (thisexp)
            {   error("e.new is only for allocating nested classes");
                goto Lerr;
            }
            else if (fdn)
            {
                // make sure the parent context fdn of cd is reachable from sc
                for (Dsymbol *sp = sc->parent; 1; sp = sp->parent)
                {
                    if (fdn == sp)
                        break;
                    FuncDeclaration *fsp = sp ? sp->isFuncDeclaration() : NULL;
                    if (!sp || (fsp && fsp->isStatic()))
                    {
                        error("outer function context of %s is needed to 'new' nested class %s", fdn->toPrettyChars(), cd->toPrettyChars());
                        goto Lerr;
                    }
                }
            }
            else
                assert(0);
        }
        else if (thisexp)
        {   error("e.new is only for allocating nested classes");
            goto Lerr;
        }

        FuncDeclaration *f = NULL;
        if (cd->ctor)
            f = resolveFuncCall(sc, loc, cd->ctor, NULL, NULL, arguments, 0);
        if (f)
        {
            checkDeprecated(sc, f);
            member = f->isCtorDeclaration();
            assert(member);

            cd->accessCheck(loc, sc, member);

            TypeFunction *tf = (TypeFunction *)f->type;

            if (!arguments)
                arguments = new Expressions();
            unsigned olderrors = global.errors;
            functionParameters(loc, sc, tf, NULL, arguments, f);
            if (olderrors != global.errors)
                return new ErrorExp();
            type = type->addMod(tf->nextOf()->mod);
        }
        else
        {
            if (arguments && arguments->dim)
            {   error("no constructor for %s", cd->toChars());
                goto Lerr;
            }
        }

        if (cd->aggNew)
        {
            // Prepend the size argument to newargs[]
            Expression *e = new IntegerExp(loc, cd->size(loc), Type::tsize_t);
            if (!newargs)
                newargs = new Expressions();
            newargs->shift(e);

            f = cd->aggNew->overloadResolve(loc, NULL, newargs);
            allocator = f->isNewDeclaration();
            assert(allocator);

            TypeFunction *tf = (TypeFunction *)f->type;
            unsigned olderrors = global.errors;
            functionParameters(loc, sc, tf, NULL, newargs, f);
            if (olderrors != global.errors)
                return new ErrorExp();

        }
        else
        {
            if (newargs && newargs->dim)
            {   error("no allocator for %s", cd->toChars());
                goto Lerr;
            }
        }
    }
    else if (tb->ty == Tstruct)
    {
        TypeStruct *ts = (TypeStruct *)tb;
        StructDeclaration *sd = ts->sym;
        if (sd->scope)
            sd->semantic(NULL);
        if (sd->noDefaultCtor && (!arguments || !arguments->dim))
        {   error("default construction is disabled for type %s", sd->toChars());
            goto Lerr;
        }

        if (sd->aggNew)
        {
            // Prepend the uint size argument to newargs[]
            Expression *e = new IntegerExp(loc, sd->size(loc), Type::tuns32);
            if (!newargs)
                newargs = new Expressions();
            newargs->shift(e);

            FuncDeclaration *f = sd->aggNew->overloadResolve(loc, NULL, newargs);
            allocator = f->isNewDeclaration();
            assert(allocator);

            TypeFunction *tf = (TypeFunction *)f->type;
            unsigned olderrors = global.errors;
            functionParameters(loc, sc, tf, NULL, newargs, f);
            if (olderrors != global.errors)
                return new ErrorExp();
        }
        else
        {
            if (newargs && newargs->dim)
            {   error("no allocator for %s", sd->toChars());
                goto Lerr;
            }
        }

        FuncDeclaration *f = NULL;
        if (sd->ctor)
            f = resolveFuncCall(sc, loc, sd->ctor, NULL, NULL, arguments, 0);
        if (f)
        {
            checkDeprecated(sc, f);
            member = f->isCtorDeclaration();
            assert(member);

            sd->accessCheck(loc, sc, member);

            TypeFunction *tf = (TypeFunction *)f->type;
            type = tf->next;

            if (!arguments)
                arguments = new Expressions();
            unsigned olderrors = global.errors;
            functionParameters(loc, sc, tf, NULL, arguments, f);
            if (olderrors != global.errors)
                return new ErrorExp();
        }
        else if (arguments && arguments->dim)
        {
            Type *tptr = type->pointerTo();

            /* Rewrite:
            *   new S(arguments)
             * as:
            *   (((S* __newsl = new S()), (*__newsl = S(arguments))), __newsl)
             */
            Identifier *id = Lexer::uniqueId("__newsl");
            ExpInitializer *ei = new ExpInitializer(loc, this);
            VarDeclaration *v = new VarDeclaration(loc, tptr, id, ei);
            v->storage_class |= STCctfe;
            Expression *e = new DeclarationExp(loc, v);
            Expression *ve = new VarExp(loc, v);
            Expression *se = new StructLiteralExp(loc, sd, arguments, type);
            Expression *ae = new ConstructExp(loc, new PtrExp(loc, ve), se);
            e = new CommaExp(loc, e, ae);
            e = new CommaExp(loc, e, ve);

            // rewrite this
            this->arguments = NULL;
            this->type = tptr;

            return e->semantic(sc);
        }

        type = type->pointerTo();
    }
    else if (tb->ty == Tarray && (arguments && arguments->dim))
    {
        for (size_t i = 0; i < arguments->dim; i++)
        {
            if (tb->ty != Tarray)
            {   error("too many arguments for array");
                goto Lerr;
            }

            Expression *arg = (*arguments)[i];
            arg = resolveProperties(sc, arg);
            arg = arg->implicitCastTo(sc, Type::tsize_t);
            arg = arg->optimize(WANTvalue);
            if (arg->op == TOKint64 && (sinteger_t)arg->toInteger() < 0)
            {   error("negative array index %s", arg->toChars());
                goto Lerr;
            }
            (*arguments)[i] =  arg;
            tb = ((TypeDArray *)tb)->next->toBasetype();
        }
    }
    else if (tb->isscalar())
    {
        if (arguments && arguments->dim)
        {   error("no constructor for %s", type->toChars());
            goto Lerr;
        }

        type = type->pointerTo();
    }
    else
    {
        error("new can only create structs, dynamic arrays or class objects, not %s's", type->toChars());
        goto Lerr;
    }

//printf("NewExp: '%s'\n", toChars());
//printf("NewExp:type '%s'\n", type->toChars());

    return this;

Lerr:
    return new ErrorExp();
}


void NewExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    if (thisexp)
    {   expToCBuffer(buf, hgs, thisexp, PREC_primary);
        buf->writeByte('.');
    }
    buf->writestring("new ");
    if (newargs && newargs->dim)
    {
        buf->writeByte('(');
        argsToCBuffer(buf, newargs, hgs);
        buf->writeByte(')');
    }
    newtype->toCBuffer(buf, NULL, hgs);
    if (arguments && arguments->dim)
    {
        buf->writeByte('(');
        argsToCBuffer(buf, arguments, hgs);
        buf->writeByte(')');
    }
}

/********************** NewAnonClassExp **************************************/

NewAnonClassExp::NewAnonClassExp(Loc loc, Expression *thisexp,
        Expressions *newargs, ClassDeclaration *cd, Expressions *arguments)
    : Expression(loc, TOKnewanonclass, sizeof(NewAnonClassExp))
{
    this->thisexp = thisexp;
    this->newargs = newargs;
    this->cd = cd;
    this->arguments = arguments;
}

Expression *NewAnonClassExp::syntaxCopy()
{
    return new NewAnonClassExp(loc,
        thisexp ? thisexp->syntaxCopy() : NULL,
        arraySyntaxCopy(newargs),
        (ClassDeclaration *)cd->syntaxCopy(NULL),
        arraySyntaxCopy(arguments));
}


Expression *NewAnonClassExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("NewAnonClassExp::semantic() %s\n", toChars());
    //printf("thisexp = %p\n", thisexp);
    //printf("type: %s\n", type->toChars());
#endif

    Expression *d = new DeclarationExp(loc, cd);
    d = d->semantic(sc);

    Expression *n = new NewExp(loc, thisexp, newargs, cd->type, arguments);

    Expression *c = new CommaExp(loc, d, n);
    return c->semantic(sc);
}


void NewAnonClassExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    if (thisexp)
    {   expToCBuffer(buf, hgs, thisexp, PREC_primary);
        buf->writeByte('.');
    }
    buf->writestring("new");
    if (newargs && newargs->dim)
    {
        buf->writeByte('(');
        argsToCBuffer(buf, newargs, hgs);
        buf->writeByte(')');
    }
    buf->writestring(" class ");
    if (arguments && arguments->dim)
    {
        buf->writeByte('(');
        argsToCBuffer(buf, arguments, hgs);
        buf->writeByte(')');
    }
    //buf->writestring(" { }");
    if (cd)
    {
        cd->toCBuffer(buf, hgs);
    }
}

/********************** SymbolExp **************************************/

#if DMDV2
SymbolExp::SymbolExp(Loc loc, enum TOK op, int size, Declaration *var, int hasOverloads)
    : Expression(loc, op, size)
{
    assert(var);
    this->var = var;
    this->hasOverloads = hasOverloads;
}
#endif

/********************** SymOffExp **************************************/

SymOffExp::SymOffExp(Loc loc, Declaration *var, unsigned offset, int hasOverloads)
    : SymbolExp(loc, TOKsymoff, sizeof(SymOffExp), var, hasOverloads)
{
    this->offset = offset;
    VarDeclaration *v = var->isVarDeclaration();
    if (v && v->needThis())
        error("need 'this' for address of %s", v->toChars());
}

Expression *SymOffExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("SymOffExp::semantic('%s')\n", toChars());
#endif
    //var->semantic(sc);
    if (!type)
        type = var->type->pointerTo();
    VarDeclaration *v = var->isVarDeclaration();
    if (v)
        v->checkNestedReference(sc, loc);
    FuncDeclaration *f = var->isFuncDeclaration();
    if (f)
        f->checkNestedReference(sc, loc);
    return this;
}

int SymOffExp::isBool(int result)
{
    return result ? TRUE : FALSE;
}

void SymOffExp::checkEscape()
{
    VarDeclaration *v = var->isVarDeclaration();
    if (v)
    {
        if (!v->isDataseg() && !(v->storage_class & (STCref | STCout)))
        {   /* BUG: This should be allowed:
             *   void foo()
             *   { int a;
             *     int* bar() { return &a; }
             *   }
             */
            error("escaping reference to local %s", v->toChars());
        }
    }
}

void SymOffExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    if (offset)
        buf->printf("(& %s+%u)", var->toChars(), offset);
    else
        buf->printf("& %s", var->toChars());
}

/******************************** VarExp **************************/

VarExp::VarExp(Loc loc, Declaration *var, int hasOverloads)
    : SymbolExp(loc, TOKvar, sizeof(VarExp), var, hasOverloads)
{
    //printf("VarExp(this = %p, '%s', loc = %s)\n", this, var->toChars(), loc.toChars());
    //if (strcmp(var->ident->toChars(), "func") == 0) halt();
    this->type = var->type;
}

int VarExp::equals(Object *o)
{   VarExp *ne;

    if (this == o ||
        (((Expression *)o)->op == TOKvar &&
         ((ne = (VarExp *)o), type->toHeadMutable()->equals(ne->type->toHeadMutable())) &&
         var == ne->var))
        return 1;
    return 0;
}

Expression *VarExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("VarExp::semantic(%s)\n", toChars());
#endif
    if (FuncDeclaration *f = var->isFuncDeclaration())
    {
        //printf("L%d fd = %s\n", __LINE__, f->toChars());
        if (!f->functionSemantic())
            return new ErrorExp();
    }

    if (!type)
        type = var->type;

    if (type && !type->deco)
        type = type->semantic(loc, sc);

    /* Fix for 1161 doesn't work because it causes protection
     * problems when instantiating imported templates passing private
     * variables as alias template parameters.
     */
    //accessCheck(loc, sc, NULL, var);

    VarDeclaration *v = var->isVarDeclaration();
    if (v)
    {
        v->checkNestedReference(sc, loc);
#if DMDV2
        checkPurity(sc, v, NULL);
#endif
    }
    FuncDeclaration *f = var->isFuncDeclaration();
    if (f)
        f->checkNestedReference(sc, loc);

    return this;
}

char *VarExp::toChars()
{
    return var->toChars();
}

void VarExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring(var->toChars());
}

void VarExp::checkEscape()
{
    VarDeclaration *v = var->isVarDeclaration();
    if (v)
    {   Type *tb = v->type->toBasetype();
        // if reference type
        if (tb->ty == Tarray || tb->ty == Tsarray || tb->ty == Tclass || tb->ty == Tdelegate)
        {
            if (v->isScope() && (!v->noscope || tb->ty == Tclass))
                error("escaping reference to scope local %s", v->toChars());
            else if (v->storage_class & STCvariadic)
                error("escaping reference to variadic parameter %s", v->toChars());
        }
    }
}

void VarExp::checkEscapeRef()
{
    VarDeclaration *v = var->isVarDeclaration();
    if (v)
    {
        if (!v->isDataseg() && !(v->storage_class & (STCref | STCout)))
            error("escaping reference to local variable %s", v->toChars());
    }
}

int VarExp::isLvalue()
{
    if (var->storage_class & (STClazy | STCtemp))
        return 0;
    return 1;
}

Expression *VarExp::toLvalue(Scope *sc, Expression *e)
{
    if (var->storage_class & STClazy)
    {   error("lazy variables cannot be lvalues");
        return new ErrorExp();
    }
    if (var->ident == Id::ctfe)
    {
        error("compiler-generated variable __ctfe is not an lvalue");
        return new ErrorExp();
    }
    return this;
}

int VarExp::checkModifiable(Scope *sc, int flag)
{
    //printf("VarExp::checkModifiable %s", toChars());
    assert(type);
    return var->checkModify(loc, sc, type, NULL, flag);
}

Expression *VarExp::modifiableLvalue(Scope *sc, Expression *e)
{
    //printf("VarExp::modifiableLvalue('%s')\n", var->toChars());
    //if (type && type->toBasetype()->ty == Tsarray)
        //error("cannot change reference to static array '%s'", var->toChars());

#if (BUG6652 == 1)
    VarDeclaration *v = var->isVarDeclaration();
    if (v && (v->storage_class & STCbug6652) && v->type->isMutable())
        warning("variable modified in foreach body requires ref storage class");
#elif (BUG6652 == 2)
    VarDeclaration *v = var->isVarDeclaration();
    if (v && (v->storage_class & STCbug6652) && v->type->isMutable())
        deprecation("variable modified in foreach body requires ref storage class");
#endif

    // See if this expression is a modifiable lvalue (i.e. not const)
    return Expression::modifiableLvalue(sc, e);
}


/******************************** OverExp **************************/

#if DMDV2
OverExp::OverExp(OverloadSet *s)
        : Expression(loc, TOKoverloadset, sizeof(OverExp))
{
    //printf("OverExp(this = %p, '%s')\n", this, var->toChars());
    vars = s;
    type = Type::tvoid;
}

int OverExp::isLvalue()
{
    return 1;
}

Expression *OverExp::toLvalue(Scope *sc, Expression *e)
{
    return this;
}
#endif


/******************************** TupleExp **************************/

TupleExp::TupleExp(Loc loc, Expressions *exps)
        : Expression(loc, TOKtuple, sizeof(TupleExp))
{
    //printf("TupleExp(this = %p)\n", this);
    this->exps = exps;
    this->type = NULL;
}


TupleExp::TupleExp(Loc loc, TupleDeclaration *tup)
        : Expression(loc, TOKtuple, sizeof(TupleExp))
{
    exps = new Expressions();
    type = NULL;

    exps->reserve(tup->objects->dim);
    for (size_t i = 0; i < tup->objects->dim; i++)
    {   Object *o = (*tup->objects)[i];
        if (o->dyncast() == DYNCAST_EXPRESSION)
        {
            Expression *e = (Expression *)o;
            if (e->op == TOKdsymbol)
                e = e->syntaxCopy();
            exps->push(e);
        }
        else if (o->dyncast() == DYNCAST_DSYMBOL)
        {
            Dsymbol *s = (Dsymbol *)o;
            Expression *e = new DsymbolExp(loc, s);
            exps->push(e);
        }
        else if (o->dyncast() == DYNCAST_TYPE)
        {
            Type *t = (Type *)o;
            Expression *e = new TypeExp(loc, t);
            exps->push(e);
        }
        else
        {
            error("%s is not an expression", o->toChars());
        }
    }
}

int TupleExp::equals(Object *o)
{
    if (this == o)
        return 1;
    if (((Expression *)o)->op == TOKtuple)
    {
        TupleExp *te = (TupleExp *)o;
        if (exps->dim != te->exps->dim)
            return 0;
        for (size_t i = 0; i < exps->dim; i++)
        {   Expression *e1 = (*exps)[i];
            Expression *e2 = (*te->exps)[i];

            if (!e1->equals(e2))
                return 0;
        }
        return 1;
    }
    return 0;
}

Expression *TupleExp::syntaxCopy()
{
    return new TupleExp(loc, arraySyntaxCopy(exps));
}

Expression *TupleExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("+TupleExp::semantic(%s)\n", toChars());
#endif
    if (type)
        return this;

    // Run semantic() on each argument
    for (size_t i = 0; i < exps->dim; i++)
    {   Expression *e = (*exps)[i];

        e = e->semantic(sc);
        if (!e->type)
        {   error("%s has no value", e->toChars());
            return new ErrorExp();
        }
        (*exps)[i] = e;
    }

    expandTuples(exps);
    type = new TypeTuple(exps);
    type = type->semantic(loc, sc);
    //printf("-TupleExp::semantic(%s)\n", toChars());
    return this;
}

void TupleExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("tuple(");
    argsToCBuffer(buf, exps, hgs);
    buf->writeByte(')');
}


void TupleExp::checkEscape()
{
    for (size_t i = 0; i < exps->dim; i++)
    {   Expression *e = (*exps)[i];
        e->checkEscape();
    }
}

/******************************** FuncExp *********************************/

FuncExp::FuncExp(Loc loc, FuncLiteralDeclaration *fd, TemplateDeclaration *td)
        : Expression(loc, TOKfunction, sizeof(FuncExp))
{
    this->fd = fd;
    this->td = td;
    tok = fd->tok;  // save original kind of function/delegate/(infer)
}

Expression *FuncExp::syntaxCopy()
{
    TemplateDeclaration *td2 = td ? (TemplateDeclaration *)td->syntaxCopy(NULL) : NULL;
    return new FuncExp(loc, (FuncLiteralDeclaration *)fd->syntaxCopy(NULL), td2);
}

Expression *FuncExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("FuncExp::semantic(%s)\n", toChars());
    if (fd->treq) printf("  treq = %s\n", fd->treq->toChars());
#endif
    if (!type || type == Type::tvoid)
    {
        /* fd->treq might be incomplete type,
         * so should not semantic it.
         * void foo(T)(T delegate(int) dg){}
         * foo(a=>a); // in IFTI, treq == T delegate(int)
         */
        //if (fd->treq)
        //    fd->treq = fd->treq->semantic(loc, sc);

        // Set target of return type inference
        if (fd->treq && !fd->type->nextOf())
        {   TypeFunction *tfv = NULL;
            if (fd->treq->ty == Tdelegate ||
                (fd->treq->ty == Tpointer && fd->treq->nextOf()->ty == Tfunction))
                tfv = (TypeFunction *)fd->treq->nextOf();
            if (tfv)
            {   TypeFunction *tfl = (TypeFunction *)fd->type;
                tfl->next = tfv->nextOf();
            }
        }

        //printf("td = %p, treq = %p\n", td, fd->treq);
        if (td)
        {
            assert(td->parameters && td->parameters->dim);
            td->semantic(sc);
            type = Type::tvoid; // temporary type

            if (!fd->treq)  // defer type determination
                return this;

            return inferType(fd->treq);
        }

        unsigned olderrors = global.errors;
        fd->semantic(sc);
        //fd->parent = sc->parent;
        if (olderrors != global.errors)
        {
        }
        else
        {
            fd->semantic2(sc);
            if ( (olderrors == global.errors) ||
                // need to infer return type
                (fd->type && fd->type->ty == Tfunction && !fd->type->nextOf()))
            {
                fd->semantic3(sc);
            }
        }

        // need to infer return type
        if ((olderrors != global.errors) && fd->type && fd->type->ty == Tfunction && !fd->type->nextOf())
            ((TypeFunction *)fd->type)->next = Type::terror;

        // Type is a "delegate to" or "pointer to" the function literal
        if ((fd->isNested() && fd->tok == TOKdelegate) ||
            (tok == TOKreserved && fd->treq && fd->treq->ty == Tdelegate))
        {
            type = new TypeDelegate(fd->type);
            type = type->semantic(loc, sc);

            fd->tok = TOKdelegate;
        }
        else
        {
            type = new TypePointer(fd->type);
            type = type->semantic(loc, sc);
            //type = fd->type->pointerTo();

            /* A lambda expression deduced to function pointer might become
             * to a delegate literal implicitly.
             *
             *   auto foo(void function() fp) { return 1; }
             *   assert(foo({}) == 1);
             *
             * So, should keep fd->tok == TOKreserve if fd->treq == NULL.
             */
            if (fd->treq && fd->treq->ty == Tpointer)
            {   // change to non-nested
                fd->tok = TOKfunction;
                fd->vthis = NULL;
            }
        }
        fd->tookAddressOf++;
    }
    return this;
}

// used from CallExp::semantic()
Expression *FuncExp::semantic(Scope *sc, Expressions *arguments)
{
    if ((!type || type == Type::tvoid) && td && arguments && arguments->dim)
    {
        for (size_t k = 0; k < arguments->dim; k++)
        {   Expression *checkarg = (*arguments)[k];
            if (checkarg->op == TOKerror)
                return checkarg;
        }

        assert(td->parameters && td->parameters->dim);
        td->semantic(sc);

        TypeFunction *tfl = (TypeFunction *)fd->type;
        size_t dim = Parameter::dim(tfl->parameters);
        if (arguments->dim < dim)
        {   // Default arguments are always typed, so they don't need inference.
            Parameter *p = Parameter::getNth(tfl->parameters, arguments->dim);
            if (p->defaultArg)
                dim = arguments->dim;
        }

        if ((!tfl->varargs && arguments->dim == dim) ||
            ( tfl->varargs && arguments->dim >= dim))
        {
            Objects *tiargs = new Objects();
            tiargs->reserve(td->parameters->dim);

            for (size_t i = 0; i < td->parameters->dim; i++)
            {
                TemplateParameter *tp = (*td->parameters)[i];
                for (size_t u = 0; u < dim; u++)
                {   Parameter *p = Parameter::getNth(tfl->parameters, u);
                    if (p->type->ty == Tident &&
                        ((TypeIdentifier *)p->type)->ident == tp->ident)
                    {   Expression *e = (*arguments)[u];
                        tiargs->push(e->type);
                        u = dim;    // break inner loop
                    }
                }
            }

            TemplateInstance *ti = new TemplateInstance(loc, td, tiargs);
            return (new ScopeExp(loc, ti))->semantic(sc);
        }
        error("cannot infer function literal type");
        return new ErrorExp();
    }
    return semantic(sc);
}

char *FuncExp::toChars()
{
    return fd->toChars();
}

void FuncExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    fd->toCBuffer(buf, hgs);
    //buf->writestring(fd->toChars());
}


/******************************** DeclarationExp **************************/

DeclarationExp::DeclarationExp(Loc loc, Dsymbol *declaration)
        : Expression(loc, TOKdeclaration, sizeof(DeclarationExp))
{
    this->declaration = declaration;
}

Expression *DeclarationExp::syntaxCopy()
{
    return new DeclarationExp(loc, declaration->syntaxCopy(NULL));
}

Expression *DeclarationExp::semantic(Scope *sc)
{
    if (type)
        return this;

#if LOGSEMANTIC
    printf("DeclarationExp::semantic() %s\n", toChars());
#endif

    unsigned olderrors = global.errors;

    /* This is here to support extern(linkage) declaration,
     * where the extern(linkage) winds up being an AttribDeclaration
     * wrapper.
     */
    Dsymbol *s = declaration;

    while (1)
    {
        AttribDeclaration *ad = s->isAttribDeclaration();
        if (ad)
        {
            if (ad->decl && ad->decl->dim == 1)
            {   s = (*ad->decl)[0];
                continue;
            }
        }
        break;
    }

    if (s->isVarDeclaration())
    {   // Do semantic() on initializer first, so:
        //      int a = a;
        // will be illegal.
        declaration->semantic(sc);
        s->parent = sc->parent;
    }

    //printf("inserting '%s' %p into sc = %p\n", s->toChars(), s, sc);
    // Insert into both local scope and function scope.
    // Must be unique in both.
    if (s->ident)
    {
        if (!sc->insert(s))
        {   error("declaration %s is already defined", s->toPrettyChars());
            return new ErrorExp();
        }
        else if (sc->func)
        {   VarDeclaration *v = s->isVarDeclaration();
            if ( (s->isFuncDeclaration() || s->isTypedefDeclaration() ||
                s->isAggregateDeclaration() || s->isEnumDeclaration() ||
                s->isInterfaceDeclaration()) &&
                !sc->func->localsymtab->insert(s))
            {
                error("declaration %s is already defined in another scope in %s",
                    s->toPrettyChars(), sc->func->toChars());
                return new ErrorExp();
            }
            else
            {   // Disallow shadowing

                for (Scope *scx = sc->enclosing; scx && scx->func == sc->func; scx = scx->enclosing)
                {   Dsymbol *s2;

                    if (scx->scopesym && scx->scopesym->symtab &&
                        (s2 = scx->scopesym->symtab->lookup(s->ident)) != NULL &&
                        s != s2)
                    {
                        error("is shadowing declaration %s", s->toPrettyChars());
                        return new ErrorExp();
                    }
                }
            }
        }
    }
    if (!s->isVarDeclaration())
    {
        Scope *sc2 = sc;
        if (sc2->stc & (STCpure | STCnothrow))
            sc2 = sc->push();
        sc2->stc &= ~(STCpure | STCnothrow);
        declaration->semantic(sc2);
        if (sc2 != sc)
            sc2->pop();
        s->parent = sc->parent;
    }
    if (global.errors == olderrors)
    {
        declaration->semantic2(sc);
        if (global.errors == olderrors)
        {
            declaration->semantic3(sc);
        }
    }

    type = Type::tvoid;
    return this;
}


void DeclarationExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    declaration->toCBuffer(buf, hgs);
}


/************************ TypeidExp ************************************/

/*
 *      typeid(int)
 */

TypeidExp::TypeidExp(Loc loc, Object *o)
    : Expression(loc, TOKtypeid, sizeof(TypeidExp))
{
    this->obj = o;
}


Expression *TypeidExp::syntaxCopy()
{
    return new TypeidExp(loc, objectSyntaxCopy(obj));
}


Expression *TypeidExp::semantic(Scope *sc)
{   Expression *e;

#if LOGSEMANTIC
    printf("TypeidExp::semantic() %s\n", toChars());
#endif
    Type *ta = isType(obj);
    Expression *ea = isExpression(obj);
    Dsymbol *sa = isDsymbol(obj);

    //printf("ta %p ea %p sa %p\n", ta, ea, sa);

    if (ta)
    {
        ta->resolve(loc, sc, &ea, &ta, &sa);
    }

    if (ea)
    {
        Dsymbol *sym = getDsymbol(ea);
        if (sym)
            ea = new DsymbolExp(loc, sym);
        ea = ea->semantic(sc);
        ea = resolveProperties(sc, ea);
        ta = ea->type;
        if (ea->op == TOKtype)
            ea = NULL;
    }

    if (!ta)
    {
        //printf("ta %p ea %p sa %p\n", ta, ea, sa);
        error("no type for typeid(%s)", ea ? ea->toChars() : (sa ? sa->toChars() : ""));
        return new ErrorExp();
    }

    if (ea && ta->toBasetype()->ty == Tclass)
    {   /* Get the dynamic type, which is .classinfo
         */
        e = new DotIdExp(ea->loc, ea, Id::classinfo);
        e = e->semantic(sc);
    }
    else
    {   /* Get the static type
         */
        e = ta->getTypeInfo(sc);
        if (e->loc.linnum == 0)
            e->loc = loc;               // so there's at least some line number info
        if (ea)
        {
            e = new CommaExp(loc, ea, e);       // execute ea
            e = e->semantic(sc);
        }
    }
    return e;
}

void TypeidExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("typeid(");
    ObjectToCBuffer(buf, hgs, obj);
    buf->writeByte(')');
}

/************************ TraitsExp ************************************/
#if DMDV2
/*
 *      __traits(identifier, args...)
 */

TraitsExp::TraitsExp(Loc loc, Identifier *ident, Objects *args)
    : Expression(loc, TOKtraits, sizeof(TraitsExp))
{
    this->ident = ident;
    this->args = args;
}


Expression *TraitsExp::syntaxCopy()
{
    return new TraitsExp(loc, ident, TemplateInstance::arraySyntaxCopy(args));
}


void TraitsExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("__traits(");
    buf->writestring(ident->toChars());
    if (args)
    {
        for (size_t i = 0; i < args->dim; i++)
        {
            buf->writeByte(',');
            Object *oarg = (*args)[i];
            ObjectToCBuffer(buf, hgs, oarg);
        }
    }
    buf->writeByte(')');
}
#endif

/************************************************************/

HaltExp::HaltExp(Loc loc)
        : Expression(loc, TOKhalt, sizeof(HaltExp))
{
}

Expression *HaltExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("HaltExp::semantic()\n");
#endif
    type = Type::tvoid;
    return this;
}


void HaltExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("halt");
}

/************************************************************/

IsExp::IsExp(Loc loc, Type *targ, Identifier *id, enum TOK tok,
        Type *tspec, enum TOK tok2, TemplateParameters *parameters)
        : Expression(loc, TOKis, sizeof(IsExp))
{
    this->targ = targ;
    this->id = id;
    this->tok = tok;
    this->tspec = tspec;
    this->tok2 = tok2;
    this->parameters = parameters;
}

Expression *IsExp::syntaxCopy()
{
    // This section is identical to that in TemplateDeclaration::syntaxCopy()
    TemplateParameters *p = NULL;
    if (parameters)
    {
        p = new TemplateParameters();
        p->setDim(parameters->dim);
        for (size_t i = 0; i < p->dim; i++)
        {   TemplateParameter *tp = (*parameters)[i];
            (*p)[i] = tp->syntaxCopy();
        }
    }

    return new IsExp(loc,
        targ->syntaxCopy(),
        id,
        tok,
        tspec ? tspec->syntaxCopy() : NULL,
        tok2,
        p);
}

Expression *IsExp::semantic(Scope *sc)
{   Type *tded;

    /* is(targ id tok tspec)
     * is(targ id :  tok2)
     * is(targ id == tok2)
     */

    //printf("IsExp::semantic(%s)\n", toChars());
    if (id && !(sc->flags & (SCOPEstaticif | SCOPEstaticassert)))
    {   error("can only declare type aliases within static if conditionals or static asserts");
        return new ErrorExp();
    }

    Type *t = targ->trySemantic(loc, sc);
    if (!t)
        goto Lno;                       // errors, so condition is false
    targ = t;
    if (tok2 != TOKreserved)
    {
        switch (tok2)
        {
            case TOKtypedef:
                if (targ->ty != Ttypedef)
                    goto Lno;
                tded = ((TypeTypedef *)targ)->sym->basetype;
                break;

            case TOKstruct:
                if (targ->ty != Tstruct)
                    goto Lno;
                if (((TypeStruct *)targ)->sym->isUnionDeclaration())
                    goto Lno;
                tded = targ;
                break;

            case TOKunion:
                if (targ->ty != Tstruct)
                    goto Lno;
                if (!((TypeStruct *)targ)->sym->isUnionDeclaration())
                    goto Lno;
                tded = targ;
                break;

            case TOKclass:
                if (targ->ty != Tclass)
                    goto Lno;
                if (((TypeClass *)targ)->sym->isInterfaceDeclaration())
                    goto Lno;
                tded = targ;
                break;

            case TOKinterface:
                if (targ->ty != Tclass)
                    goto Lno;
                if (!((TypeClass *)targ)->sym->isInterfaceDeclaration())
                    goto Lno;
                tded = targ;
                break;
#if DMDV2
            case TOKconst:
                if (!targ->isConst())
                    goto Lno;
                tded = targ;
                break;

            case TOKinvariant:
                deprecation("use of 'invariant' rather than 'immutable' is deprecated");
            case TOKimmutable:
                if (!targ->isImmutable())
                    goto Lno;
                tded = targ;
                break;

            case TOKshared:
                if (!targ->isShared())
                    goto Lno;
                tded = targ;
                break;

            case TOKwild:
                if (!targ->isWild())
                    goto Lno;
                tded = targ;
                break;
#endif

            case TOKsuper:
                // If class or interface, get the base class and interfaces
                if (targ->ty != Tclass)
                    goto Lno;
                else
                {   ClassDeclaration *cd = ((TypeClass *)targ)->sym;
                    Parameters *args = new Parameters;
                    args->reserve(cd->baseclasses->dim);
                    for (size_t i = 0; i < cd->baseclasses->dim; i++)
                    {   BaseClass *b = (*cd->baseclasses)[i];
                        args->push(new Parameter(STCin, b->type, NULL, NULL));
                    }
                    tded = new TypeTuple(args);
                }
                break;

            case TOKenum:
                if (targ->ty != Tenum)
                    goto Lno;
                tded = ((TypeEnum *)targ)->sym->memtype;
                break;

            case TOKdelegate:
                if (targ->ty != Tdelegate)
                    goto Lno;
                tded = ((TypeDelegate *)targ)->next;    // the underlying function type
                break;

            case TOKfunction:
            case TOKparameters:
            {
                if (targ->ty != Tfunction)
                    goto Lno;
                tded = targ;

                /* Generate tuple from function parameter types.
                 */
                assert(tded->ty == Tfunction);
                Parameters *params = ((TypeFunction *)tded)->parameters;
                size_t dim = Parameter::dim(params);
                Parameters *args = new Parameters;
                args->reserve(dim);
                for (size_t i = 0; i < dim; i++)
                {   Parameter *arg = Parameter::getNth(params, i);
                    assert(arg && arg->type);
                    /* If one of the default arguments was an error,
                       don't return an invalid tuple
                    */
                    if (tok2 == TOKparameters && arg->defaultArg &&
                        arg->defaultArg->op == TOKerror)
                        return new ErrorExp();
                    args->push(new Parameter(arg->storageClass, arg->type,
                        (tok2 == TOKparameters) ? arg->ident : NULL,
                        (tok2 == TOKparameters) ? arg->defaultArg : NULL));
                }
                tded = new TypeTuple(args);
                break;
            }
            case TOKreturn:
                /* Get the 'return type' for the function,
                 * delegate, or pointer to function.
                 */
                if (targ->ty == Tfunction)
                    tded = ((TypeFunction *)targ)->next;
                else if (targ->ty == Tdelegate)
                {   tded = ((TypeDelegate *)targ)->next;
                    tded = ((TypeFunction *)tded)->next;
                }
                else if (targ->ty == Tpointer &&
                         ((TypePointer *)targ)->next->ty == Tfunction)
                {   tded = ((TypePointer *)targ)->next;
                    tded = ((TypeFunction *)tded)->next;
                }
                else
                    goto Lno;
                break;

            case TOKargTypes:
                /* Generate a type tuple of the equivalent types used to determine if a
                 * function argument of this type can be passed in registers.
                 * The results of this are highly platform dependent, and intended
                 * primarly for use in implementing va_arg().
                 */
                tded = targ->toArgTypes();
                if (!tded)
                    goto Lno;           // not valid for a parameter
                break;

            default:
                assert(0);
        }
        goto Lyes;
    }
    else if (id && tspec)
    {
        /* Evaluate to TRUE if targ matches tspec.
         * If TRUE, declare id as an alias for the specialized type.
         */

        assert(parameters && parameters->dim);

        Objects dedtypes;
        dedtypes.setDim(parameters->dim);
        dedtypes.zero();

        MATCH m = targ->deduceType(sc, tspec, parameters, &dedtypes);
//printf("targ: %s\n", targ->toChars());
//printf("tspec: %s\n", tspec->toChars());
        if (m == MATCHnomatch ||
            (m != MATCHexact && tok == TOKequal))
        {
            goto Lno;
        }
        else
        {
            tded = (Type *)dedtypes[0];
            if (!tded)
                tded = targ;
#if DMDV2
            Objects tiargs;
            tiargs.setDim(1);
            tiargs[0] = targ;

            /* Declare trailing parameters
             */
            for (size_t i = 1; i < parameters->dim; i++)
            {   TemplateParameter *tp = (*parameters)[i];
                Declaration *s = NULL;

                m = tp->matchArg(sc, &tiargs, i, parameters, &dedtypes, &s);
                if (m == MATCHnomatch)
                    goto Lno;
                s->semantic(sc);
                if (sc->sd)
                    s->addMember(sc, sc->sd, 1);
                else if (!sc->insert(s))
                    error("declaration %s is already defined", s->toChars());
            }
#endif
            goto Lyes;
        }
    }
    else if (id)
    {
        /* Declare id as an alias for type targ. Evaluate to TRUE
         */
        tded = targ;
        goto Lyes;
    }
    else if (tspec)
    {
        /* Evaluate to TRUE if targ matches tspec
         * is(targ == tspec)
         * is(targ : tspec)
         */
        tspec = tspec->semantic(loc, sc);
        //printf("targ  = %s, %s\n", targ->toChars(), targ->deco);
        //printf("tspec = %s, %s\n", tspec->toChars(), tspec->deco);
        if (tok == TOKcolon)
        {   if (targ->implicitConvTo(tspec))
                goto Lyes;
            else
                goto Lno;
        }
        else /* == */
        {   if (targ->equals(tspec))
                goto Lyes;
            else
                goto Lno;
        }
    }

Lyes:
    if (id)
    {
        Dsymbol *s;
        Tuple *tup = isTuple(tded);
        if (tup)
            s = new TupleDeclaration(loc, id, &(tup->objects));
        else
            s = new AliasDeclaration(loc, id, tded);
        s->semantic(sc);
        /* The reason for the !tup is unclear. It fails Phobos unittests if it is not there.
         * More investigation is needed.
         */
        if (!tup && !sc->insert(s))
            error("declaration %s is already defined", s->toChars());
        if (sc->sd)
            s->addMember(sc, sc->sd, 1);
    }
    //printf("Lyes\n");
    return new IntegerExp(loc, 1, Type::tbool);

Lno:
    //printf("Lno\n");
    return new IntegerExp(loc, 0, Type::tbool);
}

void IsExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("is(");
    targ->toCBuffer(buf, id, hgs);
    if (tok2 != TOKreserved)
    {
        buf->printf(" %s %s", Token::toChars(tok), Token::toChars(tok2));
    }
    else if (tspec)
    {
        if (tok == TOKcolon)
            buf->writestring(" : ");
        else
            buf->writestring(" == ");
        tspec->toCBuffer(buf, NULL, hgs);
    }
#if DMDV2
    if (parameters)
    {   // First parameter is already output, so start with second
        for (size_t i = 1; i < parameters->dim; i++)
        {
            buf->writeByte(',');
            TemplateParameter *tp = (*parameters)[i];
            tp->toCBuffer(buf, hgs);
        }
    }
#endif
    buf->writeByte(')');
}


/************************************************************/

UnaExp::UnaExp(Loc loc, enum TOK op, int size, Expression *e1)
        : Expression(loc, op, size)
{
    this->e1 = e1;
}

Expression *UnaExp::syntaxCopy()
{
    UnaExp *e = (UnaExp *)copy();
    e->type = NULL;
    e->e1 = e->e1->syntaxCopy();
    return e;
}

Expression *UnaExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("UnaExp::semantic('%s')\n", toChars());
#endif
    e1 = e1->semantic(sc);
//    if (!e1->type)
//      error("%s has no value", e1->toChars());
    return this;
}

Expression *UnaExp::resolveLoc(Loc loc, Scope *sc)
{
    e1 = e1->resolveLoc(loc, sc);
    return this;
}

void UnaExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring(Token::toChars(op));
    expToCBuffer(buf, hgs, e1, precedence[op]);
}

/************************************************************/

BinExp::BinExp(Loc loc, enum TOK op, int size, Expression *e1, Expression *e2)
        : Expression(loc, op, size)
{
    this->e1 = e1;
    this->e2 = e2;
}

Expression *BinExp::syntaxCopy()
{
    BinExp *e = (BinExp *)copy();
    e->type = NULL;
    e->e1 = e->e1->syntaxCopy();
    e->e2 = e->e2->syntaxCopy();
    return e;
}

Expression *BinExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("BinExp::semantic('%s')\n", toChars());
#endif
    e1 = e1->semantic(sc);
    e2 = e2->semantic(sc);
    if (e1->op == TOKerror || e2->op == TOKerror)
        return new ErrorExp();
    return this;
}

Expression *BinExp::semanticp(Scope *sc)
{
    BinExp::semantic(sc);
    e1 = resolveProperties(sc, e1);
    e2 = resolveProperties(sc, e2);
    return this;
}


Expression *BinExp::checkComplexOpAssign(Scope *sc)
{
    // generate an error if this is a nonsensical *=,/=, or %=, eg real *= imaginary
    if (op == TOKmulass || op == TOKdivass || op == TOKmodass)
    {
        // Any multiplication by an imaginary or complex number yields a complex result.
        // r *= c, i*=c, r*=i, i*=i are all forbidden operations.
        const char *opstr = Token::toChars(op);
        if ( e1->type->isreal() && e2->type->iscomplex())
        {
            error("%s %s %s is undefined. Did you mean %s %s %s.re ?",
                e1->type->toChars(), opstr, e2->type->toChars(),
                e1->type->toChars(), opstr, e2->type->toChars());
        }
        else if (e1->type->isimaginary() && e2->type->iscomplex())
        {
            error("%s %s %s is undefined. Did you mean %s %s %s.im ?",
                e1->type->toChars(), opstr, e2->type->toChars(),
                e1->type->toChars(), opstr, e2->type->toChars());
        }
        else if ((e1->type->isreal() || e1->type->isimaginary()) &&
            e2->type->isimaginary())
        {
            error("%s %s %s is an undefined operation", e1->type->toChars(),
                    opstr, e2->type->toChars());
        }
    }

    // generate an error if this is a nonsensical += or -=, eg real += imaginary
    if (op == TOKaddass || op == TOKminass)
    {
        // Addition or subtraction of a real and an imaginary is a complex result.
        // Thus, r+=i, r+=c, i+=r, i+=c are all forbidden operations.
        if ( (e1->type->isreal() && (e2->type->isimaginary() || e2->type->iscomplex())) ||
             (e1->type->isimaginary() && (e2->type->isreal() || e2->type->iscomplex()))
            )
        {
            error("%s %s %s is undefined (result is complex)",
                e1->type->toChars(), Token::toChars(op), e2->type->toChars());
        }
        if (type->isreal() || type->isimaginary())
        {
            assert(global.errors || e2->type->isfloating());
            e2 = e2->castTo(sc, e1->type);
        }
    }

    if (op == TOKmulass)
    {
        if (e2->type->isfloating())
        {
            Type *t1 = e1->type;
            Type *t2 = e2->type;
            if (t1->isreal())
            {
                if (t2->isimaginary() || t2->iscomplex())
                {
                    e2 = e2->castTo(sc, t1);
                }
            }
            else if (t1->isimaginary())
            {
                if (t2->isimaginary() || t2->iscomplex())
                {
                    switch (t1->ty)
                    {
                        case Timaginary32: t2 = Type::tfloat32; break;
                        case Timaginary64: t2 = Type::tfloat64; break;
                        case Timaginary80: t2 = Type::tfloat80; break;
                        default:
                            assert(0);
                    }
                    e2 = e2->castTo(sc, t2);
                }
            }
        }
    } else if (op == TOKdivass)
    {
        if (e2->type->isimaginary())
        {
            Type *t1 = e1->type;
            if (t1->isreal())
            {   // x/iv = i(-x/v)
                // Therefore, the result is 0
                e2 = new CommaExp(loc, e2, new RealExp(loc, ldouble(0.0), t1));
                e2->type = t1;
                Expression *e = new AssignExp(loc, e1, e2);
                e->type = t1;
                return e;
            }
            else if (t1->isimaginary())
            {   Type *t2;

                switch (t1->ty)
                {
                    case Timaginary32: t2 = Type::tfloat32; break;
                    case Timaginary64: t2 = Type::tfloat64; break;
                    case Timaginary80: t2 = Type::tfloat80; break;
                    default:
                        assert(0);
                }
                e2 = e2->castTo(sc, t2);
                Expression *e = new AssignExp(loc, e1, e2);
                e->type = t1;
                return e;
            }
        }
    } else if (op == TOKmodass)
    {
        if (e2->type->iscomplex())
        {
            error("cannot perform modulo complex arithmetic");
            return new ErrorExp();
        }
    }
    return this;
}

void BinExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    expToCBuffer(buf, hgs, e1, precedence[op]);
    buf->writeByte(' ');
    buf->writestring(Token::toChars(op));
    buf->writeByte(' ');
    expToCBuffer(buf, hgs, e2, (enum PREC)(precedence[op] + 1));
}

int BinExp::isunsigned()
{
    return e1->type->isunsigned() || e2->type->isunsigned();
}

Expression *BinExp::incompatibleTypes()
{
    if (e1->type->toBasetype() != Type::terror &&
        e2->type->toBasetype() != Type::terror
       )
    {
        // CondExp uses 'a ? b : c' but we're comparing 'b : c'
        TOK thisOp = (op == TOKquestion) ? TOKcolon : op;
        if (e1->op == TOKtype || e2->op == TOKtype)
        {
            error("incompatible types for ((%s) %s (%s)): cannot use '%s' with types",
                e1->toChars(), Token::toChars(thisOp), e2->toChars(), Token::toChars(op));
        }
        else
        {
            error("incompatible types for ((%s) %s (%s)): '%s' and '%s'",
             e1->toChars(), Token::toChars(thisOp), e2->toChars(),
             e1->type->toChars(), e2->type->toChars());
        }
        return new ErrorExp();
    }
    return this;
}

/********************** BinAssignExp **************************************/

Expression *BinAssignExp::semantic(Scope *sc)
{
    Expression *e;

    if (type)
        return this;

    e = op_overload(sc);
    if (e)
        return e;

    if (e1->op == TOKarraylength)
    {
        e = ArrayLengthExp::rewriteOpAssign(this);
        e = e->semantic(sc);
        return e;
    }

    if (e1->op == TOKslice)
    {
        // T[] op= ...
        e = typeCombine(sc);
        if (e->op == TOKerror)
            return e;
        type = e1->type;
        return arrayOp(sc);
    }

    e1 = e1->modifiableLvalue(sc, e1);
    e1 = e1->semantic(sc);
    type = e1->type;
    checkScalar();

    int arith = (op == TOKaddass || op == TOKminass || op == TOKmulass ||
                 op == TOKdivass || op == TOKmodass || op == TOKpowass);
    int bitwise = (op == TOKandass || op == TOKorass || op == TOKxorass);
    int shift = (op == TOKshlass || op == TOKshrass || op == TOKushrass);

    if (bitwise && type->toBasetype()->ty == Tbool)
         e2 = e2->implicitCastTo(sc, type);
    else
        checkNoBool();

    if ((op == TOKaddass || op == TOKminass) &&
        e1->type->toBasetype()->ty == Tpointer &&
        e2->type->toBasetype()->isintegral())
        return scaleFactor(sc);

    typeCombine(sc);
    if (arith)
    {
        e1 = e1->checkArithmetic();
        e2 = e2->checkArithmetic();
    }
    if (bitwise || shift)
    {
        e1 = e1->checkIntegral();
        e2 = e2->checkIntegral();
    }
    if (shift)
    {
#if IN_LLVM
        e2 = e2->castTo(sc, e1->type);
#else
        e2 = e2->castTo(sc, Type::tshiftcnt);
#endif
    }

    // vectors
    if (shift && (e1->type->toBasetype()->ty == Tvector ||
                  e2->type->toBasetype()->ty == Tvector))
        return incompatibleTypes();

    int isvector = type->toBasetype()->ty == Tvector;

    if (op == TOKmulass && isvector && !e2->type->isfloating() &&
        ((TypeVector *)type->toBasetype())->elementType()->size(loc) != 2)
        return incompatibleTypes(); // Only short[8] and ushort[8] work with multiply

    if (op == TOKdivass && isvector && !e1->type->isfloating())
        return incompatibleTypes();

    if (op == TOKmodass && isvector)
        return incompatibleTypes();

    if (e1->op == TOKerror || e2->op == TOKerror)
        return new ErrorExp();

    return checkComplexOpAssign(sc);
}

#if DMDV2
int BinAssignExp::isLvalue()
{
    return 1;
}

Expression *BinAssignExp::toLvalue(Scope *sc, Expression *ex)
{   Expression *e;

    if (e1->op == TOKvar)
    {
        /* Convert (e1 op= e2) to
         *    e1 op= e2;
         *    e1
         */
        e = e1->copy();
        e = new CommaExp(loc, this, e);
        e = e->semantic(sc);
    }
    else
    {
        /* Convert (e1 op= e2) to
         *    ref v = e1;
         *    v op= e2;
         *    v
         */

        // ref v = e1;
        Identifier *id = Lexer::uniqueId("__assignop");
        ExpInitializer *ei = new ExpInitializer(loc, e1);
        VarDeclaration *v = new VarDeclaration(loc, e1->type, id, ei);
        v->storage_class |= STCref | STCforeach;
        Expression *de = new DeclarationExp(loc, v);

        // v op= e2
        e1 = new VarExp(e1->loc, v);

        e = new CommaExp(loc, de, this);
        e = new CommaExp(loc, e, new VarExp(loc, v));
        e = e->semantic(sc);
    }
    return e;
}

Expression *BinAssignExp::modifiableLvalue(Scope *sc, Expression *e)
{
    // should check e1->checkModifiable() ?
    return toLvalue(sc, this);
}

#endif

/************************************************************/

CompileExp::CompileExp(Loc loc, Expression *e)
        : UnaExp(loc, TOKmixin, sizeof(CompileExp), e)
{
}

Expression *CompileExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("CompileExp::semantic('%s')\n", toChars());
#endif
    UnaExp::semantic(sc);
    e1 = resolveProperties(sc, e1);
    if (e1->op == TOKerror)
        return e1;
    if (!e1->type->isString())
    {
        error("argument to mixin must be a string type, not %s", e1->type->toChars());
        return new ErrorExp();
    }
    e1 = e1->ctfeInterpret();
    StringExp *se = e1->toString();
    if (!se)
    {   error("argument to mixin must be a string, not (%s)", e1->toChars());
        return new ErrorExp();
    }
    se = se->toUTF8(sc);
    Parser p(sc->module, (unsigned char *)se->string, se->len, 0);
    p.loc = loc;
    p.nextToken();
    //printf("p.loc.linnum = %d\n", p.loc.linnum);
    Expression *e = p.parseExpression();
    if (p.token.value != TOKeof)
    {   error("incomplete mixin expression (%s)", se->toChars());
        return new ErrorExp();
    }
    return e->semantic(sc);
}

void CompileExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("mixin(");
    expToCBuffer(buf, hgs, e1, PREC_assign);
    buf->writeByte(')');
}

/************************************************************/

FileExp::FileExp(Loc loc, Expression *e)
        : UnaExp(loc, TOKmixin, sizeof(FileExp), e)
{
}

Expression *FileExp::semantic(Scope *sc)
{
    const char *name;
    StringExp *se;

#if LOGSEMANTIC
    printf("FileExp::semantic('%s')\n", toChars());
#endif
    UnaExp::semantic(sc);
    e1 = resolveProperties(sc, e1);
    e1 = e1->ctfeInterpret();
    if (e1->op != TOKstring)
    {   error("file name argument must be a string, not (%s)", e1->toChars());
        goto Lerror;
    }
    se = (StringExp *)e1;
    se = se->toUTF8(sc);
    name = (char *)se->string;

    if (!global.params.fileImppath)
    {   error("need -Jpath switch to import text file %s", name);
        goto Lerror;
    }

    /* Be wary of CWE-22: Improper Limitation of a Pathname to a Restricted Directory
     * ('Path Traversal') attacks.
     * http://cwe.mitre.org/data/definitions/22.html
     */

    name = FileName::safeSearchPath(global.filePath, name);
    if (!name)
    {   error("file %s cannot be found or not in a path specified with -J", se->toChars());
        goto Lerror;
    }

    if (global.params.verbose)
        printf("file      %s\t(%s)\n", (char *)se->string, name);

    {   File f(name);
        if (f.read())
        {   error("cannot read file %s", f.toChars());
            goto Lerror;
        }
        else
        {
            f.ref = 1;
            se = new StringExp(loc, f.buffer, f.len);
        }
    }
    return se->semantic(sc);

  Lerror:
    return new ErrorExp();
}

void FileExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("import(");
    expToCBuffer(buf, hgs, e1, PREC_assign);
    buf->writeByte(')');
}

/************************************************************/

AssertExp::AssertExp(Loc loc, Expression *e, Expression *msg)
        : UnaExp(loc, TOKassert, sizeof(AssertExp), e)
{
    this->msg = msg;
}

Expression *AssertExp::syntaxCopy()
{
    AssertExp *ae = new AssertExp(loc, e1->syntaxCopy(),
                                       msg ? msg->syntaxCopy() : NULL);
    return ae;
}

Expression *AssertExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("AssertExp::semantic('%s')\n", toChars());
#endif
    UnaExp::semantic(sc);
    e1 = resolveProperties(sc, e1);
    // BUG: see if we can do compile time elimination of the Assert
    e1 = e1->optimize(WANTvalue);
    e1 = e1->checkToBoolean(sc);
    if (msg)
    {
        msg = msg->semantic(sc);
        msg = resolveProperties(sc, msg);
        msg = msg->implicitCastTo(sc, Type::tchar->constOf()->arrayOf());
        msg = msg->optimize(WANTvalue);
    }
    if (e1->isBool(FALSE))
    {
        FuncDeclaration *fd = sc->parent->isFuncDeclaration();
        if (fd)
            fd->hasReturnExp |= 4;

        if (!global.params.useAssert)
        {   Expression *e = new HaltExp(loc);
            e = e->semantic(sc);
            return e;
        }
    }
    type = Type::tvoid;
    return this;
}


void AssertExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("assert(");
    expToCBuffer(buf, hgs, e1, PREC_assign);
    if (msg)
    {
        buf->writeByte(',');
        expToCBuffer(buf, hgs, msg, PREC_assign);
    }
    buf->writeByte(')');
}

/************************************************************/

DotIdExp::DotIdExp(Loc loc, Expression *e, Identifier *ident)
        : UnaExp(loc, TOKdot, sizeof(DotIdExp), e)
{
    this->ident = ident;
}

Expression *DotIdExp::semantic(Scope *sc)
{
    // Indicate we need to resolve by UFCS.
    return semantic(sc, 0);
}

Expression *DotIdExp::semantic(Scope *sc, int flag)
{   Expression *e;
    Expression *eleft;
    Expression *eright;

#if LOGSEMANTIC
    printf("DotIdExp::semantic(this = %p, '%s')\n", this, toChars());
    //printf("e1->op = %d, '%s'\n", e1->op, Token::toChars(e1->op));
#endif

//{ static int z; fflush(stdout); if (++z == 10) *(char*)0=0; }

    /* Special case: rewrite this.id and super.id
     * to be classtype.id and baseclasstype.id
     * if we have no this pointer.
     */
    if ((e1->op == TOKthis || e1->op == TOKsuper) && !hasThis(sc))
    {   ClassDeclaration *cd;
        StructDeclaration *sd;
        AggregateDeclaration *ad;

        ad = sc->getStructClassScope();
        if (ad)
        {
            cd = ad->isClassDeclaration();
            if (cd)
            {
                if (e1->op == TOKthis)
                {
                    e = typeDotIdExp(loc, cd->type, ident);
                    return e->semantic(sc);
                }
                else if (cd->baseClass && e1->op == TOKsuper)
                {
                    e = typeDotIdExp(loc, cd->baseClass->type, ident);
                    return e->semantic(sc);
                }
            }
            else
            {
                sd = ad->isStructDeclaration();
                if (sd)
                {
                    if (e1->op == TOKthis)
                    {
                        e = typeDotIdExp(loc, sd->type, ident);
                        return e->semantic(sc);
                    }
                }
            }
        }
    }

//    Type *t1save = e1->type;

    UnaExp::semantic(sc);

    if (ident == Id::mangleof)
    {   // symbol.mangleof
        Dsymbol *ds;
        switch (e1->op)
        {
            case TOKimport: ds = ((ScopeExp *)e1)->sds;     goto L1;
            case TOKvar:    ds = ((VarExp *)e1)->var;       goto L1;
            case TOKdotvar: ds = ((DotVarExp *)e1)->var;    goto L1;
            default: break;
        L1:
                char* s = ds->mangle();
                e = new StringExp(loc, s, strlen(s), 'c');
                e = e->semantic(sc);
                return e;
        }
    }

    if (e1->op == TOKdotexp)
    {
        DotExp *de = (DotExp *)e1;
        eleft = de->e1;
        eright = de->e2;
    }
    else
    {
        e1 = resolveProperties(sc, e1);
        eleft = NULL;
        eright = e1;
    }
#if DMDV2
    if (e1->op == TOKtuple && ident == Id::offsetof)
    {   /* 'distribute' the .offsetof to each of the tuple elements.
         */
        TupleExp *te = (TupleExp *)e1;
        Expressions *exps = new Expressions();
        exps->setDim(te->exps->dim);
        for (size_t i = 0; i < exps->dim; i++)
        {   Expression *e = (*te->exps)[i];
            e = e->semantic(sc);
            e = new DotIdExp(e->loc, e, Id::offsetof);
            (*exps)[i] = e;
        }
        e = new TupleExp(loc, exps);
        e = e->semantic(sc);
        return e;
    }
#endif

    if (e1->op == TOKtuple && ident == Id::length)
    {
        TupleExp *te = (TupleExp *)e1;
        e = new IntegerExp(loc, te->exps->dim, Type::tsize_t);
        return e;
    }

    if (e1->op == TOKdottd)
    {
        error("template %s does not have property %s", e1->toChars(), ident->toChars());
        return new ErrorExp();
    }

    if (!e1->type)
    {
        error("expression %s does not have property %s", e1->toChars(), ident->toChars());
        return new ErrorExp();
    }

    Type *t1b = e1->type->toBasetype();

    if (eright->op == TOKimport)        // also used for template alias's
    {
        ScopeExp *ie = (ScopeExp *)eright;

        /* Disable access to another module's private imports.
         * The check for 'is sds our current module' is because
         * the current module should have access to its own imports.
         */
        Dsymbol *s = ie->sds->search(loc, ident,
            (ie->sds->isModule() && ie->sds != sc->module) ? 1 : 0);
        if (s)
        {
            /* Check for access before resolving aliases because public
             * aliases to private symbols are public.
             */
            if (Declaration *d = s->isDeclaration())
                accessCheck(loc, sc, 0, d);

            s = s->toAlias();
            checkDeprecated(sc, s);

            EnumMember *em = s->isEnumMember();
            if (em)
            {
                e = em->value;
                e = e->semantic(sc);
                return e;
            }

            VarDeclaration *v = s->isVarDeclaration();
            if (v)
            {
                //printf("DotIdExp:: Identifier '%s' is a variable, type '%s'\n", toChars(), v->type->toChars());
                if (v->inuse)
                {
                    error("circular reference to '%s'", v->toChars());
                    return new ErrorExp();
                }
                type = v->type;
                if (v->needThis())
                {
                    if (!eleft)
                        eleft = new ThisExp(loc);
                    e = new DotVarExp(loc, eleft, v);
                    e = e->semantic(sc);
                }
                else
                {
                    e = new VarExp(loc, v);
                    if (eleft)
                    {   e = new CommaExp(loc, eleft, e);
                        e->type = v->type;
                    }
                }
                e = e->deref();
                return e->semantic(sc);
            }

            FuncDeclaration *f = s->isFuncDeclaration();
            if (f)
            {
                //printf("it's a function\n");
                if (f->needThis())
                {
                    if (!eleft)
                        eleft = new ThisExp(loc);
                    e = new DotVarExp(loc, eleft, f);
                    e = e->semantic(sc);
                }
                else
                {
                    e = new VarExp(loc, f, 1);
                    if (eleft)
                    {   e = new CommaExp(loc, eleft, e);
                        e->type = f->type;
                    }
                }
                return e;
            }
#if DMDV2
            OverloadSet *o = s->isOverloadSet();
            if (o)
            {   //printf("'%s' is an overload set\n", o->toChars());
                return new OverExp(o);
            }
#endif

            Type *t = s->getType();
            if (t)
            {
                return new TypeExp(loc, t);
            }

            TupleDeclaration *tup = s->isTupleDeclaration();
            if (tup)
            {
                if (eleft)
                {   error("cannot have e.tuple");
                    return new ErrorExp();
                }
                e = new TupleExp(loc, tup);
                e = e->semantic(sc);
                return e;
            }

            ScopeDsymbol *sds = s->isScopeDsymbol();
            if (sds)
            {
                //printf("it's a ScopeDsymbol\n");
                e = new ScopeExp(loc, sds);
                e = e->semantic(sc);
                if (eleft)
                    e = new DotExp(loc, eleft, e);
                return e;
            }

            Import *imp = s->isImport();
            if (imp)
            {
                ScopeExp *ie;

                ie = new ScopeExp(loc, imp->pkg);
                return ie->semantic(sc);
            }

            // BUG: handle other cases like in IdentifierExp::semantic()
#ifdef DEBUG
            printf("s = '%s', kind = '%s'\n", s->toChars(), s->kind());
#endif
            assert(0);
        }
        else if (ident == Id::stringof)
        {   char *s = ie->toChars();
            e = new StringExp(loc, s, strlen(s), 'c');
            e = e->semantic(sc);
            return e;
        }
        s = ie->sds->search_correct(ident);
        if (s)
            error("undefined identifier '%s', did you mean '%s %s'?",
                  ident->toChars(), s->kind(), s->toChars());
        else
            error("undefined identifier '%s'", ident->toChars());
        return new ErrorExp();
    }
    else if (t1b->ty == Tpointer && e1->type->ty != Tenum &&
             ident != Id::init && ident != Id::__sizeof &&
             ident != Id::__xalignof && ident != Id::offsetof &&
             ident != Id::mangleof && ident != Id::stringof)
    {   /* Rewrite:
         *   p.ident
         * as:
         *   (*p).ident
         */
        e = new PtrExp(loc, e1);
        e->type = ((TypePointer *)t1b)->next;
        return e->type->dotExp(sc, e, ident);
    }
#if DMDV2
    else if (!flag)
    {   /* If ident is not a valid property, rewrite:
         *   e1.ident
         * as:
         *   .ident(e1)
         */
        if (e1->op == TOKtype ||
            t1b->ty == Tvoid ||
            (t1b->ty == Tarray || t1b->ty == Tsarray || t1b->ty == Taarray) &&
            (ident == Id::sort || ident == Id::reverse || ident == Id::dup || ident == Id::idup))
        {   goto L2;
        }

        /* This would be much better if we added a "hasProperty" method to types,
         * i.e. the gagging is a bad way.
         */

        if (t1b->ty == Taarray)
        {
            TypeAArray *taa = (TypeAArray *)t1b;
            if (!taa->impl &&
                ident != Id::__sizeof &&
                ident != Id::__xalignof &&
                ident != Id::init &&
                ident != Id::mangleof &&
                ident != Id::stringof &&
                ident != Id::offsetof)
            {
                // Find out about these errors when not gagged
                taa->getImpl();
            }
        }

        Type *t1 = e1->type;
        unsigned errors = global.startGagging();
        e = t1->dotExp(sc, e1, ident);
        if (global.endGagging(errors))  // if failed to find the property
        {
            e1->type = t1;              // kludge to restore type
            errors = global.startGagging();
            e = resolveUFCSProperties(sc, this);
            if (global.endGagging(errors))
            {
                // both lookups failed, lookup property again for better error message
                e1->type = t1;  // restore type
                e = t1->dotExp(sc, e1, ident);
            }
        }
        e = e->semantic(sc);
        return e;
    }
#endif
    else
    {
    L2:
        e = e1->type->dotExp(sc, e1, ident);
        e = e->semantic(sc);
        return e;
    }
}

void DotIdExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    //printf("DotIdExp::toCBuffer()\n");
    expToCBuffer(buf, hgs, e1, PREC_primary);
    buf->writeByte('.');
    buf->writestring(ident->toChars());
}

/********************** DotTemplateExp ***********************************/

// Mainly just a placeholder

DotTemplateExp::DotTemplateExp(Loc loc, Expression *e, TemplateDeclaration *td)
        : UnaExp(loc, TOKdottd, sizeof(DotTemplateExp), e)

{
    this->td = td;
}

void DotTemplateExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    expToCBuffer(buf, hgs, e1, PREC_primary);
    buf->writeByte('.');
    buf->writestring(td->toChars());
}


/************************************************************/

DotVarExp::DotVarExp(Loc loc, Expression *e, Declaration *v, int hasOverloads)
        : UnaExp(loc, TOKdotvar, sizeof(DotVarExp), e)
{
    //printf("DotVarExp()\n");
    this->var = v;
    this->hasOverloads = hasOverloads;
}

Expression *DotVarExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("DotVarExp::semantic('%s')\n", toChars());
#endif
    if (!type)
    {
        var = var->toAlias()->isDeclaration();

        TupleDeclaration *tup = var->isTupleDeclaration();
        if (tup)
        {   /* Replace:
             *  e1.tuple(a, b, c)
             * with:
             *  tuple(e1.a, e1.b, e1.c)
             */
            Expressions *exps = new Expressions;
            Expression *ev = e1;

            exps->reserve(tup->objects->dim);
            for (size_t i = 0; i < tup->objects->dim; i++)
            {   Object *o = (*tup->objects)[i];
                Expression *e;
                if (o->dyncast() == DYNCAST_EXPRESSION)
                {
                    e = (Expression *)o;
                    if (e->op == TOKdsymbol)
                    {
                        Dsymbol *s = ((DsymbolExp *)e)->s;
                        if (i == 0 && sc->func && tup->objects->dim > 1 &&
                            e1->hasSideEffect())
                        {
                            Identifier *id = Lexer::uniqueId("__tup");
                            ExpInitializer *ei = new ExpInitializer(e1->loc, e1);
                            VarDeclaration *v = new VarDeclaration(e1->loc, NULL, id, ei);
                            v->storage_class |= STCctfe | STCref | STCforeach;

                            ev = new VarExp(e->loc, v);
                            e = new CommaExp(e1->loc, new DeclarationExp(e1->loc, v), ev);
                            e = new DotVarExp(loc, e, s->isDeclaration());
                        }
                        else
                            e = new DotVarExp(loc, ev, s->isDeclaration());
                    }
                }
                else if (o->dyncast() == DYNCAST_DSYMBOL)
                {
                    e = new DsymbolExp(loc, (Dsymbol *)o);
                }
                else if (o->dyncast() == DYNCAST_TYPE)
                {
                    e = new TypeExp(loc, (Type *)o);
                }
                else
                {
                    error("%s is not an expression", o->toChars());
                    goto Lerr;
                }
                exps->push(e);
            }
            Expression *e = new TupleExp(loc, exps);
            e = e->semantic(sc);
            return e;
        }

        e1 = e1->semantic(sc);
        e1 = e1->addDtorHook(sc);

        Type *t1 = e1->type;
        FuncDeclaration *f = var->isFuncDeclaration();
        if (f)  // for functions, do checks after overload resolution
        {
            //printf("L%d fd = %s\n", __LINE__, f->toChars());
            if (!f->functionSemantic())
                return new ErrorExp();

            type = f->type;
            assert(type);
        }
        else
        {
            type = var->type;
            if (!type && global.errors)
            {   // var is goofed up, just return 0
                goto Lerr;
            }
            assert(type);

            if (t1->ty == Tpointer)
                t1 = t1->nextOf();

            type = type->addMod(t1->mod);

            Dsymbol *vparent = var->toParent();
            AggregateDeclaration *ad = vparent ? vparent->isAggregateDeclaration() : NULL;
            e1 = getRightThis(loc, sc, ad, e1, var);
            if (!sc->noaccesscheck)
                accessCheck(loc, sc, e1, var);

            VarDeclaration *v = var->isVarDeclaration();
            Expression *e = expandVar(WANTvalue, v);
            if (e)
                return e;
        }
        Dsymbol *s;
        if (sc->func && !sc->intypeof && t1->hasPointers() &&
            (s = t1->toDsymbol(sc)) != NULL)
        {
            AggregateDeclaration *ad = s->isAggregateDeclaration();
            if (ad && ad->hasUnions)
            {
                if (sc->func->setUnsafe())
                {   error("union %s containing pointers are not allowed in @safe functions", t1->toChars());
                    goto Lerr;
                }
            }
        }
    }

    //printf("-DotVarExp::semantic('%s')\n", toChars());
    return this;

Lerr:
    return new ErrorExp();
}

int DotVarExp::isLvalue()
{
    return 1;
}

Expression *DotVarExp::toLvalue(Scope *sc, Expression *e)
{
    //printf("DotVarExp::toLvalue(%s)\n", toChars());
    return this;
}

/***********************************************
 * Mark variable v as modified if it is inside a constructor that var
 * is a field in.
 */
int modifyFieldVar(Loc loc, Scope *sc, VarDeclaration *var, Expression *e1)
{
    //printf("modifyFieldVar(var = %s)\n", var->toChars());
    Dsymbol *s = sc->func;
    while (1)
    {
        FuncDeclaration *fd = NULL;
        if (s)
            fd = s->isFuncDeclaration();
        if (fd &&
            ((fd->isCtorDeclaration() && var->storage_class & STCfield) ||
             (fd->isStaticCtorDeclaration() && !(var->storage_class & STCfield))) &&
            fd->toParent2() == var->toParent2() &&
            (!e1 || e1->op == TOKthis)
           )
        {
            var->ctorinit = 1;
            //printf("setting ctorinit\n");
            return TRUE;
        }
        else
        {
            if (s)
            {   s = s->toParent2();
                continue;
            }
        }
        break;
    }
    return FALSE;
}

int DotVarExp::checkModifiable(Scope *sc, int flag)
{
    //printf("DotVarExp::checkModifiable %s %s\n", toChars(), type->toChars());
    if (e1->op == TOKthis)
        return var->checkModify(loc, sc, type, e1, flag);

    //printf("\te1 = %s\n", e1->toChars());
    return e1->checkModifiable(sc, flag);
}

Expression *DotVarExp::modifiableLvalue(Scope *sc, Expression *e)
{
#if 0
    printf("DotVarExp::modifiableLvalue(%s)\n", toChars());
    printf("e1->type = %s\n", e1->type->toChars());
    printf("var->type = %s\n", var->type->toChars());
#endif

    return Expression::modifiableLvalue(sc, e);
}

void DotVarExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    expToCBuffer(buf, hgs, e1, PREC_primary);
    buf->writeByte('.');
    buf->writestring(var->toChars());
}

/************************************************************/

/* Things like:
 *      foo.bar!(args)
 */

DotTemplateInstanceExp::DotTemplateInstanceExp(Loc loc, Expression *e, Identifier *name, Objects *tiargs)
        : UnaExp(loc, TOKdotti, sizeof(DotTemplateInstanceExp), e)
{
    //printf("DotTemplateInstanceExp()\n");
    this->ti = new TemplateInstance(loc, name);
    this->ti->tiargs = tiargs;
}

Expression *DotTemplateInstanceExp::syntaxCopy()
{
    DotTemplateInstanceExp *de = new DotTemplateInstanceExp(loc,
        e1->syntaxCopy(),
        ti->name,
        TemplateInstance::arraySyntaxCopy(ti->tiargs));
    return de;
}

TemplateDeclaration *DotTemplateInstanceExp::getTempdecl(Scope *sc)
{
#if LOGSEMANTIC
    printf("DotTemplateInstanceExp::getTempdecl('%s')\n", toChars());
#endif
    if (!ti->tempdecl)
    {
        Expression *e = new DotIdExp(loc, e1, ti->name);
        e = e->semantic(sc);
        if (e->op == TOKdottd)
        {
            DotTemplateExp *dte = (DotTemplateExp *)e;
            ti->tempdecl = dte->td;
        }
        else if (e->op == TOKimport)
        {   ScopeExp *se = (ScopeExp *)e;
            ti->tempdecl = se->sds->isTemplateDeclaration();
        }
    }
    return ti->tempdecl;
}

Expression *DotTemplateInstanceExp::semantic(Scope *sc)
{
    // Indicate we need to resolve by UFCS.
    return semantic(sc, 0);
}
Expression *DotTemplateInstanceExp::semantic(Scope *sc, int flag)
{
#if LOGSEMANTIC
    printf("DotTemplateInstanceExp::semantic('%s')\n", toChars());
#endif

    UnaExp::semantic(sc);
    if (e1->op == TOKerror)
        return e1;

    Expression *e;
    DotIdExp *die = new DotIdExp(loc, e1, ti->name);

    if (flag || !e1->type || e1->op == TOKtype ||
        (e1->op == TOKimport && ((ScopeExp *)e1)->sds->isModule()))
    {
        e = die->semantic(sc, 1);
    }
    else
    {
        Type *t1b = e1->type->toBasetype();
        if (t1b->ty == Tarray || t1b->ty == Tsarray || t1b->ty == Taarray ||
            t1b->ty == Tnull  || (t1b->isTypeBasic() && t1b->ty != Tvoid))
        {
            /* No built-in type has templatized property, so can short cut.
             */
            return resolveUFCSProperties(sc, this);
        }

        unsigned errors = global.startGagging();
        e = die->semantic(sc, 1);
        Type *t = e1->type;
        if (global.endGagging(errors))
        {
            errors = global.startGagging();
            e = resolveUFCSProperties(sc, this);
            if (!global.endGagging(errors))
                return e;

            // both lookups failed, lookup property again for better error message
            e->type = t;  // restore type
            e = die->semantic(sc, 1);
        }
    }

L1:
    if (e->op == TOKerror)
        return e;
    if (e->op == TOKdottd)
    {
        if (ti->errors)
            return new ErrorExp();
        DotTemplateExp *dte = (DotTemplateExp *)e;
        TemplateDeclaration *td = dte->td;
        Expression *eleft = dte->e1;
        ti->tempdecl = td;
        if (ti->needsTypeInference(sc))
        {
            e1 = eleft;                 // save result of semantic()
            return this;
        }
        else
            ti->semantic(sc);
        if (!ti->inst)                  // if template failed to expand
            return new ErrorExp();
        Dsymbol *s = ti->inst->toAlias();
        Declaration *v = s->isDeclaration();
        if (v)
        {
            /* Fix for Bugzilla 4003
             * The problem is a class template member function v returning a reference to the same
             * type as the enclosing template instantiation. This results in a nested instantiation,
             * which of course gets short circuited. The return type then gets set to
             * the template instance type before instantiation, rather than after.
             * We can detect this by the deco not being set. If so, go ahead and retry
             * the return type semantic.
             * The offending code is the return type from std.typecons.Tuple.slice:
             *    ref Tuple!(Types[from .. to]) slice(uint from, uint to)()
             *    {
             *        return *cast(typeof(return) *) &(field[from]);
             *    }
             * and this line from the following unittest:
             *    auto s = a.slice!(1, 3);
             * where s's type wound up not having semantic() run on it.
             */
            if (v->type && !v->type->deco)
                v->type = v->type->semantic(v->loc, sc);

            e = new DotVarExp(loc, eleft, v);
            e = e->semantic(sc);
            return e;
        }
        e = new ScopeExp(loc, ti);
        e = new DotExp(loc, eleft, e);
        e = e->semantic(sc);
        return e;
    }
    else if (e->op == TOKimport)
    {   ScopeExp *se = (ScopeExp *)e;
        TemplateDeclaration *td = se->sds->isTemplateDeclaration();
        if (!td)
        {   error("%s is not a template", e->toChars());
            return new ErrorExp();
        }
        ti->tempdecl = td;
        e = new ScopeExp(loc, ti);
        e = e->semantic(sc);
        return e;
    }
    else if (e->op == TOKdotexp)
    {   DotExp *de = (DotExp *)e;

        if (de->e2->op == TOKoverloadset)
        {
            return e;
        }

        if (de->e2->op == TOKimport)
        {   // This should *really* be moved to ScopeExp::semantic()
            ScopeExp *se = (ScopeExp *)de->e2;
            de->e2 = new DsymbolExp(loc, se->sds);
            de->e2 = de->e2->semantic(sc);
        }

        if (de->e2->op == TOKtemplate)
        {   TemplateExp *te = (TemplateExp *) de->e2;
            e = new DotTemplateExp(loc,de->e1,te->td);
        }
        else
            goto Lerr;

        e = e->semantic(sc);
        if (e == de)
            goto Lerr;
        goto L1;
    }
Lerr:
    error("%s isn't a template", e->toChars());
    return new ErrorExp();
}

void DotTemplateInstanceExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    expToCBuffer(buf, hgs, e1, PREC_primary);
    buf->writeByte('.');
    ti->toCBuffer(buf, hgs);
}

/************************************************************/

DelegateExp::DelegateExp(Loc loc, Expression *e, FuncDeclaration *f, int hasOverloads)
        : UnaExp(loc, TOKdelegate, sizeof(DelegateExp), e)
{
    this->func = f;
    this->hasOverloads = hasOverloads;
}

Expression *DelegateExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("DelegateExp::semantic('%s')\n", toChars());
#endif
    if (!type)
    {
        e1 = e1->semantic(sc);
#if IN_LLVM
        // LDC we need a copy as we store the LLVM type in TypeFunction,
        // and delegate/members have different types for 'this'
        Type *funcType = func->type->syntaxCopy();
        funcType->deco = func->type->deco;
        type = new TypeDelegate(funcType);
#else
        type = new TypeDelegate(func->type);
#endif
        type = type->semantic(loc, sc);
        AggregateDeclaration *ad = func->toParent()->isAggregateDeclaration();
        if (func->needThis())
            e1 = getRightThis(loc, sc, ad, e1, func);
        if (ad && ad->isClassDeclaration() && ad->type != e1->type)
        {   // A downcast is required for interfaces, see Bugzilla 3706
            e1 = new CastExp(loc, e1, ad->type);
            e1 = e1->semantic(sc);
        }
    }
    return this;
}

void DelegateExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writeByte('&');
    if (!func->isNested())
    {
        expToCBuffer(buf, hgs, e1, PREC_primary);
        buf->writeByte('.');
    }
    buf->writestring(func->toChars());
}

/************************************************************/

DotTypeExp::DotTypeExp(Loc loc, Expression *e, Dsymbol *s)
        : UnaExp(loc, TOKdottype, sizeof(DotTypeExp), e)
{
    this->sym = s;
    this->type = s->getType();
}

Expression *DotTypeExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("DotTypeExp::semantic('%s')\n", toChars());
#endif
    UnaExp::semantic(sc);
    return this;
}

void DotTypeExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    expToCBuffer(buf, hgs, e1, PREC_primary);
    buf->writeByte('.');
    buf->writestring(sym->toChars());
}

/************************************************************/

CallExp::CallExp(Loc loc, Expression *e, Expressions *exps)
        : UnaExp(loc, TOKcall, sizeof(CallExp), e)
{
    this->arguments = exps;
    this->f = NULL;
}

CallExp::CallExp(Loc loc, Expression *e)
        : UnaExp(loc, TOKcall, sizeof(CallExp), e)
{
    this->arguments = NULL;
}

CallExp::CallExp(Loc loc, Expression *e, Expression *earg1)
        : UnaExp(loc, TOKcall, sizeof(CallExp), e)
{
    Expressions *arguments = new Expressions();
    if (earg1)
    {   arguments->setDim(1);
        (*arguments)[0] = earg1;
    }
    this->arguments = arguments;
}

CallExp::CallExp(Loc loc, Expression *e, Expression *earg1, Expression *earg2)
        : UnaExp(loc, TOKcall, sizeof(CallExp), e)
{
    Expressions *arguments = new Expressions();
    arguments->setDim(2);
    (*arguments)[0] = earg1;
    (*arguments)[1] = earg2;

    this->arguments = arguments;
}

Expression *CallExp::syntaxCopy()
{
    return new CallExp(loc, e1->syntaxCopy(), arraySyntaxCopy(arguments));
}


Expression *CallExp::resolveUFCS(Scope *sc)
{
    Expression *e;
    Identifier *ident;
    Objects *tiargs;

    if (e1->op == TOKdot)
    {
        DotIdExp *die = (DotIdExp *)e1;
        e      = (die->e1 = die->e1->semantic(sc));
        ident  = die->ident;
        tiargs = NULL;
    }
    else if (e1->op == TOKdotti)
    {
        DotTemplateInstanceExp *dti = (DotTemplateInstanceExp *)e1;
        e      = (dti->e1 = dti->e1->semantic(sc));
        ident  = dti->ti->name;
        tiargs = dti->ti->tiargs;
    }
    else
        return NULL;

    if (e->op == TOKerror || !e->type)
        return NULL;

    if (e->op == TOKtype || e->op == TOKimport || e->op == TOKdotexp)
        return NULL;

    e = resolveProperties(sc, e);

    Type *t = e->type->toBasetype();
    //printf("resolveUCSS %s, e = %s, %s, %s\n",
    //    toChars(), Token::toChars(e->op), t->toChars(), e->toChars());
    if (t->ty == Taarray)
    {
        if (tiargs)
        {
            goto Lshift;
        }
        else if (ident == Id::remove)
        {
            /* Transform:
             *  aa.remove(arg) into delete aa[arg]
             */
            if (!arguments || arguments->dim != 1)
            {   error("expected key as argument to aa.remove()");
                return new ErrorExp();
            }
            if (!e->type->isMutable())
            {   error("cannot remove key from %s associative array %s", MODtoChars(e->type->mod), e->toChars());
                return new ErrorExp();
            }
            Expression *key = (*arguments)[0];
            key = key->semantic(sc);
            key = resolveProperties(sc, key);

            TypeAArray *taa = (TypeAArray *)t;
            key = key->implicitCastTo(sc, taa->index);

            if (!key->rvalue())
                return new ErrorExp();

            return new RemoveExp(loc, e, key);
        }
        else if (ident == Id::apply || ident == Id::applyReverse)
        {
            return NULL;
        }
        else
        {   TypeAArray *taa = (TypeAArray *)t;
            assert(taa->ty == Taarray);
            StructDeclaration *sd = taa->getImpl();
            Dsymbol *s = sd->search(0, ident, 2);
            if (s)
                return NULL;
            goto Lshift;
        }
    }
    else if (t->ty == Tarray || t->ty == Tsarray ||
             t->ty == Tnull  || (t->isTypeBasic() && t->ty != Tvoid))
    {
        /* In basic, built-in types don't have normal and templatized
         * member functions. So can short cut.
         */
Lshift:
        if (!arguments)
            arguments = new Expressions();
        arguments->shift(e);
        if (!tiargs)
        {
            /* Transform:
             *  array.id(args) into .id(array,args)
             */
            e1 = new DotIdExp(e1->loc,
                              new IdentifierExp(e1->loc, Id::empty),
                              ident);
        }
        else
        {
            /* Transform:
             *  array.foo!(tiargs)(args) into .foo!(tiargs)(array,args)
             */
            e1 = new DotTemplateInstanceExp(e1->loc,
                            new IdentifierExp(e1->loc, Id::empty),
                            ident, tiargs);
        }
    }
    else
    {
        DotIdExp *die = new DotIdExp(e->loc, e, ident);

        unsigned errors = global.startGagging();
        Expression *ex = die->semantic(sc, 1);
        if (global.endGagging(errors))
        {
            goto Lshift;
        }
    }
    return NULL;
}

Expression *CallExp::semantic(Scope *sc)
{
    Type *t1;
    int istemp;
    Objects *targsi = NULL;     // initial list of template arguments
    TemplateInstance *tierror = NULL;
    Expression *ethis = NULL;

#if LOGSEMANTIC
    printf("CallExp::semantic() %s\n", toChars());
#endif
    if (type)
        return this;            // semantic() already run
#if 0
    if (arguments && arguments->dim)
    {
        Expression *earg = (*arguments)[0];
        earg->print();
        if (earg->type) earg->type->print();
    }
#endif

    if (e1->op == TOKcomma)
    {   /* Rewrite (a,b)(args) as (a,(b(args)))
         */
        CommaExp *ce = (CommaExp *)e1;

        e1 = ce->e2;
        e1->type = ce->type;
        ce->e2 = this;
        ce->type = NULL;
        return ce->semantic(sc);
    }

    if (e1->op == TOKdelegate)
    {   DelegateExp *de = (DelegateExp *)e1;

        e1 = new DotVarExp(de->loc, de->e1, de->func);
        return semantic(sc);
    }

    if (e1->op == TOKfunction)
    {   FuncExp *fe = (FuncExp *)e1;

        arguments = arrayExpressionSemantic(arguments, sc);
        preFunctionParameters(loc, sc, arguments);
        e1 = fe->semantic(sc, arguments);
        if (e1->op == TOKerror)
            return e1;
    }

    Expression *e = resolveUFCS(sc);
    if (e)
        return e;

    /* This recognizes:
     *  foo!(tiargs)(funcargs)
     */
    if (e1->op == TOKimport && !e1->type)
    {   ScopeExp *se = (ScopeExp *)e1;
        TemplateInstance *ti = se->sds->isTemplateInstance();
        if (ti && !ti->semanticRun)
        {
            /* Attempt to instantiate ti. If that works, go with it.
             * If not, go with partial explicit specialization.
             */
            unsigned olderrors = global.errors;
            ti->semanticTiargs(sc);
            if (olderrors != global.errors)
                return new ErrorExp();
            if (ti->needsTypeInference(sc))
            {
                /* Go with partial explicit specialization
                 */
                targsi = ti->tiargs;
                tierror = ti;                   // for error reporting
                e1 = new IdentifierExp(loc, ti->name);
            }
            else
            {
                ti->semantic(sc);
            }
        }
    }

    /* This recognizes:
     *  expr.foo!(tiargs)(funcargs)
     */
Ldotti:
    if (e1->op == TOKdotti && !e1->type)
    {   DotTemplateInstanceExp *se = (DotTemplateInstanceExp *)e1;
        TemplateInstance *ti = se->ti;
        if (!ti->semanticRun)
        {
            /* Attempt to instantiate ti. If that works, go with it.
             * If not, go with partial explicit specialization.
             */
            ti->semanticTiargs(sc);
            if (!ti->tempdecl)
            {
                se->getTempdecl(sc);
            }
            if (ti->tempdecl && ti->needsTypeInference(sc))
            {
                /* Go with partial explicit specialization
                 */
                targsi = ti->tiargs;
                tierror = ti;                   // for error reporting
                e1 = new DotIdExp(loc, se->e1, ti->name);
            }
            else
            {
                e1 = e1->semantic(sc);
            }
        }
    }

    istemp = 0;
Lagain:
    //printf("Lagain: %s\n", toChars());
    f = NULL;
    if (e1->op == TOKthis || e1->op == TOKsuper)
    {
        // semantic() run later for these
    }
    else
    {
        if (e1->op == TOKdot)
        {   DotIdExp *die = (DotIdExp *)e1;
            e1 = die->semantic(sc);
            /* Look for e1 having been rewritten to expr.opDispatch!(string)
             * We handle such earlier, so go back.
             * Note that in the rewrite, we carefully did not run semantic() on e1
             */
            if (e1->op == TOKdotti && !e1->type)
            {
                goto Ldotti;
            }
        }
        else
        {
            static int nest;
            if (++nest > 500)
            {
                error("recursive evaluation of %s", toChars());
                --nest;
                return new ErrorExp();
            }
            UnaExp::semantic(sc);
            --nest;
        }

        /* Look for e1 being a lazy parameter
         */
        if (e1->op == TOKvar)
        {   VarExp *ve = (VarExp *)e1;

            if (ve->var->storage_class & STClazy)
            {
                // lazy paramaters can be called without violating purity and safety
                Type *tw = ve->var->type;
                Type *tc = ve->var->type->substWildTo(MODconst);
                TypeFunction *tf = new TypeFunction(NULL, tc, 0, LINKd, STCsafe | STCpure);
                (tf = (TypeFunction *)tf->semantic(loc, sc))->next = tw;    // hack for bug7757
                TypeDelegate *t = new TypeDelegate(tf);
                ve->type = t->semantic(loc, sc);
            }
        }

        if (e1->op == TOKimport)
        {   // Perhaps this should be moved to ScopeExp::semantic()
            ScopeExp *se = (ScopeExp *)e1;
            e1 = new DsymbolExp(loc, se->sds);
            e1 = e1->semantic(sc);
        }
        else if (e1->op == TOKsymoff && ((SymOffExp *)e1)->hasOverloads)
        {
            SymOffExp *se = (SymOffExp *)e1;
            e1 = new VarExp(se->loc, se->var, 1);
            e1 = e1->semantic(sc);
        }
        else if (e1->op == TOKdotexp)
        {
            DotExp *de = (DotExp *) e1;

            if (de->e2->op == TOKoverloadset)
            {
                ethis = de->e1;
                e1 = de->e2;
            }

            if (de->e2->op == TOKimport)
            {   // This should *really* be moved to ScopeExp::semantic()
                ScopeExp *se = (ScopeExp *)de->e2;
                de->e2 = new DsymbolExp(loc, se->sds);
                de->e2 = de->e2->semantic(sc);
            }

            if (de->e2->op == TOKtemplate)
            {   TemplateExp *te = (TemplateExp *) de->e2;
                e1 = new DotTemplateExp(loc,de->e1,te->td);
            }
        }
    }

    t1 = NULL;
    if (e1->type)
        t1 = e1->type->toBasetype();

    // Check for call operator overload
    if (t1)
    {   AggregateDeclaration *ad;

        if (t1->ty == Tstruct)
        {
            ad = ((TypeStruct *)t1)->sym;
#if DMDV2

            if (ad->sizeok == SIZEOKnone && !ad->ctor &&
                ad->search(0, Id::ctor, 0))
            {
                // The constructor hasn't been found yet, see bug 8741
                // This can happen if we are inferring type from
                // from VarDeclaration::semantic() in declaration.c
                error("cannot create a struct until its size is determined");
                return new ErrorExp();
            }

            // First look for constructor
            if (e1->op == TOKtype && ad->ctor && (ad->noDefaultCtor || arguments && arguments->dim))
            {
                // Create variable that will get constructed
                Identifier *idtmp = Lexer::uniqueId("__ctmp");

                ExpInitializer *ei = NULL;
                if (t1->needsNested())
                {
                    Expressions *args = new Expressions();
                    StructLiteralExp *se = new StructLiteralExp(loc, (StructDeclaration *)ad, args);
                    se->ctorinit = 1;
                    ei = new ExpInitializer(loc, se);
                }

                VarDeclaration *tmp = new VarDeclaration(loc, t1, idtmp, ei);
                tmp->storage_class |= STCctfe;
                Expression *av = new DeclarationExp(loc, tmp);
                av = new CommaExp(loc, av, new VarExp(loc, tmp));

                Expression *e;
                CtorDeclaration *cf = ad->ctor->isCtorDeclaration();
                if (cf)
                    e = new DotVarExp(loc, av, cf, 1);
                else
                {   TemplateDeclaration *td = ad->ctor->isTemplateDeclaration();
                    assert(td);
                    e = new DotTemplateExp(loc, av, td);
                }
                e = new CallExp(loc, e, arguments);
                e = e->semantic(sc);
                return e;
            }
#endif
            // No constructor, look for overload of opCall
            if (search_function(ad, Id::call))
                goto L1;        // overload of opCall, therefore it's a call

            if (e1->op != TOKtype)
            {
                if (ad->aliasthis)
                {
                    e1 = resolveAliasThis(sc, e1);
                    goto Lagain;
                }
                error("%s %s does not overload ()", ad->kind(), ad->toChars());
                return new ErrorExp();
            }

            /* It's a struct literal
             */
            Expression *e = new StructLiteralExp(loc, (StructDeclaration *)ad, arguments, e1->type);
            e = e->semantic(sc);
            return e;
        }
        else if (t1->ty == Tclass)
        {
            ad = ((TypeClass *)t1)->sym;
            goto L1;
        L1:
            // Rewrite as e1.call(arguments)
            Expression *e = new DotIdExp(loc, e1, Id::call);
            e = new CallExp(loc, e, arguments);
            e = e->semantic(sc);
            return e;
        }
    }

    arguments = arrayExpressionSemantic(arguments, sc);
    preFunctionParameters(loc, sc, arguments);

    // If there was an error processing any argument, or the call,
    // return an error without trying to resolve the function call.
    if (arguments && arguments->dim)
    {
        for (size_t k = 0; k < arguments->dim; k++)
        {   Expression *checkarg = (*arguments)[k];
            if (checkarg->op == TOKerror)
                return checkarg;
        }
    }
    if (e1->op == TOKerror)
        return e1;

    // If there was an error processing any template argument,
    // return an error without trying to resolve the template.
    if (targsi && targsi->dim)
    {
        for (size_t k = 0; k < targsi->dim; k++)
        {   Object *o = (*targsi)[k];
            if (isError(o))
                return new ErrorExp();
        }
    }

    if (e1->op == TOKdotvar && t1->ty == Tfunction ||
        e1->op == TOKdottd)
    {
        DotVarExp *dve;
        DotTemplateExp *dte;
        AggregateDeclaration *ad;
        UnaExp *ue = (UnaExp *)(e1);

        Expression *ue1 = ue->e1;
        VarDeclaration *v;
        if (ue1->op == TOKvar &&
            (v = ((VarExp *)ue1)->var->isVarDeclaration()) != NULL &&
            v->needThis())
        {
            ue->e1 = new TypeExp(ue1->loc, ue1->type);
            ue1 = NULL;
        }

        if (e1->op == TOKdotvar)
        {   // Do overload resolution
            dve = (DotVarExp *)(e1);

            f = dve->var->isFuncDeclaration();
            assert(f);
            f = f->overloadResolve(loc, ue1, arguments);

            ad = f->toParent()->isAggregateDeclaration();
        }
        else
        {   dte = (DotTemplateExp *)(e1);
            TemplateDeclaration *td = dte->td;
            assert(td);
            if (!arguments)
                // Should fix deduceFunctionTemplate() so it works on NULL argument
                arguments = new Expressions();
            f = td->deduceFunctionTemplate(sc, loc, targsi, ue1, arguments);
            if (!f)
                return new ErrorExp();
            ad = td->toParent()->isAggregateDeclaration();
        }
        if (f->needThis())
        {
            ue->e1 = getRightThis(loc, sc, ad, ue->e1, f);
            if (ue->e1->op == TOKerror)
                return ue->e1;
            ethis = ue->e1;
        }

        /* Cannot call public functions from inside invariant
         * (because then the invariant would have infinite recursion)
         */
        if (sc->func && sc->func->isInvariantDeclaration() &&
            ue->e1->op == TOKthis &&
            f->addPostInvariant()
           )
        {
            error("cannot call public/export function %s from invariant", f->toChars());
            return new ErrorExp();
        }

        checkDeprecated(sc, f);
#if DMDV2
        checkPurity(sc, f);
        checkSafety(sc, f);
#endif
        accessCheck(loc, sc, ue->e1, f);
        if (!f->needThis())
        {
            VarExp *ve = new VarExp(loc, f);
            if ((ue->e1)->op == TOKtype) // just a FQN
                e1 = ve;
            else // things like (new Foo).bar()
                e1 = new CommaExp(loc, ue->e1, ve);
            e1->type = f->type;
        }
        else
        {
            if (e1->op == TOKdotvar)
            {
                dve->var = f;
                e1->type = f->type;
            }
            else
            {
                e1 = new DotVarExp(loc, dte->e1, f);
                e1 = e1->semantic(sc);
                ue = (UnaExp *)e1;
            }
#if 0
            printf("ue->e1 = %s\n", ue->e1->toChars());
            printf("f = %s\n", f->toChars());
            printf("t = %s\n", t->toChars());
            printf("e1 = %s\n", e1->toChars());
            printf("e1->type = %s\n", e1->type->toChars());
#endif

            // See if we need to adjust the 'this' pointer
            AggregateDeclaration *ad = f->isThis();
            ClassDeclaration *cd = ue->e1->type->isClassHandle();
            if (ad && cd && ad->isClassDeclaration() && ad != cd &&
                ue->e1->op != TOKsuper)
            {
                ue->e1 = ue->e1->castTo(sc, ad->type); //new CastExp(loc, ue->e1, ad->type);
                ue->e1 = ue->e1->semantic(sc);
            }
        }
        t1 = e1->type;
    }
    else if (e1->op == TOKsuper)
    {
        // Base class constructor call
        ClassDeclaration *cd = NULL;

        if (sc->func && sc->func->isThis())
            cd = sc->func->isThis()->isClassDeclaration();
        if (!cd || !cd->baseClass || !sc->func->isCtorDeclaration())
        {
            error("super class constructor call must be in a constructor");
            return new ErrorExp();
        }
        else
        {
            if (!cd->baseClass->ctor)
            {   error("no super class constructor for %s", cd->baseClass->toChars());
                return new ErrorExp();
            }
            else
            {
                if (!sc->intypeof)
                {
                    if (sc->noctor || sc->callSuper & CSXlabel)
                        error("constructor calls not allowed in loops or after labels");
                    if (sc->callSuper & (CSXsuper_ctor | CSXthis_ctor))
                        error("multiple constructor calls");
                    if ((sc->callSuper & CSXreturn) && !(sc->callSuper & CSXany_ctor))
                        error("an earlier return statement skips constructor");
                    sc->callSuper |= CSXany_ctor | CSXsuper_ctor;
                }

                f = resolveFuncCall(sc, loc, cd->baseClass->ctor, NULL, NULL, arguments, 0);
                if (!f)
                    return new ErrorExp();
                accessCheck(loc, sc, NULL, f);
                checkDeprecated(sc, f);
#if DMDV2
                checkPurity(sc, f);
                checkSafety(sc, f);
#endif
                e1 = new DotVarExp(e1->loc, e1, f);
                e1 = e1->semantic(sc);
                t1 = e1->type;
            }
        }
    }
    else if (e1->op == TOKthis)
    {
        // same class constructor call
        AggregateDeclaration *cd = NULL;

        if (sc->func && sc->func->isThis())
            cd = sc->func->isThis()->isAggregateDeclaration();
        if (!cd || !sc->func->isCtorDeclaration())
        {
            error("constructor call must be in a constructor");
            return new ErrorExp();
        }
        else
        {
            if (!sc->intypeof)
            {
                if (sc->noctor || sc->callSuper & CSXlabel)
                    error("constructor calls not allowed in loops or after labels");
                if (sc->callSuper & (CSXsuper_ctor | CSXthis_ctor))
                    error("multiple constructor calls");
                if ((sc->callSuper & CSXreturn) && !(sc->callSuper & CSXany_ctor))
                    error("an earlier return statement skips constructor");
                sc->callSuper |= CSXany_ctor | CSXthis_ctor;
            }

            f = resolveFuncCall(sc, loc, cd->ctor, NULL, NULL, arguments, 0);
            if (!f)
                return new ErrorExp();
            checkDeprecated(sc, f);
#if DMDV2
            checkPurity(sc, f);
            checkSafety(sc, f);
#endif
            e1 = new DotVarExp(e1->loc, e1, f);
            e1 = e1->semantic(sc);
            t1 = e1->type;

            // BUG: this should really be done by checking the static
            // call graph
            if (f == sc->func)
            {   error("cyclic constructor call");
                return new ErrorExp();
            }
        }
    }
    else if (e1->op == TOKoverloadset)
    {
        OverExp *eo = (OverExp *)e1;
        FuncDeclaration *f = NULL;
        Dsymbol *s = NULL;
        for (size_t i = 0; i < eo->vars->a.dim; i++)
        {   s = eo->vars->a[i];
            FuncDeclaration *f2 = s->isFuncDeclaration();
            if (f2)
            {
                f2 = f2->overloadResolve(loc, ethis, arguments, 1);
            }
            else
            {   TemplateDeclaration *td = s->isTemplateDeclaration();
                assert(td);
                f2 = td->deduceFunctionTemplate(sc, loc, targsi, ethis, arguments, 1);
            }
            if (f2)
            {   if (f)
                    /* Error if match in more than one overload set,
                     * even if one is a 'better' match than the other.
                     */
                    ScopeDsymbol::multiplyDefined(loc, f, f2);
                else
                    f = f2;
            }
        }
        if (!f)
        {   /* No overload matches
             */
            error("no overload matches for %s", s->toChars());
            return new ErrorExp();
        }
        if (ethis)
            e1 = new DotVarExp(loc, ethis, f);
        else
            e1 = new VarExp(loc, f);
        goto Lagain;
    }
    else if (!t1)
    {
        error("function expected before (), not '%s'", e1->toChars());
        return new ErrorExp();
    }
    else if (t1->ty != Tfunction)
    {
        TypeFunction *tf;
        const char *p;
        if (e1->op == TOKfunction)
        {
            // function literal that direct called is always inferred.
            assert(((FuncExp *)e1)->fd);
            f = ((FuncExp *)e1)->fd;
            tf = (TypeFunction *)f->type;
            p = "function literal";

            f->checkNestedReference(sc, loc);
        }
        else if (t1->ty == Tdelegate)
        {   TypeDelegate *td = (TypeDelegate *)t1;
            assert(td->next->ty == Tfunction);
            tf = (TypeFunction *)(td->next);
            p = "delegate";
        }
        else if (t1->ty == Tpointer && ((TypePointer *)t1)->next->ty == Tfunction)
        {
            tf = (TypeFunction *)(((TypePointer *)t1)->next);
            p = "function pointer";
        }
        else if (e1->op == TOKtemplate)
        {
            TemplateExp *te = (TemplateExp *)e1;
            f = te->td->deduceFunctionTemplate(sc, loc, targsi, NULL, arguments);
            if (!f)
            {   if (tierror)
                    tierror->error("errors instantiating template");    // give better error message
                return new ErrorExp();
            }
            if (f->needThis())
            {
                if (hasThis(sc))
                {
                    // Supply an implicit 'this', as in
                    //        this.ident

                    e1 = new DotTemplateExp(loc, (new ThisExp(loc))->semantic(sc), te->td);
                    goto Lagain;
                }
                else if (!sc->intypeof && !sc->getStructClassScope())
                {
                    error("need 'this' for %s type %s", f->toChars(), f->type->toChars());
                    return new ErrorExp();
                }
            }

            e1 = new VarExp(loc, f);
            goto Lagain;
        }
        else
        {   error("function expected before (), not %s of type %s", e1->toChars(), e1->type->toChars());
            return new ErrorExp();
        }

        if (sc->func && !tf->purity && !(sc->flags & SCOPEdebug))
        {
            if (sc->func->setImpure())
                error("pure function '%s' cannot call impure %s '%s'", sc->func->toPrettyChars(), p, e1->toChars());
        }
        if (sc->func && tf->trust <= TRUSTsystem)
        {
            if (sc->func->setUnsafe())
                error("safe function '%s' cannot call system %s '%s'", sc->func->toPrettyChars(), p, e1->toChars());
        }

        if (!tf->callMatch(NULL, arguments))
        {
            OutBuffer buf;

            buf.writeByte('(');
            if (arguments)
            {
                HdrGenState hgs;

                argExpTypesToCBuffer(&buf, arguments, &hgs);
                buf.writeByte(')');
                if (ethis)
                    ethis->type->modToBuffer(&buf);
            }
            else
                buf.writeByte(')');

            //printf("tf = %s, args = %s\n", tf->deco, (*arguments)[0]->type->deco);
            ::error(loc, "%s %s %s is not callable using argument types %s",
                p, e1->toChars(), Parameter::argsTypesToChars(tf->parameters, tf->varargs),
                buf.toChars());

            return new ErrorExp();
        }

        if (t1->ty == Tpointer)
        {
            Expression *e = new PtrExp(loc, e1);
            e->type = tf;
            e1 = e;
        }
        t1 = tf;
    }
    else if (e1->op == TOKvar)
    {
        // Do overload resolution
        VarExp *ve = (VarExp *)e1;

        f = ve->var->isFuncDeclaration();
        assert(f);

        if (ve->hasOverloads)
            f = f->overloadResolve(loc, NULL, arguments, 2);
        else
        {
            TypeFunction *tf = (TypeFunction *)f->type;
            if (!tf->callMatch(NULL, arguments))
            {
                OutBuffer buf;

                buf.writeByte('(');
                if (arguments && arguments->dim)
                {
                    HdrGenState hgs;

                    argExpTypesToCBuffer(&buf, arguments, &hgs);
                }
                buf.writeByte(')');

                //printf("tf = %s, args = %s\n", tf->deco, (*arguments)[0]->type->deco);
                ::error(loc, "%s %s is not callable using argument types %s",
                    e1->toChars(), Parameter::argsTypesToChars(tf->parameters, tf->varargs),
                    buf.toChars());

                return new ErrorExp();
            }
        }

        if (f->needThis())
        {
            if (hasThis(sc))
            {
                // Supply an implicit 'this', as in
                //    this.ident

                e1 = new DotVarExp(loc, (new ThisExp(loc))->semantic(sc), ve->var);
                goto Lagain;
            }
            else if (!sc->intypeof && !sc->getStructClassScope())
            {
                error("need 'this' for %s type %s", f->toChars(), f->type->toChars());
                return new ErrorExp();
            }
        }

        checkDeprecated(sc, f);
#if DMDV2
        checkPurity(sc, f);
        checkSafety(sc, f);
#endif
        f->checkNestedReference(sc, loc);
        accessCheck(loc, sc, NULL, f);

        ethis = NULL;

        ve->var = f;
//      ve->hasOverloads = 0;
        ve->type = f->type;
        t1 = f->type;
    }
    assert(t1->ty == Tfunction);
    TypeFunction *tf = (TypeFunction *)(t1);

    if (!arguments)
        arguments = new Expressions();
    int olderrors = global.errors;
    type = functionParameters(loc, sc, tf, ethis, arguments, f);
    if (olderrors != global.errors)
        return new ErrorExp();

    if (!type)
    {
        error("forward reference to inferred return type of function call %s", toChars());
        return new ErrorExp();
    }

    if (f && f->tintro)
    {
        Type *t = type;
        int offset = 0;
        TypeFunction *tf = (TypeFunction *)f->tintro;

        if (tf->next->isBaseOf(t, &offset) && offset)
        {
            type = tf->next;
            return castTo(sc, t);
        }
    }

    return this;
}



int CallExp::isLvalue()
{
    Type *tb = e1->type->toBasetype();
    if (tb->ty == Tdelegate || tb->ty == Tpointer)
        tb = tb->nextOf();
    if (tb->ty == Tfunction && ((TypeFunction *)tb)->isref)
    {
        if (e1->op == TOKdotvar)
            if (((DotVarExp *)e1)->var->isCtorDeclaration())
                return 0;
        return 1;               // function returns a reference
    }
    return 0;
}

Expression *CallExp::toLvalue(Scope *sc, Expression *e)
{
    if (isLvalue())
        return this;
    return Expression::toLvalue(sc, e);
}

Expression *CallExp::addDtorHook(Scope *sc)
{
    /* Only need to add dtor hook if it's a type that needs destruction.
     * Use same logic as VarDeclaration::callScopeDtor()
     */

    if (e1->type && e1->type->ty == Tfunction)
    {
        TypeFunction *tf = (TypeFunction *)e1->type;
        if (tf->isref)
            return this;
    }

    Type *tv = type->toBasetype();
    while (tv->ty == Tsarray)
    {   TypeSArray *ta = (TypeSArray *)tv;
        tv = tv->nextOf()->toBasetype();
    }
    if (tv->ty == Tstruct)
    {   TypeStruct *ts = (TypeStruct *)tv;
        StructDeclaration *sd = ts->sym;
        if (sd->dtor)
        {   /* Type needs destruction, so declare a tmp
             * which the back end will recognize and call dtor on
             */
            Identifier *idtmp = Lexer::uniqueId("__tmpfordtor");
            VarDeclaration *tmp = new VarDeclaration(loc, type, idtmp, new ExpInitializer(loc, this));
            tmp->storage_class |= STCctfe;
            Expression *ae = new DeclarationExp(loc, tmp);
            Expression *e = new CommaExp(loc, ae, new VarExp(loc, tmp));
            e = e->semantic(sc);
            return e;
        }
    }
Lnone:
    return this;
}

void CallExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    if (e1->op == TOKtype)
        /* Avoid parens around type to prevent forbidden cast syntax:
         *   (sometype)(arg1)
         * This is ok since types in constructor calls
         * can never depend on parens anyway
         */
        e1->toCBuffer(buf, hgs);
    else
        expToCBuffer(buf, hgs, e1, precedence[op]);
    buf->writeByte('(');
    argsToCBuffer(buf, arguments, hgs);
    buf->writeByte(')');
}


/************************************************************/

AddrExp::AddrExp(Loc loc, Expression *e)
        : UnaExp(loc, TOKaddress, sizeof(AddrExp), e)
{
}

Expression *AddrExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("AddrExp::semantic('%s')\n", toChars());
#endif
    if (!type)
    {
        UnaExp::semantic(sc);
        Expression *olde1 = e1;
        if (e1->type == Type::terror)
            return new ErrorExp();
        int wasCond = e1->op == TOKquestion;
        e1 = e1->toLvalue(sc, NULL);
        if (e1->op == TOKerror)
            return e1;
        if (!e1->type)
        {
            error("cannot take address of %s", e1->toChars());
            return new ErrorExp();
        }
        if (!e1->type->deco)
        {
            /* No deco means semantic() was not run on the type.
             * We have to run semantic() on the symbol to get the right type:
             *  auto x = &bar;
             *  pure: int bar() { return 1;}
             * otherwise the 'pure' is missing from the type assigned to x.
             */

            if (e1->op == TOKvar)
            {
                VarExp *ve = (VarExp *)e1;
                Declaration *d = ve->var;
                error("forward reference to %s %s", d->kind(), d->toChars());
            }
            else
                error("forward reference to %s", e1->toChars());
            return new ErrorExp();
        }

        type = e1->type->pointerTo();

        // See if this should really be a delegate
        if (e1->op == TOKdotvar)
        {
            DotVarExp *dve = (DotVarExp *)e1;
            FuncDeclaration *f = dve->var->isFuncDeclaration();

            if (f)
            {
                if (!dve->hasOverloads)
                    f->tookAddressOf++;

                Expression *e;
                if ( f->needThis())
                    e = new DelegateExp(loc, dve->e1, f, dve->hasOverloads);
                else // It is a function pointer. Convert &v.f() --> (v, &V.f())
                    e = new CommaExp(loc, dve->e1, new AddrExp(loc, new VarExp(loc, f)));
                e = e->semantic(sc);
                return e;
            }
        }
        else if (e1->op == TOKvar)
        {
            VarExp *ve = (VarExp *)e1;

            VarDeclaration *v = ve->var->isVarDeclaration();
            if (v)
            {
                if (!v->canTakeAddressOf())
                {   error("cannot take address of %s", e1->toChars());
                    return new ErrorExp();
                }

                if (sc->func && !sc->intypeof && !v->isDataseg())
                {
                    if (sc->func->setUnsafe())
                    {
                        error("cannot take address of %s %s in @safe function %s",
                            v->isParameter() ? "parameter" : "local",
                            v->toChars(),
                            sc->func->toChars());
                    }
                }
            }

            FuncDeclaration *f = ve->var->isFuncDeclaration();

            if (f)
            {
#if IN_LLVM
                if (f->isIntrinsic())
                {
                    error("cannot take the address of intrinsic function %s", e1->toChars());
                    return this;
                }
#endif

                if (!ve->hasOverloads ||
                    /* Because nested functions cannot be overloaded,
                     * mark here that we took its address because castTo()
                     * may not be called with an exact match.
                     */
                    f->isNested())
                    f->tookAddressOf++;
                if (f->isNested())
                {
                    if (f->isFuncLiteralDeclaration())
                    {
                        if (!f->FuncDeclaration::isNested())
                        {   /* Supply a 'null' for a this pointer if no this is available
                             */
                            Expression *e = new DelegateExp(loc, new NullExp(loc, Type::tnull), f, ve->hasOverloads);
                            e = e->semantic(sc);
                            return e;
                        }
                    }
                    Expression *e = new DelegateExp(loc, e1, f, ve->hasOverloads);
                    e = e->semantic(sc);
                    return e;
                }
                if (f->needThis() && hasThis(sc))
                {
                    /* Should probably supply 'this' after overload resolution,
                     * not before.
                     */
                    Expression *ethis = new ThisExp(loc);
                    Expression *e = new DelegateExp(loc, ethis, f, ve->hasOverloads);
                    e = e->semantic(sc);
                    return e;
                }
            }
        }
        else if (wasCond)
        {
            /* a ? b : c was transformed to *(a ? &b : &c), but we still
             * need to do safety checks
             */
            assert(e1->op == TOKstar);
            PtrExp *pe = (PtrExp *)e1;
            assert(pe->e1->op == TOKquestion);
            CondExp *ce = (CondExp *)pe->e1;
            assert(ce->e1->op == TOKaddress);
            assert(ce->e2->op == TOKaddress);

            // Re-run semantic on the address expressions only
            ce->e1->type = NULL;
            ce->e1 = ce->e1->semantic(sc);
            ce->e2->type = NULL;
            ce->e2 = ce->e2->semantic(sc);
        }
        return optimize(WANTvalue);
    }
    return this;
}

void AddrExp::checkEscape()
{
    e1->checkEscapeRef();
}

/************************************************************/

PtrExp::PtrExp(Loc loc, Expression *e)
        : UnaExp(loc, TOKstar, sizeof(PtrExp), e)
{
//    if (e->type)
//      type = ((TypePointer *)e->type)->next;
}

PtrExp::PtrExp(Loc loc, Expression *e, Type *t)
        : UnaExp(loc, TOKstar, sizeof(PtrExp), e)
{
    type = t;
}

Expression *PtrExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("PtrExp::semantic('%s')\n", toChars());
#endif
    if (!type)
    {
        Expression *e = op_overload(sc);
        if (e)
            return e;
        Type *tb = e1->type->toBasetype();
        switch (tb->ty)
        {
            case Tpointer:
                type = ((TypePointer *)tb)->next;
                break;

            case Tsarray:
            case Tarray:
                deprecation("using * on an array is deprecated; use *(%s).ptr instead", e1->toChars());
                type = ((TypeArray *)tb)->next;
                e1 = e1->castTo(sc, type->pointerTo());
                break;

            default:
                error("can only * a pointer, not a '%s'", e1->type->toChars());
            case Terror:
                return new ErrorExp();
        }
        if (!rvalue())
            return new ErrorExp();
    }
    return this;
}

void PtrExp::checkEscapeRef()
{
    e1->checkEscape();
}

int PtrExp::isLvalue()
{
    return 1;
}

Expression *PtrExp::toLvalue(Scope *sc, Expression *e)
{
    return this;
}

int PtrExp::checkModifiable(Scope *sc, int flag)
{
    if (e1->op == TOKsymoff)
    {   SymOffExp *se = (SymOffExp *)e1;
        return se->var->checkModify(loc, sc, type, NULL, flag);
    }
    else if (e1->op == TOKaddress)
    {
        AddrExp *ae = (AddrExp *)e1;
        return ae->e1->checkModifiable(sc, flag);
    }
    return 1;
}

#if DMDV2
Expression *PtrExp::modifiableLvalue(Scope *sc, Expression *e)
{
    //printf("PtrExp::modifiableLvalue() %s, type %s\n", toChars(), type->toChars());
    return Expression::modifiableLvalue(sc, e);
}
#endif

void PtrExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writeByte('*');
    expToCBuffer(buf, hgs, e1, precedence[op]);
}

/************************************************************/

NegExp::NegExp(Loc loc, Expression *e)
        : UnaExp(loc, TOKneg, sizeof(NegExp), e)
{
}

Expression *NegExp::semantic(Scope *sc)
{   Expression *e;

#if LOGSEMANTIC
    printf("NegExp::semantic('%s')\n", toChars());
#endif
    if (!type)
    {
        e = op_overload(sc);
        if (e)
            return e;

        e1->checkNoBool();
        if (!e1->isArrayOperand())
            e1->checkArithmetic();
        type = e1->type;
    }
    return this;
}

/************************************************************/

UAddExp::UAddExp(Loc loc, Expression *e)
        : UnaExp(loc, TOKuadd, sizeof(UAddExp), e)
{
}

Expression *UAddExp::semantic(Scope *sc)
{   Expression *e;

#if LOGSEMANTIC
    printf("UAddExp::semantic('%s')\n", toChars());
#endif
    assert(!type);
    e = op_overload(sc);
    if (e)
        return e;
    e1->checkNoBool();
    e1->checkArithmetic();
    return e1;
}

/************************************************************/

ComExp::ComExp(Loc loc, Expression *e)
        : UnaExp(loc, TOKtilde, sizeof(ComExp), e)
{
}

Expression *ComExp::semantic(Scope *sc)
{   Expression *e;

    if (!type)
    {
        e = op_overload(sc);
        if (e)
            return e;

        e1->checkNoBool();
        if (!e1->isArrayOperand())
            e1 = e1->checkIntegral();
        type = e1->type;
    }
    return this;
}

/************************************************************/

NotExp::NotExp(Loc loc, Expression *e)
        : UnaExp(loc, TOKnot, sizeof(NotExp), e)
{
}

Expression *NotExp::semantic(Scope *sc)
{
    if (!type)
    {   // Note there is no operator overload
        UnaExp::semantic(sc);
        e1 = resolveProperties(sc, e1);
        e1 = e1->checkToBoolean(sc);
        type = Type::tboolean;
    }
    return this;
}

int NotExp::isBit()
{
    return TRUE;
}



/************************************************************/

BoolExp::BoolExp(Loc loc, Expression *e, Type *t)
        : UnaExp(loc, TOKtobool, sizeof(BoolExp), e)
{
    type = t;
}

Expression *BoolExp::semantic(Scope *sc)
{
    if (!type)
    {   // Note there is no operator overload
        UnaExp::semantic(sc);
        e1 = resolveProperties(sc, e1);
        e1 = e1->checkToBoolean(sc);
        type = Type::tboolean;
    }
    return this;
}

int BoolExp::isBit()
{
    return TRUE;
}

/************************************************************/

DeleteExp::DeleteExp(Loc loc, Expression *e)
        : UnaExp(loc, TOKdelete, sizeof(DeleteExp), e)
{
}

Expression *DeleteExp::semantic(Scope *sc)
{
    Type *tb;

    UnaExp::semantic(sc);
    e1 = resolveProperties(sc, e1);
    e1 = e1->modifiableLvalue(sc, NULL);
    if (e1->op == TOKerror)
        return e1;
    type = Type::tvoid;

    tb = e1->type->toBasetype();
    switch (tb->ty)
    {   case Tclass:
        {   TypeClass *tc = (TypeClass *)tb;
            ClassDeclaration *cd = tc->sym;

            if (cd->isCOMinterface())
            {   /* Because COM classes are deleted by IUnknown.Release()
                 */
                error("cannot delete instance of COM interface %s", cd->toChars());
            }
            break;
        }
        case Tpointer:
            tb = ((TypePointer *)tb)->next->toBasetype();
            if (tb->ty == Tstruct)
            {
                TypeStruct *ts = (TypeStruct *)tb;
                StructDeclaration *sd = ts->sym;
                FuncDeclaration *f = sd->aggDelete;
                FuncDeclaration *fd = sd->dtor;

                if (!f && !fd)
                    break;

                /* Construct:
                 *      ea = copy e1 to a tmp to do side effects only once
                 *      eb = call destructor
                 *      ec = call deallocator
                 */
                Expression *ea = NULL;
                Expression *eb = NULL;
                Expression *ec = NULL;
                VarDeclaration *v;

                if (fd && f)
                {   Identifier *id = Lexer::idPool("__tmp");
                    v = new VarDeclaration(loc, e1->type, id, new ExpInitializer(loc, e1));
                    v->semantic(sc);
                    v->parent = sc->parent;
                    ea = new DeclarationExp(loc, v);
                    ea->type = v->type;
                }

                if (fd)
                {   Expression *e = ea ? new VarExp(loc, v) : e1;
                    e = new DotVarExp(0, e, fd, 0);
                    eb = new CallExp(loc, e);
                    eb = eb->semantic(sc);
                }

                if (f)
                {
                    Type *tpv = Type::tvoid->pointerTo();
                    Expression *e = ea ? new VarExp(loc, v) : e1->castTo(sc, tpv);
                    e = new CallExp(loc, new VarExp(loc, f), e);
                    ec = e->semantic(sc);
                }
                ea = combine(ea, eb);
                ea = combine(ea, ec);
                assert(ea);
                return ea;
            }
            break;

        case Tarray:
            /* BUG: look for deleting arrays of structs with dtors.
             */
            break;

        default:
            if (e1->op == TOKindex)
            {
                IndexExp *ae = (IndexExp *)(e1);
                Type *tb1 = ae->e1->type->toBasetype();
                if (tb1->ty == Taarray)
                    break;
            }
            error("cannot delete type %s", e1->type->toChars());
            return new ErrorExp();
    }

    if (e1->op == TOKindex)
    {
        IndexExp *ae = (IndexExp *)(e1);
        Type *tb1 = ae->e1->type->toBasetype();
        if (tb1->ty == Taarray)
            error("delete aa[key] deprecated, use aa.remove(key)");
    }

    return this;
}


Expression *DeleteExp::checkToBoolean(Scope *sc)
{
    error("delete does not give a boolean result");
    return new ErrorExp();
}

void DeleteExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("delete ");
    expToCBuffer(buf, hgs, e1, precedence[op]);
}

/************************************************************/

CastExp::CastExp(Loc loc, Expression *e, Type *t)
        : UnaExp(loc, TOKcast, sizeof(CastExp), e)
{
    to = t;
    this->mod = ~0;
#if IN_LLVM
    disableOptimization = false;
#endif
}

#if DMDV2
/* For cast(const) and cast(immutable)
 */
CastExp::CastExp(Loc loc, Expression *e, unsigned mod)
        : UnaExp(loc, TOKcast, sizeof(CastExp), e)
{
    to = NULL;
    this->mod = mod;
#if IN_LLVM
    disableOptimization = false;
#endif
}
#endif

Expression *CastExp::syntaxCopy()
{
    return to ? new CastExp(loc, e1->syntaxCopy(), to->syntaxCopy())
              : new CastExp(loc, e1->syntaxCopy(), mod);
}


Expression *CastExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("CastExp::semantic('%s')\n", toChars());
#endif

//static int x; assert(++x < 10);

    if (type)
        return this;
    UnaExp::semantic(sc);
    if (e1->type)               // if not a tuple
    {
        e1 = resolveProperties(sc, e1);

        if (!to)
        {
            /* Handle cast(const) and cast(immutable), etc.
             */
            to = e1->type->castMod(mod);
        }
        else
            to = to->semantic(loc, sc);
        if (to == Type::terror)
            return new ErrorExp();
        if (to->ty == Ttuple)
        {
            error("cannot cast %s to tuple type %s", e1->toChars(), to->toChars());
            return new ErrorExp();
        }
        if (e1->type->ty == Terror)
            return new ErrorExp();

        if (!to->equals(e1->type))
        {
            Expression *e = op_overload(sc);
            if (e)
            {
                return e->implicitCastTo(sc, to);
            }
        }

        if (e1->op == TOKtemplate)
        {
            error("cannot cast template %s to type %s", e1->toChars(), to->toChars());
            return new ErrorExp();
        }

        Type *t1b = e1->type->toBasetype();
        Type *tob = to->toBasetype();

        if (tob->ty == Tstruct &&
            !tob->equals(t1b)
           )
        {
            /* Look to replace:
             *  cast(S)t
             * with:
             *  S(t)
             */

            // Rewrite as to.call(e1)
            Expression *e = new TypeExp(loc, to);
            e = new CallExp(loc, e, e1);
            e = e->trySemantic(sc);
            if (e)
                return e;
        }

        // Struct casts are possible only when the sizes match
        // Same with static array -> static array
        if (tob->ty == Tstruct || t1b->ty == Tstruct ||
            (tob->ty == Tsarray && t1b->ty == Tsarray))
        {
            size_t fromsize = t1b->size(loc);
            size_t tosize = tob->size(loc);
            if (fromsize != tosize)
            {
                error("cannot cast from %s to %s", e1->type->toChars(), to->toChars());
                return new ErrorExp();
            }
        }

        // Look for casting to a vector type
        if (tob->ty == Tvector && t1b->ty != Tvector)
        {
            return new VectorExp(loc, e1, to);
        }

        if (tob->isintegral() && t1b->ty == Tarray)
            deprecation("casting %s to %s is deprecated", e1->type->toChars(), to->toChars());
    }
    else if (!to)
    {   error("cannot cast tuple");
        return new ErrorExp();
    }

    if (!e1->type)
    {   error("cannot cast %s", e1->toChars());
        return new ErrorExp();
    }

    // Check for unsafe casts
    if (sc->func && !sc->intypeof)
    {   // Disallow unsafe casts
        Type *tob = to->toBasetype();
        Type *t1b = e1->type->toBasetype();

        // Implicit conversions are always safe
        if (t1b->implicitConvTo(tob))
            goto Lsafe;

        if (!t1b->isMutable() && tob->isMutable())
            goto Lunsafe;

        if (t1b->isShared() && !tob->isShared())
            // Cast away shared
            goto Lunsafe;

        if (!tob->hasPointers())
            goto Lsafe;

        if (tob->ty == Tclass && t1b->ty == Tclass)
        {
            ClassDeclaration *cdfrom = t1b->isClassHandle();
            ClassDeclaration *cdto   = tob->isClassHandle();

            int offset;
            if (!cdfrom->isBaseOf(cdto, &offset))
                goto Lunsafe;

            if (cdfrom->isCPPinterface() ||
                cdto->isCPPinterface())
                goto Lunsafe;

            goto Lsafe;
        }

        if (tob->ty == Tarray && t1b->ty == Tarray)
        {
            Type* tobn = tob->nextOf()->toBasetype();
            Type* t1bn = t1b->nextOf()->toBasetype();
            if (!tobn->hasPointers() && MODimplicitConv(t1bn->mod, tobn->mod))
                goto Lsafe;
        }
        if (tob->ty == Tpointer && t1b->ty == Tpointer)
        {
            Type* tobn = tob->nextOf()->toBasetype();
            Type* t1bn = t1b->nextOf()->toBasetype();
            if (!tobn->hasPointers() &&
                tobn->ty != Tfunction && t1bn->ty != Tfunction &&
                tobn->size() <= t1bn->size() &&
                MODimplicitConv(t1bn->mod, tobn->mod))
                goto Lsafe;
        }

    Lunsafe:
        if (sc->func->setUnsafe())
        {   error("cast from %s to %s not allowed in safe code", e1->type->toChars(), to->toChars());
            return new ErrorExp();
        }
    }

Lsafe:
    Expression *e = e1->castTo(sc, to);
    return e;
}


void CastExp::checkEscape()
{   Type *tb = type->toBasetype();

    if (tb->ty == Tarray && e1->op == TOKvar &&
        e1->type->toBasetype()->ty == Tsarray)
    {   VarExp *ve = (VarExp *)e1;
        VarDeclaration *v = ve->var->isVarDeclaration();
        if (v)
        {
            if (!v->isDataseg() && !v->isParameter())
                error("escaping reference to local %s", v->toChars());
        }
    }
}

void CastExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("cast(");
#if DMDV1
    to->toCBuffer(buf, NULL, hgs);
#else
    if (to)
        to->toCBuffer(buf, NULL, hgs);
    else
    {
        MODtoBuffer(buf, mod);
    }
#endif
    buf->writeByte(')');
    expToCBuffer(buf, hgs, e1, precedence[op]);
}

/************************************************************/

VectorExp::VectorExp(Loc loc, Expression *e, Type *t)
        : UnaExp(loc, TOKvector, sizeof(VectorExp), e)
{
    assert(t->ty == Tvector);
    to = (TypeVector *)t;
    dim = ~0;
}

Expression *VectorExp::syntaxCopy()
{
    return new VectorExp(loc, e1->syntaxCopy(), to->syntaxCopy());
}

Expression *VectorExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("VectorExp::semantic('%s')\n", toChars());
#endif

    if (type)
        return this;
    e1 = e1->semantic(sc);
    type = to->semantic(loc, sc);
    if (e1->op == TOKerror || type->ty == Terror)
        return e1;
    Type *tb = type->toBasetype();
    assert(tb->ty == Tvector);
    TypeVector *tv = (TypeVector *)tb;
    Type *te = tv->elementType();
    dim = tv->size(loc) / te->size(loc);
    return this;
}

void VectorExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring("cast(");
    to->toCBuffer(buf, NULL, hgs);
    buf->writeByte(')');
    expToCBuffer(buf, hgs, e1, precedence[op]);
}

/************************************************************/

SliceExp::SliceExp(Loc loc, Expression *e1, Expression *lwr, Expression *upr)
        : UnaExp(loc, TOKslice, sizeof(SliceExp), e1)
{
    this->upr = upr;
    this->lwr = lwr;
    lengthVar = NULL;
}

Expression *SliceExp::syntaxCopy()
{
    Expression *lwr = NULL;
    if (this->lwr)
        lwr = this->lwr->syntaxCopy();

    Expression *upr = NULL;
    if (this->upr)
        upr = this->upr->syntaxCopy();

    SliceExp *se = new SliceExp(loc, e1->syntaxCopy(), lwr, upr);
    se->lengthVar = this->lengthVar;    // bug7871
    return se;
}

Expression *SliceExp::semantic(Scope *sc)
{   Expression *e;
    AggregateDeclaration *ad;
    //FuncDeclaration *fd;
    ScopeDsymbol *sym;

#if LOGSEMANTIC
    printf("SliceExp::semantic('%s')\n", toChars());
#endif
    if (type)
        return this;

Lagain:
    UnaExp::semantic(sc);
    e1 = resolveProperties(sc, e1);
    if (e1->op == TOKtype && e1->type->ty != Ttuple)
    {
        if (lwr || upr)
        {
            error("cannot slice type '%s'", e1->toChars());
            return new ErrorExp();
        }
        e = new TypeExp(loc, e1->type->arrayOf());
        return e->semantic(sc);
    }

    e = this;

    Type *t = e1->type->toBasetype();
    if (t->ty == Tpointer)
    {
        if (!lwr || !upr)
        {   error("need upper and lower bound to slice pointer");
            return new ErrorExp();
        }
    }
    else if (t->ty == Tarray)
    {
    }
    else if (t->ty == Tsarray)
    {
    }
    else if (t->ty == Tclass)
    {
        ad = ((TypeClass *)t)->sym;
        goto L1;
    }
    else if (t->ty == Tstruct)
    {
        ad = ((TypeStruct *)t)->sym;

    L1:
        if (search_function(ad, Id::slice))
        {
            // Rewrite as e1.slice(lwr, upr)
            SliceExp *se = resolveOpDollar(sc, this);
            Expressions *a = new Expressions();
            assert(!se->lwr || se->upr);
            if (se->lwr)
            {   a->push(se->lwr);
                a->push(se->upr);
            }
            e = new DotIdExp(loc, se->e1, Id::slice);
            e = new CallExp(loc, e, a);
            e = e->semantic(sc);
            return e;
        }
        if (ad->aliasthis)
        {
            e1 = resolveAliasThis(sc, e1);
            goto Lagain;
        }
        goto Lerror;
    }
    else if (t->ty == Ttuple)
    {
        if (!lwr && !upr)
            return e1;
        if (!lwr || !upr)
        {   error("need upper and lower bound to slice tuple");
            goto Lerror;
        }
    }
    else if (t == Type::terror)
        goto Lerr;
    else
        goto Lerror;

    {
    Scope *sc2 = sc;
    if (t->ty == Tsarray || t->ty == Tarray || t->ty == Ttuple)
    {
        sym = new ArrayScopeSymbol(sc, this);
        sym->loc = loc;
        sym->parent = sc->scopesym;
        sc2 = sc->push(sym);
    }

    if (lwr)
    {   lwr = lwr->semantic(sc2);
        lwr = resolveProperties(sc2, lwr);
        lwr = lwr->implicitCastTo(sc2, Type::tsize_t);
        if (lwr->type == Type::terror)
            goto Lerr;
    }
    if (upr)
    {   upr = upr->semantic(sc2);
        upr = resolveProperties(sc2, upr);
        upr = upr->implicitCastTo(sc2, Type::tsize_t);
        if (upr->type == Type::terror)
            goto Lerr;
    }

    if (sc2 != sc)
        sc2->pop();
    }

    if (t->ty == Ttuple)
    {
        lwr = lwr->ctfeInterpret();
        upr = upr->ctfeInterpret();
        uinteger_t i1 = lwr->toUInteger();
        uinteger_t i2 = upr->toUInteger();

        size_t length;
        TupleExp *te;
        TypeTuple *tup;

        if (e1->op == TOKtuple)         // slicing an expression tuple
        {   te = (TupleExp *)e1;
            length = te->exps->dim;
        }
        else if (e1->op == TOKtype)     // slicing a type tuple
        {   tup = (TypeTuple *)t;
            length = Parameter::dim(tup->arguments);
        }
        else
            assert(0);

        if (i1 <= i2 && i2 <= length)
        {   size_t j1 = (size_t) i1;
            size_t j2 = (size_t) i2;

            if (e1->op == TOKtuple)
            {   Expressions *exps = new Expressions;
                exps->setDim(j2 - j1);
                for (size_t i = 0; i < j2 - j1; i++)
                {   Expression *e = (*te->exps)[j1 + i];
                    (*exps)[i] = e;
                }
                if (j1 > 0 && j1 != j2 && sc->func && (*te->exps)[0]->op == TOKdotvar)
                {
                    Expression *einit = ((DotVarExp *)(*te->exps)[0])->e1->isTemp();
                    if (einit)
                        ((DotVarExp *)(*exps)[0])->e1 = einit;
                }
                e = new TupleExp(loc, exps);
            }
            else
            {   Parameters *args = new Parameters;
                args->reserve(j2 - j1);
                for (size_t i = j1; i < j2; i++)
                {   Parameter *arg = Parameter::getNth(tup->arguments, i);
                    args->push(arg);
                }
                e = new TypeExp(e1->loc, new TypeTuple(args));
            }
            e = e->semantic(sc);
        }
        else
        {
            error("string slice [%llu .. %llu] is out of bounds", i1, i2);
            goto Lerr;
        }
        return e;
    }

    type = t->nextOf()->arrayOf();
    // Allow typedef[] -> typedef[]
    if (type->equals(t))
        type = e1->type;

    return e;

Lerror:
    if (e1->op == TOKerror)
        return e1;
    char *s;
    if (t->ty == Tvoid)
        s = e1->toChars();
    else
        s = t->toChars();
    error("%s cannot be sliced with []", s);
Lerr:
    e = new ErrorExp();
    return e;
}

void SliceExp::checkEscape()
{
    e1->checkEscape();
}

void SliceExp::checkEscapeRef()
{
    e1->checkEscapeRef();
}

int SliceExp::isLvalue()
{
    return 1;
}

Expression *SliceExp::toLvalue(Scope *sc, Expression *e)
{
    return this;
}

int SliceExp::checkModifiable(Scope *sc, int flag)
{
    //printf("SliceExp::checkModifiable %s\n", toChars());
    if (e1->type->ty == Tsarray ||
        (e1->op == TOKindex && e1->type->ty != Tarray) ||
        e1->op == TOKslice)
    {
        return e1->checkModifiable(sc, flag);
    }
    return 1;
}

Expression *SliceExp::modifiableLvalue(Scope *sc, Expression *e)
{
    error("slice expression %s is not a modifiable lvalue", toChars());
    return this;
}

int SliceExp::isBool(int result)
{
    return e1->isBool(result);
}

void SliceExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    expToCBuffer(buf, hgs, e1, precedence[op]);
    buf->writeByte('[');
    if (upr || lwr)
    {
        if (lwr)
            expToCBuffer(buf, hgs, lwr, PREC_assign);
        else
            buf->writeByte('0');
        buf->writestring("..");
        if (upr)
            expToCBuffer(buf, hgs, upr, PREC_assign);
        else
            buf->writestring("length");         // BUG: should be array.length
    }
    buf->writeByte(']');
}

/********************** ArrayLength **************************************/

ArrayLengthExp::ArrayLengthExp(Loc loc, Expression *e1)
        : UnaExp(loc, TOKarraylength, sizeof(ArrayLengthExp), e1)
{
}

Expression *ArrayLengthExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("ArrayLengthExp::semantic('%s')\n", toChars());
#endif
    if (!type)
    {
        UnaExp::semantic(sc);
        e1 = resolveProperties(sc, e1);

        type = Type::tsize_t;
    }
    return this;
}

Expression *opAssignToOp(Loc loc, enum TOK op, Expression *e1, Expression *e2)
{   Expression *e;

    switch (op)
    {
        case TOKaddass:   e = new AddExp(loc, e1, e2);  break;
        case TOKminass:   e = new MinExp(loc, e1, e2);  break;
        case TOKmulass:   e = new MulExp(loc, e1, e2);  break;
        case TOKdivass:   e = new DivExp(loc, e1, e2);  break;
        case TOKmodass:   e = new ModExp(loc, e1, e2);  break;
        case TOKandass:   e = new AndExp(loc, e1, e2);  break;
        case TOKorass:    e = new OrExp (loc, e1, e2);  break;
        case TOKxorass:   e = new XorExp(loc, e1, e2);  break;
        case TOKshlass:   e = new ShlExp(loc, e1, e2);  break;
        case TOKshrass:   e = new ShrExp(loc, e1, e2);  break;
        case TOKushrass:  e = new UshrExp(loc, e1, e2); break;
        default:        assert(0);
    }
    return e;
}

/*********************
 * Rewrite:
 *    array.length op= e2
 * as:
 *    array.length = array.length op e2
 * or:
 *    auto tmp = &array;
 *    (*tmp).length = (*tmp).length op e2
 */

Expression *ArrayLengthExp::rewriteOpAssign(BinExp *exp)
{   Expression *e;

    assert(exp->e1->op == TOKarraylength);
    ArrayLengthExp *ale = (ArrayLengthExp *)exp->e1;
    if (ale->e1->op == TOKvar)
    {   e = opAssignToOp(exp->loc, exp->op, ale, exp->e2);
        e = new AssignExp(exp->loc, ale->syntaxCopy(), e);
    }
    else
    {
        /*    auto tmp = &array;
         *    (*tmp).length = (*tmp).length op e2
         */
        Identifier *id = Lexer::uniqueId("__arraylength");
        ExpInitializer *ei = new ExpInitializer(ale->loc, new AddrExp(ale->loc, ale->e1));
        VarDeclaration *tmp = new VarDeclaration(ale->loc, ale->e1->type->pointerTo(), id, ei);

        Expression *e1 = new ArrayLengthExp(ale->loc, new PtrExp(ale->loc, new VarExp(ale->loc, tmp)));
        Expression *elvalue = e1->syntaxCopy();
        e = opAssignToOp(exp->loc, exp->op, e1, exp->e2);
        e = new AssignExp(exp->loc, elvalue, e);
        e = new CommaExp(exp->loc, new DeclarationExp(ale->loc, tmp), e);
    }
    return e;
}

void ArrayLengthExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    expToCBuffer(buf, hgs, e1, PREC_primary);
    buf->writestring(".length");
}

/*********************** ArrayExp *************************************/

// e1 [ i1, i2, i3, ... ]

ArrayExp::ArrayExp(Loc loc, Expression *e1, Expressions *args)
        : UnaExp(loc, TOKarray, sizeof(ArrayExp), e1)
{
    arguments = args;
    lengthVar = NULL;
    currentDimension = 0;
}

Expression *ArrayExp::syntaxCopy()
{
    ArrayExp *ae = new ArrayExp(loc, e1->syntaxCopy(), arraySyntaxCopy(arguments));
    ae->lengthVar = this->lengthVar;    // bug7871
    return ae;
}

Expression *ArrayExp::semantic(Scope *sc)
{   Expression *e;
    Type *t1;

#if LOGSEMANTIC
    printf("ArrayExp::semantic('%s')\n", toChars());
#endif
    UnaExp::semantic(sc);
    e1 = resolveProperties(sc, e1);
    if (e1->op == TOKerror)
        return e1;

    t1 = e1->type->toBasetype();
    if (t1->ty != Tclass && t1->ty != Tstruct)
    {   // Convert to IndexExp
        if (arguments->dim != 1)
        {   error("only one index allowed to index %s", t1->toChars());
            goto Lerr;
        }
        e = new IndexExp(loc, e1, (*arguments)[0]);
        return e->semantic(sc);
    }

    e = op_overload(sc);
    if (!e)
    {   error("no [] operator overload for type %s", e1->type->toChars());
        goto Lerr;
    }
    return e;

Lerr:
    return new ErrorExp();
}


int ArrayExp::isLvalue()
{
    if (type && type->toBasetype()->ty == Tvoid)
        return 0;
    return 1;
}

Expression *ArrayExp::toLvalue(Scope *sc, Expression *e)
{
    if (type && type->toBasetype()->ty == Tvoid)
        error("voids have no value");
    return this;
}


void ArrayExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    expToCBuffer(buf, hgs, e1, PREC_primary);
    buf->writeByte('[');
    argsToCBuffer(buf, arguments, hgs);
    buf->writeByte(']');
}

/************************* DotExp ***********************************/

DotExp::DotExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKdotexp, sizeof(DotExp), e1, e2)
{
}

Expression *DotExp::semantic(Scope *sc)
{
#if LOGSEMANTIC
    printf("DotExp::semantic('%s')\n", toChars());
    if (type) printf("\ttype = %s\n", type->toChars());
#endif
    e1 = e1->semantic(sc);
    e2 = e2->semantic(sc);
    if (e2->op == TOKimport)
    {
        ScopeExp *se = (ScopeExp *)e2;
        TemplateDeclaration *td = se->sds->isTemplateDeclaration();
        if (td)
        {   Expression *e = new DotTemplateExp(loc, e1, td);
            e = e->semantic(sc);
            return e;
        }
    }
    if (!type)
        type = e2->type;
    return this;
}

void DotExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    expToCBuffer(buf, hgs, e1, PREC_primary);
    buf->writeByte('.');
    expToCBuffer(buf, hgs, e2, PREC_primary);
}

/************************* CommaExp ***********************************/

CommaExp::CommaExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKcomma, sizeof(CommaExp), e1, e2)
{
}

Expression *CommaExp::semantic(Scope *sc)
{
    if (!type)
    {   BinExp::semanticp(sc);
        e1 = e1->addDtorHook(sc);
        type = e2->type;
    }
    return this;
}

void CommaExp::checkEscape()
{
    e2->checkEscape();
}

void CommaExp::checkEscapeRef()
{
    e2->checkEscapeRef();
}

int CommaExp::isLvalue()
{
    return e2->isLvalue();
}

Expression *CommaExp::toLvalue(Scope *sc, Expression *e)
{
    e2 = e2->toLvalue(sc, NULL);
    return this;
}

int CommaExp::checkModifiable(Scope *sc, int flag)
{
    return e2->checkModifiable(sc, flag);
}

Expression *CommaExp::modifiableLvalue(Scope *sc, Expression *e)
{
    e2 = e2->modifiableLvalue(sc, e);
    return this;
}

int CommaExp::isBool(int result)
{
    return e2->isBool(result);
}


Expression *CommaExp::addDtorHook(Scope *sc)
{
    e2 = e2->addDtorHook(sc);
    return this;
}

/************************** IndexExp **********************************/

// e1 [ e2 ]

IndexExp::IndexExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKindex, sizeof(IndexExp), e1, e2)
{
    //printf("IndexExp::IndexExp('%s')\n", toChars());
    lengthVar = NULL;
    modifiable = 0;     // assume it is an rvalue
}

Expression *IndexExp::syntaxCopy()
{
    IndexExp *ie = new IndexExp(loc, e1->syntaxCopy(), e2->syntaxCopy());
    ie->lengthVar = this->lengthVar;    // bug7871
    return ie;
}

Expression *IndexExp::semantic(Scope *sc)
{   Expression *e;
    Type *t1;
    ScopeDsymbol *sym;

#if LOGSEMANTIC
    printf("IndexExp::semantic('%s')\n", toChars());
#endif
    if (type)
        return this;
    if (!e1->type)
        e1 = e1->semantic(sc);
    assert(e1->type);           // semantic() should already be run on it
    if (e1->op == TOKtype && e1->type->ty != Ttuple)
    {
        e2 = e2->semantic(sc);
        e2 = resolveProperties(sc, e2);
        Type *nt;
        if (e2->op == TOKtype)
            nt = new TypeAArray(e1->type, e2->type);
        else
            nt = new TypeSArray(e1->type, e2);
        e = new TypeExp(loc, nt);
        return e->semantic(sc);
    }
    if (e1->op == TOKerror)
        goto Lerr;
    e = this;

    // Note that unlike C we do not implement the int[ptr]

    t1 = e1->type->toBasetype();

    if (t1->ty == Tsarray || t1->ty == Tarray || t1->ty == Ttuple)
    {   // Create scope for 'length' variable
        sym = new ArrayScopeSymbol(sc, this);
        sym->loc = loc;
        sym->parent = sc->scopesym;
        sc = sc->push(sym);
    }

    e2 = e2->semantic(sc);
    e2 = resolveProperties(sc, e2);
    if (e2->type == Type::terror)
        goto Lerr;
    if (e2->type->ty == Ttuple && ((TupleExp *)e2)->exps->dim == 1) // bug 4444 fix
        e2 = (*((TupleExp *)e2)->exps)[0];

    if (t1->ty == Tsarray || t1->ty == Tarray || t1->ty == Ttuple)
        sc = sc->pop();

    switch (t1->ty)
    {
        case Tpointer:
            e2 = e2->implicitCastTo(sc, Type::tsize_t);
            e2 = e2->optimize(WANTvalue);
            if (e2->op == TOKint64 && e2->toInteger() == 0)
                ;
            else if (sc->func->setUnsafe())
            {
                error("safe function '%s' cannot index pointer '%s'",
                    sc->func->toPrettyChars(), e1->toChars());
                return new ErrorExp();
            }
            e->type = ((TypeNext *)t1)->next;
            break;

        case Tarray:
            e2 = e2->implicitCastTo(sc, Type::tsize_t);
            e->type = ((TypeNext *)t1)->next;
            break;

        case Tsarray:
        {
            e2 = e2->implicitCastTo(sc, Type::tsize_t);
            TypeSArray *tsa = (TypeSArray *)t1;
            e->type = t1->nextOf();
            break;
        }

        case Taarray:
        {   TypeAArray *taa = (TypeAArray *)t1;
            /* We can skip the implicit conversion if they differ only by
             * constness (Bugzilla 2684, see also bug 2954b)
             */
            if (!arrayTypeCompatibleWithoutCasting(e2->loc, e2->type, taa->index))
            {
                e2 = e2->implicitCastTo(sc, taa->index);        // type checking
            }
            type = taa->next;
            break;
        }

        case Ttuple:
        {
            e2 = e2->implicitCastTo(sc, Type::tsize_t);
            e2 = e2->ctfeInterpret();
            uinteger_t index = e2->toUInteger();
            size_t length;
            TupleExp *te;
            TypeTuple *tup;

            if (e1->op == TOKtuple)
            {   te = (TupleExp *)e1;
                length = te->exps->dim;
            }
            else if (e1->op == TOKtype)
            {
                tup = (TypeTuple *)t1;
                length = Parameter::dim(tup->arguments);
            }
            else
                assert(0);

            if (index < length)
            {

                if (e1->op == TOKtuple)
                {
                    e = (*te->exps)[(size_t)index];
                    if (sc->func && (*te->exps)[0]->op == TOKdotvar)
                    {
                        Expression *einit = ((DotVarExp *)(*te->exps)[0])->e1->isTemp();
                        if (einit)
                            ((DotVarExp *)e)->e1 = einit;
                    }
                }
                else
                    e = new TypeExp(e1->loc, Parameter::getNth(tup->arguments, (size_t)index)->type);
            }
            else
            {
                error("array index [%llu] is outside array bounds [0 .. %llu]",
                        index, (ulonglong)length);
                e = e1;
            }
            break;
        }

        default:
            if (e1->op == TOKerror)
                goto Lerr;
            error("%s must be an array or pointer type, not %s",
                e1->toChars(), e1->type->toChars());
        case Terror:
            goto Lerr;
    }
    return e;

Lerr:
    return new ErrorExp();
}

int IndexExp::isLvalue()
{
    return 1;
}

Expression *IndexExp::toLvalue(Scope *sc, Expression *e)
{
//    if (type && type->toBasetype()->ty == Tvoid)
//      error("voids have no value");
    return this;
}

int IndexExp::checkModifiable(Scope *sc, int flag)
{
    if (e1->type->ty == Tsarray ||
        (e1->op == TOKindex && e1->type->ty != Tarray) ||
        e1->op == TOKslice)
    {
        return e1->checkModifiable(sc, flag);
    }
    return 1;
}

Expression *IndexExp::modifiableLvalue(Scope *sc, Expression *e)
{
    //printf("IndexExp::modifiableLvalue(%s)\n", toChars());
    modifiable = 1;
    Type *t1 = e1->type->toBasetype();
    if (t1->ty == Taarray)
    {   TypeAArray *taa = (TypeAArray *)t1;
        Type *t2b = e2->type->toBasetype();
        if (t2b->ty == Tarray && t2b->nextOf()->isMutable())
            error("associative arrays can only be assigned values with immutable keys, not %s", e2->type->toChars());
        e1 = e1->modifiableLvalue(sc, e1);
        return toLvalue(sc, e);
    }

    return Expression::modifiableLvalue(sc, e);
}

void IndexExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    expToCBuffer(buf, hgs, e1, PREC_primary);
    buf->writeByte('[');
    expToCBuffer(buf, hgs, e2, PREC_assign);
    buf->writeByte(']');
}


/************************* PostExp ***********************************/

PostExp::PostExp(enum TOK op, Loc loc, Expression *e)
        : BinExp(loc, op, sizeof(PostExp), e,
          new IntegerExp(loc, 1, Type::tint32))
{
}

Expression *PostExp::semantic(Scope *sc)
{   Expression *e = this;

    if (!type)
    {
        BinExp::semantic(sc);
        e1 = resolveProperties(sc, e1);

        e = op_overload(sc);
        if (e)
            return e;

        if (e1->op == TOKslice)
        {
            const char *s = op == TOKplusplus ? "increment" : "decrement";
            error("cannot post-%s array slice '%s', use pre-%s instead", s, e1->toChars(), s);
            return new ErrorExp();
        }

        if (e1->op != TOKarraylength)
            e1 = e1->modifiableLvalue(sc, e1);

        Type *t1 = e1->type->toBasetype();
        if (t1->ty == Tclass || t1->ty == Tstruct || e1->op == TOKarraylength)
        {   /* Check for operator overloading,
             * but rewrite in terms of ++e instead of e++
             */

            /* If e1 is not trivial, take a reference to it
             */
            Expression *de = NULL;
            if (e1->op != TOKvar && e1->op != TOKarraylength)
            {
                // ref v = e1;
                Identifier *id = Lexer::uniqueId("__postref");
                ExpInitializer *ei = new ExpInitializer(loc, e1);
                VarDeclaration *v = new VarDeclaration(loc, e1->type, id, ei);
                v->storage_class |= STCref | STCforeach;
                de = new DeclarationExp(loc, v);
                e1 = new VarExp(e1->loc, v);
            }

            /* Rewrite as:
             * auto tmp = e1; ++e1; tmp
             */
            Identifier *id = Lexer::uniqueId("__pitmp");
            ExpInitializer *ei = new ExpInitializer(loc, e1);
            VarDeclaration *tmp = new VarDeclaration(loc, e1->type, id, ei);
            Expression *ea = new DeclarationExp(loc, tmp);

            Expression *eb = e1->syntaxCopy();
            eb = new PreExp(op == TOKplusplus ? TOKpreplusplus : TOKpreminusminus, loc, eb);

            Expression *ec = new VarExp(loc, tmp);

            // Combine de,ea,eb,ec
            if (de)
                ea = new CommaExp(loc, de, ea);
            e = new CommaExp(loc, ea, eb);
            e = new CommaExp(loc, e, ec);
            e = e->semantic(sc);
            return e;
        }

        e = this;
        e1->checkScalar();
        e1->checkNoBool();
        if (e1->type->ty == Tpointer)
            e = scaleFactor(sc);
        else
            e2 = e2->castTo(sc, e1->type);
        e->type = e1->type;
    }
    return e;
}

void PostExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    expToCBuffer(buf, hgs, e1, precedence[op]);
    buf->writestring(Token::toChars(op));
}

/************************* PreExp ***********************************/

PreExp::PreExp(enum TOK op, Loc loc, Expression *e)
        : UnaExp(loc, op, sizeof(PreExp), e)
{
}

Expression *PreExp::semantic(Scope *sc)
{
    Expression *e;

    e = op_overload(sc);
    if (e)
        return e;

    // Rewrite as e1+=1 or e1-=1
    if (op == TOKpreplusplus)
        e = new AddAssignExp(loc, e1, new IntegerExp(loc, 1, Type::tint32));
    else
        e = new MinAssignExp(loc, e1, new IntegerExp(loc, 1, Type::tint32));
    return e->semantic(sc);
}

void PreExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring(Token::toChars(op));
    expToCBuffer(buf, hgs, e1, precedence[op]);
}

/************************************************************/

/* op can be TOKassign, TOKconstruct, or TOKblit */

AssignExp::AssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKassign, sizeof(AssignExp), e1, e2)
{
    ismemset = 0;
}

Expression *AssignExp::semantic(Scope *sc)
{
    Expression *e1old = e1;

#if LOGSEMANTIC
    printf("AssignExp::semantic('%s')\n", toChars());
#endif
    //printf("e1->op = %d, '%s'\n", e1->op, Token::toChars(e1->op));
    //printf("e2->op = %d, '%s'\n", e2->op, Token::toChars(e2->op));

    if (type)
        return this;

    if (e2->op == TOKcomma)
    {   /* Rewrite to get rid of the comma from rvalue
         */
        AssignExp *ea = new AssignExp(loc, e1, ((CommaExp *)e2)->e2);
        ea->op = op;
        Expression *e = new CommaExp(loc, ((CommaExp *)e2)->e1, ea);
        return e->semantic(sc);
    }

    /* Look for operator overloading of a[i]=value.
     * Do it before semantic() otherwise the a[i] will have been
     * converted to a.opIndex() already.
     */
    if (e1->op == TOKarray)
    {
        ArrayExp *ae = (ArrayExp *)e1;
        AggregateDeclaration *ad = NULL;
        Identifier *id = Id::index;

        ae->e1 = ae->e1->semantic(sc);
        ae->e1 = resolveProperties(sc, ae->e1);
        Type *t1 = ae->e1->type->toBasetype();
        if (t1->ty == Tstruct)
        {
            ad = ((TypeStruct *)t1)->sym;
            goto L1;
        }
        else if (t1->ty == Tclass)
        {
            ad = ((TypeClass *)t1)->sym;
          L1:
            // Rewrite (a[i] = value) to (a.opIndexAssign(value, i))
            if (search_function(ad, Id::indexass))
            {
                // Deal with $
                ae = resolveOpDollar(sc, ae);
                Expressions *a = (Expressions *)ae->arguments->copy();
                a->insert(0, e2);

                Expression *e = new DotIdExp(loc, ae->e1, Id::indexass);
                e = new CallExp(loc, e, a);
                e = e->semantic(sc);
                return e;
            }
        }

        // No opIndexAssign found yet, but there might be an alias this to try.
        if (ad && ad->aliasthis)
        {   Expression *e = resolveAliasThis(sc, ae->e1);
            Type *t = e->type->toBasetype();

            if (t->ty == Tstruct)
            {
                ad = ((TypeStruct *)t)->sym;
                goto L1;
            }
            else if (t->ty == Tclass)
            {
                ad = ((TypeClass *)t)->sym;
                goto L1;
            }
        }
    }
    /* Look for operator overloading of a[i..j]=value.
     * Do it before semantic() otherwise the a[i..j] will have been
     * converted to a.opSlice() already.
     */
    if (e1->op == TOKslice)
    {   Type *t1;
        SliceExp *ae = (SliceExp *)e1;
        AggregateDeclaration *ad = NULL;
        Identifier *id = Id::index;

        ae->e1 = ae->e1->semantic(sc);
        ae->e1 = resolveProperties(sc, ae->e1);
        t1 = ae->e1->type->toBasetype();
        if (t1->ty == Tstruct)
        {
            ad = ((TypeStruct *)t1)->sym;
            goto L2;
        }
        else if (t1->ty == Tclass)
        {
            ad = ((TypeClass *)t1)->sym;
          L2:
            // Rewrite (a[i..j] = value) to (a.opSliceAssign(value, i, j))
            if (search_function(ad, Id::sliceass))
            {
                ae = resolveOpDollar(sc, ae);
                Expressions *a = new Expressions();
                a->push(e2);
                assert(!ae->lwr || ae->upr);
                if (ae->lwr)
                {   a->push(ae->lwr);
                    a->push(ae->upr);
                }
                Expression *e = new DotIdExp(loc, ae->e1, Id::sliceass);
                e = new CallExp(loc, e, a);
                e = e->semantic(sc);
                return e;
            }
        }

        // No opSliceAssign found yet, but there might be an alias this to try.
        if (ad && ad->aliasthis)
        {   Expression *e = resolveAliasThis(sc, ae->e1);
            Type *t = e->type->toBasetype();

            if (t->ty == Tstruct)
            {
                ad = ((TypeStruct *)t)->sym;
                goto L2;
            }
            else if (t->ty == Tclass)
            {
                ad = ((TypeClass *)t)->sym;
                goto L2;
            }
        }
    }

    /* With UFCS, e.f = value
     * Could mean:
     *      .f(e, value)
     * or:
     *      .f(e) = value
     */
    if (e1->op == TOKdotti)
    {
        Expression *e = resolveProperty(sc, &e1, e2);
        if (e) return e;
    }
    else if (e1->op == TOKdot)
    {
        Expression *e = resolveProperty(sc, &e1, e2);
        if (e) return e;

        VarDeclaration * vd = NULL;
        if (e1->op == TOKvar)
            vd = ((VarExp *)e1)->var->isVarDeclaration();

        if (vd && vd->needThis())
        {
            error("need 'this' to access member %s", e1->toChars());
            return new ErrorExp();
        }
    }
    e1 = e1->semantic(sc);
    if (e1->op == TOKerror)
        return new ErrorExp();

    /* We have f = value.
     * Could mean:
     *      f(value)
     * or:
     *      f() = value
     */
    TemplateDeclaration *td;
    Objects *targsi;
    FuncDeclaration *fd;
    Expression *ethis;
    if (e1->op == TOKdotti)
    {
        DotTemplateInstanceExp* dti = (DotTemplateInstanceExp *)e1;
        td     = dti->getTempdecl(sc);
                 dti->ti->semanticTiargs(sc);
        targsi = dti->ti->tiargs;
        ethis  = dti->e1;
        goto L3;
    }
    else if (e1->op == TOKdottd)
    {
        DotTemplateExp *dte = (DotTemplateExp *)e1;
        td     = dte->td;
        targsi = NULL;
        ethis  = dte->e1;
        goto L3;
    }
    else if (e1->op == TOKtemplate)
    {
        td     = ((TemplateExp *)e1)->td;
        targsi = NULL;
        ethis  = NULL;
    L3:
    {
        e2 = e2->semantic(sc);
        if (e2->op == TOKerror)
            return new ErrorExp();
        e2 = resolveProperties(sc, e2);

        assert(td);
        Expressions a;
        a.push(e2);

        fd = td->deduceFunctionTemplate(sc, loc, targsi, ethis, &a, 1);
        if (fd && fd->type)
            goto Lsetter;

        fd = td->deduceFunctionTemplate(sc, loc, targsi, ethis, NULL, 1);
        if (fd && fd->type)
            goto Lgetter;
    }
        goto Leprop;
    }
    else if (e1->op == TOKdotvar && e1->type->toBasetype()->ty == Tfunction)
    {
        DotVarExp *dve = (DotVarExp *)e1;
        fd    = dve->var->isFuncDeclaration();
        ethis = dve->e1;
        goto L4;
    }
    else if (e1->op == TOKvar && e1->type->toBasetype()->ty == Tfunction)
    {
        fd = ((VarExp *)e1)->var->isFuncDeclaration();
        ethis = NULL;
    L4:
    {
        e2 = e2->semantic(sc);
        if (e2->op == TOKerror)
            return new ErrorExp();
        e2 = resolveProperties(sc, e2);

        assert(fd);
        FuncDeclaration *f = fd;
        Expressions a;
        a.push(e2);

        fd = f->overloadResolve(loc, ethis, &a, 1);
        if (fd && fd->type)
            goto Lsetter;

        fd = f->overloadResolve(loc, ethis, NULL, 1);
        if (fd && fd->type)
            goto Lgetter;

        goto Leprop;
    }

        Expression *e;
        TypeFunction *tf;

    Lsetter:
        assert(fd->type->ty == Tfunction);
        tf = (TypeFunction *)fd->type;
        if (!tf->isproperty && global.params.enforcePropertySyntax)
            goto Leprop;
        e = new CallExp(loc, e1, e2);
        return e->semantic(sc);

    Lgetter:
        assert(fd->type->ty == Tfunction);
        tf = (TypeFunction *)fd->type;
        if (!tf->isref)
            goto Leprop;
        if (!tf->isproperty && global.params.enforcePropertySyntax)
            goto Leprop;
        e = new CallExp(loc, e1);
        e = new AssignExp(loc, e, e2);
        return e->semantic(sc);

    Leprop:
        ::error(e1->loc, "not a property %s", e1->toChars());
        return new ErrorExp();
    }

    assert(e1->type);
    Type *t1 = e1->type->toBasetype();

    e2 = e2->inferType(t1);

    e2 = e2->semantic(sc);
    if (e2->op == TOKerror)
        return new ErrorExp();
    e2 = resolveProperties(sc, e2);
    if (!e2->rvalue())
        return new ErrorExp();

    /* Rewrite tuple assignment as a tuple of assignments.
     */
Ltupleassign:
    if (e1->op == TOKtuple && e2->op == TOKtuple)
    {   TupleExp *tup1 = (TupleExp *)e1;
        TupleExp *tup2 = (TupleExp *)e2;
        size_t dim = tup1->exps->dim;
        if (dim != tup2->exps->dim)
        {
            error("mismatched tuple lengths, %d and %d", (int)dim, (int)tup2->exps->dim);
            return new ErrorExp();
        }
        else
        {   Expressions *exps = new Expressions;
            exps->setDim(dim);

            for (size_t i = 0; i < dim; i++)
            {   Expression *ex1 = (*tup1->exps)[i];
                Expression *ex2 = (*tup2->exps)[i];
                (*exps)[i] =  new AssignExp(loc, ex1, ex2);
            }
            Expression *e = new TupleExp(loc, exps);
            e = e->semantic(sc);
            return e;
        }
    }

    if (e1->op == TOKtuple)
    {
        if (TupleDeclaration *td = isAliasThisTuple(e2))
        {
            assert(e1->type->ty == Ttuple);
            TypeTuple *tt = (TypeTuple *)e1->type;

            Identifier *id = Lexer::uniqueId("__tup");
            ExpInitializer *ei = new ExpInitializer(e2->loc, e2);
            VarDeclaration *v = new VarDeclaration(e2->loc, NULL, id, ei);
            v->storage_class = STCctfe | STCref | STCforeach;
            Expression *ve = new VarExp(e2->loc, v);
            ve->type = e2->type;

            Expressions *iexps = new Expressions();
            iexps->push(ve);

            for (size_t u = 0; u < iexps->dim ; u++)
            {
            Lexpand:
                Expression *e = (*iexps)[u];

                Parameter *arg = Parameter::getNth(tt->arguments, u);
                //printf("[%d] iexps->dim = %d, ", u, iexps->dim);
                //printf("e = (%s %s, %s), ", Token::tochars[e->op], e->toChars(), e->type->toChars());
                //printf("arg = (%s, %s)\n", arg->toChars(), arg->type->toChars());

                if (!e->type->implicitConvTo(arg->type))
                {
                    // expand initializer to tuple
                    if (expandAliasThisTuples(iexps, u) != -1)
                        goto Lexpand;

                    goto Lnomatch;
                }
            }
            (*iexps)[0] = new CommaExp(loc, new DeclarationExp(e2->loc, v), (*iexps)[0]);
            e2 = new TupleExp(e2->loc, iexps);
            e2 = e2->semantic(sc);
            goto Ltupleassign;

        Lnomatch:
            ;
        }
    }

    // Determine if this is an initialization of a reference
    int refinit = 0;
    if (op == TOKconstruct && e1->op == TOKvar)
    {   VarExp *ve = (VarExp *)e1;
        VarDeclaration *v = ve->var->isVarDeclaration();
        if (v->storage_class & (STCout | STCref))
            refinit = 1;
    }

    /* If it is an assignment from a 'foreign' type,
     * check for operator overloading.
     */
    if (t1->ty == Tstruct)
    {
        StructDeclaration *sd = ((TypeStruct *)t1)->sym;
        if (op == TOKassign)
        {
            Expression *e = op_overload(sc);
            if (e && e1->op == TOKindex &&
                ((IndexExp *)e1)->e1->type->toBasetype()->ty == Taarray)
            {
                // Deal with AAs (Bugzilla 2451)
                // Rewrite as:
                // e1 = (typeof(aa.value) tmp = void, tmp = e2, tmp);
                Type * aaValueType = ((TypeAArray *)((IndexExp*)e1)->e1->type->toBasetype())->next;
                Identifier *id = Lexer::uniqueId("__aatmp");
                VarDeclaration *v = new VarDeclaration(loc, aaValueType,
                    id, new VoidInitializer(0));
                v->storage_class |= STCctfe;
                v->semantic(sc);
                v->parent = sc->parent;

                Expression *de = new DeclarationExp(loc, v);
                VarExp *ve = new VarExp(loc, v);

                AssignExp *ae = new AssignExp(loc, ve, e2);
                e = ae->op_overload(sc);
                e2 = new CommaExp(loc, new CommaExp(loc, de, e), ve);
                e2 = e2->semantic(sc);

                e1 = e1->optimize(WANTvalue);
                e1 = e1->modifiableLvalue(sc, e1);
                e2 = e2->implicitCastTo(sc, e1->type);
                type = e1->type;
                assert(type);
                e = this;
            }
            if (e)
            {
                /* See if we need to set ctorinit, i.e. track
                 * assignments to fields. An assignment to a field counts even
                 * if done through an opAssign overload.
                 */
                e1->checkModifiable(sc);
                return e;
            }
        }
        else if (op == TOKconstruct && !refinit)
        {   Type *t2 = e2->type->toBasetype();
            if (t2->ty == Tstruct &&
                sd == ((TypeStruct *)t2)->sym &&
                sd->cpctor)
            {   /* We have a copy constructor for this
                 */
                if (e2->op == TOKquestion)
                {   /* Write as:
                     *  a ? e1 = b : e1 = c;
                     */
                    CondExp *econd = (CondExp *)e2;
                    AssignExp *ea1 = new AssignExp(econd->e1->loc, e1, econd->e1);
                    ea1->op = op;
                    AssignExp *ea2 = new AssignExp(econd->e1->loc, e1, econd->e2);
                    ea2->op = op;
                    Expression *e = new CondExp(loc, econd->econd, ea1, ea2);
                    return e->semantic(sc);
                }
                else if (e2->isLvalue())
                {   /* Write as:
                     *  e1.cpctor(e2);
                     */
                    if (!e2->type->implicitConvTo(e1->type))
                        error("conversion error from %s to %s", e2->type->toChars(), e1->type->toChars());

                    Expression *e = new DotVarExp(loc, e1, sd->cpctor, 0);
                    e = new CallExp(loc, e, e2);
                    return e->semantic(sc);
                }
                else if (e2->op == TOKcall)
                {
                    /* The struct value returned from the function is transferred
                     * so should not call the destructor on it.
                     */
                    valueNoDtor(e2);
                }
            }
        }
    }
    else if (t1->ty == Tclass)
    {   // Disallow assignment operator overloads for same type
        if (op == TOKassign && !e2->implicitConvTo(e1->type))
        {
            Expression *e = op_overload(sc);
            if (e)
                return e;
        }
    }

    if (t1->ty == Tsarray && !refinit)
    {
        Type *t2 = e2->type->toBasetype();

        if (e1->op == TOKindex &&
            ((IndexExp *)e1)->e1->type->toBasetype()->ty == Taarray)
        {
            // Assignment to an AA of fixed-length arrays.
            // Convert T[n][U] = T[] into T[n][U] = T[n]
            e2 = e2->implicitCastTo(sc, e1->type);
            if (e2->type == Type::terror)
                return e2;
        }
        else
        {
            // Convert e2 to e2[], unless e2-> e1[0]
            if (t2->ty == Tsarray && !t2->implicitConvTo(t1->nextOf()))
            {
                e2 = new SliceExp(e2->loc, e2, NULL, NULL);
                e2 = e2->semantic(sc);
            }

            // Convert e1 to e1[]
            Expression *e = new SliceExp(e1->loc, e1, NULL, NULL);
            e1 = e->semantic(sc);
            t1 = e1->type->toBasetype();
        }
    }

    if (e1->op == TOKarraylength)
    {
        // e1 is not an lvalue, but we let code generator handle it
        ArrayLengthExp *ale = (ArrayLengthExp *)e1;

        ale->e1 = ale->e1->modifiableLvalue(sc, e1);
    }
    else if (e1->op == TOKslice)
    {
        Type *tn = e1->type->nextOf();
        if (op == TOKassign && e1->checkModifiable(sc) == 1 && !tn->isMutable())
        {   error("slice %s is not mutable", e1->toChars());
            return new ErrorExp();
        }
    }
    else
    {   // Try to do a decent error message with the expression
        // before it got constant folded
        if (e1->op != TOKvar)
            e1 = e1->optimize(WANTvalue);

        if (op != TOKconstruct)
            e1 = e1->modifiableLvalue(sc, e1old);
    }

    Type *t2 = e2->type->toBasetype();

    // If it is a array, get the element type. Note that it may be
    // multi-dimensional.
    Type *telem = t1;
    while (telem->ty == Tarray)
        telem = telem->nextOf();

    // Check for block assignment. If it is of type void[], void[][], etc,
    // '= null' is the only allowable block assignment (Bug 7493)
    if (e1->op == TOKslice &&
        t1->nextOf() && (telem->ty != Tvoid || e2->op == TOKnull) &&
        e2->implicitConvTo(t1->nextOf())
       )
    {   // memset
        ismemset = 1;   // make it easy for back end to tell what this is
        e2 = e2->implicitCastTo(sc, t1->nextOf());
    }
    else if (t1->ty == Tsarray)
    {
        /* Should have already converted e1 => e1[]
         * unless it is an AA
         */
        if (!(e1->op == TOKindex && t2->ty == Tsarray &&
            ((IndexExp *)e1)->e1->type->toBasetype()->ty == Taarray))
        {
            assert(op == TOKconstruct);
        }
        //error("cannot assign to static array %s", e1->toChars());
    }
    // Check element-wise assignment.
    else if (e1->op == TOKslice &&
        (t2->ty == Tarray || t2->ty == Tsarray) &&
        t2->nextOf()->implicitConvTo(t1->nextOf()))
    {
        if (((SliceExp *)e1)->lwr == NULL)
        {
            Type *tx1 = ((SliceExp *)e1)->e1->type->toBasetype();
            Type *tx2 = t2;
            if (e2->op == TOKslice && ((SliceExp *)e2)->lwr == NULL)
                tx2 = ((SliceExp *)e2)->e1->type->toBasetype();
            if (tx1->ty == Tsarray && tx2->ty == Tsarray)
            {   // sa1[] = sa2[];
                // sa1[] = sa2;
                TypeSArray *tsa1 = (TypeSArray *)tx1;
                TypeSArray *tsa2 = (TypeSArray *)tx2;
                uinteger_t dim1 = tsa1->dim->toInteger();
                uinteger_t dim2 = tsa2->dim->toInteger();
                if (dim1 != dim2)
                {
                    error("mismatched array lengths, %d and %d", (int)dim1, (int)dim2);
                    return new ErrorExp();
                }
            }
        }
        if (op != TOKblit &&
            (e2->op == TOKslice && ((UnaExp *)e2)->e1->isLvalue() ||
             e2->op == TOKcast  && ((UnaExp *)e2)->e1->isLvalue() ||
             e2->op != TOKslice && e2->isLvalue()))
        {
            checkPostblit(e2->loc, t2->nextOf());
        }
        if (op == TOKconstruct)
            e2 = e2->castTo(sc, e1->type->constOf());
        else
            e2 = e2->implicitCastTo(sc, e1->type->constOf());
    }
    else
    {
        e2 = e2->implicitCastTo(sc, e1->type);
    }
    if (e2->op == TOKerror)
        return new ErrorExp();

    /* Look for array operations
     */
    if (e1->op == TOKslice && !ismemset &&
        (e2->op == TOKadd || e2->op == TOKmin ||
         e2->op == TOKmul || e2->op == TOKdiv ||
         e2->op == TOKmod || e2->op == TOKxor ||
         e2->op == TOKand || e2->op == TOKor  ||
#if DMDV2
         e2->op == TOKpow ||
#endif
         e2->op == TOKtilde || e2->op == TOKneg))
    {
        type = e1->type;
        return arrayOp(sc);
    }

    if (e1->op == TOKvar &&
        (((VarExp *)e1)->var->storage_class & STCscope) &&
        op == TOKassign)
    {
        error("cannot rebind scope variables");
    }
    if (e1->op == TOKvar && ((VarExp*)e1)->var->ident == Id::ctfe)
    {
        error("cannot modify compiler-generated variable __ctfe");
    }

    type = e1->type;
    assert(type);
    return this;
}

Expression *AssignExp::checkToBoolean(Scope *sc)
{
    // Things like:
    //  if (a = b) ...
    // are usually mistakes.

    error("assignment cannot be used as a condition, perhaps == was meant?");
    return new ErrorExp();
}

/************************************************************/

ConstructExp::ConstructExp(Loc loc, Expression *e1, Expression *e2)
    : AssignExp(loc, e1, e2)
{
    op = TOKconstruct;
}

/************************************************************/

AddAssignExp::AddAssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinAssignExp(loc, TOKaddass, sizeof(AddAssignExp), e1, e2)
{
}

/************************************************************/

MinAssignExp::MinAssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinAssignExp(loc, TOKminass, sizeof(MinAssignExp), e1, e2)
{
}

/************************************************************/

CatAssignExp::CatAssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinAssignExp(loc, TOKcatass, sizeof(CatAssignExp), e1, e2)
{
}

Expression *CatAssignExp::semantic(Scope *sc)
{
    //printf("CatAssignExp::semantic() %s\n", toChars());
    Expression *e = op_overload(sc);
    if (e)
        return e;

    if (e1->op == TOKslice)
    {   SliceExp *se = (SliceExp *)e1;

        if (se->e1->type->toBasetype()->ty == Tsarray)
        {   error("cannot append to static array %s", se->e1->type->toChars());
            return new ErrorExp();
        }
    }

    e1 = e1->modifiableLvalue(sc, e1);
    if (e1->op == TOKerror)
        return e1;

    Type *tb1 = e1->type->toBasetype();
    Type *tb1next = tb1->nextOf();

    e2 = e2->inferType(tb1next);
    if (!e2->rvalue())
        return new ErrorExp();

    Type *tb2 = e2->type->toBasetype();

    if ((tb1->ty == Tarray) &&
        (tb2->ty == Tarray || tb2->ty == Tsarray) &&
        (e2->implicitConvTo(e1->type)
#if DMDV2
         || (tb2->nextOf()->implicitConvTo(tb1next) &&
             (tb2->nextOf()->size(0) == tb1next->size(0) ||
             tb1next->ty == Tchar || tb1next->ty == Twchar || tb1next->ty == Tdchar))
#endif
        )
       )
    {   // Append array
        checkPostblit(e1->loc, tb1next);
        e2 = e2->castTo(sc, e1->type);
        type = e1->type;
        e = this;
    }
    else if ((tb1->ty == Tarray) &&
        e2->implicitConvTo(tb1next)
       )
    {   // Append element
        checkPostblit(e2->loc, tb2);
        e2 = e2->castTo(sc, tb1next);
        type = e1->type;
        e = this;
    }
    else if (tb1->ty == Tarray &&
        (tb1next->ty == Tchar || tb1next->ty == Twchar) &&
        e2->type->ty != tb1next->ty &&
        e2->implicitConvTo(Type::tdchar)
       )
    {   // Append dchar to char[] or wchar[]
        e2 = e2->castTo(sc, Type::tdchar);
        type = e1->type;
        e = this;

        /* Do not allow appending wchar to char[] because if wchar happens
         * to be a surrogate pair, nothing good can result.
         */
    }
    else
    {
        if (tb1 != Type::terror && tb2 != Type::terror)
            error("cannot append type %s to type %s", tb2->toChars(), tb1->toChars());
        e = new ErrorExp();
    }
    return e;
}

/************************************************************/

MulAssignExp::MulAssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinAssignExp(loc, TOKmulass, sizeof(MulAssignExp), e1, e2)
{
}

/************************************************************/

DivAssignExp::DivAssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinAssignExp(loc, TOKdivass, sizeof(DivAssignExp), e1, e2)
{
}

/************************************************************/

ModAssignExp::ModAssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinAssignExp(loc, TOKmodass, sizeof(ModAssignExp), e1, e2)
{
}

/************************************************************/

ShlAssignExp::ShlAssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinAssignExp(loc, TOKshlass, sizeof(ShlAssignExp), e1, e2)
{
}

/************************************************************/

ShrAssignExp::ShrAssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinAssignExp(loc, TOKshrass, sizeof(ShrAssignExp), e1, e2)
{
}

/************************************************************/

UshrAssignExp::UshrAssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinAssignExp(loc, TOKushrass, sizeof(UshrAssignExp), e1, e2)
{
}

/************************************************************/

AndAssignExp::AndAssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinAssignExp(loc, TOKandass, sizeof(AndAssignExp), e1, e2)
{
}

/************************************************************/

OrAssignExp::OrAssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinAssignExp(loc, TOKorass, sizeof(OrAssignExp), e1, e2)
{
}

/************************************************************/

XorAssignExp::XorAssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinAssignExp(loc, TOKxorass, sizeof(XorAssignExp), e1, e2)
{
}

/***************** PowAssignExp *******************************************/

PowAssignExp::PowAssignExp(Loc loc, Expression *e1, Expression *e2)
        : BinAssignExp(loc, TOKpowass, sizeof(PowAssignExp), e1, e2)
{
}

Expression *PowAssignExp::semantic(Scope *sc)
{
    Expression *e;

    if (type)
        return this;

    e = op_overload(sc);
    if (e)
        return e;

    assert(e1->type && e2->type);
    if (e1->op == TOKslice)
    {   // T[] ^^= ...
        e = typeCombine(sc);
        if (e->op == TOKerror)
            return e;

        // Check element types are arithmetic
        Type *tb1 = e1->type->nextOf()->toBasetype();
        Type *tb2 = e2->type->toBasetype();
        if (tb2->ty == Tarray || tb2->ty == Tsarray)
            tb2 = tb2->nextOf()->toBasetype();

        if ( (tb1->isintegral() || tb1->isfloating()) &&
             (tb2->isintegral() || tb2->isfloating()))
        {
            type = e1->type;
            return arrayOp(sc);
        }
    }
    else
    {
        e1 = e1->modifiableLvalue(sc, e1);
    }

    if ( (e1->type->isintegral() || e1->type->isfloating()) &&
         (e2->type->isintegral() || e2->type->isfloating()))
    {
        if (e1->op == TOKvar)
        {   // Rewrite: e1 = e1 ^^ e2
            e = new PowExp(loc, e1->syntaxCopy(), e2);
            e = new AssignExp(loc, e1, e);
        }
        else
        {   // Rewrite: ref tmp = e1; tmp = tmp ^^ e2
            Identifier *id = Lexer::uniqueId("__powtmp");
            VarDeclaration *v = new VarDeclaration(e1->loc, e1->type, id, new ExpInitializer(loc, e1));
            v->storage_class |= STCref | STCforeach;
            Expression *de = new DeclarationExp(e1->loc, v);
            VarExp *ve = new VarExp(e1->loc, v);
            e = new PowExp(loc, ve, e2);
            e = new AssignExp(loc, new VarExp(e1->loc, v), e);
            e = new CommaExp(loc, de, e);
        }
        e = e->semantic(sc);
        if (e->type->toBasetype()->ty == Tvector)
            return incompatibleTypes();
        return e;
    }
    return incompatibleTypes();
}


/************************* AddExp *****************************/

AddExp::AddExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKadd, sizeof(AddExp), e1, e2)
{
}

Expression *AddExp::semantic(Scope *sc)
{   Expression *e;

#if LOGSEMANTIC
    printf("AddExp::semantic('%s')\n", toChars());
#endif
    if (!type)
    {
        BinExp::semanticp(sc);

        e = op_overload(sc);
        if (e)
            return e;

        Type *tb1 = e1->type->toBasetype();
        Type *tb2 = e2->type->toBasetype();

        if ((tb1->ty == Tarray || tb1->ty == Tsarray) &&
            (tb2->ty == Tarray || tb2->ty == Tsarray) &&
            tb1->nextOf()->equals(tb2->nextOf())
           )
        {
            type = e1->type;
            e = this;
        }
        else if (tb1->ty == Tpointer && e2->type->isintegral() ||
            tb2->ty == Tpointer && e1->type->isintegral())
            e = scaleFactor(sc);
        else if (tb1->ty == Tpointer && tb2->ty == Tpointer)
        {
            return incompatibleTypes();
        }
        else
        {
            typeCombine(sc);
            Type *tb1 = e1->type->toBasetype();
            if (tb1->ty == Tvector && !tb1->isscalar())
            {
                return incompatibleTypes();
            }
            if ((tb1->isreal() && e2->type->isimaginary()) ||
                (tb1->isimaginary() && e2->type->isreal()))
            {
                switch (type->toBasetype()->ty)
                {
                    case Tfloat32:
                    case Timaginary32:
                        type = Type::tcomplex32;
                        break;

                    case Tfloat64:
                    case Timaginary64:
                        type = Type::tcomplex64;
                        break;

                    case Tfloat80:
                    case Timaginary80:
                        type = Type::tcomplex80;
                        break;

                    default:
                        assert(0);
                }
            }
            e = this;
        }
        return e;
    }
    return this;
}

/************************************************************/

MinExp::MinExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKmin, sizeof(MinExp), e1, e2)
{
}

Expression *MinExp::semantic(Scope *sc)
{   Expression *e;

#if LOGSEMANTIC
    printf("MinExp::semantic('%s')\n", toChars());
#endif
    if (type)
        return this;

    BinExp::semanticp(sc);

    e = op_overload(sc);
    if (e)
        return e;

    e = this;
    Type *t1 = e1->type->toBasetype();
    Type *t2 = e2->type->toBasetype();
    if (t1->ty == Tpointer)
    {
        if (t2->ty == Tpointer)
        {   // Need to divide the result by the stride
            // Replace (ptr - ptr) with (ptr - ptr) / stride
            d_int64 stride;
            Expression *e;

            typeCombine(sc);            // make sure pointer types are compatible
            type = Type::tptrdiff_t;
            stride = t2->nextOf()->size();
            if (stride == 0)
            {
                e = new IntegerExp(loc, 0, Type::tptrdiff_t);
            }
            else
            {
                e = new DivExp(loc, this, new IntegerExp(0, stride, Type::tptrdiff_t));
                e->type = Type::tptrdiff_t;
            }
            return e;
        }
        else if (t2->isintegral())
            e = scaleFactor(sc);
        else
        {   error("can't subtract %s from pointer", t2->toChars());
            return new ErrorExp();
        }
    }
    else if (t2->ty == Tpointer)
    {
        type = e2->type;
        error("can't subtract pointer from %s", e1->type->toChars());
        return new ErrorExp();
    }
    else
    {
        typeCombine(sc);
        t1 = e1->type->toBasetype();
        t2 = e2->type->toBasetype();
        if (t1->ty == Tvector && !t1->isscalar())
        {
            return incompatibleTypes();
        }
        if ((t1->isreal() && t2->isimaginary()) ||
            (t1->isimaginary() && t2->isreal()))
        {
            switch (type->ty)
            {
                case Tfloat32:
                case Timaginary32:
                    type = Type::tcomplex32;
                    break;

                case Tfloat64:
                case Timaginary64:
                    type = Type::tcomplex64;
                    break;

                case Tfloat80:
                case Timaginary80:
                    type = Type::tcomplex80;
                    break;

                default:
                    assert(0);
            }
        }
    }
    return e;
}

/************************* CatExp *****************************/

CatExp::CatExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKcat, sizeof(CatExp), e1, e2)
{
}

Expression *CatExp::semantic(Scope *sc)
{   Expression *e;

    //printf("CatExp::semantic() %s\n", toChars());
    if (!type)
    {
        BinExp::semanticp(sc);
        e = op_overload(sc);
        if (e)
            return e;

        Type *tb1 = e1->type->toBasetype();
        Type *tb2 = e2->type->toBasetype();


        /* BUG: Should handle things like:
         *      char c;
         *      c ~ ' '
         *      ' ' ~ c;
         */

#if 0
        e1->type->print();
        e2->type->print();
#endif
        Type *tb1next = tb1->nextOf();
        Type *tb2next = tb2->nextOf();

        if (tb1next && tb2next &&
            (tb1next->implicitConvTo(tb2next) >= MATCHconst ||
             tb2next->implicitConvTo(tb1next) >= MATCHconst)
           )
        {
            /* Here to avoid the case of:
             *    void*[] a = [cast(void*)1];
             *    void*[] b = [cast(void*)2];
             *    a ~ b;
             * becoming:
             *    a ~ [cast(void*)b];
             */
        }
        else if ((tb1->ty == Tsarray || tb1->ty == Tarray) &&
            e2->implicitConvTo(tb1next) >= MATCHconvert &&
            tb2->ty != Tvoid)
        {
            checkPostblit(e2->loc, tb2);
            e2 = e2->implicitCastTo(sc, tb1next);
            type = tb1next->arrayOf();
            if (tb2->ty == Tarray)
            {   // Make e2 into [e2]
                e2 = new ArrayLiteralExp(e2->loc, e2);
                e2->type = type;
            }
            return this;
        }
        else if ((tb2->ty == Tsarray || tb2->ty == Tarray) &&
            e1->implicitConvTo(tb2next) >= MATCHconvert &&
            tb1->ty != Tvoid)
        {
            checkPostblit(e1->loc, tb1);
            e1 = e1->implicitCastTo(sc, tb2next);
            type = tb2next->arrayOf();
            if (tb1->ty == Tarray)
            {   // Make e1 into [e1]
                e1 = new ArrayLiteralExp(e1->loc, e1);
                e1->type = type;
            }
            return this;
        }

        if ((tb1->ty == Tsarray || tb1->ty == Tarray) &&
            (tb2->ty == Tsarray || tb2->ty == Tarray) &&
            (tb1next->mod || tb2next->mod) &&
            (tb1next->mod != tb2next->mod)
           )
        {
            Type *t1 = tb1next->mutableOf()->constOf()->arrayOf();
            Type *t2 = tb2next->mutableOf()->constOf()->arrayOf();
            if (e1->op == TOKstring && !((StringExp *)e1)->committed)
                e1->type = t1;
            else
                e1 = e1->castTo(sc, t1);
            if (e2->op == TOKstring && !((StringExp *)e2)->committed)
                e2->type = t2;
            else
                e2 = e2->castTo(sc, t2);
        }

        typeCombine(sc);
        type = type->toHeadMutable();

        Type *tb = type->toBasetype();
        if (tb->ty == Tsarray)
            type = tb->nextOf()->arrayOf();
        if (type->ty == Tarray && tb1next && tb2next &&
            tb1next->mod != tb2next->mod)
        {
            type = type->nextOf()->toHeadMutable()->arrayOf();
        }
        if (tb->nextOf())
        {
            checkPostblit(loc, tb->nextOf());
        }
#if 0
        e1->type->print();
        e2->type->print();
        type->print();
        print();
#endif
        Type *t1 = e1->type->toBasetype();
        Type *t2 = e2->type->toBasetype();
        if (e1->op == TOKstring && e2->op == TOKstring)
            e = optimize(WANTvalue);
        else if ((t1->ty == Tarray || t1->ty == Tsarray) &&
                 (t2->ty == Tarray || t2->ty == Tsarray))
        {
            e = this;
        }
        else
        {
            //printf("(%s) ~ (%s)\n", e1->toChars(), e2->toChars());
            incompatibleTypes();
            return new ErrorExp();
        }
        e->type = e->type->semantic(loc, sc);
        return e;
    }
    return this;
}

/************************************************************/

MulExp::MulExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKmul, sizeof(MulExp), e1, e2)
{
}

Expression *MulExp::semantic(Scope *sc)
{   Expression *e;

#if 0
    printf("MulExp::semantic() %s\n", toChars());
#endif
    if (type)
    {
        return this;
    }

    BinExp::semanticp(sc);
    e = op_overload(sc);
    if (e)
        return e;

    typeCombine(sc);
    if (!e1->isArrayOperand())
        e1->checkArithmetic();
    if (!e2->isArrayOperand())
        e2->checkArithmetic();
    if (type->isfloating())
    {   Type *t1 = e1->type;
        Type *t2 = e2->type;

        if (t1->isreal())
        {
            type = t2;
        }
        else if (t2->isreal())
        {
            type = t1;
        }
        else if (t1->isimaginary())
        {
            if (t2->isimaginary())
            {   Expression *e;

                switch (t1->toBasetype()->ty)
                {
                    case Timaginary32:  type = Type::tfloat32;  break;
                    case Timaginary64:  type = Type::tfloat64;  break;
                    case Timaginary80:  type = Type::tfloat80;  break;
                    default:            assert(0);
                }

                // iy * iv = -yv
                e1->type = type;
                e2->type = type;
                e = new NegExp(loc, this);
                e = e->semantic(sc);
                return e;
            }
            else
                type = t2;      // t2 is complex
        }
        else if (t2->isimaginary())
        {
            type = t1;  // t1 is complex
        }
    }
    else if (type->toBasetype()->ty == Tvector &&
             ((TypeVector *)type->toBasetype())->elementType()->size(loc) != 2)
    {   // Only short[8] and ushort[8] work with multiply
        return incompatibleTypes();
    }
    return this;
}

/************************************************************/

DivExp::DivExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKdiv, sizeof(DivExp), e1, e2)
{
}

Expression *DivExp::semantic(Scope *sc)
{   Expression *e;

    if (type)
        return this;

    BinExp::semanticp(sc);
    e = op_overload(sc);
    if (e)
        return e;

    typeCombine(sc);
    if (!e1->isArrayOperand())
        e1->checkArithmetic();
    if (!e2->isArrayOperand())
        e2->checkArithmetic();
    if (type->isfloating())
    {   Type *t1 = e1->type;
        Type *t2 = e2->type;

        if (t1->isreal())
        {
            type = t2;
            if (t2->isimaginary())
            {   Expression *e;

                // x/iv = i(-x/v)
                e2->type = t1;
                e = new NegExp(loc, this);
                e = e->semantic(sc);
                return e;
            }
        }
        else if (t2->isreal())
        {
            type = t1;
        }
        else if (t1->isimaginary())
        {
            if (t2->isimaginary())
            {
                switch (t1->toBasetype()->ty)
                {
                    case Timaginary32:  type = Type::tfloat32;  break;
                    case Timaginary64:  type = Type::tfloat64;  break;
                    case Timaginary80:  type = Type::tfloat80;  break;
                    default:            assert(0);
                }
            }
            else
                type = t2;      // t2 is complex
        }
        else if (t2->isimaginary())
        {
            type = t1;  // t1 is complex
        }
    }
    else if (type->toBasetype()->ty == Tvector)
    {   incompatibleTypes();
        return new ErrorExp();
    }
    return this;
}

/************************************************************/

ModExp::ModExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKmod, sizeof(ModExp), e1, e2)
{
}

Expression *ModExp::semantic(Scope *sc)
{   Expression *e;

    if (type)
        return this;

    BinExp::semanticp(sc);
    e = op_overload(sc);
    if (e)
        return e;

    typeCombine(sc);
    if (!e1->isArrayOperand())
        e1->checkArithmetic();
    if (!e2->isArrayOperand())
        e2->checkArithmetic();
    if (type->toBasetype()->ty == Tvector)
    {   incompatibleTypes();
        return new ErrorExp();
    }
    if (type->isfloating())
    {   type = e1->type;
        if (e2->type->iscomplex())
        {   error("cannot perform modulo complex arithmetic");
            return new ErrorExp();
        }
    }
    return this;
}

/************************************************************/

PowExp::PowExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKpow, sizeof(PowExp), e1, e2)
{
}

Expression *PowExp::semantic(Scope *sc)
{   Expression *e;

    if (type)
        return this;

    //printf("PowExp::semantic() %s\n", toChars());
    BinExp::semanticp(sc);
    e = op_overload(sc);
    if (e)
        return e;

    assert(e1->type && e2->type);
    typeCombine(sc);

    if (e1->op == TOKslice)
    {
        // Check element types are arithmetic
        Type *tb1 = e1->type->nextOf()->toBasetype();
        Type *tb2 = e2->type->toBasetype();
        if (tb2->ty == Tarray || tb2->ty == Tsarray)
            tb2 = tb2->nextOf()->toBasetype();

        if ( (tb1->isintegral() || tb1->isfloating()) &&
             (tb2->isintegral() || tb2->isfloating()))
        {
            type = e1->type;
            return this;
        }
    }

    if ( (e1->type->isintegral() || e1->type->isfloating()) &&
         (e2->type->isintegral() || e2->type->isfloating()))
    {
        // For built-in numeric types, there are several cases.
        // TODO: backend support, especially for  e1 ^^ 2.

        bool wantSqrt = false;

        // First, attempt to fold the expression.
        e = optimize(WANTvalue);
        if (e->op != TOKpow)
        {
            e = e->semantic(sc);
            return e;
        }

        // Determine if we're raising to an integer power.
        sinteger_t intpow = 0;
        if (e2->op == TOKint64 && ((sinteger_t)e2->toInteger() == 2 || (sinteger_t)e2->toInteger() == 3))
            intpow = e2->toInteger();
        else if (e2->op == TOKfloat64 && (e2->toReal() == (sinteger_t)(e2->toReal())))
            intpow = (sinteger_t)(e2->toReal());

        // Deal with x^^2, x^^3 immediately, since they are of practical importance.
        if (intpow == 2 || intpow == 3)
        {
            // Replace x^^2 with (tmp = x, tmp*tmp)
            // Replace x^^3 with (tmp = x, tmp*tmp*tmp)
            Identifier *idtmp = Lexer::uniqueId("__powtmp");
            VarDeclaration *tmp = new VarDeclaration(loc, e1->type->toBasetype(), idtmp, new ExpInitializer(0, e1));
            tmp->storage_class = STCctfe;
            Expression *ve = new VarExp(loc, tmp);
            Expression *ae = new DeclarationExp(loc, tmp);
            /* Note that we're reusing ve. This should be ok.
             */
            Expression *me = new MulExp(loc, ve, ve);
            if (intpow == 3)
                me = new MulExp(loc, me, ve);
            e = new CommaExp(loc, ae, me);
            e = e->semantic(sc);
            return e;
        }

        static int importMathChecked = 0;
        static bool importMath = false;
        if (!importMathChecked)
        {
            importMathChecked = 1;
            for (size_t i = 0; i < Module::amodules.dim; i++)
            {   Module *mi = Module::amodules[i];
                //printf("\t[%d] %s\n", i, mi->toChars());
                if (mi->ident == Id::math &&
                    mi->parent->ident == Id::std &&
                    !mi->parent->parent)
                {
                    importMath = true;
                    goto L1;
                }
            }
            error("must import std.math to use ^^ operator");
            return new ErrorExp();

         L1: ;
        }
        else
        {
            if (!importMath)
            {
                error("must import std.math to use ^^ operator");
                return new ErrorExp();
            }
        }

        e = new IdentifierExp(loc, Id::empty);
        e = new DotIdExp(loc, e, Id::std);
        e = new DotIdExp(loc, e, Id::math);
        if (e2->op == TOKfloat64 && e2->toReal() == 0.5)
        {   // Replace e1 ^^ 0.5 with .std.math.sqrt(x)
            e = new CallExp(loc, new DotIdExp(loc, e, Id::_sqrt), e1);
        }
        else
        {
            // Replace e1 ^^ e2 with .std.math.pow(e1, e2)
            e = new CallExp(loc, new DotIdExp(loc, e, Id::_pow), e1, e2);
        }
        e = e->semantic(sc);
        return e;
    }
    return incompatibleTypes();
}

/************************************************************/

ShlExp::ShlExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKshl, sizeof(ShlExp), e1, e2)
{
}

Expression *ShlExp::semantic(Scope *sc)
{   Expression *e;

    //printf("ShlExp::semantic(), type = %p\n", type);
    if (!type)
    {   BinExp::semanticp(sc);
        e = op_overload(sc);
        if (e)
            return e;
        e1 = e1->checkIntegral();
        e2 = e2->checkIntegral();
        if (e1->type->toBasetype()->ty == Tvector ||
            e2->type->toBasetype()->ty == Tvector)
            return incompatibleTypes();
        e1 = e1->integralPromotions(sc);
#if IN_LLVM
        e2 = e2->castTo(sc, e1->type);
#else
        e2 = e2->castTo(sc, Type::tshiftcnt);
#endif
        type = e1->type;
    }
    return this;
}

/************************************************************/

ShrExp::ShrExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKshr, sizeof(ShrExp), e1, e2)
{
}

Expression *ShrExp::semantic(Scope *sc)
{   Expression *e;

    if (!type)
    {   BinExp::semanticp(sc);
        e = op_overload(sc);
        if (e)
            return e;
        e1 = e1->checkIntegral();
        e2 = e2->checkIntegral();
        if (e1->type->toBasetype()->ty == Tvector ||
            e2->type->toBasetype()->ty == Tvector)
            return incompatibleTypes();
        e1 = e1->integralPromotions(sc);
#if IN_LLVM
        e2 = e2->castTo(sc, e1->type);
#else
        e2 = e2->castTo(sc, Type::tshiftcnt);
#endif
        type = e1->type;
    }
    return this;
}

/************************************************************/

UshrExp::UshrExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKushr, sizeof(UshrExp), e1, e2)
{
}

Expression *UshrExp::semantic(Scope *sc)
{   Expression *e;

    if (!type)
    {   BinExp::semanticp(sc);
        e = op_overload(sc);
        if (e)
            return e;
        e1 = e1->checkIntegral();
        e2 = e2->checkIntegral();
        if (e1->type->toBasetype()->ty == Tvector ||
            e2->type->toBasetype()->ty == Tvector)
            return incompatibleTypes();
        e1 = e1->integralPromotions(sc);
#if IN_LLVM
        e2 = e2->castTo(sc, e1->type);
#else
        e2 = e2->castTo(sc, Type::tshiftcnt);
#endif
        type = e1->type;
    }
    return this;
}

/************************************************************/

AndExp::AndExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKand, sizeof(AndExp), e1, e2)
{
}

Expression *AndExp::semantic(Scope *sc)
{   Expression *e;

    if (!type)
    {   BinExp::semanticp(sc);
        e = op_overload(sc);
        if (e)
            return e;
        if (e1->type->toBasetype()->ty == Tbool &&
            e2->type->toBasetype()->ty == Tbool)
        {
            type = e1->type;
            e = this;
        }
        else
        {
            typeCombine(sc);
            if (!e1->isArrayOperand())
                e1->checkIntegral();
            if (!e2->isArrayOperand())
                e2->checkIntegral();
        }
    }
    return this;
}

/************************************************************/

OrExp::OrExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKor, sizeof(OrExp), e1, e2)
{
}

Expression *OrExp::semantic(Scope *sc)
{   Expression *e;

    if (!type)
    {   BinExp::semanticp(sc);
        e = op_overload(sc);
        if (e)
            return e;
        if (e1->type->toBasetype()->ty == Tbool &&
            e2->type->toBasetype()->ty == Tbool)
        {
            type = e1->type;
            e = this;
        }
        else
        {
            typeCombine(sc);
            if (!e1->isArrayOperand())
                e1->checkIntegral();
            if (!e2->isArrayOperand())
                e2->checkIntegral();
        }
    }
    return this;
}

/************************************************************/

XorExp::XorExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKxor, sizeof(XorExp), e1, e2)
{
}

Expression *XorExp::semantic(Scope *sc)
{   Expression *e;

    if (!type)
    {   BinExp::semanticp(sc);
        e = op_overload(sc);
        if (e)
            return e;
        if (e1->type->toBasetype()->ty == Tbool &&
            e2->type->toBasetype()->ty == Tbool)
        {
            type = e1->type;
            e = this;
        }
        else
        {
            typeCombine(sc);
            if (!e1->isArrayOperand())
                e1->checkIntegral();
            if (!e2->isArrayOperand())
                e2->checkIntegral();
        }
    }
    return this;
}


/************************************************************/

OrOrExp::OrOrExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKoror, sizeof(OrOrExp), e1, e2)
{
}

Expression *OrOrExp::semantic(Scope *sc)
{
    unsigned cs1;

    // same as for AndAnd
    e1 = e1->semantic(sc);
    e1 = resolveProperties(sc, e1);
    e1 = e1->checkToPointer();
    e1 = e1->checkToBoolean(sc);
    cs1 = sc->callSuper;

    if (sc->flags & SCOPEstaticif)
    {
        /* If in static if, don't evaluate e2 if we don't have to.
         */
        e1 = e1->optimize(WANTflags);
        if (e1->isBool(TRUE))
        {
            return new IntegerExp(loc, 1, Type::tboolean);
        }
    }

    e2 = e2->semantic(sc);
    sc->mergeCallSuper(loc, cs1);
    e2 = resolveProperties(sc, e2);
    e2 = e2->checkToPointer();

    if (e2->type->ty == Tvoid)
        type = Type::tvoid;
    else
    {
        e2 = e2->checkToBoolean(sc);
        type = Type::tboolean;
    }
    if (e2->op == TOKtype || e2->op == TOKimport)
    {   error("%s is not an expression", e2->toChars());
        return new ErrorExp();
    }
    if (e1->op == TOKerror)
        return e1;
    if (e2->op == TOKerror)
        return e2;
    return this;
}

Expression *OrOrExp::checkToBoolean(Scope *sc)
{
    e2 = e2->checkToBoolean(sc);
    return this;
}

int OrOrExp::isBit()
{
    return TRUE;
}


/************************************************************/

AndAndExp::AndAndExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKandand, sizeof(AndAndExp), e1, e2)
{
}

Expression *AndAndExp::semantic(Scope *sc)
{
    unsigned cs1;

    // same as for OrOr
    e1 = e1->semantic(sc);
    e1 = resolveProperties(sc, e1);
    e1 = e1->checkToPointer();
    e1 = e1->checkToBoolean(sc);
    cs1 = sc->callSuper;

    if (sc->flags & SCOPEstaticif)
    {
        /* If in static if, don't evaluate e2 if we don't have to.
         */
        e1 = e1->optimize(WANTflags);
        if (e1->isBool(FALSE))
        {
            return new IntegerExp(loc, 0, Type::tboolean);
        }
    }

    e2 = e2->semantic(sc);
    sc->mergeCallSuper(loc, cs1);
    e2 = resolveProperties(sc, e2);
    e2 = e2->checkToPointer();

    if (e2->type->ty == Tvoid)
        type = Type::tvoid;
    else
    {
        e2 = e2->checkToBoolean(sc);
        type = Type::tboolean;
    }
    if (e2->op == TOKtype || e2->op == TOKimport)
    {   error("%s is not an expression", e2->toChars());
        return new ErrorExp();
    }
    if (e1->op == TOKerror)
        return e1;
    if (e2->op == TOKerror)
        return e2;
    return this;
}

Expression *AndAndExp::checkToBoolean(Scope *sc)
{
    e2 = e2->checkToBoolean(sc);
    return this;
}

int AndAndExp::isBit()
{
    return TRUE;
}


/************************************************************/

InExp::InExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKin, sizeof(InExp), e1, e2)
{
}

Expression *InExp::semantic(Scope *sc)
{   Expression *e;

    if (type)
        return this;

    BinExp::semanticp(sc);
    e = op_overload(sc);
    if (e)
        return e;

    //type = Type::tboolean;
    Type *t2b = e2->type->toBasetype();
    switch (t2b->ty)
    {
        case Taarray:
        {
            TypeAArray *ta = (TypeAArray *)t2b;

#if DMDV2
            // Special handling for array keys
            if (!arrayTypeCompatible(e1->loc, e1->type, ta->index))
#endif
            {
                // Convert key to type of key
                e1 = e1->implicitCastTo(sc, ta->index);
            }

            // Return type is pointer to value
            type = ta->nextOf()->pointerTo();
            break;
        }

        default:
            error("rvalue of in expression must be an associative array, not %s", e2->type->toChars());
        case Terror:
            return new ErrorExp();
    }
    return this;
}

int InExp::isBit()
{
    return FALSE;
}


/************************************************************/

/* This deletes the key e1 from the associative array e2
 */

RemoveExp::RemoveExp(Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, TOKremove, sizeof(RemoveExp), e1, e2)
{
    type = Type::tboolean;
}

void RemoveExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    expToCBuffer(buf, hgs, e1, PREC_primary);
    buf->writestring(".remove(");
    expToCBuffer(buf, hgs, e2, PREC_assign);
    buf->writestring(")");
}

/************************************************************/

CmpExp::CmpExp(enum TOK op, Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, op, sizeof(CmpExp), e1, e2)
{
}

Expression *CmpExp::semantic(Scope *sc)
{   Expression *e;

#if LOGSEMANTIC
    printf("CmpExp::semantic('%s')\n", toChars());
#endif
    if (type)
        return this;

    BinExp::semanticp(sc);

    Type *t1 = e1->type->toBasetype();
    Type *t2 = e2->type->toBasetype();
    if (t1->ty == Tclass && e2->op == TOKnull ||
        t2->ty == Tclass && e1->op == TOKnull)
    {
        error("do not use null when comparing class types");
        return new ErrorExp();
    }

    e = op_overload(sc);
    if (e)
    {
        if (!e->type->isscalar() && e->type->equals(e1->type))
        {
            error("recursive opCmp expansion");
            e = new ErrorExp();
        }
        else if (e->op == TOKcall)
        {   e = new CmpExp(op, loc, e, new IntegerExp(loc, 0, Type::tint32));
            e = e->semantic(sc);
        }
        return e;
    }

    /* Disallow comparing T[]==T and T==T[]
     */
    if (e1->op == TOKslice && t1->ty == Tarray && e2->implicitConvTo(t1->nextOf()) ||
        e2->op == TOKslice && t2->ty == Tarray && e1->implicitConvTo(t2->nextOf()))
    {
        incompatibleTypes();
        return new ErrorExp();
    }

    Expression *eb1 = e1;
    Expression *eb2 = e2;

    e = typeCombine(sc);
    if (e->op == TOKerror)
        return e;

    type = Type::tboolean;

    // Special handling for array comparisons
    t1 = e1->type->toBasetype();
    t2 = e2->type->toBasetype();
    if ((t1->ty == Tarray || t1->ty == Tsarray || t1->ty == Tpointer) &&
        (t2->ty == Tarray || t2->ty == Tsarray || t2->ty == Tpointer))
    {
        Type *t1next = t1->nextOf();
        Type *t2next = t2->nextOf();
        if (t1next->implicitConvTo(t2next) < MATCHconst &&
            t2next->implicitConvTo(t1next) < MATCHconst &&
            (t1next->ty != Tvoid && t2next->ty != Tvoid))
            error("array comparison type mismatch, %s vs %s", t1next->toChars(), t2next->toChars());
        e = this;
    }
    else if (t1->ty == Tstruct || t2->ty == Tstruct ||
             (t1->ty == Tclass && t2->ty == Tclass))
    {
        if (t2->ty == Tstruct)
            error("need member function opCmp() for %s %s to compare", t2->toDsymbol(sc)->kind(), t2->toChars());
        else
            error("need member function opCmp() for %s %s to compare", t1->toDsymbol(sc)->kind(), t1->toChars());
        e = new ErrorExp();
    }
    else if (t1->iscomplex() || t2->iscomplex())
    {
        error("compare not defined for complex operands");
        e = new ErrorExp();
    }
    else if (t1->ty == Tvector)
        return incompatibleTypes();
    else
    {   if (!e1->rvalue() || !e2->rvalue())
            return new ErrorExp();
        e = this;
    }
    //printf("CmpExp: %s, type = %s\n", e->toChars(), e->type->toChars());
    return e;
}

int CmpExp::isBit()
{
    return TRUE;
}


/************************************************************/

EqualExp::EqualExp(enum TOK op, Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, op, sizeof(EqualExp), e1, e2)
{
    assert(op == TOKequal || op == TOKnotequal);
}

int needDirectEq(Type *t1, Type *t2)
{
    assert(t1->ty == Tarray || t1->ty == Tsarray);
    assert(t2->ty == Tarray || t2->ty == Tsarray);

    Type *t1n = t1->nextOf()->toBasetype();
    Type *t2n = t2->nextOf()->toBasetype();

    if (((t1n->ty == Tchar || t1n->ty == Twchar || t1n->ty == Tdchar) &&
         (t2n->ty == Tchar || t2n->ty == Twchar || t2n->ty == Tdchar)) ||
        (t1n->ty == Tvoid || t2n->ty == Tvoid))
    {
        return FALSE;
    }

    if (t1n->constOf() != t2n->constOf())
        return TRUE;

    Type *t = t1n;
    while (t->toBasetype()->nextOf())
        t = t->nextOf()->toBasetype();
    if (t->ty != Tstruct)
        return FALSE;

    return ((TypeStruct *)t)->sym->xeq == StructDeclaration::xerreq;
}

Expression *EqualExp::semantic(Scope *sc)
{   Expression *e;

    //printf("EqualExp::semantic('%s')\n", toChars());
    if (type)
        return this;

    BinExp::semanticp(sc);

    /* Before checking for operator overloading, check to see if we're
     * comparing the addresses of two statics. If so, we can just see
     * if they are the same symbol.
     */
    if (e1->op == TOKaddress && e2->op == TOKaddress)
    {   AddrExp *ae1 = (AddrExp *)e1;
        AddrExp *ae2 = (AddrExp *)e2;

        if (ae1->e1->op == TOKvar && ae2->e1->op == TOKvar)
        {   VarExp *ve1 = (VarExp *)ae1->e1;
            VarExp *ve2 = (VarExp *)ae2->e1;

            if (ve1->var == ve2->var /*|| ve1->var->toSymbol() == ve2->var->toSymbol()*/)
            {
                // They are the same, result is 'true' for ==, 'false' for !=
                e = new IntegerExp(loc, (op == TOKequal), Type::tboolean);
                return e;
            }
        }
    }

    Type *t1 = e1->type->toBasetype();
    Type *t2 = e2->type->toBasetype();
    if (t1->ty == Tclass && e2->op == TOKnull ||
        t2->ty == Tclass && e1->op == TOKnull)
    {
        error("use '%s' instead of '%s' when comparing with null",
                Token::toChars(op == TOKequal ? TOKidentity : TOKnotidentity),
                Token::toChars(op));
        return new ErrorExp();
    }

    if ((t1->ty == Tarray || t1->ty == Tsarray) &&
        (t2->ty == Tarray || t2->ty == Tsarray))
    {
        if (needDirectEq(t1, t2))
        {   /* Rewrite as:
             * _ArrayEq(e1, e2)
             */
            Expression *eq = new IdentifierExp(loc, Id::_ArrayEq);
            Expressions *args = new Expressions();
            args->push(e1);
            args->push(e2);
            e = new CallExp(loc, eq, args);
            if (op == TOKnotequal)
                e = new NotExp(loc, e);
            e = e->trySemantic(sc); // for better error message
            if (!e)
            {   error("cannot compare %s and %s", t1->toChars(), t2->toChars());
                return new ErrorExp();
            }
            return e;
        }
    }

    //if (e2->op != TOKnull)
    {
        e = op_overload(sc);
        if (e)
        {
            if (e->op == TOKcall && op == TOKnotequal)
            {
                e = new NotExp(e->loc, e);
                e = e->semantic(sc);
            }
            return e;
        }
    }

    /* Disallow comparing T[]==T and T==T[]
     */
    if (e1->op == TOKslice && t1->ty == Tarray && e2->implicitConvTo(t1->nextOf()) ||
        e2->op == TOKslice && t2->ty == Tarray && e1->implicitConvTo(t2->nextOf()))
    {
        incompatibleTypes();
        return new ErrorExp();
    }

    e = typeCombine(sc);
    if (e->op == TOKerror)
        return e;

    type = Type::tboolean;

    // Special handling for array comparisons
    if (!arrayTypeCompatible(loc, e1->type, e2->type))
    {
        if (e1->type != e2->type && e1->type->isfloating() && e2->type->isfloating())
        {
            // Cast both to complex
            e1 = e1->castTo(sc, Type::tcomplex80);
            e2 = e2->castTo(sc, Type::tcomplex80);
        }
    }

    if (e1->type->toBasetype()->ty == Tvector)
        return incompatibleTypes();

    return e;
}

int EqualExp::isBit()
{
    return TRUE;
}



/************************************************************/

IdentityExp::IdentityExp(enum TOK op, Loc loc, Expression *e1, Expression *e2)
        : BinExp(loc, op, sizeof(IdentityExp), e1, e2)
{
}

Expression *IdentityExp::semantic(Scope *sc)
{
    if (type)
        return this;

    BinExp::semanticp(sc);
    type = Type::tboolean;

    Expression *e = typeCombine(sc);
    if (e->op == TOKerror)
        return e;

    if (e1->type != e2->type && e1->type->isfloating() && e2->type->isfloating())
    {
        // Cast both to complex
        e1 = e1->castTo(sc, Type::tcomplex80);
        e2 = e2->castTo(sc, Type::tcomplex80);
    }

    if (e1->type->toBasetype()->ty == Tvector)
        return incompatibleTypes();

    return this;
}

int IdentityExp::isBit()
{
    return TRUE;
}


/****************************************************************/

CondExp::CondExp(Loc loc, Expression *econd, Expression *e1, Expression *e2)
        : BinExp(loc, TOKquestion, sizeof(CondExp), e1, e2)
{
    this->econd = econd;
}

Expression *CondExp::syntaxCopy()
{
    return new CondExp(loc, econd->syntaxCopy(), e1->syntaxCopy(), e2->syntaxCopy());
}


Expression *CondExp::semantic(Scope *sc)
{   Type *t1;
    Type *t2;
    unsigned cs0;
    unsigned cs1;

#if LOGSEMANTIC
    printf("CondExp::semantic('%s')\n", toChars());
#endif
    if (type)
        return this;

    econd = econd->semantic(sc);
    econd = resolveProperties(sc, econd);
    econd = econd->checkToPointer();
    econd = econd->checkToBoolean(sc);

    cs0 = sc->callSuper;
    e1 = e1->semantic(sc);
    e1 = resolveProperties(sc, e1);
    cs1 = sc->callSuper;
    sc->callSuper = cs0;
    e2 = e2->semantic(sc);
    e2 = resolveProperties(sc, e2);
    sc->mergeCallSuper(loc, cs1);


    // If either operand is void, the result is void
    t1 = e1->type;
    t2 = e2->type;
    if (t1->ty == Tvoid || t2->ty == Tvoid)
        type = Type::tvoid;
    else if (t1 == t2)
        type = t1;
    else
    {
        typeCombine(sc);
        switch (e1->type->toBasetype()->ty)
        {
            case Tcomplex32:
            case Tcomplex64:
            case Tcomplex80:
                e2 = e2->castTo(sc, e1->type);
                break;
        }
        switch (e2->type->toBasetype()->ty)
        {
            case Tcomplex32:
            case Tcomplex64:
            case Tcomplex80:
                e1 = e1->castTo(sc, e2->type);
                break;
        }
        if (type->toBasetype()->ty == Tarray)
        {
            e1 = e1->castTo(sc, type);
            e2 = e2->castTo(sc, type);
        }
    }
#if 0
    printf("res: %s\n", type->toChars());
    printf("e1 : %s\n", e1->type->toChars());
    printf("e2 : %s\n", e2->type->toChars());
#endif
    return this;
}

int CondExp::isLvalue()
{
    return e1->isLvalue() && e2->isLvalue();
}


Expression *CondExp::toLvalue(Scope *sc, Expression *ex)
{
    PtrExp *e;

    // convert (econd ? e1 : e2) to *(econd ? &e1 : &e2)
    e = new PtrExp(loc, this, type);

    e1 = e1->addressOf(sc);
    e2 = e2->addressOf(sc);

    typeCombine(sc);

    type = e2->type;
    return e;
}

int CondExp::checkModifiable(Scope *sc, int flag)
{
    return e1->checkModifiable(sc, flag) && e2->checkModifiable(sc, flag);
}

Expression *CondExp::modifiableLvalue(Scope *sc, Expression *e)
{
    //error("conditional expression %s is not a modifiable lvalue", toChars());
    e1 = e1->modifiableLvalue(sc, e1);
    e2 = e2->modifiableLvalue(sc, e1);
    return toLvalue(sc, this);
}

void CondExp::checkEscape()
{
    e1->checkEscape();
    e2->checkEscape();
}

void CondExp::checkEscapeRef()
{
    e1->checkEscapeRef();
    e2->checkEscapeRef();
}


Expression *CondExp::checkToBoolean(Scope *sc)
{
    e1 = e1->checkToBoolean(sc);
    e2 = e2->checkToBoolean(sc);
    return this;
}


void CondExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    expToCBuffer(buf, hgs, econd, PREC_oror);
    buf->writestring(" ? ");
    expToCBuffer(buf, hgs, e1, PREC_expr);
    buf->writestring(" : ");
    expToCBuffer(buf, hgs, e2, PREC_cond);
}


/****************************************************************/

DefaultInitExp::DefaultInitExp(Loc loc, enum TOK subop, int size)
    : Expression(loc, TOKdefault, size)
{
    this->subop = subop;
}

void DefaultInitExp::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    buf->writestring(Token::toChars(subop));
}

/****************************************************************/

FileInitExp::FileInitExp(Loc loc)
    : DefaultInitExp(loc, TOKfile, sizeof(FileInitExp))
{
}

Expression *FileInitExp::semantic(Scope *sc)
{
    //printf("FileInitExp::semantic()\n");
    type = Type::tchar->invariantOf()->arrayOf();
    return this;
}

Expression *FileInitExp::resolveLoc(Loc loc, Scope *sc)
{
    //printf("FileInitExp::resolve() %s\n", toChars());
    const char *s = loc.filename ? loc.filename : sc->module->ident->toChars();
    Expression *e = new StringExp(loc, (char *)s);
    e = e->semantic(sc);
    e = e->castTo(sc, type);
    return e;
}

/****************************************************************/

LineInitExp::LineInitExp(Loc loc)
    : DefaultInitExp(loc, TOKline, sizeof(LineInitExp))
{
}

Expression *LineInitExp::semantic(Scope *sc)
{
    type = Type::tint32;
    return this;
}

Expression *LineInitExp::resolveLoc(Loc loc, Scope *sc)
{
    Expression *e = new IntegerExp(loc, loc.linnum, Type::tint32);
    e = e->castTo(sc, type);
    return e;
}

/**************************************
 * Runs semantic on ae->arguments. Declares temporary variables
 * if '$' was used.
 */

ArrayExp *resolveOpDollar(Scope *sc, ArrayExp *ae)
{
    assert(!ae->lengthVar);

    for (size_t i = 0; i < ae->arguments->dim; i++)
    {
        // Create scope for '$' variable for this dimension
        ArrayScopeSymbol *sym = new ArrayScopeSymbol(sc, ae);
        sym->loc = ae->loc;
        sym->parent = sc->scopesym;
        sc = sc->push(sym);
        ae->lengthVar = NULL;       // Create it only if required
        ae->currentDimension = i;   // Dimension for $, if required

        Expression *e = (*ae->arguments)[i];
        e = e->semantic(sc);
        e = resolveProperties(sc, e);
        if (!e->type)
            ae->error("%s has no value", e->toChars());
        if (ae->lengthVar)
        {   // If $ was used, declare it now
            Expression *de = new DeclarationExp(ae->loc, ae->lengthVar);
            e = new CommaExp(0, de, e);
            e = e->semantic(sc);
        }
        (*ae->arguments)[i] = e;
        sc = sc->pop();
    }
    return ae;
}

/**************************************
 * Runs semantic on se->lwr and se->upr. Declares a temporary variable
 * if '$' was used.
 */

SliceExp *resolveOpDollar(Scope *sc, SliceExp *se)
{
    assert(!se->lengthVar);
    assert(!se->lwr || se->upr);

    if (!se->lwr) return se;

    // create scope for '$'
    ArrayScopeSymbol *sym = new ArrayScopeSymbol(sc, se);
    sym->loc = se->loc;
    sym->parent = sc->scopesym;
    sc = sc->push(sym);

    for (size_t i = 0; i < 2; ++i)
    {
        Expression *e = i == 0 ? se->lwr : se->upr;
        e = e->semantic(sc);
        e = resolveProperties(sc, e);
        if (!e->type)
            se->error("%s has no value", e->toChars());
        i == 0 ? se->lwr : se->upr = e;
    }

    if (se->lengthVar)
    {   // If $ was used, declare it now
        Expression *de = new DeclarationExp(se->loc, se->lengthVar);
        se->lwr = new CommaExp(0, de, se->lwr);
        se->lwr = se->lwr->semantic(sc);
    }
    sc = sc->pop();
    return se;
}
