
// Compiler implementation of the D programming language
// Copyright (c) 1999-2011 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <stdio.h>
#include <assert.h>

#include "mars.h"
#include "expression.h"
#include "statement.h"
#include "mtype.h"
#include "utf.h"
#include "declaration.h"
#include "aggregate.h"
#include "scope.h"
#include "init.h"

/********************************************
 * Convert from expression to delegate that returns the expression,
 * i.e. convert:
 *      expr
 * to:
 *      t delegate() { return expr; }
 */

int lambdaSetParent(Expression *e, void *param);
int lambdaCheckForNestedRef(Expression *e, void *param);

Expression *Expression::toDelegate(Scope *sc, Type *t)
{
    //printf("Expression::toDelegate(t = %s) %s\n", t->toChars(), toChars());
    Type *tw = t->semantic(loc, sc);
    Type *tc = t->substWildTo(MODconst)->semantic(loc, sc);
    TypeFunction *tf = new TypeFunction(NULL, tc, 0, LINKd);
    if (tw != tc) tf->mod = MODwild;                            // hack for bug7757
    (tf = (TypeFunction *)tf->semantic(loc, sc))->next = tw;    // hack for bug7757
    FuncLiteralDeclaration *fld =
        new FuncLiteralDeclaration(loc, loc, tf, TOKdelegate, NULL);
    Expression *e;
    sc = sc->push();
    sc->parent = fld;           // set current function to be the delegate
    e = this;
    e->apply(&lambdaSetParent, sc);
    e->apply(&lambdaCheckForNestedRef, sc);
    sc = sc->pop();
    Statement *s;
    if (t->ty == Tvoid)
        s = new ExpStatement(loc, e);
    else
        s = new ReturnStatement(loc, e);
    fld->fbody = s;
    e = new FuncExp(loc, fld);
    e = e->semantic(sc);
    return e;
}

/******************************************
 * Patch the parent of declarations to be the new function literal.
 */
int lambdaSetParent(Expression *e, void *param)
{
    Scope *sc = (Scope *)param;
    /* We could use virtual functions instead of a switch,
     * but it doesn't seem worth the bother.
     */
    switch (e->op)
    {
        case TOKdeclaration:
        {   DeclarationExp *de = (DeclarationExp *)e;
            de->declaration->parent = sc->parent;
            break;
        }

        case TOKindex:
        {   IndexExp *de = (IndexExp *)e;
            if (de->lengthVar)
            {   //printf("lengthVar\n");
                de->lengthVar->parent = sc->parent;
            }
            break;
        }

        case TOKslice:
        {   SliceExp *se = (SliceExp *)e;
            if (se->lengthVar)
            {   //printf("lengthVar\n");
                se->lengthVar->parent = sc->parent;
            }
            break;
        }

        default:
            break;
    }
     return 0;
}

/*******************************************
 * Look for references to variables in a scope enclosing the new function literal.
 */
int lambdaCheckForNestedRef(Expression *e, void *param)
{
    Scope *sc = (Scope *)param;
    /* We could use virtual functions instead of a switch,
     * but it doesn't seem worth the bother.
     */
    switch (e->op)
    {
        case TOKsymoff:
        {   SymOffExp *se = (SymOffExp *)e;
            VarDeclaration *v = se->var->isVarDeclaration();
            if (v)
                v->checkNestedReference(sc, Loc());
            break;
        }

        case TOKvar:
        {   VarExp *ve = (VarExp *)e;
            VarDeclaration *v = ve->var->isVarDeclaration();
            if (v)
                v->checkNestedReference(sc, Loc());
            break;
        }

        case TOKthis:
        case TOKsuper:
        {   ThisExp *te = (ThisExp *)e;
            VarDeclaration *v = te->var->isVarDeclaration();
            if (v)
                v->checkNestedReference(sc, Loc());
            break;
        }

        case TOKdeclaration:
        {   DeclarationExp *de = (DeclarationExp *)e;
            VarDeclaration *v = de->declaration->isVarDeclaration();
            if (v)
            {
                v->checkNestedReference(sc, Loc());

                /* Some expressions cause the frontend to create a temporary.
                 * For example, structs with cpctors replace the original
                 * expression e with:
                 *  __cpcttmp = __cpcttmp.cpctor(e);
                 *
                 * In this instance, we need to ensure that the original
                 * expression e does not have any nested references by
                 * checking the declaration initializer too.
                 */
                if (v->init && v->init->isExpInitializer())
                {   Expression *ie = v->init->toExpression();
                    ie->apply (&lambdaCheckForNestedRef, param);
                }
            }
            break;
        }

        default:
            break;
    }
    return 0;
}

