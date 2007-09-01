

// Copyright (c) 1999-2004 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <cstdio>
#include <cassert>

#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "scope.h"
#include "init.h"
#include "expression.h"
#include "attrib.h"
#include "declaration.h"
#include "template.h"
#include "id.h"
#include "enum.h"
#include "import.h"
#include "aggregate.h"

#include "logger.h"

/*******************************************

 * Get a canonicalized form of the TypeInfo for use with the internal

 * runtime library routines. Canonicalized in that static arrays are

 * represented as dynamic arrays, enums are represented by their

 * underlying type, etc. This reduces the number of TypeInfo's needed,

 * so we can use the custom internal ones more.

 */



Expression *Type::getInternalTypeInfo(Scope *sc)

{   TypeInfoDeclaration *tid;

    Expression *e;

    Type *t;

    static TypeInfoDeclaration *internalTI[TMAX];



    //printf("Type::getInternalTypeInfo() %s\n", toChars());

    t = toBasetype();

    switch (t->ty)

    {

    case Tsarray:

        t = t->next->arrayOf(); // convert to corresponding dynamic array type

        break;



    case Tclass:

        if (((TypeClass *)t)->sym->isInterfaceDeclaration())

        break;

        goto Linternal;



    case Tarray:

        if (t->next->ty != Tclass)

        break;

        goto Linternal;



    case Tfunction:

    case Tdelegate:

    case Tpointer:

    Linternal:

        tid = internalTI[t->ty];

        if (!tid)

        {   tid = new TypeInfoDeclaration(t, 1);

        internalTI[t->ty] = tid;

        }

        e = new VarExp(0, tid);

        e = e->addressOf(sc);

        e->type = tid->type;    // do this so we don't get redundant dereference

        return e;



    default:

        break;

    }

    //printf("\tcalling getTypeInfo() %s\n", t->toChars());

    return t->getTypeInfo(sc);

}





/****************************************************

 * Get the exact TypeInfo.

 */



Expression *Type::getTypeInfo(Scope *sc)

{

    Expression *e;

    Type *t;



    //printf("Type::getTypeInfo() %p, %s\n", this, toChars());

    t = merge();    // do this since not all Type's are merge'd

    if (!t->vtinfo)

    {   t->vtinfo = t->getTypeInfoDeclaration();

    assert(t->vtinfo);



    /* If this has a custom implementation in std/typeinfo, then

     * do not generate a COMDAT for it.

     */

    if (!t->builtinTypeInfo())

    {   // Generate COMDAT

        if (sc)         // if in semantic() pass

        {   // Find module that will go all the way to an object file

        Module *m = sc->module->importedFrom;

        m->members->push(t->vtinfo);

        }

        else            // if in obj generation pass

        {

        t->vtinfo->toObjFile();

        }

    }

    }

    e = new VarExp(0, t->vtinfo);

    //e = e->addressOf(sc);
    e->type = t->vtinfo->type;      // do this so we don't get redundant dereference

    return e;

}



TypeInfoDeclaration *Type::getTypeInfoDeclaration()

{

    //printf("Type::getTypeInfoDeclaration() %s\n", toChars());

    return new TypeInfoDeclaration(this, 0);

}



TypeInfoDeclaration *TypeTypedef::getTypeInfoDeclaration()

{

    return new TypeInfoTypedefDeclaration(this);

}



TypeInfoDeclaration *TypePointer::getTypeInfoDeclaration()

{

    return new TypeInfoPointerDeclaration(this);

}



TypeInfoDeclaration *TypeDArray::getTypeInfoDeclaration()

{

    return new TypeInfoArrayDeclaration(this);

}



TypeInfoDeclaration *TypeSArray::getTypeInfoDeclaration()

{

    return new TypeInfoStaticArrayDeclaration(this);

}



TypeInfoDeclaration *TypeAArray::getTypeInfoDeclaration()

{

    return new TypeInfoAssociativeArrayDeclaration(this);

}



TypeInfoDeclaration *TypeStruct::getTypeInfoDeclaration()

{

    return new TypeInfoStructDeclaration(this);

}



TypeInfoDeclaration *TypeClass::getTypeInfoDeclaration()

{

    if (sym->isInterfaceDeclaration())

    return new TypeInfoInterfaceDeclaration(this);

    else

    return new TypeInfoClassDeclaration(this);

}



TypeInfoDeclaration *TypeEnum::getTypeInfoDeclaration()

{

    return new TypeInfoEnumDeclaration(this);

}



TypeInfoDeclaration *TypeFunction::getTypeInfoDeclaration()

{

    return new TypeInfoFunctionDeclaration(this);

}
enum RET TypeFunction::retStyle()

{

    return RETstack;

}


TypeInfoDeclaration *TypeDelegate::getTypeInfoDeclaration()

{

    return new TypeInfoDelegateDeclaration(this);

}



TypeInfoDeclaration *TypeTuple::getTypeInfoDeclaration()

{

    return new TypeInfoTupleDeclaration(this);

}


void TypeInfoDeclaration::toDt(dt_t **pdt)
{
}

void TypeInfoTypedefDeclaration::toDt(dt_t **pdt)
{
}

void TypeInfoStructDeclaration::toDt(dt_t **pdt)
{
}

void TypeInfoClassDeclaration::toDt(dt_t **pdt)
{
}

void TypeInfoDeclaration::toObjFile()
{
    Logger::println("TypeInfoDeclaration::toObjFile()");
    LOG_SCOPE;
    Logger::println("type = '%s'", tinfo->toChars());

    
}

/* ========================================================================= */

/* These decide if there's an instance for them already in std.typeinfo,
 * because then the compiler doesn't need to build one.
 */

int Type::builtinTypeInfo()
{
    return 0;
}

int TypeBasic::builtinTypeInfo()
{
    return 1;
}

int TypeDArray::builtinTypeInfo()
{
    return 0;
}

/* ========================================================================= */

/***************************************
 * Create a static array of TypeInfo references
 * corresponding to an array of Expression's.
 * Used to supply hidden _arguments[] value for variadic D functions.
 */

Expression *createTypeInfoArray(Scope *sc, Expression *args[], int dim)
{
    assert(0);
    return 0;
}

