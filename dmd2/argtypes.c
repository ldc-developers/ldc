
// Compiler implementation of the D programming language
// Copyright (c) 2010-2012 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <stdio.h>
#include <assert.h>

#include "mars.h"
#include "dsymbol.h"
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
#include "hdrgen.h"

#define tfloat2 tfloat64
//#define tfloat2 tcomplex32

/****************************************************
 * This breaks a type down into 'simpler' types that can be passed to a function
 * in registers, and returned in registers.
 * It's highly platform dependent.
 * Returning a tuple of zero length means the type cannot be passed/returned in registers.
 */


TypeTuple *Type::toArgTypes()
{
    return NULL;        // not valid for a parameter
}

TypeTuple *TypeError::toArgTypes()
{
    return new TypeTuple(Type::terror);
}

TypeTuple *TypeBasic::toArgTypes()
{   Type *t1 = NULL;
    Type *t2 = NULL;
    switch (ty)
    {
        case Tvoid:
             return NULL;

        case Tbool:
        case Tint8:
        case Tuns8:
        case Tint16:
        case Tuns16:
        case Tint32:
        case Tuns32:
        case Tfloat32:
        case Tint64:
        case Tuns64:
        case Tfloat64:
        case Tfloat80:
            t1 = this;
            break;

        case Timaginary32:
            t1 = Type::tfloat32;
            break;

        case Timaginary64:
            t1 = Type::tfloat64;
            break;

        case Timaginary80:
            t1 = Type::tfloat80;
            break;

        case Tcomplex32:
            if (global.params.is64bit)
                t1 = Type::tfloat2;
            else
            {
                t1 = Type::tfloat64;
                t2 = Type::tfloat64;
            }
            break;

        case Tcomplex64:
            t1 = Type::tfloat64;
            t2 = Type::tfloat64;
            break;

        case Tcomplex80:
            t1 = Type::tfloat80;
            t2 = Type::tfloat80;
            break;

        case Tascii:
            t1 = Type::tuns8;
            break;

        case Twchar:
            t1 = Type::tuns16;
            break;

        case Tdchar:
            t1 = Type::tuns32;
            break;

        default:        assert(0);
    }

    TypeTuple *t;
    if (t1)
    {
        if (t2)
            t = new TypeTuple(t1, t2);
        else
            t = new TypeTuple(t1);
    }
    else
        t = new TypeTuple();
    return t;
}

#if DMDV2
TypeTuple *TypeVector::toArgTypes()
{
    return new TypeTuple(this);
}
#endif

TypeTuple *TypeSArray::toArgTypes()
{
#if DMDV2
    if (dim)
    {
        /* Should really be done as if it were a struct with dim members
         * of the array's elements.
         * I.e. int[2] should be done like struct S { int a; int b; }
         */
        dinteger_t sz = dim->toInteger();
        if (sz == 1)
            // T[1] should be passed like T
            return next->toArgTypes();
    }
    return new TypeTuple();     // pass on the stack for efficiency
#else
    return new TypeTuple();     // pass on the stack for efficiency
#endif
}

TypeTuple *TypeDArray::toArgTypes()
{
    /* Should be done as if it were:
     * struct S { size_t length; void* ptr; }
     */
    return new TypeTuple(Type::tsize_t, Type::tvoidptr);
}

TypeTuple *TypeAArray::toArgTypes()
{
    return new TypeTuple(Type::tvoidptr);
}

TypeTuple *TypePointer::toArgTypes()
{
    return new TypeTuple(Type::tvoidptr);
}

TypeTuple *TypeDelegate::toArgTypes()
{
    /* Should be done as if it were:
     * struct S { void* ptr; void* funcptr; }
     */
    return new TypeTuple(Type::tvoidptr, Type::tvoidptr);
}

/*************************************
 * Convert a floating point type into the equivalent integral type.
 */

Type *mergeFloatToInt(Type *t)
{
    switch (t->ty)
    {
        case Tfloat32:
        case Timaginary32:
            t = Type::tint32;
            break;
        case Tfloat64:
        case Timaginary64:
        case Tcomplex32:
            t = Type::tint64;
            break;
        default:
#ifdef DEBUG
            printf("mergeFloatToInt() %s\n", t->toChars());
#endif
            assert(0);
    }
    return t;
}

/*************************************
 * This merges two types into an 8byte type.
 */

Type *argtypemerge(Type *t1, Type *t2, unsigned offset2)
{
    //printf("argtypemerge(%s, %s, %d)\n", t1 ? t1->toChars() : "", t2 ? t2->toChars() : "", offset2);
    if (!t1)
    {   assert(!t2 || offset2 == 0);
        return t2;
    }
    if (!t2)
        return t1;

    unsigned sz1 = t1->size(0);
    unsigned sz2 = t2->size(0);

    if (t1->ty != t2->ty &&
        (t1->ty == Tfloat80 || t2->ty == Tfloat80))
        return NULL;

    // [float,float] => [cfloat]
    if (t1->ty == Tfloat32 && t2->ty == Tfloat32 && offset2 == 4)
        return Type::tfloat2;

    // Merging floating and non-floating types produces the non-floating type
    if (t1->isfloating())
    {
        if (!t2->isfloating())
            t1 = mergeFloatToInt(t1);
    }
    else if (t2->isfloating())
        t2 = mergeFloatToInt(t2);

    Type *t;

    // Pick type with larger size
    if (sz1 < sz2)
        t = t2;
    else
        t = t1;

    // If t2 does not lie within t1, need to increase the size of t to enclose both
    if (offset2 && sz1 < offset2 + sz2)
    {
        switch (offset2 + sz2)
        {
            case 2:
                t = Type::tint16;
                break;
            case 3:
            case 4:
                t = Type::tint32;
                break;
            case 5:
            case 6:
            case 7:
            case 8:
                t = Type::tint64;
                break;
            default:
                assert(0);
        }
    }
    return t;
}

TypeTuple *TypeStruct::toArgTypes()
{
    //printf("TypeStruct::toArgTypes() %s\n", toChars());
    if (!sym->isPOD())
    {
     Lmemory:
        //printf("\ttoArgTypes() %s => [ ]\n", toChars());
        return new TypeTuple();         // pass on the stack
    }
    Type *t1 = NULL;
    Type *t2 = NULL;
    d_uns64 sz = size(0);
    assert(sz < 0xFFFFFFFF);
    switch ((unsigned)sz)
    {
        case 1:
            t1 = Type::tint8;
            break;
        case 2:
            t1 = Type::tint16;
            break;
        case 4:
            t1 = Type::tint32;
            break;
        case 8:
            t1 = Type::tint64;
            break;
        case 16:
            t1 = NULL;                   // could be a TypeVector
            break;
        default:
            goto Lmemory;
    }
    if (global.params.is64bit && sym->fields.dim)
    {
#if 1
        unsigned sz1 = 0;
        unsigned sz2 = 0;
        t1 = NULL;
        for (size_t i = 0; i < sym->fields.dim; i++)
        {   VarDeclaration *f = sym->fields[i];
            //printf("f->type = %s\n", f->type->toChars());

            TypeTuple *tup = f->type->toArgTypes();
            if (!tup)
                goto Lmemory;
            size_t dim = tup->arguments->dim;
            Type *ft1 = NULL;
            Type *ft2 = NULL;
            switch (dim)
            {
                case 2:
                    ft1 = (*tup->arguments)[0]->type;
                    ft2 = (*tup->arguments)[1]->type;
                    break;
                case 1:
                    if (f->offset < 8)
                        ft1 = (*tup->arguments)[0]->type;
                    else
                        ft2 = (*tup->arguments)[0]->type;
                    break;
                default:
                    goto Lmemory;
            }

            if (f->offset & 7)
            {
                // Misaligned fields goto Lmemory
                unsigned alignsz = f->type->alignsize();
                if (f->offset & (alignsz - 1))
                    goto Lmemory;

                // Fields that overlap the 8byte boundary goto Lmemory
                unsigned fieldsz = f->type->size(0);
                if (f->offset < 8 && (f->offset + fieldsz) > 8)
                    goto Lmemory;
            }

            // First field in 8byte must be at start of 8byte
            assert(t1 || f->offset == 0);

            if (ft1)
            {
                t1 = argtypemerge(t1, ft1, f->offset);
                if (!t1)
                    goto Lmemory;
            }

            if (ft2)
            {
                unsigned off2 = f->offset;
                if (ft1)
                    off2 = 8;
                if (!t2 && off2 != 8)
                    goto Lmemory;
                assert(t2 || off2 == 8);
                t2 = argtypemerge(t2, ft2, off2 - 8);
                if (!t2)
                    goto Lmemory;
            }
        }

        if (t2)
        {
            if (t1->isfloating() && t2->isfloating())
            {
                if (t1->ty == Tfloat64 && t2->ty == Tfloat64)
                    ;
                else
                    goto Lmemory;
            }
            else if (t1->isfloating())
                goto Lmemory;
            else if (t2->isfloating())
                goto Lmemory;
            else
                ;
        }
#else
        if (sym->fields.dim == 1)
        {   VarDeclaration *f = sym->fields[0];
            //printf("f->type = %s\n", f->type->toChars());
            TypeTuple *tup = f->type->toArgTypes();
            if (tup)
            {
                size_t dim = tup->arguments->dim;
                if (dim == 1)
                    t1 = (*tup->arguments)[0]->type;
            }
        }
#endif
    }

    //printf("\ttoArgTypes() %s => [%s,%s]\n", toChars(), t1 ? t1->toChars() : "", t2 ? t2->toChars() : "");

    TypeTuple *t;
    if (t1)
    {
        //if (t1) printf("test1: %s => %s\n", toChars(), t1->toChars());
        if (t2)
            t = new TypeTuple(t1, t2);
        else
            t = new TypeTuple(t1);
    }
    else
        goto Lmemory;
    return t;
}

TypeTuple *TypeEnum::toArgTypes()
{
    return toBasetype()->toArgTypes();
}

TypeTuple *TypeTypedef::toArgTypes()
{
    return sym->basetype->toArgTypes();
}

TypeTuple *TypeClass::toArgTypes()
{
    return new TypeTuple(Type::tvoidptr);
}

