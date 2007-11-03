
// Copyright (c) 1999-2005 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

// stubbed out for dmdfe. Original is in dmd/todt.c

#include	"lexer.h"
#include	"mtype.h"
#include	"expression.h"
#include	"init.h"
#include	"enum.h"
#include	"aggregate.h"
#include	"declaration.h"

struct dt_t {};

dt_t *Initializer::toDt()
{
    return 0;
}


dt_t *StructInitializer::toDt()
{
    return 0;
}


dt_t *ArrayInitializer::toDt()
{
    return 0;
}


dt_t *ArrayInitializer::toDtBit()
{
    return 0;
}


dt_t *ExpInitializer::toDt()
{
    return 0;
}

dt_t *VoidInitializer::toDt()
{
    return 0;
}

/* ================================================================ */

dt_t **Expression::toDt(dt_t **pdt)
{
    return 0;
}

dt_t **IntegerExp::toDt(dt_t **pdt)
{
    return 0;
}

dt_t **RealExp::toDt(dt_t **pdt)
{
    return 0;
}

dt_t **ComplexExp::toDt(dt_t **pdt)
{
    return 0;
}

dt_t **NullExp::toDt(dt_t **pdt)
{
    return 0;
}

dt_t **StringExp::toDt(dt_t **pdt)
{
    return 0;
}

dt_t **SymOffExp::toDt(dt_t **pdt)
{
    return 0;
}

dt_t **VarExp::toDt(dt_t **pdt)
{
    return 0;
}

dt_t **ArrayLiteralExp::toDt(dt_t **pdt)
{
    return 0;
}
dt_t **StructLiteralExp::toDt(dt_t **pdt)
{
    return 0;
}

void ClassDeclaration::toDt(dt_t **pdt)
{
}

void ClassDeclaration::toDt2(dt_t **pdt, ClassDeclaration *cd)
{
}

void StructDeclaration::toDt(dt_t **pdt)
{
}

dt_t **Type::toDt(dt_t **pdt)
{
    return 0;
}

dt_t **TypeSArray::toDt(dt_t **pdt)
{
    return 0;
}

dt_t **TypeStruct::toDt(dt_t **pdt)
{
    return 0;
}

dt_t **TypeTypedef::toDt(dt_t **pdt)
{
    return 0;
}



