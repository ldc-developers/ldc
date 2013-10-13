//===-- target.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include <assert.h>

#include "target.h"
#include "gen/irstate.h"
#include "mars.h"
#include "mtype.h"

int Target::ptrsize;
int Target::realsize;
int Target::realpad;
int Target::realalignsize;

void Target::init()
{
    ptrsize = gDataLayout->getPointerSize(ADDRESS_SPACE);

    llvm::Type* real = DtoType(Type::basic[Tfloat80]);
    realsize = gDataLayout->getTypeAllocSize(real);
    realpad = realsize - gDataLayout->getTypeStoreSize(real);
    realalignsize = gDataLayout->getABITypeAlignment(real);
}

/******************************
 * Return memory alignment size of type.
 */

unsigned Target::alignsize (Type* type)
{
    assert (type->isTypeBasic());
    if (type->ty == Tvoid) return 1;
    return gDataLayout->getABITypeAlignment(DtoType(type));
}

/******************************
 * Return field alignment size of type.
 */

unsigned Target::fieldalign (Type* type)
{
    // LDC_FIXME: Verify this.
    return type->alignsize();
}
