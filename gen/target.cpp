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
#include "mars.h"
#include "mtype.h"

unsigned GetTypeAlignment(Type* t);
unsigned GetPointerSize();
unsigned GetTypeStoreSize(Type* t);
unsigned GetTypeAllocSize(Type* t);

int Target::ptrsize;
int Target::realsize;
int Target::realpad;
int Target::realalignsize;

void Target::init()
{
    ptrsize = GetPointerSize();
    realsize = GetTypeAllocSize(Type::basic[Tfloat80]);
    realpad = realsize - GetTypeStoreSize(Type::basic[Tfloat80]);
    realalignsize = GetTypeAlignment(Type::basic[Tfloat80]);
}

/******************************
 * Return memory alignment size of type.
 */

unsigned Target::alignsize (Type* type)
{
    assert (type->isTypeBasic());
    if (type->ty == Tvoid) return 1;
    return GetTypeAlignment(type);
}

/******************************
 * Return field alignment size of type.
 */

unsigned Target::fieldalign (Type* type)
{
    // LDC_FIXME: Verify this.
    return type->alignsize();
}
