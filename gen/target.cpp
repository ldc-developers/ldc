//===-- target.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "target.h"
#include "gen/irstate.h"
#include "mars.h"
#include "mtype.h"
#include <assert.h>

#if defined(_MSC_VER)
#include <windows.h>
#else
#include <pthread.h>
#endif

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

// sizes based on those from tollvm.cpp:DtoMutexType()
unsigned Target::critsecsize()
{
#if defined(_MSC_VER)
    // Return sizeof(RTL_CRITICAL_SECTION)
    return global.params.is64bit ? 40 : 24;
#else
    if (global.params.targetTriple.isOSWindows())
        return global.params.is64bit ? 40 : 24;
    else if (global.params.targetTriple.getOS() == llvm::Triple::FreeBSD)
        return sizeof(size_t);
    else
        return sizeof(pthread_mutex_t);
#endif
}
