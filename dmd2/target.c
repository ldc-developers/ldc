
// Copyright (c) 2013 by Digital Mars
// All Rights Reserved
// written by Iain Buclaw
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <assert.h>

#include "target.h"
#include "mars.h"
#include "mtype.h"

#if IN_LLVM
unsigned GetTypeAlignment(Type* t);
unsigned GetPointerSize();
unsigned GetTypeStoreSize(Type* t);
unsigned GetTypeAllocSize(Type* t);
#endif

int Target::ptrsize;
int Target::realsize;
int Target::realpad;
int Target::realalignsize;

void Target::init()
{
#if IN_LLVM
    ptrsize = GetPointerSize();
    realsize = GetTypeAllocSize(Type::basic[Tfloat80]);
    realpad = realsize - GetTypeStoreSize(Type::basic[Tfloat80]);
    realalignsize = GetTypeAlignment(Type::basic[Tfloat80]);
#else
    // These have default values for 32 bit code, they get
    // adjusted for 64 bit code.
    ptrsize = 4;

    if (global.params.isLinux || global.params.isFreeBSD
        || global.params.isOpenBSD || global.params.isSolaris)
    {
        realsize = 12;
        realpad = 2;
        realalignsize = 4;
    }
    else if (global.params.isOSX)
    {
        realsize = 16;
        realpad = 6;
        realalignsize = 16;
    }
    else if (global.params.isWindows)
    {
        realsize = 10;
        realpad = 0;
        realalignsize = 2;
    }
    else
        assert(0);

    if (global.params.is64bit)
    {
        ptrsize = 8;
        if (global.params.isLinux || global.params.isFreeBSD || global.params.isSolaris)
        {
            realsize = 16;
            realpad = 6;
            realalignsize = 16;
        }
    }
#endif
}

/******************************
 * Return memory alignment size of type.
 */

unsigned Target::alignsize (Type* type)
{
    assert (type->isTypeBasic());

#if IN_LLVM
    if (type->ty == Tvoid) return 1;
    return GetTypeAlignment(type);
#else
    switch (type->ty)
    {
        case Tfloat80:
        case Timaginary80:
        case Tcomplex80:
            return Target::realalignsize;

        case Tcomplex32:
            if (global.params.isLinux || global.params.isOSX || global.params.isFreeBSD
                || global.params.isOpenBSD || global.params.isSolaris)
                return 4;
            break;

        case Tint64:
        case Tuns64:
        case Tfloat64:
        case Timaginary64:
        case Tcomplex64:
            if (global.params.isLinux || global.params.isOSX || global.params.isFreeBSD
                || global.params.isOpenBSD || global.params.isSolaris)
                return global.params.is64bit ? 8 : 4;
            break;

        default:
            break;
    }
    return type->size(Loc());
#endif
}

/******************************
 * Return field alignment size of type.
 */

unsigned Target::fieldalign (Type* type)
{
    // LDC_FIXME: Verify this.
    return type->alignsize();
}
