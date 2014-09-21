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
int Target::longsize;
bool Target::reverseCppOverloads;

void Target::init()
{
    ptrsize = gDataLayout->getPointerSize(ADDRESS_SPACE);

    llvm::Type* real = DtoType(Type::basic[Tfloat80]);
    realsize = gDataLayout->getTypeAllocSize(real);
    realpad = realsize - gDataLayout->getTypeStoreSize(real);
    realalignsize = gDataLayout->getABITypeAlignment(real);
    longsize = global.params.is64bit ? 8 : 4;

    reverseCppOverloads = false; // DMC is not supported.
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

Type *Target::va_listType()
{
    return Type::tchar->pointerTo();
}

/******************************
 * Encode the given expression, which is assumed to be an rvalue literal
 * as another type for use in CTFE.
 * This corresponds roughly to the idiom *(Type *)&e.
 */

Expression *Target::paintAsType(Expression *e, Type *type)
{
    union
    {
        d_int32 int32value;
        d_int64 int64value;
        float float32value;
        double float64value;
    } u;

    assert(e->type->size() == type->size());

    switch (e->type->ty)
    {
        case Tint32:
        case Tuns32:
            u.int32value = (d_int32)e->toInteger();
            break;

        case Tint64:
        case Tuns64:
            u.int64value = (d_int64)e->toInteger();
            break;

        case Tfloat32:
            u.float32value = e->toReal();
            break;

        case Tfloat64:
            u.float64value = e->toReal();
            break;

        default:
            assert(0);
    }

    switch (type->ty)
    {
        case Tint32:
        case Tuns32:
            return new IntegerExp(e->loc, u.int32value, type);

        case Tint64:
        case Tuns64:
            return new IntegerExp(e->loc, u.int64value, type);

        case Tfloat32:
            return new RealExp(e->loc, ldouble(u.float32value), type);

        case Tfloat64:
            return new RealExp(e->loc, ldouble(u.float64value), type);

        default:
            assert(0);
    }

    return NULL;    // avoid warning
}
