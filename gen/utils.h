#ifndef __LDC_GEN_UTILS_H__
#define __LDC_GEN_UTILS_H__

#include "root.h"

/// Very simple templated iterator for DMD ArrayS.
template<class C>
struct ArrayIter
{
    Array& array;
    size_t index;

    ArrayIter(Array& arr, size_t idx = 0)
    :   array(arr), index(idx)
    { }

    bool done()
    {
        return index >= array.dim;
    }
    bool more()
    {
        return index < array.dim;
    }

    C* get()
    {
        return static_cast<C*>(array.data[index]);
    }
    C* operator->()
    {
        return static_cast<C*>(array.data[index]);
    }

    void next()
    {
        ++index;
    }
};

// some aliases
typedef ArrayIter<Dsymbol> DsymbolIter;
typedef ArrayIter<FuncDeclaration> FuncDeclarationIter;
typedef ArrayIter<VarDeclaration> VarDeclarationIter;

#endif
