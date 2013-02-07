//===-- gen/utils.h - Utilities for handling frontend types -----*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Some utilities for handling front-end types in a more C++-like fashion.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_UTILS_H
#define LDC_GEN_UTILS_H

#include "root.h"

/// Very simple templated iterator for DMD ArrayS.
template<class C>
struct ArrayIter
{
    Array* array;
    size_t index;

    ArrayIter(Array& arr, size_t idx = 0)
    :   array(&arr), index(idx)
    { }
    ArrayIter(Array* arr, size_t idx = 0)
    :   array(arr), index(idx)
    { assert(arr && "null array"); }

    ArrayIter<C>& operator=(const Array& arr)
    {
        array = const_cast<Array*>(&arr);
        index = 0;
        return *this;
    }
    ArrayIter<C>& operator=(const Array* arr)
    {
        assert(arr && "null array");
        array = const_cast<Array*>(arr);
        index = 0;
        return *this;
    }

    bool done()
    {
        return index >= array->dim;
    }
    bool more()
    {
        return index < array->dim;
    }

    C* get() {
        return static_cast<C*>(array->data[index]);
    }
    C* operator->() {
        return get();
    }
    C* operator*() {
        return get();
    }

    void next()
    {
        ++index;
    }

    bool operator==(const ArrayIter<C>& other) {
        return &array->data[index] == &other.array->data[other.index];
    }
};

// some aliases
typedef ArrayIter<Dsymbol> DsymbolIter;
typedef ArrayIter<FuncDeclaration> FuncDeclarationIter;
typedef ArrayIter<VarDeclaration> VarDeclarationIter;

#endif
