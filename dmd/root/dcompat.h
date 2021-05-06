
/* Copyright (C) 1999-2021 by The D Language Foundation, All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/dlang/dmd/blob/master/src/dmd/root/dcompat.h
 */

#pragma once

#include <string>
#include <cstring>
#include "dsystem.h"

/// Represents a D [ ] array
template<typename T>
struct DArray
{
    size_t length;
    T *ptr;

    DArray() : length(0), ptr(NULL) { }

    DArray(size_t length_in, T *ptr_in)
        : length(length_in), ptr(ptr_in) { }
};

struct DString : public DArray<const char>
{
    DString() : DArray<const char>() { }

    DString(const char *ptr)
        : DArray<const char>(ptr ? strlen(ptr) : 0, ptr) { }

    DString(size_t length, const char *ptr)
        : DArray<const char>(length, ptr) { }

    int compare (std::string str)
    {
        return this->length >= str.length() ?
            strncmp(this->ptr, str.c_str(), this->length) : -1;
    }
};

/// Corresponding C++ type that maps to D size_t
#if __APPLE__ && __i386__
// size_t is 'unsigned long', which makes it mangle differently than D's 'uint'
typedef unsigned d_size_t;
#elif MARS && DMD_VERSION >= 2079 && DMD_VERSION <= 2081 && \
        __APPLE__ && __SIZEOF_SIZE_T__ == 8
// DMD versions between 2.079 and 2.081 mapped D ulong to uint64_t on OS X.
typedef uint64_t d_size_t;
#elif __APPLE__ && __LP64__ && LDC_HOST_DigitalMars && LDC_HOST_FE_VER >= 2079 && LDC_HOST_FE_VER <= 2081
typedef uint64_t d_size_t;
#else
typedef size_t d_size_t;
#endif
