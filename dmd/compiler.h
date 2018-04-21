
/* Compiler implementation of the D programming language
 * Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/dlang/dmd/blob/master/src/compiler.h
 */

#ifndef DMD_COMPILER_H
#define DMD_COMPILER_H

// This file contains a data structure that describes a back-end compiler
// and implements compiler-specific actions.

struct Compiler
{
    const char *vendor;     // Compiler backend name
};

#endif /* DMD_COMPILER_H */
