
/* Compiler implementation of the D programming language
 * Copyright (c) 1999-2014 by Digital Mars
 * All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/D-Programming-Language/dmd/blob/master/src/root/int128.h
 */

#ifndef DMD_INT128_H
#define DMD_INT128_H

#define WANT_CENT __GNUC__

#if WANT_CENT
#if __GNUC__ || __clang__
typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;
#else

#endif

// 128bit literals are not supported
#define INT128C(hi,lo) ((((int128_t)hi)<<64)|lo)
#define UINT128C(hi,lo) ((((uint128_t)hi)<<64)|lo)

void sprintf_i128(char* buffer, int128_t v);
void sprintf_u128(char* buffer, uint128_t v);

#endif

#endif // DMD_INT128_H
