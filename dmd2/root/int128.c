
/* Compiler implementation of the D programming language
 * Copyright (c) 1999-2014 by Digital Mars
 * All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/D-Programming-Language/dmd/blob/master/src/root/int128.c
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "int128.h"

#if WANT_CENT
// see http://stackoverflow.com/question/11656241/how-to-print-uint128-t-number-using-gcc
#ifndef UINT64_MAX
#define UINT64_MAX 0xFFFFFFFFFFFFFFFFULL
#endif

#define INT128_MIN INT128C(0x8000000000000000ULL, 0x0000000000000000ULL)

#define P10_UINT64 10000000000000000000ULL

void sprintf_u128(char *buffer, uint128_t u128)
{
    if (u128 > UINT64_MAX)
    {
        uint128_t leading = u128 / P10_UINT64;
        uint64_t trailing = u128 % P10_UINT64;
        sprintf_u128(buffer, leading);
        sprintf(&buffer[strlen(buffer)], "%.19llu", trailing);
    }
    else
    {
         sprintf(buffer,"%llu",(uint64_t) u128);
    }
}

void sprintf_i128(char *buffer, int128_t i128)
{
    if (i128 == INT128_MIN)
        strcpy(buffer, "-170141183460469231731687303715884105728");
    else
    {
        if (i128 < 0)
        {
            *buffer++ = '-';
            i128 = -i128;
        }
        sprintf_u128(buffer, (uint128_t) i128);
    }
}

#endif
