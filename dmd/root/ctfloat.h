
/* Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * All Rights Reserved, written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 * https://github.com/dlang/dmd/blob/master/src/root/ctfloat.h
 */

#ifndef CTFLOAT_H
#define CTFLOAT_H

#include "longdouble.h"

// Type used by the front-end for compile-time reals
typedef longdouble real_t;

#if IN_LLVM
namespace llvm { class APFloat; }
#endif

// Compile-time floating-point helper
struct CTFloat
{
    static bool yl2x_supported;
    static bool yl2xp1_supported;

    static void yl2x(const real_t *x, const real_t *y, real_t *res);
    static void yl2xp1(const real_t *x, const real_t *y, real_t *res);

    static real_t sin(real_t x);
    static real_t cos(real_t x);
    static real_t tan(real_t x);
    static real_t sqrt(real_t x);
    static real_t fabs(real_t x);
    static real_t ldexp(real_t n, int exp);

    static real_t round(real_t x);
    static real_t floor(real_t x);
    static real_t ceil(real_t x);
    static real_t trunc(real_t x);
    static real_t log(real_t x);
    static real_t log2(real_t x);
    static real_t log10(real_t x);
    static real_t pow(real_t x, real_t y);
    static real_t exp(real_t x);
    static real_t expm1(real_t x);
    static real_t exp2(real_t x);

    static real_t fmin(real_t x, real_t y);
    static real_t fmax(real_t x, real_t y);
    static real_t copysign(real_t x, real_t s);

    static real_t fma(real_t x, real_t y, real_t z);

#if IN_LLVM
    static real_t rint(real_t x);
    static real_t nearbyint(real_t x);

    // implemented in gen/ctfloat.cpp
    static void _init();
    static void toAPFloat(real_t src, llvm::APFloat &dst);
    static real_t fromAPFloat(const llvm::APFloat &src);

    static bool isFloat32LiteralOutOfRange(const char *literal);
    static bool isFloat64LiteralOutOfRange(const char *literal);
#endif

    static bool isIdentical(real_t a, real_t b);
    static bool isNaN(real_t r);
#if !IN_LLVM
    static bool isSNaN(real_t r);
#endif
    static bool isInfinity(real_t r);

    static real_t parse(const char *literal, bool *isOutOfRange = NULL);
    static int sprint(char *str, char fmt, real_t x);

    static size_t hash(real_t a);

    // Constant real values 0, 1, -1 and 0.5.
    static real_t zero;
    static real_t one;
    static real_t minusone;
    static real_t half;
#if IN_LLVM
    static real_t initVal;
    static real_t nan;
    static real_t infinity;
#endif
};

#endif
