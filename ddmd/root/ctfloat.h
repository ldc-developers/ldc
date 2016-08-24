//===-- ddmd/root/ctfloat.h -  CTFloat implementation for LDC ---*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Front-end compile-time floating-point implementation for LDC.
//
//===----------------------------------------------------------------------===//

#ifndef CTFLOAT_H
#define CTFLOAT_H

#include "gen/real_t.h"

// Type used by the front-end for compile-time reals
typedef ldc::real_t real_t;

// Compile-time floating-point helper
struct CTFloat
{
    static bool yl2x_supported;
    static bool yl2xp1_supported;

    static void yl2x(const real_t *x, const real_t *y, real_t *res);
    static void yl2xp1(const real_t *x, const real_t *y, real_t *res);

    static real_t parse(const char *literal, bool *isOutOfRange = NULL);

    static real_t sinImpl(const real_t &x);
    static real_t cosImpl(const real_t &x);
    static real_t tanImpl(const real_t &x);
    static real_t sqrtImpl(const real_t &x);
    static real_t fabsImpl(const real_t &x);

    // additional LDC built-ins
    static real_t logImpl(const real_t &x);
    static real_t fminImpl(const real_t &l, const real_t &r);
    static real_t fmaxImpl(const real_t &l, const real_t &r);
    static real_t floorImpl(const real_t &x);
    static real_t ceilImpl(const real_t &x);
    static real_t truncImpl(const real_t &x);
    static real_t roundImpl(const real_t &x);

    static bool isIdenticalImpl(const real_t &a, const real_t &b);
    static bool isNaNImpl(const real_t &r);
    static bool isSNaNImpl(const real_t &r);
    static bool isInfinityImpl(const real_t &r);

    static int sprintImpl(char *str, char fmt, const real_t &x);
};

#endif
