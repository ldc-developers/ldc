module ldc.longdouble;

/**
* Implementation of support routines for real data type on Windows 64.
*
* Copyright: Copyright The LDC Developers 2012
* License:   <a href="http://www.boost.org/LICENSE_1_0.txt">Boost License 1.0</a>.
* Authors:   Kai Nacke <kai@redstar.de>
*/

/*          Copyright The LDC Developers 2012.
* Distributed under the Boost Software License, Version 1.0.
*    (See accompanying file LICENSE or copy at
*          http://www.boost.org/LICENSE_1_0.txt)
*/

version(Windows):
extern(C):

import ldc.llvmasm;


private:
    enum {
        MASK_ROUNDING = 0xf3ffu,
    }

    enum {
        ROUND_TO_NEAREST = 0x0000u,
        ROUND_TO_MINUS_INF = 0x0400u,
        ROUND_TO_PLUS_INF = 0x0800u,
        ROUND_TO_ZERO = 0x0C00u,

        EXC_PRECISION = 0x0020u,
    }

/*
 * The C runtime environment of Visual C++ has no support for long double
 * aka real datatype. This file adds the missing functions.
 */

// Computes the cosine.
real cosl(real x)
{
    return __asm!(real)("fcos", "={st},{st}", x);
}

// Computes the sine.
real sinl(real x)
{
    return __asm!(real)("fsin", "={st},{st}", x);
}

unittest
{
    real v = 3.0L;
    real s = sinl(v);
    real c = cosl(v);
    assert(s*s + c*c == 1.0L);
}

// Computes the tangent.
real tanl(real x)
{
    return __asm!(real)("fptan", "={st},{st}", x);
}

// Computes the square root.
real sqrtl(real x)
{
    return __asm!(real)("fsqrt", "={st},{st}", x);
}

unittest
{
    assert(sqrtl(4.0L) == 2.0L);
    assert(sqrtl(16.0L) == 4.0L);
}

// Round to the nearest integer value.
long llroundl(real x)
{
    return 0;
}

// Returns an unbiased exponent.
real ilogbl(real x)
{
    return 0.0;
}

// Loads exponent of a floating-point number.
real ldexpl(real arg, int exp)
{
//    See http://llvm.org/bugs/show_bug.cgi?id=15773
//    return __asm!(real)("fscale", "={st},{st(1)},{st},~{st(1)}", exp, arg);
    return __asm!(real)("fildl $2 ; fxch %st(1) ; fscale", "={st},{st},*m}", arg, &exp);
}

// Computes the natural logarithm.
real logl(real x)
{
    // Computes log_e(x) = 1/(log_2 e) * log_2 x
    return __asm!(real)("fldl2e; fld1 ; fdivp ; fxch %st(1); fyl2x", "={st},{st}", x);
}

// Computes the base 10 logarithm.
real log10l(real x)
{
    return __asm!(real)("fldlg2 ; fxch %st(1) ; fyl2x", "={st},{st}", x);
}

// Computes a natural logarithm.
real log1pl(real x)
{
    // Computes log_e(1.0 + x) = 1/(log_2 e) * log_2 (1.0 + x)
    // FIXME: Check input rang of x and use fyl2x if not in range
    return __asm!(real)("fldl2e; fld1 ; fdivp ; fxch %st(1); fyl2xp1", "={st},{st}", x);
}

// Computes the base 2 logarithm.
real log2l(real x)
{
    return __asm!(real)("fld1 ; fxch %st(1) ; fyl2x", "={st},{st}", x);
}

// Computes the radix-independent exponent.
real logbl(real x)
{
    return 0.0;
}

// Computes the base-2 exponential of x (2^x).
real exp2l(real x)
{
    return 0.0;
}

// Computes the base-2 exponential of x (2^x).
double exp2(double x)
{
    return 0.0;
}

// Computes the base-2 exponential of x (2^x).
float exp2f(float x)
{
    return 0.0;
}

// Computes the floating-point remainder value.
real fmodl(real arg1, real arg2)
{
    return 0.0;
}

// Decomposes a floating-point number.
real modfl(real x, real *iptr)
{
    return 0.0;
}

// Computes the cube root.
real cbrtl(real arg)
{
    return 0.0;
}

// Determines the absolute value.
real fabsl(real x)
{
    return __asm!(real)("fabs", "={st},{st}", x);
}

unittest
{
    assert(42.0L == fabsl(-42.0L));
    assert(42.0L == fabsl(42.0L));
}

// Compute the ceiling value.
real ceill(real x)
{
    // Store the current FPU control word.
    uint fpuctl = void;
    __asm!(void)("fstcw $0", "*m", &fpuctl);
    // Load new FPU control word with rounding mode set toward +oo (bit 11/10 = 10b)
    uint newctl = (fpuctl & MASK_ROUNDING) | ROUND_TO_PLUS_INF;
    __asm!(void)("fldcw $0", "*m", &newctl);
    // Round to integer
    real res = __asm!(real)("frndint", "={st},{st}", x);
    // Restore FPU control word
    __asm!(void)("fldcw $0", "*m", &fpuctl);
    return res;
}

unittest
{
    assert(5.0L == ceill(4.1L));
    assert(-4.0L == ceill(-4.1L));
}

// Return the largest floating-point integer value that is not greater than the parameter.
real floorl(real x)
{
    // Store the current FPU control word.
    uint fpuctl = void;
    __asm!(void)("fstcw $0", "*m", &fpuctl);
    // Load new FPU control word with rounding mode set toward -oo (bit 11/10 = 01b)
    uint newctl = (fpuctl & MASK_ROUNDING) | ROUND_TO_MINUS_INF;
    __asm!(void)("fldcw $0", "*m", &newctl);
    // Round to integer
    real res = __asm!(real)("frndint", "={st},{st}", x);
    // Restore FPU control word
    __asm!(void)("fldcw $0", "*m", &fpuctl);
    return res;
}

unittest
{
    assert(4.0L == floorl(4.1L));
    assert(-5.0L == floorl(-4.1L));
}

// Round argument to an integer value in floating-point format, using the current rounding
// direction and without raising the inexact floating-point exception.
real nearbyintl(real x)
{
    // Store the current FPU control word.
    uint fpuctl = void;
    __asm!(void)("fnstcw $0", "*m", &fpuctl);
    // Load new FPU control word with precision exception = blocked (bit 5 set = 0x20)
    uint newctl = fpuctl | EXC_PRECISION;
    __asm!(void)("fldcw $0", "*m", &newctl);
    // Round to integer
    real res = __asm!(real)("frndint", "={st},{st}", x);
    // Restore FPU control word
    __asm!(void)("fldcw $0", "*m", &fpuctl);
    return res;
}

// Returns the integral value (represented as a floating-point number) nearest
// arg in the direction of the current rounding mode.
real rintl(real x)
{
    return __asm!(real)("frndint", "={st},{st}", x);
}

// Round to the nearest integer value using current rounding direction.
long llrintl(real x)
{
    long res = void;
    __asm!(void)("fistpl $0", "=*m,{st},~{st}", &res, x);
    return  res;
}

// Rounds to the nearest integer value in a floating-point format.
real roundl(real arg)
{
    return 0.0;
}

// Rounds to truncated integer value.
real truncl(real x)
{
    // Store the current FPU control word.
    uint fpuctl = void;
    __asm!(void)("fstcw $0", "*m", &fpuctl);
    // Load new FPU control word with rounding mode set toward zero (bit 11/10 = 11b)
    uint newctl = (fpuctl & MASK_ROUNDING) | ROUND_TO_ZERO;
    __asm!(void)("fldcw $0", "*m", &newctl);
    // Round to integer
    real res = __asm!(real)("frndint", "={st},{st}", x);
    // Restore FPU control word
    __asm!(void)("fldcw $0", "*m", &fpuctl);
    return res;
}

// Returns the floating-point remainder.
real remainderl(real x, real y)
{
    return 0.0;
}

// Computes power.
real powl(real x, real y)
{
    return 0.0;
}

// Computes the exponent using FLT_RADIX=2.
real scalbnl (real x, int n)
{
    return __asm!(real)("fildl $2 ; fxch %st(1) ; fscale", "={st},{st},*m}", x, &n);
}


