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

version(Win64):
extern(C):

/*
 * The C runtime environment of Visual C++ has no support for long double
 * aka real datatype. This file adds the missing functions.
 */

// Computes the cosine.
real cosl(real arg)
{
    real res;
    asm
    {
        fld arg;
        fcos;
        fstp res;
    }
    return res;
}

// Computes the sine.
real sinl(real arg)
{
    real res;
    asm
    {
        fld arg;
        fsin;
        fstp res;
    }
    return res;
}

// Computes the tangent.
real tanl(real arg)
{
    real res;
    asm
    {
        fld arg;
        fptan;
        fstp res;
    }
    return res;
}

// Computes the square root.
real sqrtl(real arg)
{
    real res;
    asm
    {
        fld arg;
        fsqrt;
        fstp res;
    }
    return res;
}

// Round to the nearest integer value.
long llroundl(real arg)
{
    return 0;
}

// Returns an unbiased exponent.
real ilogbl(real arg)
{
    return 0.0;
}

// Loads exponent of a floating-point number.
real ldexpl(real arg, int exp)
{
    real res;
    asm
    {
        fild exp;
        fld arg;
        fscale;
        fstp res;
    }
    return res;
}

// Computes the natural logarithm.
real logl(real arg)
{
    real res;
    asm
    {
        fld1;
        fldl2e;
        fdivrp;
        fld arg;
        fyl2x;
        fstp res;
    }
    return res;
}

// Computes the base 10 logarithm.
real log10l(real arg)
{
    return 0.0;
}

// Computes a natural logarithm.
real log1pl(real arg)
{
    return 0.0;
}

// Computes the base 2 logarithm.
real log2l(real arg)
{
    return 0.0;
}

// Computes the radix-independent exponent.
real logbl(real arg)
{
    return 0.0;
}

// Computes the base 2 exponential.
real exp2l(real x)
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
real fabsl(real arg)
{
    return 0.0;
}

// Compute the ceiling value.
real ceill(real arg)
{
    return 0.0;
}

// Return the largest floating-point integer value that is not greater than the parameter.
real floorl(real arg)
{
    return 0.0;
}

// Round numbers to an integer value in floating-point format.
real nearbyintl(real arg)
{
    return 0.0;
}

// Returns the integral value (represented as a floating-point number) nearest
// arg in the direction of the current rounding mode.
real rintl(real arg)
{
    real res;
    asm
    {
        fld arg;
        frndint;
        fstp res;
    }
    return res;
}

// Round to the nearest integer value using current rounding direction.
long llrintl(real arg)
{
    long res;
    asm
    {
        fld arg;
        fistp res;
    }
    return res;
}

// Rounds to the nearest integer value in a floating-point format.
real roundl(real arg)
{
    return 0.0;
}

// Rounds to truncated integer value.
real truncl(real arg)
{
    return 0.0;
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
	return 0.0;
}
