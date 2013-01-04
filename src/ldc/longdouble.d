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
real cosl(real x)
{
    asm
    {
        fld x;
        fcos;
        fstp x;
    }
    return x;
}

// Computes the sine.
real sinl(real x)
{
    asm
    {
        fld x;
        fsin;
        fstp x;
    }
    return x;
}

// Computes the tangent.
real tanl(real x)
{
    asm
    {
        fld x;
        fptan;
        fstp x;
    }
    return x;
}

// Computes the square root.
real sqrtl(real x)
{
    asm
    {
        fld x;
        fsqrt;
        fstp x;
    }
    return x;
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
    asm
    {
        fild exp;
        fld arg;
        fscale;
        fstp arg;
    }
    return arg;
}

// Computes the natural logarithm.
real logl(real x)
{
    asm
    {
        fld1;
        fldl2e;
        fdivrp;
        fld x;
        fyl2x;
        fstp x;
    }
    return x;
}

// Computes the base 10 logarithm.
real log10l(real x)
{
    asm
    {
        fldlg2;
        fld x;
        fyl2x;
        fstp x;
    }
    return x;
}

// Computes a natural logarithm.
real log1pl(real x)
{
    return 0.0;
}

// Computes the base 2 logarithm.
real log2l(real x)
{
    asm
    {
        fld1;
        fld x;
        fyl2x;
        fstp x;
    }
    return x;
}

// Computes the radix-independent exponent.
real logbl(real x)
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
real fabsl(real x)
{
    asm
    {
        fld x;
        fabs;
        fstp x;
    }
    return x;
}

// Compute the ceiling value.
real ceill(real arg)
{
    return 0.0;
}

// Return the largest floating-point integer value that is not greater than the parameter.
real floorl(real x)
{
    return 0.0;
}

// Round numbers to an integer value in floating-point format.
real nearbyintl(real x)
{
    return 0.0;
}

// Returns the integral value (represented as a floating-point number) nearest
// arg in the direction of the current rounding mode.
real rintl(real x)
{
    asm
    {
        fld x;
        frndint;
        fstp x;
    }
    return x;
}

// Round to the nearest integer value using current rounding direction.
long llrintl(real x)
{
    long res;
    asm
    {
        fld x;
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
/*
    asm
    {
        mov [RBP+0x10],ECX; // Use shadow area
        fild [RBP+0x10];
        fistp x;
        fstp x;
    }
    return x;
*/
    return 0.0;
}
