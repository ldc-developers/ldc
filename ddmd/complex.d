// Compiler implementation of the D programming language
// Copyright (c) 1999-2015 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.complex;

import ddmd.root.ctfloat;

struct complex_t
{
    real_t re;
    real_t im;

    this() @disable;

    this(real_t re)
    {
        this(re, real_t(0));
    }

    this(real_t re, real_t im)
    {
        this.re = re;
        this.im = im;
    }

    complex_t opAdd(complex_t y)
    {
        return complex_t(re + y.re, im + y.im);
    }

    complex_t opSub(complex_t y)
    {
        return complex_t(re - y.re, im - y.im);
    }

    complex_t opNeg()
    {
        return complex_t(-re, -im);
    }

    complex_t opMul(complex_t y)
    {
        return complex_t(re * y.re - im * y.im, im * y.re + re * y.im);
    }

    complex_t opMul_r(real_t x)
    {
        return complex_t(x) * this;
    }

    complex_t opMul(real_t y)
    {
        return this * complex_t(y);
    }

    complex_t opDiv(real_t y)
    {
        return this / complex_t(y);
    }

    complex_t opDiv(complex_t y)
    {
        if (CTFloat.fabs(y.re) < CTFloat.fabs(y.im))
        {
            const r = y.re / y.im;
            const den = y.im + r * y.re;
            return complex_t((re * r + im) / den, (im * r - re) / den);
        }
        else
        {
            const r = y.im / y.re;
            const den = y.re + r * y.im;
            return complex_t((re + r * im) / den, (im - r * re) / den);
        }
    }

    bool opCast(T : bool)()
    {
        return re || im;
    }

    int opEquals(complex_t y)
    {
        return re == y.re && im == y.im;
    }
}

extern (C++) real_t creall(complex_t x)
{
    return x.re;
}

extern (C++) real_t cimagl(complex_t x)
{
    return x.im;
}
