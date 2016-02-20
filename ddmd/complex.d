// Compiler implementation of the D programming language
// Copyright (c) 1999-2015 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.complex;

import ddmd.root.real_t;

struct complex_t
{
    real_t re = 0;
    real_t im = 0;

    this(real_t re)
    {
        this.re = re;
        this.im = 0;
    }

    this(real_t re, real_t im)
    {
        this.re = re;
        this.im = im;
    }

    complex_t opAdd(complex_t y)
    {
        complex_t r;
        r.re = re + y.re;
        r.im = im + y.im;
        return r;
    }

    complex_t opSub(complex_t y)
    {
        complex_t r;
        r.re = re - y.re;
        r.im = im - y.im;
        return r;
    }

    complex_t opNeg()
    {
        complex_t r;
        r.re = -re;
        r.im = -im;
        return r;
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
        real_t abs_y_re = y.re < 0 ? -y.re : y.re;
        real_t abs_y_im = y.im < 0 ? -y.im : y.im;

        if (abs_y_re < abs_y_im)
        {
            real_t r = y.re / y.im;
            real_t den = y.im + r * y.re;
            return complex_t((re * r + im) / den, (im * r - re) / den);
        }
        else
        {
            real_t r = y.im / y.re;
            real_t den = y.re + r * y.im;
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

real_t creall(complex_t x)
{
    return x.re;
}

real_t cimagl(complex_t x)
{
    return x.im;
}
