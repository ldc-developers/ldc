/***
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/root/ctfloat.d, root/_ctfloat.d)
 * Documentation: https://dlang.org/phobos/dmd_root_ctfloat.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/root/ctfloat.d
 */

module dmd.root.ctfloat;

static import core.math, core.stdc.math;
import core.stdc.errno;
import core.stdc.stdio;
import core.stdc.stdlib;
import core.stdc.string;

nothrow:

// Type used by the front-end for compile-time reals
public import dmd.root.longdouble : real_t = longdouble;

private
{
    version(CRuntime_DigitalMars) __gshared extern (C) extern const(char)* __locale_decpoint;

    version(CRuntime_Microsoft) extern (C++)
    {
        public import dmd.root.longdouble : longdouble_soft, ld_sprint;
        longdouble_soft strtold_dm(const(char)* p, char** endp);
    }
}

// Compile-time floating-point helper
extern (C++) struct CTFloat
{
  nothrow:
    version (GNU)
        enum yl2x_supported = false;
    else
        enum yl2x_supported = __traits(compiles, core.math.yl2x(1.0L, 2.0L));
    enum yl2xp1_supported = yl2x_supported;

    static void yl2x(const real_t* x, const real_t* y, real_t* res)
    {
        static if (yl2x_supported)
            *res = core.math.yl2x(*x, *y);
        else
            assert(0);
    }

    static void yl2xp1(const real_t* x, const real_t* y, real_t* res)
    {
        static if (yl2xp1_supported)
            *res = core.math.yl2xp1(*x, *y);
        else
            assert(0);
    }

    static if (!is(real_t == real))
    {
        alias sin = dmd.root.longdouble.sinl;
        alias cos = dmd.root.longdouble.cosl;
        alias tan = dmd.root.longdouble.tanl;
        alias sqrt = dmd.root.longdouble.sqrtl;
        alias fabs = dmd.root.longdouble.fabsl;
        alias ldexp = dmd.root.longdouble.ldexpl;
    }
    else
    {
        static real_t sin(real_t x) { return core.math.sin(x); }
        static real_t cos(real_t x) { return core.math.cos(x); }
        static real_t tan(real_t x) { return core.stdc.math.tanl(x); }
        static real_t sqrt(real_t x) { return core.math.sqrt(x); }
        static real_t fabs(real_t x) { return core.math.fabs(x); }
        static real_t ldexp(real_t n, int exp) { return core.math.ldexp(n, exp); }
    }

    static if (!is(real_t == real))
    {
        static real_t round(real_t x) { return real_t(cast(double)core.stdc.math.roundl(cast(double)x)); }
        static real_t floor(real_t x) { return real_t(cast(double)core.stdc.math.floor(cast(double)x)); }
        static real_t ceil(real_t x) { return real_t(cast(double)core.stdc.math.ceil(cast(double)x)); }
        static real_t trunc(real_t x) { return real_t(cast(double)core.stdc.math.trunc(cast(double)x)); }
        static real_t log(real_t x) { return real_t(cast(double)core.stdc.math.logl(cast(double)x)); }
        static real_t log2(real_t x) { return real_t(cast(double)core.stdc.math.log2l(cast(double)x)); }
        static real_t log10(real_t x) { return real_t(cast(double)core.stdc.math.log10l(cast(double)x)); }
        static real_t pow(real_t x, real_t y) { return real_t(cast(double)core.stdc.math.powl(cast(double)x, cast(double)y)); }
        static real_t exp(real_t x) { return real_t(cast(double)core.stdc.math.expl(cast(double)x)); }
        static real_t expm1(real_t x) { return real_t(cast(double)core.stdc.math.expm1l(cast(double)x)); }
        static real_t exp2(real_t x) { return real_t(cast(double)core.stdc.math.exp2l(cast(double)x)); }
        static real_t copysign(real_t x, real_t s) { return real_t(cast(double)core.stdc.math.copysignl(cast(double)x, cast(double)s)); }
    }
    else
    {
        static real_t round(real_t x) { return core.stdc.math.roundl(x); }
        static real_t floor(real_t x) { return core.stdc.math.floor(x); }
        static real_t ceil(real_t x) { return core.stdc.math.ceil(x); }
        static real_t trunc(real_t x) { return core.stdc.math.trunc(x); }
        static real_t log(real_t x) { return core.stdc.math.logl(x); }
        static real_t log2(real_t x) { return core.stdc.math.log2l(x); }
        static real_t log10(real_t x) { return core.stdc.math.log10l(x); }
        static real_t pow(real_t x, real_t y) { return core.stdc.math.powl(x, y); }
        static real_t exp(real_t x) { return core.stdc.math.expl(x); }
        static real_t expm1(real_t x) { return core.stdc.math.expm1l(x); }
        static real_t exp2(real_t x) { return core.stdc.math.exp2l(x); }
        static real_t copysign(real_t x, real_t s) { return core.stdc.math.copysignl(x, s); }
    }

    static real_t fmin(real_t x, real_t y) { return x < y ? x : y; }
    static real_t fmax(real_t x, real_t y) { return x > y ? x : y; }

    static real_t fma(real_t x, real_t y, real_t z) { return (x * y) + z; }

  version(IN_LLVM)
  {
    static import std.math;

    static if (!is(real_t == real))
    {
        static real_t rint(real_t x) { return real_t(cast(double)std.math.rint(cast(double)x)); }
        static real_t nearbyint(real_t x) { return real_t(cast(double)std.math.nearbyint(cast(double)x)); }
    }
    else
    {
        static real_t rint(real_t x) { return std.math.rint(x); }
        static real_t nearbyint(real_t x) { return std.math.nearbyint(x); }
    }

    static void _init();

    static bool isFloat32LiteralOutOfRange(const(char)* literal);
    static bool isFloat64LiteralOutOfRange(const(char)* literal);
  }

    static bool isIdentical(real_t a, real_t b)
    {
        // don't compare pad bytes in extended precision
        enum sz = (real_t.mant_dig == 64) ? 10 : real_t.sizeof;
        return memcmp(&a, &b, sz) == 0;
    }

    static size_t hash(real_t a)
    {
        import dmd.root.hash : calcHash;

        if (isNaN(a))
            a = real_t.nan;
        enum sz = (real_t.mant_dig == 64) ? 10 : real_t.sizeof;
        return calcHash(cast(ubyte*) &a, sz);
    }

    static bool isNaN(real_t r)
    {
        return !(r == r);
    }

  version(IN_LLVM)
  {
    // LDC doesn't need isSNaN(). The upstream implementation is tailored for
    // DMD/x86 and only supports x87 real_t types.
  }
  else
  {
    static bool isSNaN(real_t r)
    {
        return isNaN(r) && !(((cast(ubyte*)&r)[7]) & 0x40);
    }

    // the implementation of longdouble for MSVC is a struct, so mangling
    //  doesn't match with the C++ header.
    // add a wrapper just for isSNaN as this is the only function called from C++
    version(CRuntime_Microsoft) static if (is(real_t == real))
        static bool isSNaN(longdouble_soft ld)
        {
            return isSNaN(cast(real)ld);
        }
  }

    static bool isInfinity(real_t r)
    {
        return isIdentical(fabs(r), real_t.infinity);
    }

version (IN_LLVM)
{
    // implemented in gen/ctfloat.cpp
    static real_t parse(const(char)* literal, bool* isOutOfRange = null);
}
else
{
    static real_t parse(const(char)* literal, bool* isOutOfRange = null)
    {
        errno = 0;
        version(CRuntime_DigitalMars)
        {
            auto save = __locale_decpoint;
            __locale_decpoint = ".";
        }
        version(CRuntime_Microsoft)
        {
            auto r = cast(real_t) strtold_dm(literal, null);
        }
        else
            auto r = strtold(literal, null);
        version(CRuntime_DigitalMars) __locale_decpoint = save;
        if (isOutOfRange)
            *isOutOfRange = (errno == ERANGE);
        return r;
    }
}

    static int sprint(char* str, char fmt, real_t x)
    {
        version(CRuntime_Microsoft)
        {
            return cast(int)ld_sprint(str, fmt, longdouble_soft(x));
        }
        else
        {
            if (real_t(cast(ulong)x) == x)
            {
                // ((1.5 -> 1 -> 1.0) == 1.5) is false
                // ((1.0 -> 1 -> 1.0) == 1.0) is true
                // see http://en.cppreference.com/w/cpp/io/c/fprintf
                char[5] sfmt = "%#Lg\0";
                sfmt[3] = fmt;
                return sprintf(str, sfmt.ptr, x);
            }
            else
            {
                char[4] sfmt = "%Lg\0";
                sfmt[2] = fmt;
                return sprintf(str, sfmt.ptr, x);
            }
        }
    }

    // Constant real values 0, 1, -1 and 0.5.
    __gshared real_t zero;
    __gshared real_t one;
    __gshared real_t minusone;
    __gshared real_t half;
  version(IN_LLVM)
  {
    // Initialized via LLVM in C++.
    __gshared real_t initVal;
    __gshared real_t nan;
    __gshared real_t infinity;
  }

    static void initialize()
    {
        zero = real_t(0);
        one = real_t(1);
        minusone = real_t(-1);
        half = real_t(0.5);
    }
}

version(IN_LLVM)
{
    shared static this()
    {
        CTFloat._init();
    }
}
