/**
 * Collects functions for compile-time floating-point calculations.
 *
 * Copyright:   Copyright (C) 1999-2024 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 https://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 https://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
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
version (IN_LLVM) {} else
{
        import dmd.root.strtold;
}
    }
}

// Compile-time floating-point helper
extern (C++) struct CTFloat
{
  nothrow:
  @nogc:
  @safe:

    version (GNU)
        enum yl2x_supported = false;
    else
        enum yl2x_supported = __traits(compiles, core.math.yl2x(1.0L, 2.0L));
    enum yl2xp1_supported = yl2x_supported;

    static void yl2x(const real_t* x, const real_t* y, real_t* res) // IN_LLVM: impure because of log2
    {
        static if (yl2x_supported)
            *res = core.math.yl2x(*x, *y);
        else version (IN_LLVM)
            *res = *y * log2(*x); // fall back to generic version
        else
            assert(0);
    }

    static void yl2xp1(const real_t* x, const real_t* y, real_t* res) // IN_LLVM: impure because of log2
    {
        static if (yl2xp1_supported)
            *res = core.math.yl2xp1(*x, *y);
        else version (IN_LLVM)
            *res = *y * log2(*x + real_t(1)); // fall back to generic version
        else
            assert(0);
    }

    static if (!is(real_t == real))
    {
        static import dmd.root.longdouble;
        alias sin = dmd.root.longdouble.sinl;
        alias cos = dmd.root.longdouble.cosl;
        alias tan = dmd.root.longdouble.tanl;
        alias sqrt = dmd.root.longdouble.sqrtl;
        alias fabs = dmd.root.longdouble.fabsl;
        alias ldexp = dmd.root.longdouble.ldexpl;
    }
    else
    {
        pure static real_t sin(real_t x) { return core.math.sin(x); }
        pure static real_t cos(real_t x) { return core.math.cos(x); }
        static real_t tan(real_t x) { return core.stdc.math.tanl(x); }
        pure static real_t sqrt(real_t x) { return core.math.sqrt(x); }
        pure static real_t fabs(real_t x) { return core.math.fabs(x); }
        pure static real_t ldexp(real_t n, int exp) { return core.math.ldexp(n, exp); }
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

    pure
    static real_t fmin(real_t x, real_t y) { return x < y ? x : y; }
    pure
    static real_t fmax(real_t x, real_t y) { return x > y ? x : y; }

    pure
    static real_t fma(real_t x, real_t y, real_t z) { return (x * y) + z; }

  version (IN_LLVM)
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

    static bool isFloat32LiteralOutOfRange(const(char)* literal) @nogc;
    static bool isFloat64LiteralOutOfRange(const(char)* literal) @nogc;
  }

    pure @trusted
    static bool isIdentical(real_t a, real_t b)
    {
        // don't compare pad bytes in extended precision
        enum sz = (real_t.mant_dig == 64) ? 10 : real_t.sizeof;
        return memcmp(&a, &b, sz) == 0;
    }

    pure @trusted
    static size_t hash(real_t a)
    {
        import dmd.root.hash : calcHash;

        if (isNaN(a))
            a = real_t.nan;
        enum sz = (real_t.mant_dig == 64) ? 10 : real_t.sizeof;
        return calcHash((cast(ubyte*) &a)[0 .. sz]);
    }

    pure
    static bool isNaN(real_t r)
    {
        return !(r == r);
    }

  version (IN_LLVM)
  {
    // LDC doesn't need isSNaN(). The upstream implementation is tailored for
    // DMD/x86 and only supports x87 real_t types.
  }
  else
  {
    pure @trusted
    static bool isSNaN(real_t r)
    {
        return isNaN(r) && !(((cast(ubyte*)&r)[7]) & 0x40);
    }

    // the implementation of longdouble for MSVC is a struct, so mangling
    //  doesn't match with the C++ header.
    // add a wrapper just for isSNaN as this is the only function called from C++
    version(CRuntime_Microsoft) static if (is(real_t == real))
        pure @trusted
        static bool isSNaN(longdouble_soft ld)
        {
            return isSNaN(cast(real)ld);
        }
  }

    static bool isInfinity(real_t r) pure
    {
        return isIdentical(fabs(r), real_t.infinity);
    }

  version (IN_LLVM)
  {
    // implemented in gen/ctfloat.cpp
    @system
    static real_t parse(const(char)* literal, out bool isOutOfRange);
    @system
    static int sprint(char* str, size_t size, char fmt, real_t x);
  }
  else
  {
    @system
    static real_t parse(const(char)* literal, out bool isOutOfRange)
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
        isOutOfRange = (errno == ERANGE);
        return r;
    }

    @system
    static int sprint(char* str, size_t size, char fmt, real_t x)
    {
        version(CRuntime_Microsoft)
        {
            auto len = cast(int) ld_sprint(str, size, fmt, longdouble_soft(x));
        }
        else
        {
            char[4] sfmt = "%Lg\0";
            sfmt[2] = fmt;
            auto len = snprintf(str, size, sfmt.ptr, x);
        }

        if (fmt != 'a' && fmt != 'A')
        {
            assert(fmt == 'g');

            // 1 => 1.0 to distinguish from integers
            bool needsFPSuffix = true;
            foreach (char c; str[0 .. len])
            {
                // str might be `nan` or `inf`...
                if (c != '-' && !(c >= '0' && c <= '9'))
                {
                    needsFPSuffix = false;
                    break;
                }
            }

            if (needsFPSuffix)
            {
                str[len .. len+3] = ".0\0";
                len += 2;
            }
        }

        return len;
    }
  }

    // Constant real values 0, 1, -1 and 0.5.
    __gshared real_t zero;
    __gshared real_t one;
    __gshared real_t minusone;
    __gshared real_t half;
  version (IN_LLVM)
  {
    __gshared real_t nan;
    __gshared real_t infinity;
  }

  version (IN_LLVM)
  {
    // implemented in gen/ctfloat.cpp
    @trusted
    static void initialize();
  }
  else
  {
    @trusted
    static void initialize()
    {
        zero = real_t(0);
        one = real_t(1);
        minusone = real_t(-1);
        half = real_t(0.5);
    }
  }
}

version (IN_LLVM)
{
    version (Android) { /* double/quadruple real_t */ } else
    {
        version (X86)    version = real_t_X87;
        version (X86_64) version = real_t_X87;
    }

    // Test parsing and printing of real_t values.
    unittest
    {
        CTFloat.initialize();

        static void printAndCheck(char format, real_t x, string expected) nothrow
        {
            char[32] buffer = void;
            const length = CTFloat.sprint(buffer.ptr, buffer.length, format, x);
            assert(length < buffer.length);
            printf("'%s', expected '%.*s'\n", buffer.ptr, cast(int) expected.length, expected.ptr);
            assert(buffer[0 .. length] == expected);
            assert(buffer[length] == 0);
        }

        static struct T
        {
            nothrow:

            real_t x;
            string expected_g, expected_a, expected_A;

            this(real_t x, string g, string a, string A)
            {
                this.x = x;
                expected_g = g;
                expected_a = a;
                expected_A = A;
            }

            this(string x, string g, string a, string A)
            {
                bool isOutOfRange;
                this.x = CTFloat.parse(x.ptr, isOutOfRange);
                expected_g = g;
                expected_a = a;
                expected_A = A;
            }

            void test() const
            {
                printAndCheck('g', x, expected_g);
                printAndCheck('a', x, expected_a);
                printAndCheck('A', x, expected_A);
            }
        }

        immutable T[] generic_ts = [
            T( CTFloat.nan,       "nan",  "nan",  "NAN"),
            T(-CTFloat.nan,      "-nan", "-nan", "-NAN"),
            T( CTFloat.infinity,  "inf",  "inf",  "INF"),
            T(-CTFloat.infinity, "-inf", "-inf", "-INF"),
            T( "0.0",      "0.0",    "0x0p+0",    "0X0P+0"),
            T("-0.0",     "-0.0",   "-0x0p+0",   "-0X0P+0"),
            T( "0x1p-1",   "0.5",    "0x1p-1",    "0X1P-1"),
            T( "0x3p-3",   "0.375",  "0x1.8p-2",  "0X1.8P-2"),
            T( "0x1p+0",   "1.0",    "0x1p+0",    "0X1P+0"),
            T( "0x3p-1",   "1.5",    "0x1.8p+0",  "0X1.8P+0"),
            T("-0x3p-2",  "-0.75",  "-0x1.8p-1", "-0X1.8P-1"),
            T(  "100.0", "100.0",    "0x1.9p+6",  "0X1.9P+6"),
        ];

        foreach (t; generic_ts)
        {
            version (AArch64)
            {
                // FPU may not preserve NaN sign (depending on 'default NaN mode' control bit)
                if (t.expected_g == "-nan")
                    continue;
            }
            t.test();
        }

        version (real_t_X87)
        {
            immutable T[] x87_ts = [
                T(                         "1e+300",  "1e+300",       "0x1.7e43c8800759ba5ap+996",  "0X1.7E43C8800759BA5AP+996"),
                T(                         "1e-300",  "1e-300",       "0x1.56e1fc2f8f358d94p-997",  "0X1.56E1FC2F8F358D94P-997"),
                T( "1.2345678901234567890123456789",  "1.23457",      "0x1.3c0ca428c59fb71ap+0",    "0X1.3C0CA428C59FB71AP+0"),
                T("-12.345678901234567890123456789", "-12.3457",     "-0x1.8b0fcd32f707a4e2p+3",   "-0X1.8B0FCD32F707A4E2P+3"),
                T( "123456.78901234567890123456789",  "123457.0",     "0x1.e240c9fcb68cd4c4p+16",   "0X1.E240C9FCB68CD4C4P+16"),
                T("-123456.78901234567890123456789", "-123457.0",    "-0x1.e240c9fcb68cd4c4p+16",  "-0X1.E240C9FCB68CD4C4P+16"),
                T( "1234567.8901234567890123456789",  "1.23457e+06",  "0x1.2d687e3df21804fap+20",   "0X1.2D687E3DF21804FAP+20"),
                T("-1234567.8901234567890123456789", "-1.23457e+06", "-0x1.2d687e3df21804fap+20",  "-0X1.2D687E3DF21804FAP+20"),
                T( "0.0001234567890123456789012345",  "0.000123457",  "0x1.02e85be180b7447cp-13",   "0X1.02E85BE180B7447CP-13"),
                T("-0.0001234567890123456789012345", "-0.000123457", "-0x1.02e85be180b7447cp-13",  "-0X1.02E85BE180B7447CP-13"),
                T( "0.0000123456789012345678901234",  "1.23457e-05",  "0x1.9e409302678ba0c8p-17",   "0X1.9E409302678BA0C8P-17"),
                T("-0.0000123456789012345678901234", "-1.23457e-05", "-0x1.9e409302678ba0c8p-17",  "-0X1.9E409302678BA0C8P-17"),
            ];

            foreach (t; x87_ts)
                t.test();
        }
    }
}
