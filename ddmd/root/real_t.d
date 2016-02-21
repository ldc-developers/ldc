// Compiler implementation of the D programming language
// Copyright (c) 1999-2016 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.root.real_t;

static import core.math;
import core.stdc.errno;
import core.stdc.stdio;
import core.stdc.string;

version(IN_LLVM_MSVC)
{
    alias real_t = double;
}
else
{
    alias real_t = real;
}

private
{
    version(CRuntime_DigitalMars) __gshared extern (C) extern const(char)* __locale_decpoint;

    version(CRuntime_Microsoft)   extern(C++) struct longdouble { real_t r; }
    version(CRuntime_Microsoft)   extern(C++) size_t ld_sprint(char* str, int fmt, longdouble x);

    version(IN_LLVM_MSVC)
        extern (C) double strtold(const(char)* p, char** endp);
    else
    version(CRuntime_Microsoft)
        extern (C++) longdouble strtold_dm(const(char)* p, char** endp);
    else
        extern (C) real strtold(const(char)* p, char** endp);

    extern (C) float  strtof(const(char)* p, char** endp);
    extern (C) double strtod(const(char)* p, char** endp);

    version(CRuntime_Microsoft)
    {
        enum _OVERFLOW = 3;    /* overflow range error */
        enum _UNDERFLOW = 4;   /* underflow range error */

        extern (C) int _atoflt(float* value, const(char)* str);
        extern (C) int _atodbl(double* value, const(char)* str);
    }
}

extern(C++, target_fp)
{
    version(IN_LLVM)
    {
        __gshared bool yl2x_supported = false;
        __gshared bool yl2xp1_supported = false;
    }
    else
    version(DigitalMars)
    {
        __gshared bool yl2x_supported = true;
        __gshared bool yl2xp1_supported = true;
    }
    else
    {
        __gshared bool yl2x_supported = false;
        __gshared bool yl2xp1_supported = false;
    }

    __gshared real_t snan;

    real_t sqrt(real_t x)
    {
        return core.math.sqrt(x);
    }

    void yl2x(real_t* x, real_t* y, real_t* res)
    {
        version(DigitalMars)
            *res = core.math.yl2x(*x, *y);
        version(IN_LLVM)
            assert(0);
    }

    void yl2xp1(real_t* x, real_t* y, real_t* res)
    {
        version(DigitalMars)
            *res = core.math.yl2xp1(*x, *y);
        version(IN_LLVM)
            assert(0);
    }

    bool areBitwiseEqual(real_t a, real_t b)
    {
        //TODO: proper support for non-x87
        enum unpaddedSize = (is(real_t == double) ? 8 : 10);
        return memcmp(&a, &b, unpaddedSize) == 0;
    }

    bool isNaN(real_t r)
    {
        return !(r == r);
    }

    bool isInfinity(real_t r)
    {
        return r is real_t.infinity || r is -real_t.infinity;
    }

    real_t parseTargetReal(const(char)* literal)
    {
        version (CRuntime_DigitalMars)
        {
            auto save = __locale_decpoint;
            __locale_decpoint = ".";
        }

        version(IN_LLVM_MSVC)
            auto r = strtold(literal, null);  // C99 conformant since VS 2015
        else
        version (CRuntime_Microsoft)
            auto r = strtold_dm(literal, null).r;
        else
            auto r = strtold(literal, null);
        version (CRuntime_DigitalMars) __locale_decpoint = save;
        return r;
    }

    bool isFloat32LiteralOutOfRange(const(char)* literal)
    {
        errno = 0;
        version (CRuntime_DigitalMars)
        {
            auto save = __locale_decpoint;
            __locale_decpoint = ".";
        }
        version (CRuntime_Microsoft)
        {
            float r;
            int res = _atoflt(&r, literal);
            if (res == _UNDERFLOW || res == _OVERFLOW)
                errno = ERANGE;
        }
        else
        {
            strtof(literal, null);
        }
        version (CRuntime_DigitalMars) __locale_decpoint = save;
        return errno == ERANGE;
    }

    bool isFloat64LiteralOutOfRange(const(char)* literal)
    {
        errno = 0;
        version (CRuntime_DigitalMars)
        {
            auto save = __locale_decpoint;
            __locale_decpoint = ".";
        }
        version (CRuntime_Microsoft)
        {
            double r;
            int res = _atodbl(&r, literal);
            if (res == _UNDERFLOW || res == _OVERFLOW)
                errno = ERANGE;
        }
        else
        {
            strtod(literal, null);
        }
        version (CRuntime_DigitalMars) __locale_decpoint = save;
        return errno == ERANGE;
    }

    size_t sprint(char* str, char fmt, real_t x)
    {
        version(IN_LLVM_MSVC)
        {
            if (real_t(cast(ulong)x) == x)
            {
                // ((1.5 -> 1 -> 1.0) == 1.5) is false
                // ((1.0 -> 1 -> 1.0) == 1.0) is true
                // see http://en.cppreference.com/w/cpp/io/c/fprintf
                char[4] sfmt = "%#g\0";
                sfmt[2] = cast(char)fmt;
                return sprintf(str, sfmt.ptr, double(x));
            }
            else
            {
                char[3] sfmt = "%g\0";
                sfmt[1] = cast(char)fmt;
                return sprintf(str, sfmt.ptr, double(x));
            }
        }
        else
        version(CRuntime_Microsoft)
        {
            return ld_sprint(str, fmt, longdouble(x));
        }
        else
        {
            if (real_t(cast(ulong)x) == x)
            {
                // ((1.5 -> 1 -> 1.0) == 1.5) is false
                // ((1.0 -> 1 -> 1.0) == 1.0) is true
                // see http://en.cppreference.com/w/cpp/io/c/fprintf
                char[5] sfmt = "%#Lg\0";
                sfmt[3] = cast(char)fmt;
                return sprintf(str, sfmt.ptr, x);
            }
            else
            {
                char[4] sfmt = "%Lg\0";
                sfmt[2] = cast(char)fmt;
                return sprintf(str, sfmt.ptr, x);
            }
        }
    }
}

static this()
{
    /*
     * Use a payload which is different from the machine NaN,
     * so that uninitialised variables can be
     * detected even if exceptions are disabled.
     */
    //TODO: proper support for non-x87
    static if (is(real_t == double))
        snan = real_t.nan;
    else
    {
        ushort* us = cast(ushort*)&snan;
        us[0] = 0;
        us[1] = 0;
        us[2] = 0;
        us[3] = 0xA000;
        us[4] = 0x7FFF;
    }
}
