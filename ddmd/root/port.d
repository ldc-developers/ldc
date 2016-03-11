// Compiler implementation of the D programming language
// Copyright (c) 1999-2015 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.root.port;

import core.stdc.ctype;
import core.stdc.string;
import core.stdc.stdio;
import core.stdc.errno;
import core.math;

version(CRuntime_DigitalMars) __gshared extern (C) extern const(char)* __locale_decpoint;
version(CRuntime_Microsoft)   extern(C++) struct longdouble { real r; }
version(CRuntime_Microsoft)   extern(C++) size_t ld_sprint(char* str, int fmt, longdouble x);

extern (C) float strtof(const(char)* p, char** endp);
extern (C) double strtod(const(char)* p, char** endp);

version(IN_LLVM_MSVC)
    extern (C) double strtold(const(char)* p, char** endp);
else
version(CRuntime_Microsoft)
    extern (C++) longdouble strtold_dm(const(char)* p, char** endp);
else
    extern (C) real strtold(const(char)* p, char** endp);

version(CRuntime_Microsoft)
{
    enum _OVERFLOW = 3;   /* overflow range error */
    enum _UNDERFLOW = 4;   /* underflow range error */

    extern (C) int _atoflt(float* value, const char * str);
    extern (C) int _atodbl(double* value, const char * str);
}

extern (C++) struct Port
{
    enum nan = double.nan;
    enum infinity = double.infinity;
    version(IN_LLVM_MSVC)
        private alias ldbl = double;
    else
        private alias ldbl = real;

    version(IN_LLVM)
    {
        enum ldbl_min_normal = ldbl.min_normal;
        enum ldbl_max = ldbl.max;
        enum ldbl_nan = ldbl.nan;
        enum ldbl_infinity = ldbl.infinity;
        enum ldbl_dig = ldbl.dig;
        enum ldbl_epsilon = ldbl.epsilon;
        enum ldbl_mant_dig = ldbl.mant_dig;
        enum ldbl_max_10_exp = ldbl.max_10_exp;
        enum ldbl_max_exp = ldbl.max_exp;
        enum ldbl_min_10_exp = ldbl.min_10_exp;
        enum ldbl_min_exp = ldbl.min_exp;
    }
    else
    {
        enum ldbl_max = real.max;
        enum ldbl_nan = real.nan;
        enum ldbl_infinity = real.infinity;
    }
    version(IN_LLVM)
    {
        static __gshared bool yl2x_supported = false;
        static __gshared bool yl2xp1_supported = false;
    }
    else
    version(DigitalMars)
    {
        static __gshared bool yl2x_supported = true;
        static __gshared bool yl2xp1_supported = true;
    }
    else
    {
        static __gshared bool yl2x_supported = false;
        static __gshared bool yl2xp1_supported = false;
    }
    static __gshared real snan;

    static bool isNan(double r)
    {
        return !(r == r);
    }

    static real sqrt(real x)
    {
        return .sqrt(x);
    }

    static real fmodl(real a, real b)
    {
        return a % b;
    }

    static bool fequal(real a, real b)
    {
        // don't compare pad bytes in extended precision
        enum sz = (real.mant_dig == 64) ? 10 : real.sizeof;
        return memcmp(&a, &b, sz) == 0;
    }

    static int memicmp(const char* s1, const char* s2, size_t n)
    {
        int result = 0;

        for (int i = 0; i < n; i++)
        {
            char c1 = s1[i];
            char c2 = s2[i];

            result = c1 - c2;
            if (result)
            {
                result = toupper(c1) - toupper(c2);
                if (result)
                    break;
            }
        }
        return result;
    }

    static char* strupr(char* s)
    {
        char* t = s;

        while (*s)
        {
            *s = cast(char)toupper(*s);
            s++;
        }

        return t;
    }

    static int isSignallingNan(double r)
    {
        return isNan(r) && !(((cast(ubyte*)&r)[6]) & 8);
    }

    static int isSignallingNan(real r)
    {
        return isNan(r) && !(((cast(ubyte*)&r)[7]) & 0x40);
    }

    version(CRuntime_Microsoft)
    {
        static int isSignallingNan(longdouble ld)
        {
            return isSignallingNan(*cast(real*)&ld);
        }
    }

    static int isInfinity(double r)
    {
        return r is double.infinity || r is -double.infinity;
    }

    static float strtof(const(char)* p, char** endp)
    {
        version (CRuntime_DigitalMars)
        {
            auto save = __locale_decpoint;
            __locale_decpoint = ".";
        }
        version (CRuntime_Microsoft)
        {
            float r;
            if(endp)
            {
                r = .strtod(p, endp); // does not set errno for underflows, but unused
            }
            else
            {
                int res = _atoflt(&r, p);
                if (res == _UNDERFLOW || res == _OVERFLOW)
                    errno = ERANGE;
            }
        }
        else
        {
            auto r = .strtof(p, endp);
        }
        version (CRuntime_DigitalMars) __locale_decpoint = save;
        return r;
    }

    static double strtod(const(char)* p, char** endp)
    {
        version (CRuntime_DigitalMars)
        {
            auto save = __locale_decpoint;
            __locale_decpoint = ".";
        }
        version (CRuntime_Microsoft)
        {
            double r;
            if(endp)
            {
                r = .strtod(p, endp); // does not set errno for underflows, but unused
            }
            else
            {
                int res = _atodbl(&r, p);
                if (res == _UNDERFLOW || res == _OVERFLOW)
                    errno = ERANGE;
            }
        }
        else
        {
            auto r = .strtod(p, endp);
        }
        version (CRuntime_DigitalMars) __locale_decpoint = save;
        return r;
    }

    static real strtold(const(char)* p, char** endp)
    {
        version (CRuntime_DigitalMars)
        {
            auto save = __locale_decpoint;
            __locale_decpoint = ".";
        }

        version(IN_LLVM_MSVC)
            auto r = .strtold(p, endp);  // C99 conformant since VS 2015
        else
        version (CRuntime_Microsoft)
            auto r = .strtold_dm(p, endp).r;
        else
            auto r = .strtold(p, endp);
        version (CRuntime_DigitalMars) __locale_decpoint = save;
        return r;
    }

    static size_t ld_sprint(char* str, int fmt, real x)
    {
        version(IN_LLVM_MSVC)
        {
            if ((cast(real)cast(ulong)x) == x)
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
            return .ld_sprint(str, fmt, longdouble(x));
        }
        else
        {
            if ((cast(real)cast(ulong)x) == x)
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

    static void yl2x_impl(real* x, real* y, real* res)
    {
        version(DigitalMars)
            *res = yl2x(*x, *y);
        version(IN_LLVM)
            assert(0);
    }

    static void yl2xp1_impl(real* x, real* y, real* res)
    {
        version(DigitalMars)
            *res = yl2xp1(*x, *y);
        version(IN_LLVM)
            assert(0);
    }

    // Little endian
    static void writelongLE(uint value, void* buffer)
    {
        auto p = cast(ubyte*)buffer;
        p[3] = cast(ubyte)(value >> 24);
        p[2] = cast(ubyte)(value >> 16);
        p[1] = cast(ubyte)(value >> 8);
        p[0] = cast(ubyte)(value);
    }

    // Little endian
    static uint readlongLE(void* buffer)
    {
        auto p = cast(ubyte*)buffer;
        return (((((p[3] << 8) | p[2]) << 8) | p[1]) << 8) | p[0];
    }

    // Big endian
    static void writelongBE(uint value, void* buffer)
    {
        auto p = cast(ubyte*)buffer;
        p[0] = cast(ubyte)(value >> 24);
        p[1] = cast(ubyte)(value >> 16);
        p[2] = cast(ubyte)(value >> 8);
        p[3] = cast(ubyte)(value);
    }

    // Big endian
    static uint readlongBE(void* buffer)
    {
        auto p = cast(ubyte*)buffer;
        return (((((p[0] << 8) | p[1]) << 8) | p[2]) << 8) | p[3];
    }

    // Little endian
    static uint readwordLE(void* buffer)
    {
        auto p = cast(ubyte*)buffer;
        return (p[1] << 8) | p[0];
    }

    // Big endian
    static uint readwordBE(void* buffer)
    {
        auto p = cast(ubyte*)buffer;
        return (p[0] << 8) | p[1];
    }

    version(IN_LLVM)
    {   // LDC_FIXME: look at the old port.c how to implement this with system calls for some OSses
        static int stricmp(const(char)* s1, const(char)* s2)
        {
            int result = 0;

            for (;;)
            {   char c1 = *s1;
                char c2 = *s2;

                result = c1 - c2;
                if (result)
                {
                    result = toupper(c1) - toupper(c2);
                    if (result)
                        break;
                }
                if (!c1)
                    break;
                s1++;
                s2++;
            }
            return result;
        }
    }
}
