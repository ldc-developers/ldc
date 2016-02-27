//===-- ctfloat.d ---------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module ddmd.root.ctfloat;

// Publicly import LDC real_t type used by the front-end for compile-time reals
public import ddmd.root.real_t : real_t;

// Compile-time floating-point helper
extern (C++) struct CTFloat
{
    static __gshared bool yl2x_supported = false;
    static __gshared bool yl2xp1_supported = false;

    static void yl2x(const real_t* x, const real_t* y, real_t* res)
    {
        assert(0);
    }

    static void yl2xp1(const real_t* x, const real_t* y, real_t* res)
    {
        assert(0);
    }

    static real_t parse(const(char)* literal, bool* isOutOfRange = null);

    static real_t sinImpl(const ref real_t x);
    static real_t cosImpl(const ref real_t x);
    static real_t tanImpl(const ref real_t x);
    static real_t sqrtImpl(const ref real_t x);
    static real_t fabsImpl(const ref real_t x);

    // additional LDC built-ins
    static real_t logImpl(const ref real_t x);
    static real_t fminImpl(const ref real_t l, const ref real_t r);
    static real_t fmaxImpl(const ref real_t l, const ref real_t r);
    static real_t floorImpl(const ref real_t x);
    static real_t ceilImpl(const ref real_t x);
    static real_t truncImpl(const ref real_t x);
    static real_t roundImpl(const ref real_t x);

    static bool isIdenticalImpl(const ref real_t a, const ref real_t b);
    static bool isNaNImpl(const ref real_t r);
    static bool isSNaNImpl(const ref real_t r);
    static bool isInfinityImpl(const ref real_t r);

    static int sprintImpl(char* str, char fmt, const ref real_t x);

    extern(D):

    static real_t sin(real_t x) { return sinImpl(x); }
    static real_t cos(real_t x) { return cosImpl(x); }
    static real_t tan(real_t x) { return tanImpl(x); }
    static real_t sqrt(real_t x) { return sqrtImpl(x); }
    static real_t fabs(real_t x) { return fabsImpl(x); }

    static real_t log(real_t x) { return logImpl(x); }
    static real_t fmin(real_t l, real_t r) { return fminImpl(l, r); }
    static real_t fmax(real_t l, real_t r) { return fmaxImpl(l, r); }
    static real_t floor(real_t x) { return floorImpl(x); }
    static real_t ceil(real_t x) { return ceilImpl(x); }
    static real_t trunc(real_t x) { return truncImpl(x); }
    static real_t round(real_t x) { return roundImpl(x); }

    static bool isIdentical(real_t a, real_t b) { return isIdenticalImpl(a, b); }
    static bool isNaN(real_t r) { return isNaNImpl(r); }
    static bool isSNaN(real_t r) { return isSNaNImpl(r); }
    static bool isInfinity(real_t r) { return isInfinityImpl(r); }

    static int sprint(char* str, char fmt, real_t x)
    {
        return sprintImpl(str, fmt, x);
    }
}
