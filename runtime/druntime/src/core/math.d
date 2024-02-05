// Written in the D programming language.

/**
 * Builtin mathematical intrinsics
 *
 * Source: $(DRUNTIMESRC core/_math.d)
 * Macros:
 *      TABLE_SV = <table border="1" cellpadding="4" cellspacing="0">
 *              <caption>Special Values</caption>
 *              $0</table>
 *
 *      NAN = $(RED NAN)
 *      SUP = <span style="vertical-align:super;font-size:smaller">$0</span>
 *      POWER = $1<sup>$2</sup>
 *      PLUSMN = &plusmn;
 *      INFIN = &infin;
 *      PLUSMNINF = &plusmn;&infin;
 *      LT = &lt;
 *      GT = &gt;
 *
 * Copyright: Copyright Digital Mars 2000 - 2011.
 * License:   $(HTTP www.boost.org/LICENSE_1_0.txt, Boost License 1.0).
 * Authors:   $(HTTP digitalmars.com, Walter Bright),
 *                        Don Clugston
 */
module core.math;

version (LDC)
{
    import ldc.intrinsics;

    private enum isRealX87 = (real.mant_dig == 64);
}

public:
@nogc:
nothrow:
@safe:

pure:
/***********************************
 * Returns cosine of x. x is in radians.
 *
 *      $(TABLE_SV
 *      $(TR $(TH x)                 $(TH cos(x)) $(TH invalid?))
 *      $(TR $(TD $(NAN))            $(TD $(NAN)) $(TD yes)     )
 *      $(TR $(TD $(PLUSMN)$(INFIN)) $(TD $(NAN)) $(TD yes)     )
 *      )
 * Bugs:
 *      Results are undefined if |x| >= $(POWER 2,64).
 */

version (LDC)
{
    alias cos = llvm_cos!float;
    alias cos = llvm_cos!double;
    alias cos = llvm_cos!real;
}
else
{
    float cos(float x);     /* intrinsic */
    double cos(double x);   /* intrinsic */ /// ditto
    real cos(real x);       /* intrinsic */ /// ditto
}

/***********************************
 * Returns sine of x. x is in radians.
 *
 *      $(TABLE_SV
 *      $(TR $(TH x)               $(TH sin(x))      $(TH invalid?))
 *      $(TR $(TD $(NAN))          $(TD $(NAN))      $(TD yes))
 *      $(TR $(TD $(PLUSMN)0.0)    $(TD $(PLUSMN)0.0) $(TD no))
 *      $(TR $(TD $(PLUSMNINF))    $(TD $(NAN))      $(TD yes))
 *      )
 * Bugs:
 *      Results are undefined if |x| >= $(POWER 2,64).
 */

version (LDC)
{
    alias sin = llvm_sin!float;
    alias sin = llvm_sin!double;
    alias sin = llvm_sin!real;
}
else
{
    float sin(float x);     /* intrinsic */
    double sin(double x);   /* intrinsic */ /// ditto
    real sin(real x);       /* intrinsic */ /// ditto
}

/*****************************************
 * Returns x rounded to a long value using the current rounding mode.
 * If the integer value of x is
 * greater than long.max, the result is
 * indeterminate.
 */

version (LDC)
{
    private extern(C)
    {
        long llroundf(float x);
        long llround(double x);
        long llroundl(real x);
    }

    alias rndtol = llroundf;
    alias rndtol = llround;
    alias rndtol = llroundl;
}
else
{
    long rndtol(float x);   /* intrinsic */
    long rndtol(double x);  /* intrinsic */ /// ditto
    long rndtol(real x);    /* intrinsic */ /// ditto
}

/***************************************
 * Compute square root of x.
 *
 *      $(TABLE_SV
 *      $(TR $(TH x)         $(TH sqrt(x))   $(TH invalid?))
 *      $(TR $(TD -0.0)      $(TD -0.0)      $(TD no))
 *      $(TR $(TD $(LT)0.0)  $(TD $(NAN))    $(TD yes))
 *      $(TR $(TD +$(INFIN)) $(TD +$(INFIN)) $(TD no))
 *      )
 */

version (LDC)
{
    pragma(inline, true):

    // http://llvm.org/docs/LangRef.html#llvm-sqrt-intrinsic
    // sqrt(x) when x is less than zero is undefined
    float  sqrt(float  x) { return x < 0 ? float.nan  : llvm_sqrt(x); }
    double sqrt(double x) { return x < 0 ? double.nan : llvm_sqrt(x); }
    real   sqrt(real   x) { return x < 0 ? real.nan   : llvm_sqrt(x); }
}
else
{
    float sqrt(float x);    /* intrinsic */
    double sqrt(double x);  /* intrinsic */ /// ditto
    real sqrt(real x);      /* intrinsic */ /// ditto
}

/*******************************************
 * Compute n * 2$(SUPERSCRIPT exp)
 * References: frexp
 */

version (LDC)
{
    pragma(inline, true):

    // Implementation from libmir:
    // https://github.com/libmir/mir-core/blob/master/source/mir/math/ieee.d
    private T ldexpImpl(T)(const T n, int exp) @trusted pure nothrow
    {
        enum RealFormat { ieeeSingle, ieeeDouble, ieeeExtended, ieeeQuadruple }

             static if (T.mant_dig ==  24) enum realFormat = RealFormat.ieeeSingle;
        else static if (T.mant_dig ==  53) enum realFormat = RealFormat.ieeeDouble;
        else static if (T.mant_dig ==  64) enum realFormat = RealFormat.ieeeExtended;
        else static if (T.mant_dig == 113) enum realFormat = RealFormat.ieeeQuadruple;
        else static assert(false, "Unsupported format for " ~ T.stringof);

        version (LittleEndian)
        {
            enum MANTISSA_LSB = 0;
            enum MANTISSA_MSB = 1;
        }
        else
        {
            enum MANTISSA_LSB = 1;
            enum MANTISSA_MSB = 0;
        }

        static if (realFormat == RealFormat.ieeeExtended)
        {
            alias S = int;
            alias U = ushort;
            enum sig_mask = U(1) << (U.sizeof * 8 - 1);
            enum exp_shft = 0;
            enum man_mask = 0;
            version (LittleEndian)
                enum idx = 4;
            else
                enum idx = 0;
        }
        else
        {
            static if (realFormat == RealFormat.ieeeQuadruple || realFormat == RealFormat.ieeeDouble && double.sizeof == size_t.sizeof)
            {
                alias S = long;
                alias U = ulong;
            }
            else
            {
                alias S = int;
                alias U = uint;
            }
            static if (realFormat == RealFormat.ieeeQuadruple)
                alias M = ulong;
            else
                alias M = U;
            enum sig_mask = U(1) << (U.sizeof * 8 - 1);
            enum uint exp_shft = T.mant_dig - 1 - (T.sizeof > U.sizeof ? U.sizeof * 8 : 0);
            enum man_mask = (U(1) << exp_shft) - 1;
            enum idx = T.sizeof > U.sizeof ? MANTISSA_MSB : 0;
        }
        enum exp_mask = (U.max >> (exp_shft + 1)) << exp_shft;
        enum int exp_msh = exp_mask >> exp_shft;
        enum intPartMask = man_mask + 1;

        import core.checkedint : adds;
        alias _expect = llvm_expect;

        enum norm_factor = 1 / T.epsilon;
        T vf = n;

        auto u = (cast(U*)&vf)[idx];
        int e = (u & exp_mask) >> exp_shft;
        if (_expect(e != exp_msh, true))
        {
            if (_expect(e == 0, false)) // subnormals input
            {
                bool overflow;
                vf *= norm_factor;
                u = (cast(U*)&vf)[idx];
                e = int((u & exp_mask) >> exp_shft) - (T.mant_dig - 1);
            }
            bool overflow;
            exp = adds(exp, e, overflow);
            if (_expect(overflow || exp >= exp_msh, false)) // infs
            {
                static if (realFormat == RealFormat.ieeeExtended)
                {
                    return vf * T.infinity;
                }
                else
                {
                    u &= sig_mask;
                    u ^= exp_mask;
                    static if (realFormat == RealFormat.ieeeExtended)
                    {
                        version (LittleEndian)
                            auto mp = cast(ulong*)&vf;
                        else
                            auto mp = cast(ulong*)((cast(ushort*)&vf) + 1);
                        *mp = 0;
                    }
                    else
                    static if (T.sizeof > U.sizeof)
                    {
                        (cast(U*)&vf)[MANTISSA_LSB] = 0;
                    }
                }
            }
            else
            if (_expect(exp > 0, true)) // normal
            {
                u = cast(U)((u & ~exp_mask) ^ (cast(typeof(U.init + 0))exp << exp_shft));
            }
            else // subnormal output
            {
                exp = 1 - exp;
                static if (realFormat != RealFormat.ieeeExtended)
                {
                    auto m = u & man_mask;
                    if (exp > T.mant_dig)
                    {
                        exp = T.mant_dig;
                        static if (T.sizeof > U.sizeof)
                            (cast(U*)&vf)[MANTISSA_LSB] = 0;
                    }
                }
                u &= sig_mask;
                static if (realFormat == RealFormat.ieeeExtended)
                {
                    version (LittleEndian)
                        auto mp = cast(ulong*)&vf;
                    else
                        auto mp = cast(ulong*)((cast(ushort*)&vf) + 1);
                    if (exp >= ulong.sizeof * 8)
                        *mp = 0;
                    else
                        *mp >>>= exp;
                }
                else
                {
                    m ^= intPartMask;
                    static if (T.sizeof > U.sizeof)
                    {
                        int exp2 = exp - int(U.sizeof) * 8;
                        if (exp2 < 0)
                        {
                            (cast(U*)&vf)[MANTISSA_LSB] = ((cast(U*)&vf)[MANTISSA_LSB] >> exp) ^ (m << (U.sizeof * 8 - exp));
                            m >>>= exp;
                            u ^= cast(U) m;
                        }
                        else
                        {
                            exp = exp2;
                            (cast(U*)&vf)[MANTISSA_LSB] = (exp < U.sizeof * 8) ? m >> exp : 0;
                        }
                    }
                    else
                    {
                        m >>>= exp;
                        u ^= cast(U) m;
                    }
                }
            }
            (cast(U*)&vf)[idx] = u;
        }
        return vf;
    }

    float  ldexp(float  n, int exp) { return ldexpImpl(n, exp); }
    double ldexp(double n, int exp) { return ldexpImpl(n, exp); }
    static if (isRealX87)
    {
        // Roughly 20% faster than ldexpImpl() on an i5-3550 CPU.
        real ldexp(real n, int exp)
        {
            real r = void;
            asm @trusted pure nothrow @nogc
            {
                `fildl  %1       # push exp
                 fxch   %%st(1)  # swap ST(0) and ST(1)
                 fscale          # ST(0) := ST(0) * (2 ^^ ST(1))
                 fstp   %%st(1)  # pop and keep ST(0) value on top`
                : "=st" (r)
                : "m" (exp), "st" (n)
                : "flags"; // might clobber x87 flags
            }
            return r;
        }
    }
    else
    {
        real ldexp(real n, int exp) { return ldexpImpl(n, exp); }
    }
}
else
{
    float ldexp(float n, int exp);   /* intrinsic */
    double ldexp(double n, int exp); /* intrinsic */ /// ditto
    real ldexp(real n, int exp);     /* intrinsic */ /// ditto
}

unittest {
    static if (real.mant_dig == 113)
    {
        assert(ldexp(1.0L, -16384) == 0x1p-16384L);
        assert(ldexp(1.0L, -16382) == 0x1p-16382L);
    }
    else static if (real.mant_dig == 106)
    {
        assert(ldexp(1.0L,  1023) == 0x1p1023L);
        assert(ldexp(1.0L, -1022) == 0x1p-1022L);
        assert(ldexp(1.0L, -1021) == 0x1p-1021L);
    }
    else static if (real.mant_dig == 64)
    {
        assert(ldexp(1.0L, -16384) == 0x1p-16384L);
        assert(ldexp(1.0L, -16382) == 0x1p-16382L);
    }
    else static if (real.mant_dig == 53)
    {
        assert(ldexp(1.0L,  1023) == 0x1p1023L);
        assert(ldexp(1.0L, -1022) == 0x1p-1022L);
        assert(ldexp(1.0L, -1021) == 0x1p-1021L);
    }
    else
        assert(false, "Only 128bit, 80bit and 64bit reals expected here");
}

/*******************************
 * Compute the absolute value.
 *      $(TABLE_SV
 *      $(TR $(TH x)                 $(TH fabs(x)))
 *      $(TR $(TD $(PLUSMN)0.0)      $(TD +0.0) )
 *      $(TR $(TD $(PLUSMN)$(INFIN)) $(TD +$(INFIN)) )
 *      )
 * It is implemented as a compiler intrinsic.
 * Params:
 *      x = floating point value
 * Returns: |x|
 * References: equivalent to `std.math.fabs`
 */
version (LDC)
{
    alias fabs = llvm_fabs!float;
    alias fabs = llvm_fabs!double;
    alias fabs = llvm_fabs!real;
}
else @safe pure nothrow @nogc
{
    float  fabs(float  x);
    double fabs(double x); /// ditto
    real   fabs(real   x); /// ditto
}

/**********************************
 * Rounds x to the nearest integer value, using the current rounding
 * mode.
 * If the return value is not equal to x, the FE_INEXACT
 * exception is raised.
 * $(B nearbyint) performs
 * the same operation, but does not set the FE_INEXACT exception.
 */
version (LDC)
{
    alias rint = llvm_rint!float;
    alias rint = llvm_rint!double;
    alias rint = llvm_rint!real;
}
else
{
    float rint(float x);    /* intrinsic */
    double rint(double x);  /* intrinsic */ /// ditto
    real rint(real x);      /* intrinsic */ /// ditto
}

/***********************************
 * Building block functions, they
 * translate to a single x87 instruction.
 */

version (LDC)
{
    static if (isRealX87)
    {
        pragma(inline, true):

        // y * log2(x)
        real yl2x(real x, real y)
        {
            real r = void;
            asm @trusted pure nothrow @nogc { "fyl2x" : "=st" (r) : "st(1)" (y), "st" (x) : "st(1)", "flags"; }
            return r;
        }

        // y * log2(x + 1)
        real yl2xp1(real x, real y)
        {
            real r = void;
            asm @trusted pure nothrow @nogc { "fyl2xp1" : "=st" (r) : "st(1)" (y), "st" (x) : "st(1)", "flags"; }
            return r;
        }
    }
}
else
{
    // y * log2(x)
    float yl2x(float x, float y);    /* intrinsic */
    double yl2x(double x, double y);  /* intrinsic */ /// ditto
    real yl2x(real x, real y);      /* intrinsic */ /// ditto
    // y * log2(x +1)
    float yl2xp1(float x, float y);    /* intrinsic */
    double yl2xp1(double x, double y);  /* intrinsic */ /// ditto
    real yl2xp1(real x, real y);      /* intrinsic */ /// ditto
}

unittest
{
    version (INLINE_YL2X)
    {
        assert(yl2x(1024.0L, 1) == 10);
        assert(yl2xp1(1023.0L, 1) == 10);
    }
}

/*************************************
 * Round argument to a specific precision.
 *
 * D language types specify only a minimum precision, not a maximum. The
 * `toPrec()` function forces rounding of the argument `f` to the precision
 * of the specified floating point type `T`.
 * The rounding mode used is inevitably target-dependent, but will be done in
 * a way to maximize accuracy. In most cases, the default is round-to-nearest.
 *
 * Params:
 *      T = precision type to round to
 *      f = value to convert
 * Returns:
 *      f in precision of type `T`
 */
T toPrec(T:float)(float f) { pragma(inline, false); return f; }
/// ditto
T toPrec(T:float)(double f) { pragma(inline, false); return cast(T) f; }
/// ditto
T toPrec(T:float)(real f)  { pragma(inline, false); return cast(T) f; }
/// ditto
T toPrec(T:double)(float f) { pragma(inline, false); return f; }
/// ditto
T toPrec(T:double)(double f) { pragma(inline, false); return f; }
/// ditto
T toPrec(T:double)(real f)  { pragma(inline, false); return cast(T) f; }
/// ditto
T toPrec(T:real)(float f) { pragma(inline, false); return f; }
/// ditto
T toPrec(T:real)(double f) { pragma(inline, false); return f; }
/// ditto
T toPrec(T:real)(real f)  { pragma(inline, false); return f; }

@safe unittest
{
    // Test all instantiations work with all combinations of float.
    float f = 1.1f;
    double d = 1.1;
    real r = 1.1L;
    f = toPrec!float(f + f);
    f = toPrec!float(d + d);
    f = toPrec!float(r + r);
    d = toPrec!double(f + f);
    d = toPrec!double(d + d);
    d = toPrec!double(r + r);
    r = toPrec!real(f + f);
    r = toPrec!real(d + d);
    r = toPrec!real(r + r);

    // Comparison tests.
    bool approxEqual(T)(T lhs, T rhs)
    {
        return fabs((lhs - rhs) / rhs) <= 1e-2 || fabs(lhs - rhs) <= 1e-5;
    }

    enum real PIR = 0xc.90fdaa22168c235p-2;
    enum double PID = 0x1.921fb54442d18p+1;
    enum float PIF = 0x1.921fb6p+1;
    static assert(approxEqual(toPrec!float(PIR), PIF));
    static assert(approxEqual(toPrec!double(PIR), PID));
    static assert(approxEqual(toPrec!real(PIR), PIR));
    static assert(approxEqual(toPrec!float(PID), PIF));
    static assert(approxEqual(toPrec!double(PID), PID));
    static assert(approxEqual(toPrec!real(PID), PID));
    static assert(approxEqual(toPrec!float(PIF), PIF));
    static assert(approxEqual(toPrec!double(PIF), PIF));
    static assert(approxEqual(toPrec!real(PIF), PIF));

    assert(approxEqual(toPrec!float(PIR), PIF));
    assert(approxEqual(toPrec!double(PIR), PID));
    assert(approxEqual(toPrec!real(PIR), PIR));
    assert(approxEqual(toPrec!float(PID), PIF));
    assert(approxEqual(toPrec!double(PID), PID));
    assert(approxEqual(toPrec!real(PID), PID));
    assert(approxEqual(toPrec!float(PIF), PIF));
    assert(approxEqual(toPrec!double(PIF), PIF));
    assert(approxEqual(toPrec!real(PIF), PIF));
}
