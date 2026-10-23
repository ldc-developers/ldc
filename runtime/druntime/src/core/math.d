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
    import ldc.attributes : llvmAttr;
    import ldc.llvmasm : __irEx_pure;

    private enum isRealX87 = (real.mant_dig == 64);
}

version (LDC) version (Posix)
@system unittest
{
    import core.stdc.fenv : fenv_t, fegetenv, fesetenv, fesetround,
        FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO;

    fenv_t saved;
    assert(fegetenv(&saved) == 0);
    scope(exit) fesetenv(&saved);
    // A stack address is not a compile-time input. Both values are in [1,2).
    real input = 1;
    static if (real.mant_dig == 64)
    {
        version (LittleEndian)
            enum lowWord = 0;
        else
            enum lowWord = 1;
        (cast(ulong*)&input)[lowWord] |= cast(size_t)&saved & 0xffff;
        fesetround(FE_TONEAREST);
        real nearest = ldexpImpl(input, real.min_exp - real.mant_dig - 2);
        fesetround(FE_UPWARD);
        real upward = ldexpImpl(input, real.min_exp - real.mant_dig - 2);
        assert(nearest == 0);
        assert(upward == real.min_normal * real.epsilon);

        // Exercise the public x87 instruction and the generic implementation
        // with the same exact references, including the lowest fraction bit.
        struct Case { real value; int exponent; uint[4] units; }
        enum e = real.min_exp - real.mant_dig;
        Case[] cases = [
            Case(0.609375L, e + 3, [5, 5, 4, 4]),
            Case(0.5625L, e + 3, [4, 5, 4, 4]),
            Case(0.6875L, e + 3, [6, 6, 5, 5]),
            Case(0.625L, e + 3, [5, 5, 5, 5]),
            Case(1.0L + real.epsilon, e - 1, [1, 1, 0, 0]),
            Case(1.0L, e - 2, [0, 1, 0, 0]),
        ];
        foreach (negative; [false, true])
        foreach (mi, mode; [FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO])
        foreach (c; cases)
        {
            real value = negative ? -c.value : c.value;
            assert(fesetround(mode) == 0);
            auto magnitudeMode = negative && (mi == 1 || mi == 2) ? 3 - mi : mi;
            real expected = c.units[magnitudeMode] * (real.min_normal * real.epsilon);
            if (negative)
                expected = -expected;
            assert(ldexpImpl(value, c.exponent) is expected);
            assert(ldexp(value, c.exponent) is expected);
        }
    }
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
    alias rndtol = llvm_llround!float;
    alias rndtol = llvm_llround!double;
    alias rndtol = llvm_llround!real;
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
            enum sig_mask = U(1) << (U.sizeof * 8 - 1);
            enum uint exp_shft = T.mant_dig - 1 - (T.sizeof > U.sizeof ? U.sizeof * 8 : 0);
            enum man_mask = (U(1) << exp_shft) - 1;
            enum idx = T.sizeof > U.sizeof ? MANTISSA_MSB : 0;
        }
        enum exp_mask = (U.max >> (exp_shft + 1)) << exp_shft;
        enum int exp_msh = exp_mask >> exp_shft;

        import core.checkedint : adds;
        alias _expect = llvm_expect;

        enum norm_factor = 1 / T.epsilon;
        T vf = n;

        auto u = (cast(U*)&vf)[idx];
        int e = (u & exp_mask) >> exp_shft;
        if (_expect(e != exp_msh, true))
        {
            if (_expect(e == 0, false)) // zero or subnormal input
            {
                if (vf == 0)
                    return vf; // preserve the sign of zero

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
                static if (T.mant_dig <= 53)
                {
                    return ldexpSubnormal(vf, exp);
                }
                else
                {
                    // Keep all significand bits in an exact normal intermediate.
                    // The final multiplication performs the subnormal rounding.
                    // Clamping deeper underflow preserves directed rounding.
                    if (exp < -T.mant_dig)
                        exp = -T.mant_dig;
                    exp += T.mant_dig + 1;
                    u = cast(U)((u & ~exp_mask) | (cast(typeof(U.init + 0))exp << exp_shft));
                    (cast(U*)&vf)[idx] = u;
                    return scaleSubnormal(vf, T.epsilon / 4);
                }
            }
            (cast(U*)&vf)[idx] = u;
        }
        return vf;
    }

    // Round the desired result y in units of the smallest subnormal q, then
    // encode the integer. For precision p, bias B and adjusted biased exponent E,
    // y/q has biased exponent E+B+p-2. Rewriting that exponent preserves the significand
    // exactly. nearbyint rounds the signed units in the active mode; integer
    // encoding preserves the historical float/double exception behavior.
    // A rounded magnitude of 2^(p-1) encodes minimum normal directly, so the
    // subnormal-to-normal carry needs no floating-point correction.
    // Keep this rare path out of normal calls, including library-call setup
    // on targets where nearbyint does not lower to an instruction.
    // strictfp plus constrained operations prevent reuse across fenv changes.
    // Do not introduce ordinary FP comparisons/conversions in this helper.
    pragma(inline, false)
    @llvmAttr("strictfp")
    private T ldexpSubnormal(T)(T value, int exponent) @trusted
    {
        static if (T.mant_dig == 24)
            alias Bits = uint;
        else
            alias Bits = ulong;
        static assert((T.mant_dig == 24 || T.mant_dig == 53) && T.sizeof == Bits.sizeof);
        enum shift = T.mant_dig - 1;
        enum signMask = Bits(1) << (Bits.sizeof * 8 - 1);
        enum exponentMask = (Bits.max >> (shift + 1)) << shift;
        auto bits = *cast(Bits*)&value;
        const sign = bits & signMask;

        // All still smaller magnitudes round like a signed value below half
        // a unit: zero for nearest/toward-zero, or one unit when directed away
        // from zero. The clamp also keeps exponent arithmetic in range.
        if (exponent < -T.mant_dig)
            exponent = -T.mant_dig;
        enum bias = T.max_exp - 1;
        const scaledExponent = exponent + bias + T.mant_dig - 2;
        bits = (bits & ~exponentMask) | (Bits(scaledExponent) << shift);
        *cast(Bits*)&value = bits;

        // Use the active rounding mode without introducing an inexact flag
        // where the previous integer-shift implementation raised none.
        // The rounded magnitude is at most 2^(p-1), so conversion through
        // signed long is exact and bounded. 2^(p-1) is also the bit encoding
        // of the smallest normal, handling a carry without a special case.
        // Clear the rounded sign with integer operations (including -0).
        enum irType = is(T == float) ? "float" : "double";
        enum irSuffix = is(T == float) ? "f32" : "f64";
        enum intType = is(T == float) ? "i32" : "i64";
        enum magnitudeMask = is(T == float) ? "2147483647" : "9223372036854775807";
        const magnitude = cast(Bits)__irEx_pure!(
            "declare " ~ irType ~ " @llvm.experimental.constrained.nearbyint." ~ irSuffix ~ "(" ~ irType ~ ", metadata, metadata) strictfp\n" ~
            "declare i64 @llvm.experimental.constrained.fptosi.i64." ~ irSuffix ~ "(" ~ irType ~ ", metadata) strictfp",
            "%r = call " ~ irType ~ " @llvm.experimental.constrained.nearbyint." ~ irSuffix ~ "(" ~ irType ~ " %0, metadata !\"round.dynamic\", metadata !\"fpexcept.strict\") strictfp\n" ~
            "%b = bitcast " ~ irType ~ " %r to " ~ intType ~ "\n" ~
            "%absbits = and " ~ intType ~ " %b, " ~ magnitudeMask ~ "\n" ~
            "%abs = bitcast " ~ intType ~ " %absbits to " ~ irType ~ "\n" ~
            "%mag = call i64 @llvm.experimental.constrained.fptosi.i64." ~ irSuffix ~ "(" ~ irType ~ " %abs, metadata !\"fpexcept.strict\") strictfp\nret i64 %mag",
            "", long, T)(value);
        *cast(Bits*)&value = sign | magnitude;
        return value;
    }


    // Isolate the wider-format final rounding from ordinary FP operations in
    // ldexpImpl. On soft-float targets the runtime providing multiplication
    // must also honor the active mode; constrained IR cannot supply that
    // guarantee for a nearest-only __multf3 implementation.
    pragma(inline, false)
    @llvmAttr("strictfp")
    private T scaleSubnormal(T)(T value, T factor) @trusted pure nothrow @nogc
    {
        static if (T.mant_dig == 64)
        {
            enum irType = "x86_fp80";
            enum irSuffix = "f80";
        }
        else static if (T.mant_dig == 113)
        {
            enum irType = "fp128";
            enum irSuffix = "f128";
        }
        else
            static assert(false, "Unsupported wider format");
        return __irEx_pure!(
            "declare " ~ irType ~ " @llvm.experimental.constrained.fmul." ~ irSuffix ~ "(" ~ irType ~ ", " ~ irType ~ ", metadata, metadata) strictfp",
            "%r = call " ~ irType ~ " @llvm.experimental.constrained.fmul." ~ irSuffix ~ "(" ~ irType ~ " %0, " ~ irType ~ " %1, metadata !\"round.dynamic\", metadata !\"fpexcept.strict\") strictfp\nret " ~ irType ~ " %r",
            "", T, T, T)(value, factor);
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

// https://github.com/ldc-developers/ldc/issues/5302
@safe pure nothrow @nogc unittest
{
    // Force a runtime call even in optimized builds.
    pragma(inline, false)
    T scale(T)(T value, int exponent)
    {
        return ldexp(value, exponent);
    }

    void check(T)()
    {
        foreach (zero; [T(0.0), T(-0.0)])
        {
            foreach (exponent; [int.min, -1510, -75, 0, 23, 24, 52, 53, 75, 1510, int.max])
            {
                const result = scale(zero, exponent);
                assert(result is zero);
            }
        }

        // Nonzero subnormals still need normalization.
        const smallest = T.min_normal * T.epsilon;
        foreach (sign; [T(1), T(-1)])
        {
            assert(scale(sign * smallest, T.mant_dig - 1) == sign * T.min_normal);
            assert(scale(sign * T.min_normal, 1 - T.mant_dig) == sign * smallest);
            assert(scale(sign * T(1.5), 3) == sign * T(12));
        }
    }

    check!float();
    check!double();
    check!real();
}

// Subnormal outputs must round rather than discard the low significand bits.
unittest
{
    pragma(inline, false)
    T scale(T, bool generic)(T value, int exponent)
    {
        static if (generic)
            return ldexpImpl(value, exponent);
        else
            return ldexp(value, exponent);
    }

    void check(T, bool generic = false)()
    {
        const unit = T.min_normal * T.epsilon;
        enum exponent = T.min_exp - T.mant_dig + 3;
        foreach (sign; [T(1), T(-1)])
        {
            // Exactly 4.875, 4.5, 5.5, and 5 minimum-subnormal units.
            assert(scale!(T, generic)(sign * T(0.609375), exponent) == sign * (5 * unit));
            assert(scale!(T, generic)(sign * T(0.5625), exponent) == sign * (4 * unit));
            assert(scale!(T, generic)(sign * T(0.6875), exponent) == sign * (6 * unit));
            assert(scale!(T, generic)(sign * T(0.625), exponent) == sign * (5 * unit));
            // The midpoint below the smallest normal rounds up to that value.
            assert(scale!(T, generic)(sign * (T(2) - T.epsilon), T.min_exp - 2) == sign * T.min_normal);
            const zero = scale!(T, generic)(sign * T(0.5), T.min_exp - T.mant_dig);
            assert(zero == 0);
            assert(T(1) / zero == sign * T.infinity);
            assert(scale!(T, generic)(sign * T(0.75), T.min_exp - T.mant_dig) == sign * unit);
        }
    }
    check!float();
    check!double();
    check!real();
    version (LDC)
        check!(real, true)(); // Exercise the generic wider-format path as well.
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
