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
    import stdc = core.stdc.math;
    import ldc.intrinsics;
    import ldc.llvmasm;
}

public:
@nogc:

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
    alias cos = llvm_cos!real;
else
real cos(real x) @safe pure nothrow;       /* intrinsic */

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
    alias sin = llvm_sin!real;
else
real sin(real x) @safe pure nothrow;       /* intrinsic */

/*****************************************
 * Returns x rounded to a long value using the current rounding mode.
 * If the integer value of x is
 * greater than long.max, the result is
 * indeterminate.
 */
version (LDC)
    alias rndtol = stdc.llroundl;
else
long rndtol(real x) @safe pure nothrow;    /* intrinsic */


/*****************************************
 * Returns x rounded to a long value using the FE_TONEAREST rounding mode.
 * If the integer value of x is
 * greater than long.max, the result is
 * indeterminate.
 */
extern (C) real rndtonl(real x);

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

@safe pure nothrow
{
  version (LDC)
  {
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
}

/*******************************************
 * Compute n * 2$(SUPERSCRIPT exp)
 * References: frexp
 */

version (LDC)
{
    version (MinGW)
    {
        real ldexp(real n, int exp) @safe pure nothrow
        {
            // The MinGW runtime only provides a double precision ldexp, and
            // it doesn't seem to reliably possible to express the fscale
            // semantics (two FP stack inputs/returns) in an inline asm
            // expression clobber list.
            version (D_InlineAsm_X86_64)
            {
                asm @trusted pure nothrow
                {
                    naked;
                    push RCX;                // push exp (8 bytes), passed in ECX
                    fild int ptr [RSP];      // push exp onto FPU stack
                    pop RCX;                 // return stack to initial state
                    fld real ptr [RDX];      // push n   onto FPU stack, passed in [RDX]
                    fscale;                  // ST(0) = ST(0) * 2^ST(1)
                    fstp ST(1);              // pop stack maintaining top value => function return value
                    ret;                     // no arguments passed via stack
                }
            }
            else
            {
                asm @trusted pure nothrow
                {
                    naked;
                    push EAX;
                    fild int ptr [ESP];
                    fld real ptr [ESP+8];
                    fscale;
                    fstp ST(1);
                    pop EAX;
                    ret 12;
                }
            }
        }
    }
    else // !MinGW
    {
        alias ldexp = stdc.ldexpl;
    }
}
else
real ldexp(real n, int exp) @safe pure nothrow;    /* intrinsic */

unittest {
    static if (real.mant_dig == 113)
    {
        assert(ldexp(1, -16384) == 0x1p-16384L);
        assert(ldexp(1, -16382) == 0x1p-16382L);
    }
    else static if (real.mant_dig == 106)
    {
        assert(ldexp(1,  1023) == 0x1p1023L);
        assert(ldexp(1, -1022) == 0x1p-1022L);
        assert(ldexp(1, -1021) == 0x1p-1021L);
    }
    else static if (real.mant_dig == 64)
    {
        assert(ldexp(1, -16384) == 0x1p-16384L);
        assert(ldexp(1, -16382) == 0x1p-16382L);
    }
    else static if (real.mant_dig == 53)
    {
        assert(ldexp(1,  1023) == 0x1p1023L);
        assert(ldexp(1, -1022) == 0x1p-1022L);
        assert(ldexp(1, -1021) == 0x1p-1021L);
    }
    else
        assert(false, "Only 128bit, 80bit and 64bit reals expected here");
}

/*******************************
 * Returns |x|
 *
 *      $(TABLE_SV
 *      $(TR $(TH x)                 $(TH fabs(x)))
 *      $(TR $(TD $(PLUSMN)0.0)      $(TD +0.0) )
 *      $(TR $(TD $(PLUSMN)$(INFIN)) $(TD +$(INFIN)) )
 *      )
 */
version (LDC)
    alias fabs = llvm_fabs!real;
else
real fabs(real x) @safe pure nothrow;      /* intrinsic */

/**********************************
 * Rounds x to the nearest integer value, using the current rounding
 * mode.
 * If the return value is not equal to x, the FE_INEXACT
 * exception is raised.
 * $(B nearbyint) performs
 * the same operation, but does not set the FE_INEXACT exception.
 */
version (LDC)
    alias rint = llvm_rint!real;
else
real rint(real x) @safe pure nothrow;      /* intrinsic */

/***********************************
 * Building block functions, they
 * translate to a single x87 instruction.
 */

version (LDC)
{
    version (X86)    version = X86_Any;
    version (X86_64) version = X86_Any;

    version (X86_Any)
    {
        static if (real.mant_dig == 64)
        {
            // y * log2(x)
            real yl2x(real x, real y)   @safe pure nothrow
            {
                return __asm_trusted!real("fyl2x", "={st},{st(1)},{st},~{st(1)}", y, x);
            }

            // y * log2(x + 1)
            real yl2xp1(real x, real y) @safe pure nothrow
            {
                return __asm_trusted!real("fyl2xp1", "={st},{st(1)},{st},~{st(1)}", y, x);
            }
        }
    }
}
else
{
real yl2x(real x, real y)   @safe pure nothrow;       // y * log2(x)
real yl2xp1(real x, real y) @safe pure nothrow;       // y * log2(x + 1)
}

unittest
{
    version (INLINE_YL2X)
    {
        assert(yl2x(1024, 1) == 10);
        assert(yl2xp1(1023, 1) == 10);
    }
}

/*************************************
 * Round argument to a specific precision.
 *
 * D language types specify a minimum precision, not a maximum. The
 * `toPrec()` function forces rounding of the argument `f` to the
 * precision of the specified floating point type `T`.
 *
 * Params:
 *      T = precision type to round to
 *      f = value to convert
 * Returns:
 *      f in precision of type `T`
 */
@safe pure nothrow
T toPrec(T:float)(float f) { pragma(inline, false); return f; }
/// ditto
@safe pure nothrow
T toPrec(T:float)(double f) { pragma(inline, false); return cast(T) f; }
/// ditto
@safe pure nothrow
T toPrec(T:float)(real f)  { pragma(inline, false); return cast(T) f; }
/// ditto
@safe pure nothrow
T toPrec(T:double)(float f) { pragma(inline, false); return f; }
/// ditto
@safe pure nothrow
T toPrec(T:double)(double f) { pragma(inline, false); return f; }
/// ditto
@safe pure nothrow
T toPrec(T:double)(real f)  { pragma(inline, false); return cast(T) f; }
/// ditto
@safe pure nothrow
T toPrec(T:real)(float f) { pragma(inline, false); return f; }
/// ditto
@safe pure nothrow
T toPrec(T:real)(double f) { pragma(inline, false); return f; }
/// ditto
@safe pure nothrow
T toPrec(T:real)(real f)  { pragma(inline, false); return f; }

@safe unittest
{
    static float f = 1.1f;
    static double d = 1.1;
    static real r = 1.1L;
    f = toPrec!float(f + f);
    f = toPrec!float(d + d);
    f = toPrec!float(r + r);
    d = toPrec!double(f + f);
    d = toPrec!double(d + d);
    d = toPrec!double(r + r);
    r = toPrec!real(f + f);
    r = toPrec!real(d + d);
    r = toPrec!real(r + r);

    /+ Uncomment these once compiler support has been added.
    enum real PIR = 0xc.90fdaa22168c235p-2;
    enum double PID = 0x1.921fb54442d18p+1;
    enum float PIF = 0x1.921fb6p+1;

    assert(toPrec!float(PIR) == PIF);
    assert(toPrec!double(PIR) == PID);
    assert(toPrec!real(PIR) == PIR);
    assert(toPrec!float(PID) == PIF);
    assert(toPrec!double(PID) == PID);
    assert(toPrec!real(PID) == PID);
    assert(toPrec!float(PIF) == PIF);
    assert(toPrec!double(PIF) == PIF);
    assert(toPrec!real(PIF) == PIF);
    +/
}
