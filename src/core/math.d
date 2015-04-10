// Written in the D programming language.

/**
 * Builtin mathematical intrinsics
 *
 * Source: $(DRUNTIMESRC core/_math.d)
 * Macros:
 *      TABLE_SV = <table border=1 cellpadding=4 cellspacing=0>
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
 * License:   $(WEB www.boost.org/LICENSE_1_0.txt, Boost License 1.0).
 * Authors:   $(WEB digitalmars.com, Walter Bright),
 *                        Don Clugston
 */
module core.math;

version (LDC)
{
    import stdc = core.stdc.math;
    import ldc.intrinsics;
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
    real cos(real x) @safe pure nothrow { return llvm_cos(x); }
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
    real sin(real x) @safe pure nothrow { return llvm_sin(x); }
else
real sin(real x) @safe pure nothrow;       /* intrinsic */

/*****************************************
 * Returns x rounded to a long value using the current rounding mode.
 * If the integer value of x is
 * greater than long.max, the result is
 * indeterminate.
 */
version (LDC)
{
    long rndtol(real x) @safe pure nothrow { return stdc.llroundl(x); }
}
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
    extern(D) float  sqrt(float  x) { return llvm_sqrt(x); }
    extern(D) double sqrt(double x) { return llvm_sqrt(x); }
    extern(D) real   sqrt(real   x) { return llvm_sqrt(x); }
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
    real ldexp(real n, int exp) @trusted pure nothrow
    {
        version (MinGW)
        {
            // The MinGW runtime only provides a double precision ldexp, and
            // it doesn't seem to reliably possible to express the fscale
            // semantics (two FP stack inputs/returns) in an inline asm
            // expression clobber list.
            version (D_InlineAsm_X86_64)
            {
                asm
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
                asm
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
        else
        {
            return stdc.ldexpl(n, exp);
        }
    }
}
else
real ldexp(real n, int exp) @safe pure nothrow;    /* intrinsic */

unittest {
    static if (real.mant_dig == 64)
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
        assert(false, "Only 80bit and 64bit reals expected here");
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
{
    static if (__traits(compiles, llvm_abs(3.14L)))
        real fabs(real x) @safe pure nothrow { return llvm_fabs(x); }
    else
        real fabs(real x) @safe pure nothrow { return stdc.fabsl(x); }
}
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
{
  static if (__traits(compiles, llvm_rint(3.14L)))
    real rint(real x) @safe pure nothrow { return llvm_rint(x); }
  else
    real rint(real x) @safe pure nothrow { return stdc.rintl(x); }
}
else
real rint(real x) @safe pure nothrow;      /* intrinsic */

/***********************************
 * Building block functions, they
 * translate to a single x87 instruction.
 */

version (LDC) {}
else
{
real yl2x(real x, real y)   @safe pure nothrow;       // y * log2(x)
real yl2xp1(real x, real y) @safe pure nothrow;       // y * log2(x + 1)

unittest
{
    version (INLINE_YL2X)
    {
        assert(yl2x(1024, 1) == 10);
        assert(yl2xp1(1023, 1) == 10);
    }
}
}
