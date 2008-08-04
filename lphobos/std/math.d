// Written in the D programming language

/**
 * Macros:
 *	WIKI = Phobos/StdMath
 *
 *	TABLE_SV = <table border=1 cellpadding=4 cellspacing=0>
 *		<caption>Special Values</caption>
 *		$0</table>
 *	SVH = $(TR $(TH $1) $(TH $2))
 *	SV  = $(TR $(TD $1) $(TD $2))
 *
 *	NAN = $(RED NAN)
 *	SUP = <span style="vertical-align:super;font-size:smaller">$0</span>
 *	GAMMA =  &#915;
 *	INTEGRAL = &#8747;
 *	INTEGRATE = $(BIG &#8747;<sub>$(SMALL $1)</sub><sup>$2</sup>)
 *	POWER = $1<sup>$2</sup>
 *	BIGSUM = $(BIG &Sigma; <sup>$2</sup><sub>$(SMALL $1)</sub>)
 *	CHOOSE = $(BIG &#40;) <sup>$(SMALL $1)</sup><sub>$(SMALL $2)</sub> $(BIG &#41;)
 *	PLUSMN = &plusmn;
 *	INFIN = &infin;
 *	PI = &pi;
 *	LT = &lt;
 *	GT = &gt;
 */

/*
 * Author:
 *	Walter Bright
 * Copyright:
 *	Copyright (c) 2001-2005 by Digital Mars,
 *	All Rights Reserved,
 *	www.digitalmars.com
 *  Copyright (c) 2007 by Tomas Lindquist Olsen
 * License:
 *  This software is provided 'as-is', without any express or implied
 *  warranty. In no event will the authors be held liable for any damages
 *  arising from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute it
 *  freely, subject to the following restrictions:
 *
 *  <ul>
 *  <li> The origin of this software must not be misrepresented; you must not
 *       claim that you wrote the original software. If you use this software
 *       in a product, an acknowledgment in the product documentation would be
 *       appreciated but is not required.
 *  </li>
 *  <li> Altered source versions must be plainly marked as such, and must not
 *       be misrepresented as being the original software.
 *  </li>
 *  <li> This notice may not be removed or altered from any source
 *       distribution.
 *  </li>
 *  </ul>
 */


module std.math;

//debug=math;		// uncomment to turn on debugging printf's

private import std.c.stdio;
private import std.c.math;

class NotImplemented : Error
{
    this(string msg)
    {
	super(msg ~ "not implemented");
    }
}

const real E =		2.7182818284590452354L;  /** e */
const real LOG2T =	0x1.a934f0979a3715fcp+1; /** log<sub>2</sub>10 */ // 3.32193 fldl2t
const real LOG2E =	0x1.71547652b82fe178p+0; /** log<sub>2</sub>e */ // 1.4427 fldl2e
const real LOG2 =	0x1.34413509f79fef32p-2; /** log<sub>10</sub>2 */ // 0.30103 fldlg2
const real LOG10E =	0.43429448190325182765;  /** log<sub>10</sub>e */
const real LN2 =	0x1.62e42fefa39ef358p-1; /** ln 2 */	// 0.693147 fldln2
const real LN10 =	2.30258509299404568402;  /** ln 10 */
const real PI =		0x1.921fb54442d1846ap+1; /** $(PI) */ // 3.14159 fldpi
const real PI_2 =	1.57079632679489661923;  /** $(PI) / 2 */
const real PI_4 =	0.78539816339744830962;  /** $(PI) / 4 */
const real M_1_PI =	0.31830988618379067154;  /** 1 / $(PI) */
const real M_2_PI =	0.63661977236758134308;  /** 2 / $(PI) */
const real M_2_SQRTPI =	1.12837916709551257390;  /** 2 / &radic;$(PI) */
const real SQRT2 =	1.41421356237309504880;  /** &radic;2 */
const real SQRT1_2 =	0.70710678118654752440;  /** &radic;&frac12; */

/*
	Octal versions:
	PI/64800	0.00001 45530 36176 77347 02143 15351 61441 26767
	PI/180		0.01073 72152 11224 72344 25603 54276 63351 22056
	PI/8		0.31103 75524 21026 43021 51423 06305 05600 67016
	SQRT(1/PI)	0.44067 27240 41233 33210 65616 51051 77327 77303
	2/PI		0.50574 60333 44710 40522 47741 16537 21752 32335
	PI/4		0.62207 73250 42055 06043 23046 14612 13401 56034
	SQRT(2/PI)	0.63041 05147 52066 24106 41762 63612 00272 56161

	PI		3.11037 55242 10264 30215 14230 63050 56006 70163
	LOG2		0.23210 11520 47674 77674 61076 11263 26013 37111
 */


/***********************************
 * Calculates the absolute value
 *
 * For complex numbers, abs(z) = sqrt( $(POWER z.re, 2) + $(POWER z.im, 2) )
 * = hypot(z.re, z.im).
 */
real abs(real x)
{
    return fabs(x);
}

/** ditto */
long abs(long x)
{
    return x>=0 ? x : -x;
}

/** ditto */
int abs(int x)
{
    return x>=0 ? x : -x;
}

/** ditto */
real abs(creal z)
{
    return hypot(z.re, z.im);
}

/** ditto */
real abs(ireal y)
{
    return fabs(y.im);
}


unittest
{
    assert(isPosZero(abs(-0.0L)));
    assert(isnan(abs(real.nan)));
    assert(abs(-real.infinity) == real.infinity);
    assert(abs(-3.2Li) == 3.2L);
    assert(abs(71.6Li) == 71.6L);
    assert(abs(-56) == 56);
    assert(abs(2321312L)  == 2321312L);
    assert(abs(-1+1i) == sqrt(2.0));
}

/***********************************
 * Complex conjugate
 *
 *  conj(x + iy) = x - iy
 *
 * Note that z * conj(z) = $(POWER z.re, 2) - $(POWER z.im, 2)
 * is always a real number
 */
creal conj(creal z)
{
    return z.re - z.im*1i;
}

/** ditto */
ireal conj(ireal y)
{
    return -y;
}

unittest
{
    assert(conj(7 + 3i) == 7-3i);
    ireal z = -3.2Li;
    assert(conj(z) == -z);
}

/***********************************
 * Returns cosine of x. x is in radians.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)                 $(TH cos(x)) $(TH invalid?))
 *	$(TR $(TD $(NAN))            $(TD $(NAN)) $(TD yes)	)
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD $(NAN)) $(TD yes)	)
 *	)
 * Bugs:
 *	Results are undefined if |x| >= $(POWER 2,64).
 */

pragma(intrinsic, "llvm.cos.f32")
float cos(float x);

pragma(intrinsic, "llvm.cos.f64")
double cos(double x); // ditto

pragma(intrinsic, "llvm.cos.f80")
real cos(real x); /// ditto


/***********************************
 * Returns sine of x. x is in radians.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)               $(TH sin(x))      $(TH invalid?))
 *	$(TR $(TD $(NAN))          $(TD $(NAN))      $(TD yes))
 *	$(TR $(TD $(PLUSMN)0.0)    $(TD $(PLUSMN)0.0) $(TD no))
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD $(NAN))      $(TD yes))
 *	)
 * Bugs:
 *	Results are undefined if |x| >= $(POWER 2,64).
 */

pragma(intrinsic, "llvm.sin.f32")
float sin(float x);

pragma(intrinsic, "llvm.sin.f64")
double sin(double x); // ditto

pragma(intrinsic, "llvm.sin.f80")
real sin(real x); /// ditto


/****************************************************************************
 * Returns tangent of x. x is in radians.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)               $(TH tan(x))       $(TH invalid?))
 *	$(TR $(TD $(NAN))          $(TD $(NAN))       $(TD yes))
 *	$(TR $(TD $(PLUSMN)0.0)    $(TD $(PLUSMN)0.0) $(TD no))
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD $(NAN))     $(TD yes))
 *	)
 */

version(D_InlineAsm_X86)
real tan(real x)
{

    asm
    {
	fld	x[EBP]			; // load theta
	fxam				; // test for oddball values
	fstsw	AX			;
	sahf				;
	jc	trigerr			; // x is NAN, infinity, or empty
					  // 387's can handle denormals
SC18:	fptan				;
	fstp	ST(0)			; // dump X, which is always 1
	fstsw	AX			;
	sahf				;
	jnp	Lret			; // C2 = 1 (x is out of range)

	// Do argument reduction to bring x into range
	fldpi				;
	fxch				;
SC17:	fprem1				;
	fstsw	AX			;
	sahf				;
	jp	SC17			;
	fstp	ST(1)			; // remove pi from stack
	jmp	SC18			;

trigerr:
	jnp	Lret			; // if theta is NAN, return theta
	fstp	ST(0)			; // dump theta
    }
    return real.nan;

Lret:
    ;
}
else
{
real tan(real x) { return std.c.math.atan(x); }
}


unittest
{
    static real vals[][2] =	// angle,tan
    [
	    [   0,   0],
	    [   .5,  .5463024898],
	    [   1,   1.557407725],
	    [   1.5, 14.10141995],
	    [   2,  -2.185039863],
	    [   2.5,-.7470222972],
	    [   3,  -.1425465431],
	    [   3.5, .3745856402],
	    [   4,   1.157821282],
	    [   4.5, 4.637332055],
	    [   5,  -3.380515006],
	    [   5.5,-.9955840522],
	    [   6,  -.2910061914],
	    [   6.5, .2202772003],
	    [   10,  .6483608275],

	    // special angles
	    [   PI_4,	1],
	    //[	PI_2,	real.infinity],
	    [   3*PI_4,	-1],
	    [   PI,	0],
	    [   5*PI_4,	1],
	    //[	3*PI_2,	-real.infinity],
	    [   7*PI_4,	-1],
	    [   2*PI,	0],

	    // overflow
	    [   real.infinity,	real.nan],
	    [   real.nan,	real.nan],
	    //[   1e+100,	real.nan],
    ];
    int i;

    for (i = 0; i < vals.length; i++)
    {
	real x = vals[i][0];
	real r = vals[i][1];
	real t = tan(x);

	//printf("tan(%Lg) = %Lg, should be %Lg\n", x, t, r);
	assert(mfeq(r, t, .0000001));

	x = -x;
	r = -r;
	t = tan(x);
	//printf("tan(%Lg) = %Lg, should be %Lg\n", x, t, r);
	assert(mfeq(r, t, .0000001));
    }
}

/***************
 * Calculates the arc cosine of x,
 * returning a value ranging from -$(PI)/2 to $(PI)/2.
 *
 *	$(TABLE_SV
 *      $(TR $(TH x)         $(TH acos(x)) $(TH invalid?))
 *      $(TR $(TD $(GT)1.0)  $(TD $(NAN))  $(TD yes))
 *      $(TR $(TD $(LT)-1.0) $(TD $(NAN))  $(TD yes))
 *      $(TR $(TD $(NAN))    $(TD $(NAN))  $(TD yes))
 *      )
 */
real acos(real x)		{ return std.c.math.acos(x); }

/***************
 * Calculates the arc sine of x,
 * returning a value ranging from -$(PI)/2 to $(PI)/2.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)            $(TH asin(x))      $(TH invalid?))
 *	$(TR $(TD $(PLUSMN)0.0) $(TD $(PLUSMN)0.0) $(TD no))
 *	$(TR $(TD $(GT)1.0)     $(TD $(NAN))       $(TD yes))
 *	$(TR $(TD $(LT)-1.0)    $(TD $(NAN))       $(TD yes))
 *       )
 */
real asin(real x)		{ return std.c.math.asin(x); }

/***************
 * Calculates the arc tangent of x,
 * returning a value ranging from -$(PI)/2 to $(PI)/2.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)                 $(TH atan(x))      $(TH invalid?))
 *	$(TR $(TD $(PLUSMN)0.0)      $(TD $(PLUSMN)0.0) $(TD no))
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD $(NAN))       $(TD yes))
 *       )
 */
real atan(real x)		{ return std.c.math.atan(x); }

/***************
 * Calculates the arc tangent of y / x,
 * returning a value ranging from -$(PI)/2 to $(PI)/2.
 *
 *      $(TABLE_SV
 *      $(TR $(TH y)                 $(TH x)            $(TH atan(y, x)))
 *      $(TR $(TD $(NAN))            $(TD anything)     $(TD $(NAN)) )
 *      $(TR $(TD anything)          $(TD $(NAN))       $(TD $(NAN)) )
 *      $(TR $(TD $(PLUSMN)0.0)      $(TD $(GT)0.0)     $(TD $(PLUSMN)0.0) )
 *      $(TR $(TD $(PLUSMN)0.0)      $(TD $(PLUSMN)0.0) $(TD $(PLUSMN)0.0) )
 *      $(TR $(TD $(PLUSMN)0.0)      $(TD $(LT)0.0)     $(TD $(PLUSMN)$(PI)))
 *      $(TR $(TD $(PLUSMN)0.0)      $(TD -0.0)         $(TD $(PLUSMN)$(PI)))
 *      $(TR $(TD $(GT)0.0)          $(TD $(PLUSMN)0.0) $(TD $(PI)/2) )
 *      $(TR $(TD $(LT)0.0)          $(TD $(PLUSMN)0.0) $(TD $(PI)/2))
 *      $(TR $(TD $(GT)0.0)          $(TD $(INFIN))     $(TD $(PLUSMN)0.0) )
 *      $(TR $(TD $(PLUSMN)$(INFIN)) $(TD anything)     $(TD $(PLUSMN)$(PI)/2))
 *      $(TR $(TD $(GT)0.0)          $(TD -$(INFIN))    $(TD $(PLUSMN)$(PI)) )
 *      $(TR $(TD $(PLUSMN)$(INFIN)) $(TD $(INFIN))     $(TD $(PLUSMN)$(PI)/4))
 *      $(TR $(TD $(PLUSMN)$(INFIN)) $(TD -$(INFIN))    $(TD $(PLUSMN)3$(PI)/4))
 *      )
 */
real atan2(real y, real x)      { return std.c.math.atan2(y,x); }

/***********************************
 * Calculates the hyperbolic cosine of x.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)                 $(TH cosh(x))      $(TH invalid?))
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD $(PLUSMN)0.0) $(TD no) )
 *      )
 */
real cosh(real x)		{ return std.c.math.cosh(x); }

/***********************************
 * Calculates the hyperbolic sine of x.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)                 $(TH sinh(x))           $(TH invalid?))
 *	$(TR $(TD $(PLUSMN)0.0)      $(TD $(PLUSMN)0.0)      $(TD no))
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD $(PLUSMN)$(INFIN)) $(TD no))
 *      )
 */
real sinh(real x)		{ return std.c.math.sinh(x); }

/***********************************
 * Calculates the hyperbolic tangent of x.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)                 $(TH tanh(x))      $(TH invalid?))
 *	$(TR $(TD $(PLUSMN)0.0)	     $(TD $(PLUSMN)0.0) $(TD no) )
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD $(PLUSMN)1.0) $(TD no))
 *      )
 */
real tanh(real x)		{ return std.c.math.tanh(x); }

//real acosh(real x)		{ return std.c.math.acoshl(x); }
//real asinh(real x)		{ return std.c.math.asinhl(x); }
//real atanh(real x)		{ return std.c.math.atanhl(x); }

/***********************************
 * Calculates the inverse hyperbolic cosine of x.
 *
 *  Mathematically, acosh(x) = log(x + sqrt( x*x - 1))
 *
 * $(TABLE_DOMRG
 *  $(DOMAIN 1..$(INFIN))
 *  $(RANGE  1..log(real.max), $(INFIN)) )
 *	$(TABLE_SV
 *    $(SVH  x,     acosh(x) )
 *    $(SV  $(NAN), $(NAN) )
 *    $(SV  <1,     $(NAN) )
 *    $(SV  1,      0       )
 *    $(SV  +$(INFIN),+$(INFIN))
 *  )
 */   
real acosh(real x)
{
    if (x > 1/real.epsilon)
	return LN2 + log(x);
    else
	return log(x + sqrt(x*x - 1));
}

unittest
{
    assert(isnan(acosh(0.9)));
    assert(isnan(acosh(real.nan)));
    assert(acosh(1)==0.0);
    assert(acosh(real.infinity) == real.infinity);
}

/***********************************
 * Calculates the inverse hyperbolic sine of x.
 *
 *  Mathematically,
 *  ---------------
 *  asinh(x) =  log( x + sqrt( x*x + 1 )) // if x >= +0
 *  asinh(x) = -log(-x + sqrt( x*x + 1 )) // if x <= -0
 *  -------------
 *
 *	$(TABLE_SV
 *    $(SVH x,                asinh(x)       )
 *    $(SV  $(NAN),           $(NAN)         )
 *    $(SV  $(PLUSMN)0,       $(PLUSMN)0      )
 *    $(SV  $(PLUSMN)$(INFIN),$(PLUSMN)$(INFIN))
 *  )
 */
real asinh(real x)
{   
    if (fabs(x) > 1 / real.epsilon)   // beyond this point, x*x + 1 == x*x
	return copysign(LN2 + log(fabs(x)), x);
    else
    {
	// sqrt(x*x + 1) ==  1 + x * x / ( 1 + sqrt(x*x + 1) )
	return copysign(log1p(fabs(x) + x*x / (1 + sqrt(x*x + 1)) ), x);
    }
}

unittest
{
    assert(isPosZero(asinh(0.0)));
    assert(isNegZero(asinh(-0.0)));
    assert(asinh(real.infinity) == real.infinity);
    assert(asinh(-real.infinity) == -real.infinity);
    assert(isnan(asinh(real.nan)));
}

/***********************************
 * Calculates the inverse hyperbolic tangent of x,
 * returning a value from ranging from -1 to 1.
 *  
 * Mathematically, atanh(x) = log( (1+x)/(1-x) ) / 2
 *  
 *
 * $(TABLE_DOMRG
 *  $(DOMAIN -$(INFIN)..$(INFIN))
 *  $(RANGE  -1..1) )
 *	$(TABLE_SV
 *    $(SVH  x,     acosh(x) )
 *    $(SV  $(NAN), $(NAN) )
 *    $(SV  $(PLUSMN)0, $(PLUSMN)0)
 *    $(SV  -$(INFIN), -0)
 *  )
 */   
real atanh(real x)
{
    // log( (1+x)/(1-x) ) == log ( 1 + (2*x)/(1-x) )
    return  0.5 * log1p( 2 * x / (1 - x) );
}

unittest
{
    assert(isPosZero(atanh(0.0)));
    assert(isNegZero(atanh(-0.0)));
    assert(isnan(atanh(real.nan)));
    assert(isnan(atanh(-real.infinity))); 
}

/*****************************************
 * Returns x rounded to a long value using the current rounding mode.
 * If the integer value of x is
 * greater than long.max, the result is
 * indeterminate.
 */
long rndtol(real x);	/* intrinsic */


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
 *	$(TABLE_SV
 *	$(TR $(TH x)         $(TH sqrt(x))   $(TH invalid?))
 *	$(TR $(TD -0.0)      $(TD -0.0)      $(TD no))
 *	$(TR $(TD $(LT)0.0)  $(TD $(NAN))    $(TD yes))
 *	$(TR $(TD +$(INFIN)) $(TD +$(INFIN)) $(TD no))
 *	)
 */

pragma(intrinsic, "llvm.sqrt.f32")
float sqrt(float x);	/* intrinsic */

pragma(intrinsic, "llvm.sqrt.f64")
double sqrt(double x);	/* intrinsic */	/// ditto

pragma(intrinsic, "llvm.sqrt.f80")
real sqrt(real x);	/* intrinsic */ /// ditto

creal sqrt(creal z)
{
    creal c;
    real x,y,w,r;

    if (z == 0)
    {
	c = 0 + 0i;
    }
    else
    {	real z_re = z.re;
	real z_im = z.im;

	x = fabs(z_re);
	y = fabs(z_im);
	if (x >= y)
	{
	    r = y / x;
	    w = sqrt(x) * sqrt(0.5 * (1 + sqrt(1 + r * r)));
	}
	else
	{
	    r = x / y;
	    w = sqrt(y) * sqrt(0.5 * (r + sqrt(1 + r * r)));
	}

	if (z_re >= 0)
	{
	    c = w + (z_im / (w + w)) * 1.0i;
	}
	else
	{
	    if (z_im < 0)
		w = -w;
	    c = z_im / (w + w) + w * 1.0i;
	}
    }
    return c;
}

/**********************
 * Calculates e$(SUP x).
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)         $(TH exp(x)))
 *	$(TR $(TD +$(INFIN)) $(TD +$(INFIN)) )
 *	$(TR $(TD -$(INFIN)) $(TD +0.0) )
 *	)
 */
real exp(real x)		{ return std.c.math.exp(x); }

/**********************
 * Calculates 2$(SUP x).
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)         $(TH exp2(x)))
 *	$(TR $(TD +$(INFIN)) $(TD +$(INFIN)))
 *	$(TR $(TD -$(INFIN)) $(TD +0.0))
 *	)
 */
real exp2(real x)		{ return std.c.math.exp2(x); }

/******************************************
 * Calculates the value of the natural logarithm base (e)
 * raised to the power of x, minus 1.
 *
 * For very small x, expm1(x) is more accurate 
 * than exp(x)-1. 
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)            $(TH e$(SUP x)-1))
 *	$(TR $(TD $(PLUSMN)0.0) $(TD $(PLUSMN)0.0))
 *	$(TR $(TD +$(INFIN))    $(TD +$(INFIN)))
 *	$(TR $(TD -$(INFIN))    $(TD -1.0))
 *	)
 */

real expm1(real x)		{ return std.c.math.expm1(x); }


/*********************************************************************
 * Separate floating point value into significand and exponent.
 *
 * Returns:
 *	Calculate and return <i>x</i> and exp such that
 *	value =<i>x</i>*2$(SUP exp) and
 *	.5 $(LT)= |<i>x</i>| $(LT) 1.0<br>
 *	<i>x</i> has same sign as value.
 *
 *	$(TABLE_SV
 *	$(TR $(TH value)           $(TH returns)         $(TH exp))
 *	$(TR $(TD $(PLUSMN)0.0)    $(TD $(PLUSMN)0.0)    $(TD 0))
 *	$(TR $(TD +$(INFIN))       $(TD +$(INFIN))       $(TD int.max))
 *	$(TR $(TD -$(INFIN))       $(TD -$(INFIN))       $(TD int.min))
 *	$(TR $(TD $(PLUSMN)$(NAN)) $(TD $(PLUSMN)$(NAN)) $(TD int.min))
 *	)
 */


real frexp(real value, out int exp)
{
    ushort* vu = cast(ushort*)&value;
    long* vl = cast(long*)&value;
    uint ex;

    // If exponent is non-zero
    ex = vu[4] & 0x7FFF;
    if (ex)
    {
	if (ex == 0x7FFF)
	{   // infinity or NaN
	    if (*vl &  0x7FFFFFFFFFFFFFFF)	// if NaN
	    {	*vl |= 0xC000000000000000;	// convert $(NAN)S to $(NAN)Q
		exp = int.min;
	    }
	    else if (vu[4] & 0x8000)
	    {	// negative infinity
		exp = int.min;
	    }
	    else
	    {	// positive infinity
		exp = int.max;
	    }
	}
	else
	{
	    exp = ex - 0x3FFE;
	    vu[4] = cast(ushort)((0x8000 & vu[4]) | 0x3FFE);
	}
    }
    else if (!*vl)
    {
	// value is +-0.0
	exp = 0;
    }
    else
    {	// denormal
	int i = -0x3FFD;

	do
	{
	    i--;
	    *vl <<= 1;
	} while (*vl > 0);
	exp = i;
        vu[4] = cast(ushort)((0x8000 & vu[4]) | 0x3FFE);
    }
    return value;
}


unittest
{
    static real vals[][3] =	// x,frexp,exp
    [
	[0.0,	0.0,	0],
	[-0.0,	-0.0,	0],
	[1.0,	.5,	1],
	[-1.0,	-.5,	1],
	[2.0,	.5,	2],
	[155.67e20,	0x1.A5F1C2EB3FE4Fp-1,	74],	// normal
	[1.0e-320,	0.98829225,		-1063],
	[real.min,	.5,		-16381],
	[real.min/2.0L,	.5,		-16382],	// denormal

	[real.infinity,real.infinity,int.max],
	[-real.infinity,-real.infinity,int.min],
	[real.nan,real.nan,int.min],
	[-real.nan,-real.nan,int.min],

	// Don't really support signalling nan's in D
	//[real.nans,real.nan,int.min],
	//[-real.nans,-real.nan,int.min],
    ];
    int i;

    for (i = 0; i < vals.length; i++)
    {
	real x = vals[i][0];
	real e = vals[i][1];
	int exp = cast(int)vals[i][2];
	int eptr;
	real v = frexp(x, eptr);

	//printf("frexp(%Lg) = %.8Lg, should be %.8Lg, eptr = %d, should be %d\n", x, v, e, eptr, exp);
	assert(mfeq(e, v, .0000001));
	assert(exp == eptr);
    }
}


/******************************************
 * Extracts the exponent of x as a signed integral value.
 *
 * If x is not a special value, the result is the same as
 * <tt>cast(int)logb(x)</tt>.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)                $(TH ilogb(x))     $(TH Range error?))
 *	$(TR $(TD 0)                 $(TD FP_ILOGB0)   $(TD yes))
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD int.max)     $(TD no))
 *	$(TR $(TD $(NAN))            $(TD FP_ILOGBNAN) $(TD no))
 *	)
 */
int ilogb(real x)		{ return std.c.math.ilogb(x); }

alias std.c.math.FP_ILOGB0   FP_ILOGB0;
alias std.c.math.FP_ILOGBNAN FP_ILOGBNAN;


/*******************************************
 * Compute n * 2$(SUP exp)
 * References: frexp
 */

real ldexp(real n, int exp)	{ return std.c.math.ldexp(n, exp); }

/**************************************
 * Calculate the natural logarithm of x.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)            $(TH log(x))    $(TH divide by 0?) $(TH invalid?))
 *	$(TR $(TD $(PLUSMN)0.0) $(TD -$(INFIN)) $(TD yes)          $(TD no))
 *	$(TR $(TD $(LT)0.0)     $(TD $(NAN))    $(TD no)           $(TD yes))
 *	$(TR $(TD +$(INFIN))    $(TD +$(INFIN)) $(TD no)           $(TD no))
 *	)
 */

real log(real x)		{ return std.c.math.log(x); }

/**************************************
 * Calculate the base-10 logarithm of x.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)            $(TH log10(x))  $(TH divide by 0?) $(TH invalid?))
 *	$(TR $(TD $(PLUSMN)0.0) $(TD -$(INFIN)) $(TD yes)          $(TD no))
 *	$(TR $(TD $(LT)0.0)     $(TD $(NAN))    $(TD no)           $(TD yes))
 *	$(TR $(TD +$(INFIN))    $(TD +$(INFIN)) $(TD no)           $(TD no))
 *	)
 */

real log10(real x)		{ return std.c.math.log10(x); }

/******************************************
 *	Calculates the natural logarithm of 1 + x.
 *
 *	For very small x, log1p(x) will be more accurate than 
 *	log(1 + x). 
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)            $(TH log1p(x))     $(TH divide by 0?) $(TH invalid?))
 *	$(TR $(TD $(PLUSMN)0.0) $(TD $(PLUSMN)0.0) $(TD no)           $(TD no))
 *	$(TR $(TD -1.0)         $(TD -$(INFIN))    $(TD yes)          $(TD no))
 *	$(TR $(TD $(LT)-1.0)    $(TD $(NAN))       $(TD no)           $(TD yes))
 *	$(TR $(TD +$(INFIN))    $(TD -$(INFIN))    $(TD no)           $(TD no))
 *	)
 */

real log1p(real x)		{ return std.c.math.log1p(x); }

/***************************************
 * Calculates the base-2 logarithm of x:
 * log<sub>2</sub>x
 *
 *	$(TABLE_SV
 *	$(TR $(TH x) 	        $(TH log2(x))   $(TH divide by 0?) $(TH invalid?))
 *	$(TR $(TD $(PLUSMN)0.0) $(TD -$(INFIN)) $(TD yes)          $(TD no) )
 *	$(TR $(TD $(LT)0.0)     $(TD $(NAN))    $(TD no)           $(TD yes) )
 *	$(TR $(TD +$(INFIN))    $(TD +$(INFIN)) $(TD no)           $(TD no) )
 *	)
 */
real log2(real x)		{ return std.c.math.log2(x); }

/*****************************************
 * Extracts the exponent of x as a signed integral value.
 *
 * If x is subnormal, it is treated as if it were normalized.
 * For a positive, finite x: 
 *
 * -----
 * 1 <= $(I x) * FLT_RADIX$(SUP -logb(x)) < FLT_RADIX 
 * -----
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)                 $(TH logb(x))   $(TH divide by 0?) )
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD +$(INFIN)) $(TD no))
 *	$(TR $(TD $(PLUSMN)0.0)      $(TD -$(INFIN)) $(TD yes) )
 *	)
 */
real logb(real x)		{ return std.c.math.logb(x); }

/************************************
 * Calculates the remainder from the calculation x/y.
 * Returns:
 * The value of x - i * y, where i is the number of times that y can 
 * be completely subtracted from x. The result has the same sign as x. 
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)                   $(TH y)                 $(TH modf(x, y))   $(TH invalid?))
 *	$(TR $(TD $(PLUSMN)0.0)        $(TD no)t 0.0           $(TD $(PLUSMN)0.0) $(TD no))
 *	$(TR $(TD $(PLUSMN)$(INFIN))   $(TD anything)          $(TD $(NAN))       $(TD yes))
 *	$(TR $(TD anything)            $(TD $(PLUSMN)0.0)      $(TD $(NAN))       $(TD yes))
 *	$(TR $(TD !=$(PLUSMN)$(INFIN)) $(TD $(PLUSMN)$(INFIN)) $(TD x)            $(TD no))
 *	)
 */
real modf(real x, inout real y)
{
double Y = y;
auto tmp = std.c.math.modf(x,&Y);
y = Y;
return tmp;
}

/*************************************
 * Efficiently calculates x * 2$(SUP n).
 *
 * scalbn handles underflow and overflow in 
 * the same fashion as the basic arithmetic operators. 
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)                 $(TH scalb(x)))
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD $(PLUSMN)$(INFIN)) )
 *	$(TR $(TD $(PLUSMN)0.0)      $(TD $(PLUSMN)0.0) )
 *	)
 */
real scalbn(real x, int n)
{
    version (linux)
	return std.c.math.scalbn(x, n);
    else
	throw new NotImplemented("scalbn");
}

/***************
 * Calculates the cube root x.
 *
 *	$(TABLE_SV
 *	$(TR $(TH $(I x))	     $(TH cbrt(x))           $(TH invalid?))
 *	$(TR $(TD $(PLUSMN)0.0)	     $(TD $(PLUSMN)0.0)      $(TD no) )
 *	$(TR $(TD $(NAN))	     $(TD $(NAN))            $(TD yes) )
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD $(PLUSMN)$(INFIN)) $(TD no) )
 *	)
 */
real cbrt(real x)		{ return std.c.math.cbrt(x); }


/*******************************
 * Returns |x|
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)                 $(TH fabs(x)))
 *	$(TR $(TD $(PLUSMN)0.0)      $(TD +0.0) )
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD +$(INFIN)) )
 *	)
 */
real fabs(real x)	{ return std.c.math.fabs(x); }


/***********************************************************************
 * Calculates the length of the 
 * hypotenuse of a right-angled triangle with sides of length x and y. 
 * The hypotenuse is the value of the square root of 
 * the sums of the squares of x and y:
 *
 *	sqrt(x&sup2; + y&sup2;)
 *
 * Note that hypot(x, y), hypot(y, x) and
 * hypot(x, -y) are equivalent.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)                 $(TH y)            $(TH hypot(x, y)) $(TH invalid?))
 *	$(TR $(TD x)                 $(TD $(PLUSMN)0.0) $(TD |x|)         $(TD no))
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD y)            $(TD +$(INFIN))   $(TD no))
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD $(NAN))       $(TD +$(INFIN))   $(TD no))
 *	)
 */

real hypot(real x, real y)
{
    /*
     * This is based on code from:
     * Cephes Math Library Release 2.1:  January, 1989
     * Copyright 1984, 1987, 1989 by Stephen L. Moshier
     * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
     */

    const int PRECL = 32;
    const int MAXEXPL = real.max_exp; //16384;
    const int MINEXPL = real.min_exp; //-16384;

    real xx, yy, b, re, im;
    int ex, ey, e;

    // Note, hypot(INFINITY, NAN) = INFINITY.
    if (isinf(x) || isinf(y))
	return real.infinity;

    if (isnan(x))
	return x;
    if (isnan(y))
	return y;

    re = fabs(x);
    im = fabs(y);

    if (re == 0.0)
	return im;
    if (im == 0.0)
	return re;

    // Get the exponents of the numbers
    xx = frexp(re, ex);
    yy = frexp(im, ey);

    // Check if one number is tiny compared to the other
    e = ex - ey;
    if (e > PRECL)
	return re;
    if (e < -PRECL)
	return im;

    // Find approximate exponent e of the geometric mean.
    e = (ex + ey) >> 1;

    // Rescale so mean is about 1
    xx = ldexp(re, -e);
    yy = ldexp(im, -e);

    // Hypotenuse of the right triangle
    b = sqrt(xx * xx  +  yy * yy);

    // Compute the exponent of the answer.
    yy = frexp(b, ey);
    ey = e + ey;

    // Check it for overflow and underflow.
    if (ey > MAXEXPL + 2)
    {
	//return __matherr(_OVERFLOW, INFINITY, x, y, "hypotl");
	return real.infinity;
    }
    if (ey < MINEXPL - 2)
	return 0.0;

    // Undo the scaling
    b = ldexp(b, e);
    return b;
}

unittest
{
    static real vals[][3] =	// x,y,hypot
    [
	[	0,	0,	0],
	[	0,	-0,	0],
	[	3,	4,	5],
	[	-300,	-400,	500],
	[	real.min, real.min, 4.75473e-4932L],
	[	real.max/2, real.max/2, 0x1.6a09e667f3bcc908p+16383L /*8.41267e+4931L*/],
	[	real.infinity, real.nan, real.infinity],
	[	real.nan, real.nan, real.nan],
    ];
    int i;

    for (i = 0; i < vals.length; i++)
    {
	real x = vals[i][0];
	real y = vals[i][1];
	real z = vals[i][2];
	real h = hypot(x, y);

	//printf("hypot(%Lg, %Lg) = %Lg, should be %Lg\n", x, y, h, z);
	//if (!mfeq(z, h, .0000001))
	    //printf("%La\n", h);
	assert(mfeq(z, h, .0000001));
    }
}

/**********************************
 * Returns the error function of x.
 *
 * <img src="erf.gif" alt="error function">
 */
real erf(real x)		{ return std.c.math.erf(x); }

/**********************************
 * Returns the complementary error function of x, which is 1 - erf(x).
 *
 * <img src="erfc.gif" alt="complementary error function">
 */
real erfc(real x)		{ return std.c.math.erfc(x); }

/***********************************
 * Natural logarithm of gamma function.
 *
 * Returns the base e (2.718...) logarithm of the absolute
 * value of the gamma function of the argument.
 *
 * For reals, lgamma is equivalent to log(fabs(gamma(x))).
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)                 $(TH lgamma(x)) $(TH invalid?))
 *	$(TR $(TD $(NAN))            $(TD $(NAN))    $(TD yes))
 *	$(TR $(TD integer <= 0)      $(TD +$(INFIN)) $(TD yes))
 *	$(TR $(TD $(PLUSMN)$(INFIN)) $(TD +$(INFIN)) $(TD no))
 *	)
 */
/* Documentation prepared by Don Clugston */
real lgamma(real x)
{
    return std.c.math.lgamma(x);

    // Use etc.gamma.lgamma for those C systems that are missing it
}

/***********************************
 *  The Gamma function, $(GAMMA)(x)
 *
 *  $(GAMMA)(x) is a generalisation of the factorial function
 *  to real and complex numbers.
 *  Like x!, $(GAMMA)(x+1) = x*$(GAMMA)(x).
 *
 *  Mathematically, if z.re > 0 then
 *   $(GAMMA)(z) =<big>$(INTEGRAL)<sub><small>0</small></sub><sup>$(INFIN)</sup></big>t<sup>z-1</sup>e<sup>-t</sup>dt
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)              $(TH $(GAMMA)(x))       $(TH invalid?))
 *	$(TR $(TD $(NAN))         $(TD $(NAN))            $(TD yes))
 *	$(TR $(TD $(PLUSMN)0.0)   $(TD $(PLUSMN)$(INFIN)) $(TD yes))
 *	$(TR $(TD integer $(GT)0) $(TD (x-1)!)            $(TD no))
 *	$(TR $(TD integer $(LT)0) $(TD $(NAN))            $(TD yes))
 *	$(TR $(TD +$(INFIN))      $(TD +$(INFIN))         $(TD no))
 *	$(TR $(TD -$(INFIN))      $(TD $(NAN))            $(TD yes))
 *	)
 *
 *  References:
 *	$(LINK http://en.wikipedia.org/wiki/Gamma_function),
 *	$(LINK http://www.netlib.org/cephes/ldoubdoc.html#gamma)
 */
/* Documentation prepared by Don Clugston */
real tgamma(real x)
{
    return std.c.math.tgamma(x);

    // Use etc.gamma.tgamma for those C systems that are missing it
}

/**************************************
 * Returns the value of x rounded upward to the next integer
 * (toward positive infinity).
 */
real ceil(real x)		{ return std.c.math.ceil(x); }

/**************************************
 * Returns the value of x rounded downward to the next integer
 * (toward negative infinity).
 */
real floor(real x)		{ return std.c.math.floor(x); }

/******************************************
 * Rounds x to the nearest integer value, using the current rounding 
 * mode.
 *
 * Unlike the rint functions, nearbyint does not raise the 
 * FE_INEXACT exception. 
 */
real nearbyint(real x) { return std.c.math.nearbyint(x); }

/**********************************
 * Rounds x to the nearest integer value, using the current rounding
 * mode.
 * If the return value is not equal to x, the FE_INEXACT
 * exception is raised.
 * <b>nearbyint</b> performs
 * the same operation, but does not set the FE_INEXACT exception.
 */
real rint(real x)	{ return std.c.math.rint(x); }

/***************************************
 * Rounds x to the nearest integer value, using the current rounding
 * mode.
 */
long lrint(real x)
{
    version (linux)
	return std.c.math.llrint(x);
    else
	throw new NotImplemented("lrint");
}

/*******************************************
 * Return the value of x rounded to the nearest integer.
 * If the fractional part of x is exactly 0.5, the return value is rounded to
 * the even integer. 
 */
real round(real x) { return std.c.math.round(x); }

/**********************************************
 * Return the value of x rounded to the nearest integer.
 *
 * If the fractional part of x is exactly 0.5, the return value is rounded
 * away from zero.
 */
long lround(real x)
{
    version (linux)
	return std.c.math.llround(x);
    else
	throw new NotImplemented("lround");
}

/****************************************************
 * Returns the integer portion of x, dropping the fractional portion. 
 *
 * This is also known as "chop" rounding. 
 */
real trunc(real x) { return std.c.math.trunc(x); }

/****************************************************
 * Calculate the remainder x REM y, following IEC 60559.
 *
 * REM is the value of x - y * n, where n is the integer nearest the exact 
 * value of x / y.
 * If |n - x / y| == 0.5, n is even.
 * If the result is zero, it has the same sign as x.
 * Otherwise, the sign of the result is the sign of x / y.
 * Precision mode has no effect on the remainder functions.
 *
 * remquo returns n in the parameter n.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)                    $(TH y)                 $(TH remainder(x, y)) $(TH n)   $(TH invalid?))
 *	$(TR $(TD $(PLUSMN)0.0)         $(TD no)t 0.0           $(TD $(PLUSMN)0.0)    $(TD 0.0) $(TD no))
 *	$(TR $(TD $(PLUSMN)$(INFIN))    $(TD anything)          $(TD $(NAN))          $(TD ?)   $(TD yes))
 *	$(TR $(TD anything)             $(TD $(PLUSMN)0.0)      $(TD $(NAN))          $(TD ?)   $(TD yes))
 *	$(TR $(TD != $(PLUSMN)$(INFIN)) $(TD $(PLUSMN)$(INFIN)) $(TD x)               $(TD ?)   $(TD no))
 *	)
 */
real remainder(real x, real y) { return std.c.math.remainder(x, y); }

real remquo(real x, real y, out int n)	/// ditto
{
    version (linux)
	return std.c.math.remquo(x, y, &n);
    else
	throw new NotImplemented("remquo");
}

/*********************************
 * Returns !=0 if e is a NaN.
 */

int isnan(real e)
{
    ushort* pe = cast(ushort *)&e;
    ulong*  ps = cast(ulong *)&e;

    return (pe[4] & 0x7FFF) == 0x7FFF &&
	    *ps & 0x7FFFFFFFFFFFFFFF;
}

unittest
{
    assert(isnan(float.nan));
    assert(isnan(-double.nan));
    assert(isnan(real.nan));

    assert(!isnan(53.6));
    assert(!isnan(float.infinity));
}

/*********************************
 * Returns !=0 if e is finite.
 */

int isfinite(real e)
{
    ushort* pe = cast(ushort *)&e;

    return (pe[4] & 0x7FFF) != 0x7FFF;
}

unittest
{
    assert(isfinite(1.23));
    assert(!isfinite(double.infinity));
    assert(!isfinite(float.nan));
}


/*********************************
 * Returns !=0 if x is normalized.
 */

/* Need one for each format because subnormal floats might
 * be converted to normal reals.
 */

int isnormal(float x)
{
    uint *p = cast(uint *)&x;
    uint e;

    e = *p & 0x7F800000;
    //printf("e = x%x, *p = x%x\n", e, *p);
    return e && e != 0x7F800000;
}

/// ditto

int isnormal(double d)
{
    uint *p = cast(uint *)&d;
    uint e;

    e = p[1] & 0x7FF00000;
    return e && e != 0x7FF00000;
}

/// ditto

int isnormal(real e)
{
    ushort* pe = cast(ushort *)&e;
    long*   ps = cast(long *)&e;

    return (pe[4] & 0x7FFF) != 0x7FFF && *ps < 0;
}

unittest
{
    float f = 3;
    double d = 500;
    real e = 10e+48;

    assert(isnormal(f));
    assert(isnormal(d));
    assert(isnormal(e));
}

/*********************************
 * Is number subnormal? (Also called "denormal".)
 * Subnormals have a 0 exponent and a 0 most significant mantissa bit.
 */

/* Need one for each format because subnormal floats might
 * be converted to normal reals.
 */

int issubnormal(float f)
{
    uint *p = cast(uint *)&f;

    //printf("*p = x%x\n", *p);
    return (*p & 0x7F800000) == 0 && *p & 0x007FFFFF;
}

unittest
{
    float f = 3.0;

    for (f = 1.0; !issubnormal(f); f /= 2)
	assert(f != 0);
}

/// ditto

int issubnormal(double d)
{
    uint *p = cast(uint *)&d;

    return (p[1] & 0x7FF00000) == 0 && (p[0] || p[1] & 0x000FFFFF);
}

unittest
{
    double f;

    for (f = 1; !issubnormal(f); f /= 2)
	assert(f != 0);
}

/// ditto

int issubnormal(real e)
{
    ushort* pe = cast(ushort *)&e;
    long*   ps = cast(long *)&e;

    return (pe[4] & 0x7FFF) == 0 && *ps > 0;
}

unittest
{
    real f;

    for (f = 1; !issubnormal(f); f /= 2)
	assert(f != 0);
}

/*********************************
 * Return !=0 if e is $(PLUSMN)$(INFIN).
 */

int isinf(real e)
{
    ushort* pe = cast(ushort *)&e;
    ulong*  ps = cast(ulong *)&e;

    return (pe[4] & 0x7FFF) == 0x7FFF &&
	    *ps == 0x8000000000000000;
}

unittest
{
    assert(isinf(float.infinity));
    assert(!isinf(float.nan));
    assert(isinf(double.infinity));
    assert(isinf(-real.infinity));

    assert(isinf(-1.0 / 0.0));
}

/*********************************
 * Return 1 if sign bit of e is set, 0 if not.
 */

int signbit(real e)
{
    ubyte* pe = cast(ubyte *)&e;

//printf("e = %Lg\n", e);
    return (pe[9] & 0x80) != 0;
}

unittest
{
    debug (math) printf("math.signbit.unittest\n");
    assert(!signbit(float.nan));
    assert(signbit(-float.nan));
    assert(!signbit(168.1234));
    assert(signbit(-168.1234));
    assert(!signbit(0.0));
    assert(signbit(-0.0));
}

/*********************************
 * Return a value composed of to with from's sign bit.
 */

real copysign(real to, real from)
{
    ubyte* pto   = cast(ubyte *)&to;
    ubyte* pfrom = cast(ubyte *)&from;

    pto[9] &= 0x7F;
    pto[9] |= pfrom[9] & 0x80;

    return to;
}

unittest
{
    real e;

    e = copysign(21, 23.8);
    assert(e == 21);

    e = copysign(-21, 23.8);
    assert(e == 21);

    e = copysign(21, -23.8);
    assert(e == -21);

    e = copysign(-21, -23.8);
    assert(e == -21);

    e = copysign(real.nan, -23.8);
    assert(isnan(e) && signbit(e));
}

/******************************************
 * Creates a quiet NAN with the information from tagp[] embedded in it.
 */
real nan(char[] tagp) { return std.c.math.nan((tagp~\0).ptr); }

/******************************************
 * Calculates the next representable value after x in the direction of y. 
 *
 * If y $(GT) x, the result will be the next largest floating-point value;
 * if y $(LT) x, the result will be the next smallest value.
 * If x == y, the result is y.
 * The FE_INEXACT and FE_OVERFLOW exceptions will be raised if x is finite and
 * the function result is infinite. The FE_INEXACT and FE_UNDERFLOW 
 * exceptions will be raised if the function value is subnormal, and x is 
 * not equal to y. 
 */
real nextafter(real x, real y)
{
    version (linux)
	return std.c.math.nextafterl(x, y);
    else
	throw new NotImplemented("nextafter");
}

//real nexttoward(real x, real y) { return std.c.math.nexttowardl(x, y); }

/*******************************************
 * Returns the positive difference between x and y.
 * Returns:
 *	$(TABLE_SV
 *	$(TR $(TH x, y)       $(TH fdim(x, y)))
 *	$(TR $(TD x $(GT) y)  $(TD x - y))
 *	$(TR $(TD x $(LT)= y) $(TD +0.0))
 *	)
 */
real fdim(real x, real y) { return (x > y) ? x - y : +0.0; }

/****************************************
 * Returns the larger of x and y.
 */
real fmax(real x, real y) { return x > y ? x : y; }

/****************************************
 * Returns the smaller of x and y.
 */
real fmin(real x, real y) { return x < y ? x : y; }

/**************************************
 * Returns (x * y) + z, rounding only once according to the
 * current rounding mode.
 */
real fma(real x, real y, real z) { return (x * y) + z; }

/*******************************************************************
 * Fast integral powers.
 */

pragma(intrinsic, "llvm.powi.f32")
{
float pow(float x, uint n);
/// ditto
float pow(float x, int n);
}

pragma(intrinsic, "llvm.powi.f64")
{
/// ditto
double pow(double x, uint n);
/// ditto
double pow(double x, int n);
}

pragma(intrinsic, "llvm.powi.f80")
{
/// ditto
real pow(real x, uint n);
/// ditto
real pow(real x, int n);
}

/+
real pow(real x, uint n);
{
    real p;

    switch (n)
    {
	case 0:
	    p = 1.0;
	    break;

	case 1:
	    p = x;
	    break;

	case 2:
	    p = x * x;
	    break;

	default:
	    p = 1.0;
	    while (1)
	    {
		if (n & 1)
		    p *= x;
		n >>= 1;
		if (!n)
		    break;
		x *= x;
	    }
	    break;
    }
    return p;
}

/// ditto
real pow(real x, int n);
{
    if (n < 0)
	return pow(x, cast(real)n);
    else
	return pow(x, cast(uint)n);
}
+/

/*********************************************
 * Calculates x$(SUP y).
 *
 * $(TABLE_SV
 * $(TR
 * $(TH x) $(TH y) $(TH pow(x, y)) $(TH div 0) $(TH invalid?))
 * $(TR
 * $(TD anything) 	$(TD $(PLUSMN)0.0) 	$(TD 1.0) 	$(TD no) 	$(TD no) )
 * $(TR
 * $(TD |x| $(GT) 1) 	$(TD +$(INFIN)) 	$(TD +$(INFIN)) 	$(TD no) 	$(TD no) )
 * $(TR
 * $(TD |x| $(LT) 1) 	$(TD +$(INFIN)) 	$(TD +0.0) 	$(TD no) 	$(TD no) )
 * $(TR
 * $(TD |x| $(GT) 1) 	$(TD -$(INFIN)) 	$(TD +0.0) 	$(TD no) 	$(TD no) )
 * $(TR
 * $(TD |x| $(LT) 1) 	$(TD -$(INFIN)) 	$(TD +$(INFIN)) 	$(TD no) 	$(TD no) )
 * $(TR
 * $(TD +$(INFIN)) 	$(TD $(GT) 0.0) 	$(TD +$(INFIN)) 	$(TD no) 	$(TD no) )
 * $(TR
 * $(TD +$(INFIN)) 	$(TD $(LT) 0.0) 	$(TD +0.0) 	$(TD no) 	$(TD no) )
 * $(TR
 * $(TD -$(INFIN)) 	$(TD odd integer $(GT) 0.0)	$(TD -$(INFIN)) 	$(TD no) 	$(TD no) )
 * $(TR
 * $(TD -$(INFIN))  	$(TD $(GT) 0.0, not odd integer) $(TD +$(INFIN)) 	$(TD no) 	$(TD no))
 * $(TR
 * $(TD -$(INFIN)) 	$(TD odd integer $(LT) 0.0)  	$(TD -0.0) 	$(TD no) 	$(TD no) )
 * $(TR
 * $(TD -$(INFIN)) 	$(TD $(LT) 0.0, not odd integer) $(TD +0.0) 	$(TD no) 	$(TD no) )
 * $(TR
 * $(TD $(PLUSMN)1.0) 	$(TD $(PLUSMN)$(INFIN)) 	$(TD $(NAN)) 	$(TD no) 	$(TD yes) )
 * $(TR
 * $(TD $(LT) 0.0) 	$(TD finite, nonintegral) 	$(TD $(NAN)) 	$(TD no) 	$(TD yes))
 * $(TR
 * $(TD $(PLUSMN)0.0) 	$(TD odd integer $(LT) 0.0)	$(TD $(PLUSMN)$(INFIN)) $(TD yes) 	$(TD no) )
 * $(TR
 * $(TD $(PLUSMN)0.0) 	$(TD $(LT) 0.0, not odd integer) $(TD +$(INFIN)) 	$(TD yes) 	$(TD no))
 * $(TR
 * $(TD $(PLUSMN)0.0) 	$(TD odd integer $(GT) 0.0)	$(TD $(PLUSMN)0.0) $(TD no) 	$(TD no) )
 * $(TR
 * $(TD $(PLUSMN)0.0) 	$(TD $(GT) 0.0, not odd integer) $(TD +0.0) 	$(TD no) 	$(TD no) )
 * )
 */

pragma(intrinsic, "llvm.pow.f32")
float pow(float x, float y);

pragma(intrinsic, "llvm.pow.f64")
double pow(double x, double y);

pragma(intrinsic, "llvm.pow.f80")
real pow(real x, real y);

/+
real pow(real x, real y);
{
    version (linux) // C pow() often does not handle special values correctly
    {
	if (isnan(y))
	    return y;

	if (y == 0)
	    return 1;		// even if x is $(NAN)
	if (isnan(x) && y != 0)
	    return x;
	if (isinf(y))
	{
	    if (fabs(x) > 1)
	    {
		if (signbit(y))
		    return +0.0;
		else
		    return real.infinity;
	    }
	    else if (fabs(x) == 1)
	    {
		return real.nan;
	    }
	    else // < 1
	    {
		if (signbit(y))
		    return real.infinity;
		else
		    return +0.0;
	    }
	}
	if (isinf(x))
	{
	    if (signbit(x))
	    {   long i;

		i = cast(long)y;
		if (y > 0)
		{
		    if (i == y && i & 1)
			return -real.infinity;
		    else
			return real.infinity;
		}
		else if (y < 0)
		{
		    if (i == y && i & 1)
			return -0.0;
		    else
			return +0.0;
		}
	    }
	    else
	    {
		if (y > 0)
		    return real.infinity;
		else if (y < 0)
		    return +0.0;
	    }
	}

	if (x == 0.0)
	{
	    if (signbit(x))
	    {   long i;

		i = cast(long)y;
		if (y > 0)
		{
		    if (i == y && i & 1)
			return -0.0;
		    else
			return +0.0;
		}
		else if (y < 0)
		{
		    if (i == y && i & 1)
			return -real.infinity;
		    else
			return real.infinity;
		}
	    }
	    else
	    {
		if (y > 0)
		    return +0.0;
		else if (y < 0)
		    return real.infinity;
	    }
	}
    }
    return std.c.math.powl(x, y);
}
+/

unittest
{
    real x = 46;

    assert(pow(x,0) == 1.0);
    assert(pow(x,1) == x);
    assert(pow(x,2) == x * x);
    assert(pow(x,3) == x * x * x);
    assert(pow(x,8) == (x * x) * (x * x) * (x * x) * (x * x));
}

/****************************************
 * Simple function to compare two floating point values
 * to a specified precision.
 * Returns:
 *	1	match
 *	0	nomatch
 */

private int mfeq(real x, real y, real precision)
{
    if (x == y)
	return 1;
    if (isnan(x))
	return isnan(y);
    if (isnan(y))
	return 0;
    return fabs(x - y) <= precision;
}

// Returns true if x is +0.0 (This function is used in unit tests)
bool isPosZero(real x)
{
    return (x == 0) && (signbit(x) == 0);
}

// Returns true if x is -0.0 (This function is used in unit tests)
bool isNegZero(real x)
{
    return (x == 0) && signbit(x);
}

/**************************************
 * To what precision is x equal to y?
 *
 * Returns: the number of mantissa bits which are equal in x and y.
 * eg, 0x1.F8p+60 and 0x1.F1p+60 are equal to 5 bits of precision.
 *
 *	$(TABLE_SV
 *	$(TR $(TH x)      $(TH y)          $(TH feqrel(x, y)))
 *	$(TR $(TD x)      $(TD x)          $(TD real.mant_dig))
 *	$(TR $(TD x)      $(TD $(GT)= 2*x) $(TD 0))
 *	$(TR $(TD x)      $(TD $(LT)= x/2) $(TD 0))
 *	$(TR $(TD $(NAN)) $(TD any)        $(TD 0))
 *	$(TR $(TD any)    $(TD $(NAN))     $(TD 0))
 *	)
 */

int feqrel(real x, real y)
{
    /* Public Domain. Author: Don Clugston, 18 Aug 2005.
     */

    if (x == y)
	return real.mant_dig; // ensure diff!=0, cope with INF.

    real diff = fabs(x - y);

    ushort *pa = cast(ushort *)(&x);
    ushort *pb = cast(ushort *)(&y);
    ushort *pd = cast(ushort *)(&diff);

    // The difference in abs(exponent) between x or y and abs(x-y)
    // is equal to the number of mantissa bits of x which are
    // equal to y. If negative, x and y have different exponents.
    // If positive, x and y are equal to 'bitsdiff' bits.
    // AND with 0x7FFF to form the absolute value.
    // To avoid out-by-1 errors, we subtract 1 so it rounds down
    // if the exponents were different. This means 'bitsdiff' is
    // always 1 lower than we want, except that if bitsdiff==0,
    // they could have 0 or 1 bits in common.
    int bitsdiff = ( ((pa[4]&0x7FFF) + (pb[4]&0x7FFF)-1)>>1) - pd[4];

    if (pd[4] == 0)
    {	// Difference is denormal
	// For denormals, we need to add the number of zeros that
	// lie at the start of diff's mantissa.
	// We do this by multiplying by 2^real.mant_dig
	diff *= 0x1p+63;
	return bitsdiff + real.mant_dig - pd[4];
    }

    if (bitsdiff > 0)
	return bitsdiff + 1; // add the 1 we subtracted before

    // Avoid out-by-1 errors when factor is almost 2.
    return (bitsdiff == 0) ? (pa[4] == pb[4]) : 0;
}

unittest
{
   // Exact equality
   assert(feqrel(real.max,real.max)==real.mant_dig);
   assert(feqrel(0,0)==real.mant_dig);
   assert(feqrel(7.1824,7.1824)==real.mant_dig);
   assert(feqrel(real.infinity,real.infinity)==real.mant_dig);

   // a few bits away from exact equality
   real w=1;
   for (int i=1; i<real.mant_dig-1; ++i) {
      assert(feqrel(1+w*real.epsilon,1)==real.mant_dig-i);
      assert(feqrel(1-w*real.epsilon,1)==real.mant_dig-i);
      assert(feqrel(1,1+(w-1)*real.epsilon)==real.mant_dig-i+1);
      w*=2;
   }
   assert(feqrel(1.5+real.epsilon,1.5)==real.mant_dig-1);
   assert(feqrel(1.5-real.epsilon,1.5)==real.mant_dig-1);
   assert(feqrel(1.5-real.epsilon,1.5+real.epsilon)==real.mant_dig-2);

   // Numbers that are close
   assert(feqrel(0x1.Bp+84, 0x1.B8p+84)==5);
   assert(feqrel(0x1.8p+10, 0x1.Cp+10)==2);
   assert(feqrel(1.5*(1-real.epsilon), 1)==2);
   assert(feqrel(1.5, 1)==1);
   assert(feqrel(2*(1-real.epsilon), 1)==1);

   // Factors of 2
   assert(feqrel(real.max,real.infinity)==0);
   assert(feqrel(2*(1-real.epsilon), 1)==1);
   assert(feqrel(1, 2)==0);
   assert(feqrel(4, 1)==0);

   // Extreme inequality
   assert(feqrel(real.nan,real.nan)==0);
   assert(feqrel(0,-real.nan)==0);
   assert(feqrel(real.nan,real.infinity)==0);
   assert(feqrel(real.infinity,-real.infinity)==0);
   assert(feqrel(-real.max,real.infinity)==0);
   assert(feqrel(real.max,-real.max)==0);
}


/***********************************
 * Evaluate polynomial A(x) = a<sub>0</sub> + a<sub>1</sub>x + a<sub>2</sub>x&sup2; + a<sub>3</sub>x&sup3; ...
 *
 * Uses Horner's rule A(x) = a<sub>0</sub> + x(a<sub>1</sub> + x(a<sub>2</sub> + x(a<sub>3</sub> + ...)))
 * Params:
 *	A =	array of coefficients a<sub>0</sub>, a<sub>1</sub>, etc.
 */ 
real poly(real x, real[] A)
in
{
    assert(A.length > 0);
}
body
{
    version (D_InlineAsm_X86)
    {
	version (Windows)
	{
	    asm	// assembler by W. Bright
	    {
		// EDX = (A.length - 1) * real.sizeof
		mov     ECX,A[EBP]		; // ECX = A.length
		dec     ECX			;
		lea     EDX,[ECX][ECX*8]	;
		add     EDX,ECX			;
		add     EDX,A+4[EBP]		;
		fld     real ptr [EDX]		; // ST0 = coeff[ECX]
		jecxz   return_ST		;
		fld     x[EBP]			; // ST0 = x
		fxch    ST(1)			; // ST1 = x, ST0 = r
		align   4			;
	L2:     fmul    ST,ST(1)		; // r *= x
		fld     real ptr -10[EDX]	;
		sub     EDX,10			; // deg--
		faddp   ST(1),ST		;
		dec     ECX			;
		jne     L2			;
		fxch    ST(1)			; // ST1 = r, ST0 = x
		fstp    ST(0)			; // dump x
		align   4			;
	return_ST:				;
		;
	    }
	}
	else
	{
	    asm	// assembler by W. Bright
	    {
		// EDX = (A.length - 1) * real.sizeof
		mov     ECX,A[EBP]		; // ECX = A.length
		dec     ECX			;
		lea     EDX,[ECX*8]		;
		lea	EDX,[EDX][ECX*4]	;
		add     EDX,A+4[EBP]		;
		fld     real ptr [EDX]		; // ST0 = coeff[ECX]
		jecxz   return_ST		;
		fld     x[EBP]			; // ST0 = x
		fxch    ST(1)			; // ST1 = x, ST0 = r
		align   4			;
	L2:     fmul    ST,ST(1)		; // r *= x
		fld     real ptr -12[EDX]	;
		sub     EDX,12			; // deg--
		faddp   ST(1),ST		;
		dec     ECX			;
		jne     L2			;
		fxch    ST(1)			; // ST1 = r, ST0 = x
		fstp    ST(0)			; // dump x
		align   4			;
	return_ST:				;
		;
	    }
	}
    }
    else
    {
	int i = A.length - 1;
	real r = A[i];
	while (--i >= 0)
	{
	    r *= x;
	    r += A[i];
	}
	return r;
    }
}

unittest
{
    debug (math) printf("math.poly.unittest\n");
    real x = 3.1;
    static real pp[] = [56.1, 32.7, 6];

    assert( poly(x, pp) == (56.1L + (32.7L + 6L * x) * x) );
}


