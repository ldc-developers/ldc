//===-- gen/ldc-real.cpp - Interface of real_t for LDC ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Implements a real_t type for LDC.
//
//===----------------------------------------------------------------------===//

#include "gen/ldc-real.h"
#include "gen/llvmcompat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "mars.h"  // Only for warning()
#include <cmath>   // Only for sin(), cos(), tan()

static llvm::cl::opt<bool> longdouble64("long-double-64",
           llvm::cl::desc("Choose real to be 64bit double"),
           llvm::cl::ZeroOrMore,
           llvm::cl::init(false));

extern llvm::TargetMachine *gTargetMachine;

const llvm::fltSemantics &ldc::longdouble::getFltSemantics()
{
    static llvm::Triple::ArchType arch = llvm::Triple::UnknownArch;
    switch (arch)
    {
    case llvm::Triple::UnknownArch:
        arch = llvm::Triple(gTargetMachine->getTargetTriple()).getArch();
        if (arch != llvm::Triple::UnknownArch)
            return getFltSemantics();
        else
            return llvm::APFloat::IEEEdouble;
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
        return longdouble64 ? llvm::APFloat::IEEEdouble : llvm::APFloat::x87DoubleExtended;
    case llvm::Triple::ppc:
    case llvm::Triple::ppc64:
#if LDC_LLVM_VER >= 304
    case llvm::Triple::ppc64le:
#endif
        return longdouble64 ? llvm::APFloat::IEEEdouble : llvm::APFloat::PPCDoubleDouble;
    default:
        return llvm::APFloat::IEEEdouble;
    }
}

ldc::longdouble ldc::longdouble::abs() const
{
    if (value()->isNegative())
    {
        llvm::APFloat f(*value());
        f.changeSign();
        ldc::longdouble tmp;
        return tmp.init(f);
    }
    return *this;
}

static llvm::APFloat::opStatus sqrt(llvm::APFloat &the)
{
    if (!the.isNaN() && !the.isInfinity() && !the.isZero())
    {
        if (the.isNegative()) {
            the = llvm::APFloat::getNaN(the.getSemantics());
            return llvm::APFloat::opInvalidOp;
        }

        static const llvm::APFloat onehalf(the.getSemantics(), "0.5");
        llvm::APFloat x0(the);

        int loop = 0;
        do {
            llvm::APFloat x1(the);
            x1.divide(x0, llvm::APFloat::rmNearestTiesToEven);
            x1.add(x0, llvm::APFloat::rmNearestTiesToEven);
            x1.multiply(onehalf, llvm::APFloat::rmNearestTiesToEven);
            if (x1.bitwiseIsEqual(x0)) break;
            x0 = x1;
        } while (loop++ < 10);

        the = x0;
    }
    return llvm::APFloat::opOK;
}

ldc::longdouble ldc::longdouble::sqrt() const
{
    llvm::APFloat f(*value());
    ::sqrt(f);
    ldc::longdouble tmp;
    return tmp.init(f);
}

static llvm::APFloat::opStatus sin(llvm::APFloat &the)
{
// Using a Maclaurin series:
// sin(x) = x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11! + x^13/13! ...
// Also note:
// sin(-x) = - sin(x)
    static const llvm::APFloat zero(the.getSemantics());
    static const llvm::APFloat pi(the.getSemantics(), "3.1415");
    static const llvm::APFloat plusone(the.getSemantics(), "1");
    static const llvm::APFloat minusone(the.getSemantics(), "-1");
    static const llvm::APFloat S6(the.getSemantics(), "156"); // 13*12
    static const llvm::APFloat S5(the.getSemantics(), "110"); // 11*10
    static const llvm::APFloat S4(the.getSemantics(), "72");  //  9* 8
    static const llvm::APFloat S3(the.getSemantics(), "42");  //  7* 6
    static const llvm::APFloat S2(the.getSemantics(), "20");  //  4* 5
    static const llvm::APFloat S1(the.getSemantics(), "6");   //  2* 3

    bool negate = the.compare(zero) == llvm::APFloat::cmpLessThan;
    if (negate)
        the.changeSign();
    llvm::APFloat the2(the); the2.multiply(the2, llvm::APFloat::rmNearestTiesToEven);

    llvm::APFloat v(the2);
    v.divide(S6, llvm::APFloat::rmNearestTiesToEven);
    v.add(minusone, llvm::APFloat::rmNearestTiesToEven);
    v.multiply(the2, llvm::APFloat::rmNearestTiesToEven);
    v.divide(S5, llvm::APFloat::rmNearestTiesToEven);
    v.add(plusone, llvm::APFloat::rmNearestTiesToEven);
    v.multiply(the2, llvm::APFloat::rmNearestTiesToEven);
    v.divide(S4, llvm::APFloat::rmNearestTiesToEven);
    v.add(minusone, llvm::APFloat::rmNearestTiesToEven);
    v.multiply(the2, llvm::APFloat::rmNearestTiesToEven);
    v.divide(S3, llvm::APFloat::rmNearestTiesToEven);
    v.add(plusone, llvm::APFloat::rmNearestTiesToEven);
    v.multiply(the2, llvm::APFloat::rmNearestTiesToEven);
    v.divide(S2, llvm::APFloat::rmNearestTiesToEven);
    v.add(minusone, llvm::APFloat::rmNearestTiesToEven);
    v.multiply(the2, llvm::APFloat::rmNearestTiesToEven);
    v.divide(S1, llvm::APFloat::rmNearestTiesToEven);

    v.multiply(the, llvm::APFloat::rmNearestTiesToEven);
    if (negate)
        v.changeSign();
    the = v;

    return llvm::APFloat::opOK;
}

ldc::longdouble ldc::longdouble::sin() const
{
	warning(Loc(), "sin() uses only double precision");
	double d = std::sin(value()->convertToDouble());
	return ldouble(d);
#if 0
// Not good enough yet.
	llvm::APFloat f(*value());
    ::sin(f);
    ldc::longdouble tmp;
    return tmp.init(f);
#endif
}

ldc::longdouble ldc::longdouble::cos() const
{
	warning(Loc(), "cos() uses only double precision");
	double d = std::cos(value()->convertToDouble());
    return ldouble(d);
}

ldc::longdouble ldc::longdouble::tan() const
{
	warning(Loc(), "tan() uses only double precision");
	double d = std::tan(value()->convertToDouble());
	return ldouble(d);
}

ldc::longdouble ldc::longdouble::floor() const
{
    // matches man floor(3) description
    //   floor(+-0) returns +-0.
    //   floor(+-infinity) returns +-infinity.
    longdouble tmp;
    tmp.init(*this).value()->roundToIntegral(llvm::APFloat::rmTowardNegative);
    return tmp;
}

ldc::longdouble ldc::longdouble::ceil() const
{
    // matches man ceil(3) special values
    //   ceil(+-0) returns +-0.
    //   ceil(+-infinity) returns +-infinity.
    longdouble tmp;
    tmp.init(*this).value()->roundToIntegral(llvm::APFloat::rmTowardPositive);
    return tmp;
}

ldc::longdouble ldc::longdouble::trunc() const
{
    // matches man trunc(3) special values
    //   trunc(+-0) returns +-0.
    //   trunc(+-infinity) returns +-infinity.
    longdouble tmp;
    tmp.init(*this).value()->roundToIntegral(llvm::APFloat::rmTowardZero);
    return tmp;
}

ldc::longdouble ldc::longdouble::round() const
{
    // matches man round(3) special values
    //   round(+-0) returns +-0.
    //   round(+-infinity) returns +-infinity.
    longdouble tmp;
    tmp.init(*this).value()->roundToIntegral(llvm::APFloat::rmNearestTiesToAway);
    return tmp;
}

ldc::longdouble ldc::longdouble::fmin(longdouble x, longdouble y)
{
    // matches man fmin(3) special values and -0,+0 ordering
    //   If exactly one argument is a NaN, fmin() returns the other
    //   argument. If both arguments are NaNs, fmin() returns a NaN.
    return (x.isNaN() ? y :
            y.isNaN() ? x :
            x <= y ? x : y);
}

ldc::longdouble ldc::longdouble::fmax(longdouble x, longdouble y)
{
    // matches man fmax(3) special values and -0,+0 ordering
    //   If exactly one argument is a NaN, fmax() returns the other
    //   argument. If both arguments are NaNs, fmax() returns a NaN.
    return (x.isNaN() ? y :
            y.isNaN() ? x :
            x >= y ? x : y);
}

ldc::longdouble ldc::longdouble::fmod(ldc::longdouble x, ldc::longdouble y)
{
    llvm::APFloat f(*x.value());
    f.mod(*y.value(), llvm::APFloat::rmNearestTiesToEven);
    ldc::longdouble tmp;
    return tmp.init(f);
}

ldc::longdouble ldc::longdouble::ldexp(ldc::longdouble ldval, int exp)
{
    llvm_unreachable("ldc::longdouble::ldexp() not implemented");
}

int ldc::longdouble::format (char *buf) const
{
    llvm::SmallString<64> str;
    value()->toString(str, 0, 0);
    strcpy(buf, str.str().str().c_str());
    return strlen(buf);
}

int ldc::longdouble::formatHex (char *buf, bool upper) const
{
    return value()->convertToHexString(buf, 0, upper, llvm::APFloat::rmNearestTiesToEven);
}

static llvm::APFloat toLD(const llvm::APFloat &f)
{
    llvm::APFloat cvt(f);
    bool ignored;
    llvm::APFloat::opStatus status = cvt.convert(ldc::longdouble::getFltSemantics(), llvm::APFloat::rmNearestTiesToEven, &ignored);
    assert(status == llvm::APFloat::opOK);
    (void)status;
    return cvt;
}

ldc::real_properties ldc::real_limits[ldc::longdouble::NumModes];

void ldc::real_init()
{
    using ldc::longdouble;
    using llvm::APFloat;

    const llvm::fltSemantics &ldSem = longdouble::getFltSemantics();
    enum { x87, ppc, other };
    int real;
    switch (APFloat::semanticsPrecision(ldSem))
    {
        case 64: real = x87; break;
        case 106: real = ppc; break;
        case 53: real = other; break;
        default: llvm_unreachable("Floating point type not supported");
    }

    real_limits[longdouble::Float].maxval = toLD(APFloat::getLargest(APFloat::IEEEsingle));
    real_limits[longdouble::Double].maxval = toLD(APFloat::getLargest(APFloat::IEEEdouble));
    real_limits[longdouble::LongDouble].maxval = APFloat::getLargest(ldSem);

    real_limits[longdouble::Float].minval = toLD(APFloat::getSmallestNormalized(APFloat::IEEEsingle));
    real_limits[longdouble::Double].minval = toLD(APFloat::getSmallestNormalized(APFloat::IEEEdouble));
    real_limits[longdouble::LongDouble].minval = APFloat::getSmallestNormalized(ldSem);

    real_limits[longdouble::Float].epsilonval = APFloat(ldSem, "1.19209290e-07");
    real_limits[longdouble::Double].epsilonval = APFloat(ldSem, "2.2204460492503131e-016");
    switch (real)
    {
        case x87: real_limits[longdouble::LongDouble].epsilonval = APFloat(ldSem, "1.0842021724855044340e-019"); break;
        case ppc: real_limits[longdouble::LongDouble].epsilonval = APFloat(ldSem, "2.2204460492503131e-016"); break; // like double
        case other: real_limits[longdouble::LongDouble].epsilonval = APFloat(ldSem, "2.2204460492503131e-016"); break; // aka double
    }

    real_limits[longdouble::Float].dig = 6;
    real_limits[longdouble::Double].dig = 15;
    switch (real)
    {
        case x87: real_limits[longdouble::LongDouble].dig = 80; break;
        case ppc: real_limits[longdouble::LongDouble].dig = 15; break; // like double
        case other: real_limits[longdouble::LongDouble].dig = 15; break; // aka double
    }

    real_limits[longdouble::Float].mant_dig = APFloat::semanticsPrecision(APFloat::IEEEsingle);
    real_limits[longdouble::Double].mant_dig = APFloat::semanticsPrecision(APFloat::IEEEdouble);
    real_limits[longdouble::LongDouble].mant_dig = APFloat::semanticsPrecision(ldSem);

    real_limits[longdouble::Float].max_10_exp = 38;
    real_limits[longdouble::Double].max_10_exp = 308;
    switch (real)
    {
        case x87: real_limits[longdouble::LongDouble].max_10_exp = 4932; break;
        case ppc: real_limits[longdouble::LongDouble].max_10_exp = 308; break; // like double
        case other: real_limits[longdouble::LongDouble].max_10_exp = 308; break; // aka double
    }

    real_limits[longdouble::Float].min_10_exp = -37;
    real_limits[longdouble::Double].min_10_exp = -307;
    switch (real)
    {
        case x87: real_limits[longdouble::LongDouble].min_10_exp = -4932; break;
        case ppc: real_limits[longdouble::LongDouble].min_10_exp = -307; break; // like double
        case other: real_limits[longdouble::LongDouble].min_10_exp = -307; break; // aka double
    }

    real_limits[longdouble::Float].max_exp = 128;
    real_limits[longdouble::Double].max_exp = 1024;
    switch (real)
    {
        case x87: real_limits[longdouble::LongDouble].max_exp = 16384; break;
        case ppc: real_limits[longdouble::LongDouble].max_exp = 1024; break; // like double
        case other: real_limits[longdouble::LongDouble].max_exp = 1024; break; // aka double
    }

    real_limits[longdouble::Float].min_exp = -125;
    real_limits[longdouble::Double].min_exp = -1021;
    switch (real)
    {
        case x87: real_limits[longdouble::LongDouble].min_exp = -16381; break;
        case ppc: real_limits[longdouble::LongDouble].min_exp = -1021; break; // like double
        case other: real_limits[longdouble::LongDouble].min_exp = -1021; break; // aka double
    }
}
