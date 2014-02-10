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
#include "llvm/ADT/Triple.h"
#include "llvm/Target/TargetMachine.h"

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
        return llvm::APFloat::x87DoubleExtended;
    case llvm::Triple::ppc:
    case llvm::Triple::ppc64:
#if LDC_LLVM_VER >= 304
    case llvm::Triple::ppc64le:
#endif
        return llvm::APFloat::PPCDoubleDouble;
    default:
        return llvm::APFloat::IEEEdouble;
    }
}

ldc::longdouble ldc::longdouble::abs() const
{
    if (value->isNegative())
    {
        llvm::APFloat f(*value);
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

        const llvm::APFloat onehalf(the.getSemantics(), "0.5");
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
    llvm::APFloat f(*value);
    ::sqrt(f);
    ldc::longdouble tmp;
    return tmp.init(f);
}

ldc::longdouble ldc::longdouble::sin() const
{
    llvm_unreachable("ldc::longdouble::sin() not implemented");
}

ldc::longdouble ldc::longdouble::cos() const
{
    llvm_unreachable("ldc::longdouble::cos() not implemented");
}

ldc::longdouble ldc::longdouble::tan() const
{
    llvm_unreachable("ldc::longdouble::tan() not implemented");
}

ldc::longdouble ldc::longdouble::fmod(ldc::longdouble x, ldc::longdouble y)
{
    llvm::APFloat f(*x.value);
    f.mod(*y.value, llvm::APFloat::rmTowardZero);
    ldc::longdouble tmp;
    return tmp.init(f);
}

ldc::longdouble ldc::longdouble::ldexp(ldc::longdouble ldval, int exp)
{
    llvm_unreachable("ldc::longdouble::ldexp() not implemented");
}

int ldc::longdouble::format (char *buf, unsigned buf_size) const
{
    return 0;
}

int ldc::longdouble::formatHex (char *buf, unsigned buf_size) const
{
    return 0;
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

    real_limits[longdouble::Float].maxval = APFloat::getLargest(APFloat::IEEEsingle);
    real_limits[longdouble::Double].maxval = APFloat::getLargest(APFloat::IEEEdouble);
    real_limits[longdouble::LongDouble].maxval = APFloat::getLargest(ldSem);

    real_limits[longdouble::Float].minval = APFloat::getSmallestNormalized(APFloat::IEEEsingle);
    real_limits[longdouble::Double].minval = APFloat::getSmallestNormalized(APFloat::IEEEdouble);
    real_limits[longdouble::LongDouble].minval = APFloat::getSmallestNormalized(ldSem);

    real_limits[longdouble::Float].epsilonval = APFloat(APFloat::IEEEsingle); // FIXME
    real_limits[longdouble::Double].epsilonval = APFloat(APFloat::IEEEdouble); // FIXME
    real_limits[longdouble::LongDouble].epsilonval = APFloat(ldSem); // FIXME

    real_limits[longdouble::Float].dig = 6;
    real_limits[longdouble::Double].dig = 15;
    real_limits[longdouble::LongDouble].dig = 0; // FIXME

    real_limits[longdouble::Float].mant_dig = APFloat::semanticsPrecision(APFloat::IEEEsingle);
    real_limits[longdouble::Double].mant_dig = APFloat::semanticsPrecision(APFloat::IEEEdouble);
    real_limits[longdouble::LongDouble].mant_dig = APFloat::semanticsPrecision(ldSem);

    real_limits[longdouble::Float].max_10_exp = 38;
    real_limits[longdouble::Double].max_10_exp = 308;
    real_limits[longdouble::LongDouble].max_10_exp = 0; // FIXME

    real_limits[longdouble::Float].min_10_exp = -37;
    real_limits[longdouble::Double].min_10_exp = -307;
    real_limits[longdouble::LongDouble].min_10_exp = 0; // FIXME

    real_limits[longdouble::Float].max_exp = 128;
    real_limits[longdouble::Double].max_exp = 1024;
    real_limits[longdouble::LongDouble].max_exp = 0; // FIXME

    real_limits[longdouble::Float].min_exp = -125;
    real_limits[longdouble::Double].min_exp = -1021;
    real_limits[longdouble::LongDouble].min_exp = 0; // FIXME
}