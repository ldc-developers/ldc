//===-- ctfloat.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/root/ctfloat.h"
#include "gen/llvm.h"
#include "llvm/Support/Error.h"

using llvm::APFloat;

#if LDC_LLVM_VER >= 400
#define AP_SEMANTICS_PARENS ()
#else
#define AP_SEMANTICS_PARENS
#endif

namespace {

const llvm::fltSemantics *apSemantics = nullptr;

constexpr unsigned numUint64Parts = (sizeof(real_t) + 7) / 8;
union CTFloatUnion {
  real_t fp;
  uint64_t bits[numUint64Parts];
};

APFloat parseLiteral(const llvm::fltSemantics &semantics, const char *literal,
                     bool *isOutOfRange = nullptr) {
  APFloat ap(semantics, APFloat::uninitialized);
  auto r =
#if LDC_LLVM_VER >= 1000
      llvm::cantFail
#endif
      (ap.convertFromString(literal, APFloat::rmNearestTiesToEven));
  if (isOutOfRange) {
    *isOutOfRange = (
#if LDC_LLVM_VER >= 1100
                     r.get()
#else
                     r
#endif
                     & (APFloat::opOverflow | APFloat::opUnderflow)) != 0;
  }
  return ap;
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

void CTFloat::initialize() {
  if (apSemantics)
    return;

  static_assert(sizeof(real_t) >= 8, "real_t < 64 bits?");

  if (sizeof(real_t) == 8) {
    apSemantics = &(APFloat::IEEEdouble AP_SEMANTICS_PARENS);
  } else {
#if __aarch64__ || (__ANDROID__ && __LP64__)
    apSemantics = &(APFloat::IEEEquad AP_SEMANTICS_PARENS);
#elif __i386__ || __x86_64__ || _M_IX86 || _M_X64
    apSemantics = &(APFloat::x87DoubleExtended AP_SEMANTICS_PARENS);
#elif __ppc__ || __ppc64__
    apSemantics = &(APFloat::PPCDoubleDouble AP_SEMANTICS_PARENS);
#else
    llvm_unreachable("Unknown host real_t type for compile-time reals");
#endif
  }

  zero = 0;
  one = 1;
  minusone = -1;
  half = 0.5;

  nan = fromAPFloat(APFloat::getQNaN(*apSemantics));
  infinity = fromAPFloat(APFloat::getInf(*apSemantics));
}

////////////////////////////////////////////////////////////////////////////////

void CTFloat::toAPFloat(const real_t src, APFloat &dst) {
  if (sizeof(real_t) == 8) {
    dst = APFloat(static_cast<double>(src));
    return;
  }

  CTFloatUnion u;
  u.fp = src;

  const unsigned sizeInBits = APFloat::getSizeInBits(*apSemantics);
  const APInt bits = APInt(sizeInBits, numUint64Parts, u.bits);

  dst = APFloat(*apSemantics, bits);
}

////////////////////////////////////////////////////////////////////////////////

real_t CTFloat::fromAPFloat(const APFloat &src_) {
  APFloat src = src_;
  if (&src.getSemantics() != apSemantics) {
    bool ignored;
    src.convert(*apSemantics, APFloat::rmNearestTiesToEven, &ignored);
  }

  const APInt bits = src.bitcastToAPInt();

  CTFloatUnion u;
  memcpy(u.bits, bits.getRawData(), bits.getBitWidth() / 8);
  return u.fp;
}

////////////////////////////////////////////////////////////////////////////////

real_t CTFloat::parse(const char *literal, bool *isOutOfRange) {
  const APFloat ap = parseLiteral(*apSemantics, literal, isOutOfRange);
  return fromAPFloat(ap);
}

bool CTFloat::isFloat32LiteralOutOfRange(const char *literal) {
  bool isOutOfRange;
  parseLiteral(APFloat::IEEEsingle AP_SEMANTICS_PARENS, literal, &isOutOfRange);
  return isOutOfRange;
}

bool CTFloat::isFloat64LiteralOutOfRange(const char *literal) {
  bool isOutOfRange;
  parseLiteral(APFloat::IEEEdouble AP_SEMANTICS_PARENS, literal, &isOutOfRange);
  return isOutOfRange;
}
