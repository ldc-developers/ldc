//===-- ctfloat.cpp -------------------------------------------------------===//
//
//                         LDC � the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ctfloat.h"
#include "gen/llvm.h"

using llvm::APFloat;

#if LDC_LLVM_VER >= 400
#define AP_SEMANTICS_PARENS ()
#else
#define AP_SEMANTICS_PARENS
#endif

namespace {

const llvm::fltSemantics *apSemantics = nullptr;

////////////////////////////////////////////////////////////////////////////////

constexpr unsigned numUint64Parts = (sizeof(real_t) + 7) / 8;
union CTFloatUnion {
  real_t fp;
  uint64_t bits[numUint64Parts];
};

////////////////////////////////////////////////////////////////////////////////

unsigned getUnpaddedSizeInBits() {
#if LDC_LLVM_VER >= 307
  return APFloat::getSizeInBits(*apSemantics);
#else
  if (sizeof(real_t) == 8)
    return 64;
#if __i386__ || __x86_64__
  return 80;
#elif __aarch64__
  return 128;
#elif __ppc__ || __ppc64__
  return 128;
#else
  llvm_unreachable("Unknown host real_t type for compile-time reals");
  return sizeof(real_t) * 8;
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////

APFloat parseLiteral(const llvm::fltSemantics &semantics, const char *literal,
                     bool *isOutOfRange = nullptr) {
  APFloat ap(semantics, APFloat::uninitialized);
  const auto r = ap.convertFromString(literal, APFloat::rmNearestTiesToEven);
  if (isOutOfRange) {
    *isOutOfRange = (r & (APFloat::opOverflow | APFloat::opUnderflow)) != 0;
  }
  return ap;
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

void CTFloat::_init() {
  static_assert(sizeof(real_t) >= 8, "real_t < 64 bits?");

  if (sizeof(real_t) == 8) {
    apSemantics = &(APFloat::IEEEdouble AP_SEMANTICS_PARENS);
    return;
  }

#if __i386__ || __x86_64__
  apSemantics = &(APFloat::x87DoubleExtended AP_SEMANTICS_PARENS);
#elif __aarch64__
  apSemantics = &(APFloat::IEEEquad AP_SEMANTICS_PARENS);
#elif __ppc__ || __ppc64__
  apSemantics = &(APFloat::PPCDoubleDouble AP_SEMANTICS_PARENS);
#else
  llvm_unreachable("Unknown host real_t type for compile-time reals");
#endif
}

////////////////////////////////////////////////////////////////////////////////

void CTFloat::toAPFloat(const real_t src, APFloat &dst) {
  if (sizeof(real_t) == 8) {
    dst = APFloat(static_cast<double>(src));
    return;
  }

  CTFloatUnion u;
  u.fp = src;

  const APInt bits = APInt(getUnpaddedSizeInBits(), numUint64Parts, u.bits);

  dst = APFloat(*apSemantics, bits);
}

////////////////////////////////////////////////////////////////////////////////

// implemented in C++ to avoid relying on host D compiler's `real_t.infinity`
bool CTFloat::isInfinity(real_t r) {
  const real_t inf = std::numeric_limits<real_t>::infinity();
  const real_t ninf = -inf;
  const auto unpaddedSize = getUnpaddedSizeInBits() / 8;
  return memcmp(&r, &inf, unpaddedSize) == 0 ||
         memcmp(&r, &ninf, unpaddedSize) == 0;
}

////////////////////////////////////////////////////////////////////////////////

real_t CTFloat::parse(const char *literal, bool *isOutOfRange) {
  const APFloat ap = parseLiteral(*apSemantics, literal, isOutOfRange);
  const APInt bits = ap.bitcastToAPInt();

  CTFloatUnion u;
  memcpy(u.bits, bits.getRawData(), bits.getBitWidth() / 8);
  return u.fp;
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
