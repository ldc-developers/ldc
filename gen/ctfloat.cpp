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
    *isOutOfRange = (r & (APFloat::opOverflow | APFloat::opUnderflow)) != 0;
  }
  return ap;
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

void CTFloat::initialize() {
  if (apSemantics)
    return;

#ifdef _MSC_VER
  // MSVC hosts use dmd.root.longdouble (80-bit x87)
  apSemantics = &(APFloat::x87DoubleExtended AP_SEMANTICS_PARENS);
#else
  static_assert(std::numeric_limits<real_t>::is_specialized,
                "real_t is not an arithmetic type");
  constexpr int digits = std::numeric_limits<real_t>::digits;
  if (digits == 53) {
    apSemantics = &(APFloat::IEEEdouble AP_SEMANTICS_PARENS);
  } else if (digits == 64) {
    apSemantics = &(APFloat::x87DoubleExtended AP_SEMANTICS_PARENS);
  } else if (digits == 113) {
    apSemantics = &(APFloat::IEEEquad AP_SEMANTICS_PARENS);
  } else if (digits == 106) {
    apSemantics = &(APFloat::PPCDoubleDouble AP_SEMANTICS_PARENS);
  } else {
    llvm_unreachable("Unknown host real_t type for compile-time reals");
  }
#endif

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

real_t CTFloat::parseFloat(const char *literal, bool *isOutOfRange) {
  const APFloat ap = parseLiteral((APFloat::IEEEsingle AP_SEMANTICS_PARENS),
                                  literal, isOutOfRange);
  return fromAPFloat(ap);
}

real_t CTFloat::parseDouble(const char *literal, bool *isOutOfRange) {
  const APFloat ap = parseLiteral((APFloat::IEEEdouble AP_SEMANTICS_PARENS),
                                  literal, isOutOfRange);
  return fromAPFloat(ap);
}

real_t CTFloat::parseReal(const char *literal, bool *isOutOfRange) {
  const APFloat ap = parseLiteral(*apSemantics, literal, isOutOfRange);
  return fromAPFloat(ap);
}

////////////////////////////////////////////////////////////////////////////////

int CTFloat::sprint(char *str, char fmt, real_t x) {
  assert(fmt == 'g' || fmt == 'a' || fmt == 'A');
  const bool uppercase = fmt == 'A';

  // We try to keep close to C printf and handle a few divergences of the LLVM
  // to-string utility functions.

  if (isNaN(x)) {
    int length = 0;
    if (copysign(one, x) != one) {
      str[0] = '-';
      ++length;
    }
    memcpy(str + length, uppercase ? "NAN" : "nan", 3);
    length += 3;
    str[length] = 0;
    return length;
  }

  if (isInfinity(x)) { // incl. -inf
    int length = 0;
    if (x < 0) {
      str[0] = '-';
      ++length;
    }
    memcpy(str + length, uppercase ? "INF" : "inf", 3);
    length += 3;
    str[length] = 0;
    return length;
  }

  // Use LLVM for printing hex strings.
  if (fmt == 'a' || fmt == 'A') {
    APFloat ap(0.0);
    toAPFloat(x, ap);

    int length =
        ap.convertToHexString(str, 0, uppercase, APFloat::rmNearestTiesToEven);

    // insert a '+' prefix for non-negative exponents (incl. 0) as visual aid
    const char p = uppercase ? 'P' : 'p';
    for (int i = length - 2; i >= 0; --i) {
      if (str[i] == p) {
        if (str[i + 1] != '-' && str[i + 1] != '+') {
          for (int j = length - 1; j > i; --j)
            str[j + 1] = str[j];
          str[i + 1] = '+';
          ++length;
        }

        break;
      }
    }

    str[length] = 0;
    return length;
  }

  assert(fmt == 'g');

  // Use the host C runtime for printing decimal strings;
  // llvm::APFloat::toString() seems not to round correctly, e.g., with LLVM 10:
  // * powl(2.5L, 2.5L) = 9.882117688... => `9.88211` (not 9.88212)
  // * 1e-300L => `9.99999e-301`
#ifdef _MSC_VER
  int length = sprintf(str, "%g", static_cast<double>(x));
#else
  int length = sprintf(str, "%Lg", x);
#endif

  // 1 => 1.0 to distinguish from integers
  bool needsFPSuffix = true;
  for (int i = 0; i < length; ++i) {
    if (str[i] != '-' && !isdigit(str[i])) {
      needsFPSuffix = false;
      break;
    }
  }
  if (needsFPSuffix) {
    memcpy(str + length, ".0", 2);
    length += 2;
  }

  str[length] = 0;
  return length;
}
