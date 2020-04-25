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

////////////////////////////////////////////////////////////////////////////////

int CTFloat::sprint(char *str, char fmt, real_t x) {
  assert(fmt == 'g' || fmt == 'a' || fmt == 'A');
  const bool uppercase = fmt == 'A';

  APFloat ap(0.0);
  toAPFloat(x, ap);

  // We try to keep close to C printf and handle a few divergences of the LLVM
  // to-string utility functions.

  int length = 0;
  if (isNaN(x)) {
    if (copysign(one, x) != one) {
      str[0] = '-';
      ++length;
    }
    memcpy(str + length, uppercase ? "NAN" : "nan", 3);
    length += 3;
  } else if (isInfinity(x)) {
    if (!isIdentical(x, infinity)) {
      str[0] = '-';
      ++length;
    }
    memcpy(str + length, uppercase ? "INF" : "inf", 3);
    length += 3;
  } else if (fmt == 'a' || fmt == 'A') {
    length =
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
  } else {
    assert(fmt == 'g');

    llvm::SmallString<32> buffer;
    ap.toString(buffer, /*FormatPrecision=*/6, /*FormatMaxPadding=*/4);

    bool needsFPSuffix = true;
    ptrdiff_t commaIndex = -1;
    for (size_t i = 0; i < buffer.size(); ++i) {
      if (buffer[i] == '.') {
        needsFPSuffix = false;
        commaIndex = i;
      } else if (buffer[i] == 'E') {
        needsFPSuffix = false;

        if (!uppercase)
            buffer[i] = 'e';

        // remove excessive comma+zeros in scientific notation
        // (1.000000e+100 => 1e+100)
        if (commaIndex > 0) {
          bool onlyZeros = std::all_of(&buffer[commaIndex + 1], &buffer[i],
                                       [](char c) { return c == '0'; });
          if (onlyZeros) {
            buffer.erase(&buffer[commaIndex], &buffer[i]);
            i = commaIndex;
            commaIndex = -1;
          }
        }

        // ensure at least 2 digits for the exponent after the sign
        // (1e+6 => 1e+06)
        if (buffer.size() - (i + 2) == 1)
          buffer.insert(&buffer[i + 2], '0');

        break;
      }
    }

    // no comma for non-scientific notation? then add FP suffix to distinguish
    // from integers (100 => 100.0)
    if (needsFPSuffix)
      buffer += ".0";

    length = static_cast<int>(buffer.size());
    memcpy(str, buffer.data(), length);
  }

  str[length] = 0; // null-terminate
  return length;
}
