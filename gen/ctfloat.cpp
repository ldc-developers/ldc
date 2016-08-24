//===-- gen/ctfloat.cpp -  CTFloat implementation for LDC -------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Front-end compile-time floating-point implementation for LDC.
//
//===----------------------------------------------------------------------===//

#include "root/ctfloat.h"
#include "llvm/ADT/SmallString.h"
#include <cmath>

using llvm::APFloat;

namespace {

real_t fromHostReal(long double x) {
  char buffer[64];
  sprintf(buffer, "%La", x);
  return CTFloat::parse(buffer);
}

long double toHostReal(const real_t &x) {
  char buffer[64];
  CTFloat::sprintImpl(buffer, 'a', x);
  return strtold(buffer, nullptr);
}

}

real_t CTFloat::sinImpl(const real_t &x) {
  if (&real_t::getSemantics() == &APFloat::IEEEdouble)
    return std::sin(x.toDouble());
  return fromHostReal(std::sin(toHostReal(x)));
}

real_t CTFloat::cosImpl(const real_t &x) {
  if (&real_t::getSemantics() == &APFloat::IEEEdouble)
    return std::cos(x.toDouble());
  return fromHostReal(std::cos(toHostReal(x)));
}

real_t CTFloat::tanImpl(const real_t &x) {
  if (&real_t::getSemantics() == &APFloat::IEEEdouble)
    return std::tan(x.toDouble());
  return fromHostReal(std::tan(toHostReal(x)));
}

real_t CTFloat::sqrtImpl(const real_t &x) {
  if (&real_t::getSemantics() == &APFloat::IEEEdouble)
    return std::sqrt(x.toDouble());
  return fromHostReal(std::sqrt(toHostReal(x)));
}

real_t CTFloat::fabsImpl(const real_t &x) {
  if (x.value.isNegative()) {
    auto f = x.value;
    f.changeSign();
    return f;
  }
  return x;
}

// additional LDC built-ins:
real_t CTFloat::logImpl(const real_t &x) {
  if (&real_t::getSemantics() == &APFloat::IEEEdouble)
    return std::log(x.toDouble());
  return fromHostReal(std::log(toHostReal(x)));
}

real_t CTFloat::fminImpl(const real_t &l, const real_t &r) {
  if (&real_t::getSemantics() == &APFloat::IEEEdouble)
    return std::fmin(l.toDouble(), r.toDouble());
  return fromHostReal(std::fmin(toHostReal(l), toHostReal(r)));
}

real_t CTFloat::fmaxImpl(const real_t &l, const real_t &r) {
  if (&real_t::getSemantics() == &APFloat::IEEEdouble)
    return std::fmax(l.toDouble(), r.toDouble());
  return fromHostReal(std::fmax(toHostReal(l), toHostReal(r)));
}

real_t CTFloat::floorImpl(const real_t &x) {
  if (&real_t::getSemantics() == &APFloat::IEEEdouble)
    return std::floor(x.toDouble());
  return fromHostReal(std::floor(toHostReal(x)));
}

real_t CTFloat::ceilImpl(const real_t &x) {
  if (&real_t::getSemantics() == &APFloat::IEEEdouble)
    return std::ceil(x.toDouble());
  return fromHostReal(std::ceil(toHostReal(x)));
}

real_t CTFloat::truncImpl(const real_t &x) {
  if (&real_t::getSemantics() == &APFloat::IEEEdouble)
    return std::trunc(x.toDouble());
  return fromHostReal(std::trunc(toHostReal(x)));
}

real_t CTFloat::roundImpl(const real_t &x) {
  if (&real_t::getSemantics() == &APFloat::IEEEdouble)
    return std::round(x.toDouble());
  return fromHostReal(std::round(toHostReal(x)));
}

bool CTFloat::isIdenticalImpl(const real_t &a, const real_t &b) {
  return a.value.bitwiseIsEqual(b.value);
}
bool CTFloat::isNaNImpl(const real_t &r) { return r.value.isNaN(); }
bool CTFloat::isSNaNImpl(const real_t &r) { return r.value.isSignaling(); }
bool CTFloat::isInfinityImpl(const real_t &r) { return r.value.isInfinity(); }

real_t CTFloat::parse(const char *literal, bool *isOutOfRange) {
  APFloat x(real_t::getSemantics(), APFloat::uninitialized);
  auto status = x.convertFromString(literal, APFloat::rmNearestTiesToEven);
  if (isOutOfRange) {
    *isOutOfRange =
        (status == APFloat::opOverflow || status == APFloat::opUnderflow);
  }
  return x;
}

int CTFloat::sprintImpl(char *str, char fmt, const real_t &x) {
  // The signature of this method leads to buffer overflows.
  if (fmt == 'a' || fmt == 'A') {
    return x.value.convertToHexString(str, 0, fmt == 'A',
                                      APFloat::rmNearestTiesToEven);
  }

  assert(fmt == 'g');
  llvm::SmallString<64> buf;
  // printf's default precision is 6 digits
  constexpr int precision = 6;
  x.value.toString(buf, precision);

  // post-processing for printf compatibility
  if (x.value.isFinite()) {
    int exponentIndex = -1;
    bool hasDecimalPoint = false;
    for (size_t i = 0; i < buf.size(); ++i) {
      auto &c = buf[i];
      if (c == 'E') {
        c = 'e'; // lower case
        exponentIndex = int(i);
      } else if (c == '.') {
        hasDecimalPoint = true;
      }
    }

    if (exponentIndex != -1) {
      // printf prints at least 2 exponent digits
      // "1.5e+3" => "1.5e+03"
      int firstExponentDigitIndex = exponentIndex + 2; // past "e±"
      int numExponentDigits = buf.size() - firstExponentDigitIndex;
      if (numExponentDigits == 1) {
        auto digit = buf.back();
        buf.back() = '0';
        buf.push_back(digit);
      }
    } else if (!hasDecimalPoint) {
      // LLVM may print the number as integer without ".…" suffix
      // "1" => "1.00000", "10" => "10.0000"
      int numDigits = buf.size() - (buf[0] == '-' ? 1 : 0);
      buf.push_back('.');
      if (numDigits < precision)
        buf.append(precision - numDigits, '0');
    }
  }

  strcpy(str, buf.str().str().c_str());
  return buf.size();
}
