//===-- gen/real_t.cpp - Interface of real_t for LDC ------------*- C++ -*-===//
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

#include "globals.h"
#include "gen/real_t.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"
#include <type_traits>

using llvm::APFloat;

static llvm::cl::opt<bool>
    longdouble64("long-double-64",
                 llvm::cl::desc("Choose real to be 64bit double"),
                 llvm::cl::ZeroOrMore, llvm::cl::init(false));

namespace ldc {

/*** SEMANTICS ***/

const llvm::fltSemantics *real_t::semantics = nullptr;

void real_t::_init() {
  const auto &targetTriple = *global.params.targetTriple;
  if (longdouble64 || targetTriple.isWindowsMSVCEnvironment()) {
    semantics = &APFloat::IEEEdouble;
  } else {
    switch (targetTriple.getArch()) {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      semantics = &APFloat::x87DoubleExtended;
      break;
    case llvm::Triple::ppc:
    case llvm::Triple::ppc64:
    case llvm::Triple::ppc64le:
      semantics = &APFloat::PPCDoubleDouble;
      break;
    case llvm::Triple::aarch64:
    case llvm::Triple::aarch64_be:
      semantics = &APFloat::IEEEquad;
      break;
    default:
      semantics = &APFloat::IEEEdouble;
      break;
    }
  }
}



/*** TYPE CONVERSIONS ***/

template <typename T> void real_t::fromInteger(T i) {
  const llvm::integerPart tmp(i);
  value.convertFromSignExtendedInteger(&tmp, 1, std::is_signed<T>::value,
                                       APFloat::rmNearestTiesToEven);
}

template <typename T> T real_t::toInteger() const {
  llvm::APSInt trunc(8 * sizeof(T), !std::is_signed<T>::value);
  bool ignored;
  value.convertToInteger(trunc, APFloat::rmTowardZero, &ignored);
  return static_cast<T>(std::is_signed<T>::value ? trunc.getSExtValue()
                                                 : trunc.getZExtValue());
}

real_t::real_t(float f) : value(f) {
  bool ignored;
  auto status = value.convert(getSemantics(), APFloat::rmNearestTiesToEven, &ignored);
  assert(status == APFloat::opOK);
}

real_t::real_t(double f) : value(f) {
  bool ignored;
  auto status = value.convert(getSemantics(), APFloat::rmNearestTiesToEven, &ignored);
  assert(status == APFloat::opOK);
}

real_t::real_t(int32_t i) : value(getSemantics(), APFloat::uninitialized) {
  fromInteger(i);
}
real_t::real_t(int64_t i) : value(getSemantics(), APFloat::uninitialized) {
  fromInteger(i);
}
real_t::real_t(uint32_t i) : value(getSemantics(), APFloat::uninitialized) {
  fromInteger(i);
}
real_t::real_t(uint64_t i) : value(getSemantics(), APFloat::uninitialized) {
  fromInteger(i);
}

void real_t::initFrom(float f) { new (this) real_t(f); }
void real_t::initFrom(double f) { new (this) real_t(f); }
void real_t::initFrom(int32_t i) { new (this) real_t(i); }
void real_t::initFrom(int64_t i) { new (this) real_t(i); }
void real_t::initFrom(uint32_t i) { new (this) real_t(i); }
void real_t::initFrom(uint64_t i) { new (this) real_t(i); }

bool real_t::toBool() const { return !value.isZero(); }

float real_t::toFloat() const {
  auto trunc = value;
  bool ignored;
  auto status = trunc.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven,
                              &ignored);
  // assert(status == APFloat::opOK);
  return trunc.convertToFloat();
}

double real_t::toDouble() const {
  auto trunc = value;
  bool ignored;
  auto status = trunc.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven,
                              &ignored);
  // assert(status == APFloat::opOK);
  return trunc.convertToDouble();
}

int32_t real_t::toInt32() const { return toInteger<int32_t>(); }
int64_t real_t::toInt64() const { return toInteger<int64_t>(); }
uint32_t real_t::toUInt32() const { return toInteger<uint32_t>(); }
uint64_t real_t::toUInt64() const { return toInteger<uint64_t>(); }



/*** LIFETIME ***/

bool real_t::isInitialized() const { return valueSemantics != nullptr; }

void real_t::safeInit() { new (this) real_t; }

// C++
real_t::real_t(const real_t &r) : real_t() {
  if (r.isInitialized())
    value = r.value;
}
real_t::real_t(real_t &&r) : real_t() {
  if (r.isInitialized())
    value = std::move(r.value);
}

// D
void real_t::postblit() {
  if (!isInitialized()) // leave uninitialized if original was too
    return;

  // this instance is a bitcopy of the right-hand-side
  // 1) save the bitcopy
  alignas(alignof(APFloat)) char rhsBitcopy[sizeof(value)];
  memcpy(rhsBitcopy, &value, sizeof(value));
  // 2) safely initialize this APFloat
  safeInit();
  // 3) assign the bitcopy, i.e., copy the value
  value = *reinterpret_cast<const APFloat *>(rhsBitcopy);
  // 4) don't destruct the bitcopy!
}

// C++
real_t &real_t::operator=(const real_t &r) {
  if (!isInitialized())
    safeInit();
  if (r.isInitialized())
    value = r.value;
  return *this;
}
real_t &real_t::operator=(real_t &&r) {
  moveAssign(r);
  return *this;
}

// D
void real_t::moveAssign(real_t &r) {
  if (!isInitialized())
    safeInit();
  if (r.isInitialized())
    value = std::move(r.value);
}

// C++
real_t::~real_t() { destruct(); }

// D
void real_t::destruct() {
  if (isInitialized())
    value.APFloat::~APFloat();
}



/*** ARITHMETIC OPERATORS ***/

#if LDC_LLVM_VER < 306 // no llvm::APFloat operators
APFloat operator+(const APFloat &l, const APFloat &r) {
  APFloat x = l;
  x.add(r, APFloat::rmNearestTiesToEven);
  return x;
}
APFloat operator-(const APFloat &l, const APFloat &r) {
  APFloat x = l;
  x.subtract(r, APFloat::rmNearestTiesToEven);
  return x;
}
APFloat operator*(const APFloat &l, const APFloat &r) {
  APFloat x = l;
  x.multiply(r, APFloat::rmNearestTiesToEven);
  return x;
}
APFloat operator/(const APFloat &l, const APFloat &r) {
  APFloat x = l;
  x.divide(r, APFloat::rmNearestTiesToEven);
  return x;
}
#endif

real_t real_t::opNeg() const {
  auto tmp = value;
  tmp.changeSign();
  return tmp;
}

real_t real_t::add(const real_t &r) const { return value + r.value; }
real_t real_t::sub(const real_t &r) const { return value - r.value; }
real_t real_t::mul(const real_t &r) const { return value * r.value; }
real_t real_t::div(const real_t &r) const { return value / r.value; }
real_t real_t::mod(const real_t &r) const {
  auto x = value;
#if LDC_LLVM_VER >= 308
  x.mod(r.value);
#else
  x.mod(r.value, APFloat::rmNearestTiesToEven);
#endif
  return x;
}


int real_t::cmp(const real_t &r) const {
  auto res = value.compare(r.value);
  switch (res) {
  case APFloat::cmpLessThan:
    return -1;
  case APFloat::cmpEqual:
    return 0;
  case APFloat::cmpGreaterThan:
    return 1;
  default:
    assert(0);
  }
  return 0;
}

} // namespace ldc
