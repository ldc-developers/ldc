//===-- gen/real_t.h - Interface of real_t for LDC --------------*- C++ -*-===//
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

#ifndef LDC_GEN_REAL_T_H
#define LDC_GEN_REAL_T_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringRef.h"

struct CTFloat;

namespace ldc {

struct real_t {
  friend CTFloat;

  static const llvm::fltSemantics &getSemantics() { return *semantics; }

  static real_t nan() { return llvm::APFloat::getNaN(getSemantics()); }
  static real_t snan() { return llvm::APFloat::getSNaN(getSemantics()); }
  static real_t infinity() { return llvm::APFloat::getInf(getSemantics()); }

  static void _init();

  // implicit type conversions
  real_t(float f);
  real_t(double f);
  real_t(int32_t i);
  real_t(int64_t i);
  real_t(uint32_t i);
  real_t(uint64_t i);

  // type conversions
  bool toBool() const;
  float toFloat() const;
  double toDouble() const;
  int32_t toInt32() const;
  int64_t toInt64() const;
  uint32_t toUInt32() const;
  uint64_t toUInt64() const;

  // C++ cast operators
  operator bool() const { return toBool(); }
  operator float() const { return toFloat(); }
  operator double() const { return toDouble(); }
  operator int32_t() const { return toInt32(); }
  operator int64_t() const { return toInt64(); }
  operator uint32_t() const { return toUInt32(); }
  operator uint64_t() const { return toUInt64(); }
  operator const llvm::APFloat &() const { return value; }

  // arithmetic operators
  real_t opNeg() const;
  real_t add(const real_t &r) const;
  real_t sub(const real_t &r) const;
  real_t mul(const real_t &r) const;
  real_t div(const real_t &r) const;
  real_t mod(const real_t &r) const;

  // comparison
  int cmp(const real_t &r) const;

  void safeInit();

private:
  static const llvm::fltSemantics *semantics;

  template <typename T> void fromInteger(T i);
  template <typename T> T toInteger() const;

  llvm::APFloat value;

  real_t() : value(llvm::APFloat::Bogus, llvm::APFloat::uninitialized) {}
  real_t(const llvm::APFloat &value) : value(value) {}
  real_t(llvm::APFloat &&value) : value(std::move(value)) {}

  void initFrom(float f);
  void initFrom(double f);
  void initFrom(int32_t i);
  void initFrom(int64_t i);
  void initFrom(uint32_t i);
  void initFrom(uint64_t i);
  void postblit();
  void moveAssign(real_t &r);
  void destruct();
};

// C++ arithmetic and comparison operators:
inline real_t operator-(const real_t &x) { return x.opNeg(); }
inline real_t operator+(const real_t &l, const real_t &r) { return l.add(r); }
inline real_t operator-(const real_t &l, const real_t &r) { return l.sub(r); }
inline real_t operator*(const real_t &l, const real_t &r) { return l.mul(r); }
inline real_t operator/(const real_t &l, const real_t &r) { return l.div(r); }
inline real_t operator%(const real_t &l, const real_t &r) { return l.mod(r); }

inline real_t &operator+=(real_t &l, const real_t &r) { return l = l + r; }
inline real_t &operator-=(real_t &l, const real_t &r) { return l = l - r; }
inline real_t &operator*=(real_t &l, const real_t &r) { return l = l * r; }
inline real_t &operator/=(real_t &l, const real_t &r) { return l = l / r; }

inline bool operator<(const real_t &l, const real_t &r) {
  return l.cmp(r) == -1;
}
inline bool operator<=(const real_t &l, const real_t &r) {
  return l.cmp(r) <= 0;
}
inline bool operator>(const real_t &l, const real_t &r) {
  return l.cmp(r) == 1;
}
inline bool operator>=(const real_t &l, const real_t &r) {
  return l.cmp(r) >= 0;
}
inline bool operator==(const real_t &l, const real_t &r) {
  return l.cmp(r) == 0;
}
inline bool operator!=(const real_t &l, const real_t &r) {
  return l.cmp(r) != 0;
}

} // namespace ldc

#endif
