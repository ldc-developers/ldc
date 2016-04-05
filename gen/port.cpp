//===-- port.cpp ----------------------------------------------------------===//
//
//                         LDC - the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the DMD OS portability layer.
//
//===----------------------------------------------------------------------===//

#include "port.h"
#include "target.h"
#include "llvm/ADT/APFloat.h"

#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#if __FreeBSD__ && __i386__
#include <ieeefp.h>
#endif

double Port::nan;
longdouble Port::ldbl_nan;
longdouble Port::snan;

double Port::infinity;
longdouble Port::ldbl_infinity;

double Port::dbl_max;
double Port::dbl_min;
longdouble Port::ldbl_max;

bool Port::yl2x_supported = false;
bool Port::yl2xp1_supported = false;

struct PortInitializer {
  PortInitializer();
};

static PortInitializer portinitializer;

PortInitializer::PortInitializer() {
// Derive LLVM APFloat::fltSemantics from native format
#if LDBL_MANT_DIG == 53
#define FLT_SEMANTIC llvm::APFloat::IEEEdouble
#elif LDBL_MANT_DIG == 64
#define FLT_SEMANTIC llvm::APFloat::x87DoubleExtended
#elif LDBL_MANT_DIG == 106
#define FLT_SEMANTIC llvm::APFloat::PPCDoubleDouble
#elif LDBL_MANT_DIG == 113
#define FLT_SEMANTIC llvm::APFloat::IEEEquad
#else
#error "Unsupported native floating point format"
#endif

  Port::nan = *reinterpret_cast<const double *>(
      llvm::APFloat::getNaN(llvm::APFloat::IEEEdouble)
          .bitcastToAPInt()
          .getRawData());
  Port::ldbl_nan = *reinterpret_cast<const long double *>(
      llvm::APFloat::getNaN(FLT_SEMANTIC).bitcastToAPInt().getRawData());
  Port::snan = *reinterpret_cast<const long double *>(
      llvm::APFloat::getSNaN(FLT_SEMANTIC).bitcastToAPInt().getRawData());

#if __FreeBSD__ && __i386__
  // LDBL_MAX comes out as infinity. Fix.
  static unsigned char x[sizeof(longdouble)] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                                0xFF, 0xFF, 0xFF, 0xFE, 0x7F};
  Port::ldbl_max = *(longdouble *)&x[0];
  // FreeBSD defaults to double precision. Switch to extended precision.
  fpsetprec(FP_PE);
#endif

#if __i386 || __x86_64__
  Port::yl2x_supported = true;
  Port::yl2xp1_supported = true;
#endif
}

int Port::isNan(double r) {
#if __APPLE__
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 1080
  return __inline_isnand(r);
#else
  return __inline_isnan(r);
#endif
#elif __HAIKU__ || __FreeBSD__ || __OpenBSD__ || __NetBSD__ || __DragonFly__
  return isnan(r);
#else
#undef isnan
  return std::isnan(r);
#endif
}

int Port::isNan(longdouble r) {
#if __APPLE__
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 1080
  return __inline_isnanl(r);
#else
  return __inline_isnan(r);
#endif
#elif __HAIKU__ || __FreeBSD__ || __OpenBSD__ || __NetBSD__ || __DragonFly__
  return isnan(r);
#else
#undef isnan
  return std::isnan(r);
#endif
}

int Port::isSignallingNan(double r) {
  /* A signalling NaN is a NaN with 0 as the most significant bit of
  * its significand, which is bit 51 of 0..63 for 64 bit doubles.
  */
  return isNan(r) && !((((unsigned char *)&r)[6]) & 8);
}

int Port::isSignallingNan(longdouble r) {
  /* A signalling NaN is a NaN with 0 as the most significant bit of
  * its significand, which is bit 62 of 0..79 for 80 bit reals.
  */
  return isNan(r) && !((((unsigned char *)&r)[7]) & 0x40);
}

int Port::isInfinity(double r) {
#if __APPLE__
  return fpclassify(r) == FP_INFINITE;
#elif __HAIKU__ || __FreeBSD__ || __OpenBSD__ || __NetBSD__ || __DragonFly__
  return isinf(r);
#else
#undef isinf
  return std::isinf(r);
#endif
}

longdouble Port::sqrt(longdouble x) { return std::sqrt(x); }

longdouble Port::fmodl(longdouble x, longdouble y) {
#if __FreeBSD__ && __FreeBSD_version < 800000 || __OpenBSD__ || __NetBSD__ ||  \
    __DragonFly__
  return ::fmod(x, y); // hack for now, fix later
#else
  return std::fmod(x, y);
#endif
}

int Port::fequal(longdouble x, longdouble y) {
  /* In some cases, the REALPAD bytes get garbage in them,
  * so be sure and ignore them.
  */
  return std::memcmp(&x, &y, Target::realsize - Target::realpad) == 0;
}

#if __i386 || __x86_64__
void Port::yl2x_impl(longdouble *x, longdouble *y, longdouble *res) {
  __asm__ volatile("fyl2x" : "=t"(*res) : "u"(*y), "0"(*x) : "st(1)");
}

void Port::yl2xp1_impl(longdouble *x, longdouble *y, longdouble *res) {
  __asm__ volatile("fyl2xp1" : "=t"(*res) : "u"(*y), "0"(*x) : "st(1)");
}
#else
void Port::yl2x_impl(longdouble *x, longdouble *y, longdouble *res) {
  assert(0);
}

void Port::yl2xp1_impl(longdouble *x, longdouble *y, longdouble *res) {
  assert(0);
}
#endif

int Port::memicmp(const char *s1, const char *s2, int n) {
#if HAVE_MEMICMP
  return ::memicmp(s1, s2, n);
#else
  int result = 0;

  for (int i = 0; i < n; i++) {
    char c1 = s1[i];
    char c2 = s2[i];

    result = c1 - c2;
    if (result) {
      result = std::toupper(c1) - std::toupper(c2);
      if (result)
        break;
    }
  }
  return result;
#endif
}

int Port::stricmp(const char *s1, const char *s2) {
#if HAVE_STRICMP
  return ::stricmp(s1, s2);
#else
  int result = 0;

  for (;;) {
    char c1 = *s1;
    char c2 = *s2;

    result = c1 - c2;
    if (result) {
      result = std::toupper(c1) - std::toupper(c2);
      if (result)
        break;
    }
    if (!c1)
      break;
    s1++;
    s2++;
  }
  return result;
#endif
}

char *Port::strupr(char *s) {
#if HAVE_STRUPR
  return ::strupr(s);
#else
  char *t = s;

  while (*s) {
    *s = std::toupper(*s);
    s++;
  }

  return t;
#endif
}

float Port::strtof(const char *p, char **endp) {
#if HAVE_STRTOF
  return std::strtof(p, endp);
#else
  return static_cast<float>(std::strtod(p, endp));
#endif
}

double Port::strtod(const char *p, char **endp) { return std::strtod(p, endp); }

longdouble Port::strtold(const char *p, char **endp) {
  return std::strtold(p, endp);
}

// Little endian
void Port::writelongLE(unsigned value, void *buffer) {
  unsigned char *p = (unsigned char *)buffer;
  p[3] = (unsigned char)(value >> 24);
  p[2] = (unsigned char)(value >> 16);
  p[1] = (unsigned char)(value >> 8);
  p[0] = (unsigned char)(value);
}

// Little endian
unsigned Port::readlongLE(void *buffer) {
  unsigned char *p = (unsigned char *)buffer;
  return (((((p[3] << 8) | p[2]) << 8) | p[1]) << 8) | p[0];
}

// Big endian
void Port::writelongBE(unsigned value, void *buffer) {
  unsigned char *p = (unsigned char *)buffer;
  p[0] = (unsigned char)(value >> 24);
  p[1] = (unsigned char)(value >> 16);
  p[2] = (unsigned char)(value >> 8);
  p[3] = (unsigned char)(value);
}

// Big endian
unsigned Port::readlongBE(void *buffer) {
  unsigned char *p = (unsigned char *)buffer;
  return (((((p[0] << 8) | p[1]) << 8) | p[2]) << 8) | p[3];
}

// Little endian
unsigned Port::readwordLE(void *buffer) {
  unsigned char *p = (unsigned char *)buffer;
  return (p[1] << 8) | p[0];
}

// Big endian
unsigned Port::readwordBE(void *buffer) {
  unsigned char *p = (unsigned char *)buffer;
  return (p[0] << 8) | p[1];
}
