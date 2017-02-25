
/* Copyright (c) 1999-2014 by Digital Mars
 * All Rights Reserved, written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 * https://github.com/D-Programming-Language/dmd/blob/master/src/root/port.c
 */

#include "port.h"

#if __DMC__
#include <math.h>
#include <float.h>
#include <fp.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

double Port::nan = NAN;
longdouble Port::ldbl_nan = NAN;
longdouble Port::snan;

double Port::infinity = INFINITY;
longdouble Port::ldbl_infinity = INFINITY;

double Port::dbl_max = DBL_MAX;
double Port::dbl_min = DBL_MIN;
longdouble Port::ldbl_max = LDBL_MAX;

bool Port::yl2x_supported = true;
bool Port::yl2xp1_supported = true;

struct PortInitializer
{
    PortInitializer();
};

static PortInitializer portinitializer;

PortInitializer::PortInitializer()
{
    union
    {   unsigned int ui[4];
        longdouble     ld;
    } snan = {{ 0, 0xA0000000, 0x7FFF, 0 }};

    Port::snan = snan.ld;
}

int Port::isNan(double r)
{
    return ::isnan(r);
}

int Port::isNan(longdouble r)
{
    return ::isnan(r);
}

int Port::isSignallingNan(double r)
{
    /* A signalling NaN is a NaN with 0 as the most significant bit of
     * its significand, which is bit 51 of 0..63 for 64 bit doubles.
     */
    return isNan(r) && !((((unsigned char*)&r)[6]) & 8);
}

int Port::isSignallingNan(longdouble r)
{
    /* A signalling NaN is a NaN with 0 as the most significant bit of
     * its significand, which is bit 62 of 0..79 for 80 bit reals.
     */
    return isNan(r) && !((((unsigned char*)&r)[7]) & 0x40);
}

int Port::isInfinity(double r)
{
    return (::fpclassify(r) == FP_INFINITE);
}

longdouble Port::sqrt(longdouble x)
{
    return ::sqrtl(x);
}

longdouble Port::fmodl(longdouble x, longdouble y)
{
    return ::fmodl(x, y);
}

int Port::fequal(longdouble x, longdouble y)
{
    /* In some cases, the REALPAD bytes get garbage in them,
     * so be sure and ignore them.
     */
    return memcmp(&x, &y, 10) == 0;
}

void Port::yl2x_impl(longdouble* x, longdouble* y, longdouble* res)
{
    *res = _inline_yl2x(*x, *y);
}

void Port::yl2xp1_impl(longdouble* x, longdouble* y, longdouble* res)
{
    *res = _inline_yl2xp1(*x, *y);
}

char *Port::strupr(char *s)
{
    return ::strupr(s);
}

int Port::memicmp(const char *s1, const char *s2, int n)
{
    return ::memicmp(s1, s2, n);
}

int Port::stricmp(const char *s1, const char *s2)
{
    return ::stricmp(s1, s2);
}


extern "C" const char * __cdecl __locale_decpoint;

float Port::strtof(const char *buffer, char **endp)
{
    const char *save = __locale_decpoint;
    __locale_decpoint = ".";
    float result = ::strtof(buffer, endp);
    __locale_decpoint = save;
    return result;
}

double Port::strtod(const char *buffer, char **endp)
{
    const char *save = __locale_decpoint;
    __locale_decpoint = ".";
    double result = ::strtod(buffer, endp);
    __locale_decpoint = save;
    return result;
}

longdouble Port::strtold(const char *buffer, char **endp)
{
    const char *save = __locale_decpoint;
    __locale_decpoint = ".";
    longdouble result = ::strtold(buffer, endp);
    __locale_decpoint = save;
    return result;
}

#endif

#if _MSC_VER

// Disable useless warnings about unreferenced functions
#pragma warning (disable : 4514)

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/StringRef.h>
#include <math.h>
#include <float.h>  // for _isnan
#include <time.h>
#include <errno.h>
#include <string.h>
#include <ctype.h>
#include <wchar.h>
#include <stdlib.h>
#include <limits> // for std::numeric_limits
#include "target.h"

#if IN_LLVM
#include "llvm/Support/ErrorHandling.h"
#endif

double Port::nan;
longdouble Port::ldbl_nan;
longdouble Port::snan;

double Port::infinity;
longdouble Port::ldbl_infinity;

double Port::dbl_max = DBL_MAX;
double Port::dbl_min = DBL_MIN;
longdouble Port::ldbl_max = LDBL_MAX;

#if IN_LLVM
bool Port::yl2x_supported = false;
bool Port::yl2xp1_supported = false;
#else
#if _M_IX86 || _M_X64
bool Port::yl2x_supported = true;
bool Port::yl2xp1_supported = true;
#else
bool Port::yl2x_supported = false;
bool Port::yl2xp1_supported = false;
#endif
#endif

struct PortInitializer
{
    PortInitializer();
};

static PortInitializer portinitializer;

PortInitializer::PortInitializer()
{
#if IN_LLVM
    union {
        unsigned long ul[2];
        double d;
    }
    nan = { { 0, 0x7FF80000 } },
    snan = { { 0, 0x7FFC0000 } },
    inf = { { 0, 0x7FF00000 } };

    Port::nan = nan.d;
    Port::ldbl_nan = nan.d;
    Port::snan = snan.d;
    Port::infinity = inf.d;
    Port::ldbl_infinity = inf.d;
#else
    union {
        unsigned long ul[2];
        double d;
    } nan = { { 0, 0x7FF80000 } };

    Port::nan = nan.d;
    Port::ldbl_nan = ld_qnan;
    Port::snan = ld_snan;
    Port::infinity = std::numeric_limits<double>::infinity();
    Port::ldbl_infinity = ld_inf;
#endif

    _set_abort_behavior(_WRITE_ABORT_MSG, _WRITE_ABORT_MSG | _CALL_REPORTFAULT); // disable crash report
}

int Port::isNan(double r)
{
#if _MSC_VER >= 1900
    return ::isnan(r);
#else
    return ::_isnan(r);
#endif
}

int Port::isNan(longdouble r)
{
#if IN_LLVM
    return ::isnan(r);
#else
    return ::_isnan(r);
#endif
}

int Port::isSignallingNan(double r)
{
    /* A signalling NaN is a NaN with 0 as the most significant bit of
     * its significand, which is bit 51 of 0..63 for 64 bit doubles.
     */
    return isNan(r) && !((((unsigned char*)&r)[6]) & 8);
}

int Port::isSignallingNan(longdouble r)
{
    /* MSVC doesn't have 80 bit long doubles
     */
    return isSignallingNan((double) r);
}

int Port::isInfinity(double r)
{
#if _MSC_VER >= 1900
    return ::isinf(r);
#else
    return (::_fpclass(r) & (_FPCLASS_NINF | _FPCLASS_PINF));
#endif
}

longdouble Port::sqrt(longdouble x)
{
    return ::sqrtl(x);
}

longdouble Port::fmodl(longdouble x, longdouble y)
{
    return ::fmodl(x, y);
}

int Port::fequal(longdouble x, longdouble y)
{
    /* In some cases, the REALPAD bytes get garbage in them,
     * so be sure and ignore them.
     */
    return memcmp(&x, &y, Target::realsize - Target::realpad) == 0;
}

#if IN_LLVM
void Port::yl2x_impl(longdouble* x, longdouble* y, longdouble* res)
{
    llvm_unreachable("Port::yl2x_impl");
}

void Port::yl2xp1_impl(longdouble* x, longdouble* y, longdouble* res)
{
    llvm_unreachable("Port::yl2xp1_impl");
}
#else
#if _M_IX86
void Port::yl2x_impl(longdouble* x, longdouble* y, longdouble* res)
{
    __asm
    {
        mov eax, y
        mov ebx, x
        mov ecx, res
        fld tbyte ptr [eax]
        fld tbyte ptr [ebx]
        fyl2x
        fstp tbyte ptr [ecx]
    }
}

void Port::yl2xp1_impl(longdouble* x, longdouble* y, longdouble* res)
{
    __asm
    {
        mov eax, y
        mov ebx, x
        mov ecx, res
        fld tbyte ptr [eax]
        fld tbyte ptr [ebx]
        fyl2xp1
        fstp tbyte ptr [ecx]
    }
}
#elif _M_X64

//defined in ldfpu.asm
extern "C"
{
    void ld_yl2x(longdouble *x, longdouble *y, longdouble *r);
    void ld_yl2xp1(longdouble *x, longdouble *y, longdouble *r);
}

void Port::yl2x_impl(longdouble* x, longdouble* y, longdouble* res)
{
    ld_yl2x(x, y, res);
}

void Port::yl2xp1_impl(longdouble* x, longdouble* y, longdouble* res)
{
    ld_yl2xp1(x, y, res);
}
#else

void Port::yl2x_impl(longdouble* x, longdouble* y, longdouble* res)
{
    assert(0);
}

void Port::yl2xp1_impl(longdouble* x, longdouble* y, longdouble* res)
{
    assert(0);
}

#endif
#endif

char *Port::strupr(char *s)
{
    return ::strupr(s);
}

int Port::memicmp(const char *s1, const char *s2, int n)
{
    return ::memicmp(s1, s2, n);
}

int Port::stricmp(const char *s1, const char *s2)
{
    return ::stricmp(s1, s2);
}

float Port::strtof(const char *p, char **endp)
{
#if _MSC_VER >= 1900
    return ::strtof(p, endp); // C99 conformant since VS 2015
#else
    if(endp)
        return static_cast<float>(::strtod(p, endp)); // does not set errno for underflows, but unused

    _CRT_FLOAT flt;
    int res = _atoflt(&flt, (char*)p);
    if (res == _UNDERFLOW)
        errno = ERANGE;
    return flt.f;
#endif
}

double Port::strtod(const char *p, char **endp)
{
#if _MSC_VER >= 1900
    return ::strtod(p, endp); // C99 conformant since VS 2015
#else
    if(endp)
        return ::strtod(p, endp); // does not set errno for underflows, but unused

    _CRT_DOUBLE dbl;
    int res = _atodbl(&dbl, const_cast<char*> (p));
    if (res == _UNDERFLOW)
        errno = ERANGE;
    return dbl.x;
#endif
}

// from backend/strtold.c, renamed to avoid clash with decl in stdlib.h
longdouble strtold_dm(const char *p,char **endp);

longdouble Port::strtold(const char *p, char **endp)
{
#if IN_LLVM
#if _MSC_VER >= 1900
    return ::strtold(p, endp); // C99 conformant since VS 2015
#else
    // MSVC strtold() before VS 2015 does not support hex float strings. Just
    // use the function provided by LLVM because we going to use it anyway.
    llvm::APFloat val(llvm::APFloat::IEEEdouble, llvm::APFloat::uninitialized);
    val.convertFromString(llvm::StringRef(p), llvm::APFloat::rmNearestTiesToEven);
    return val.convertToDouble();
#endif
#else
    return ::strtold_dm(p, endp);
#endif
}

#endif

#if __MINGW32__

#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <wchar.h>
#include <float.h>
#include <assert.h>

double Port::nan;
longdouble Port::ldbl_nan;
longdouble Port::snan;

static double zero = 0;
double Port::infinity = 1 / zero;
longdouble Port::ldbl_infinity = 1 / zero;

double Port::dbl_max = 1.7976931348623157e308;
double Port::dbl_min = 5e-324;
longdouble Port::ldbl_max = LDBL_MAX;

#if _X86_ || __x86_64__
bool Port::yl2x_supported = true;
bool Port::yl2xp1_supported = true;
#else
bool Port::yl2x_supported = false;
bool Port::yl2xp1_supported = false;
#endif

struct PortInitializer
{
    PortInitializer();
};

static PortInitializer portinitializer;

PortInitializer::PortInitializer()
{
    union
    {   unsigned int ui[2];
        double d;
    } nan = {{ 0, 0x7FF80000 }};

    Port::nan = nan.d;
    assert(!signbit(Port::nan));

    union
    {   unsigned int ui[4];
        longdouble ld;
    } ldbl_nan = {{ 0, 0xC0000000, 0x7FFF, 0}};

    Port::ldbl_nan = ldbl_nan.ld;
    assert(!signbit(Port::ldbl_nan));

    union
    {   unsigned int ui[4];
        longdouble     ld;
    } snan = {{ 0, 0xA0000000, 0x7FFF, 0 }};

    Port::snan = snan.ld;
}

int Port::isNan(double r)
{
    return isnan(r);
}

int Port::isNan(longdouble r)
{
    return isnan(r);
}

int Port::isSignallingNan(double r)
{
    /* A signalling NaN is a NaN with 0 as the most significant bit of
     * its significand, which is bit 51 of 0..63 for 64 bit doubles.
     */
    return isNan(r) && !((((unsigned char*)&r)[6]) & 8);
}

int Port::isSignallingNan(longdouble r)
{
    /* A signalling NaN is a NaN with 0 as the most significant bit of
     * its significand, which is bit 62 of 0..79 for 80 bit reals.
     */
    return isNan(r) && !((((unsigned char*)&r)[7]) & 0x40);
}

int Port::isInfinity(double r)
{
    return isinf(r);
}

longdouble Port::sqrt(longdouble x)
{
    return ::sqrtl(x);
}

longdouble Port::fmodl(longdouble x, longdouble y)
{
    return ::fmodl(x, y);
}

int Port::fequal(longdouble x, longdouble y)
{
    /* In some cases, the REALPAD bytes get garbage in them,
     * so be sure and ignore them.
     */
    return memcmp(&x, &y, 10) == 0;
}

#if _X86_ || __x86_64__
void Port::yl2x_impl(longdouble* x, longdouble* y, longdouble* res)
{
    __asm__ volatile("fyl2x": "=t" (*res): "u" (*y), "0" (*x) : "st(1)" );
}

void Port::yl2xp1_impl(longdouble* x, longdouble* y, longdouble* res)
{
    __asm__ volatile("fyl2xp1": "=t" (*res): "u" (*y), "0" (*x) : "st(1)" );
}
#else
void Port::yl2x_impl(longdouble* x, longdouble* y, longdouble* res)
{
    assert(0);
}

void Port::yl2xp1_impl(longdouble* x, longdouble* y, longdouble* res)
{
    assert(0);
}
#endif

char *Port::strupr(char *s)
{
    char *t = s;

    while (*s)
    {
        *s = toupper(*s);
        s++;
    }

    return t;
}

int Port::memicmp(const char *s1, const char *s2, int n)
{
    int result = 0;

    for (int i = 0; i < n; i++)
    {   char c1 = s1[i];
        char c2 = s2[i];

        result = c1 - c2;
        if (result)
        {
            result = toupper(c1) - toupper(c2);
            if (result)
                break;
        }
    }
    return result;
}

int Port::stricmp(const char *s1, const char *s2)
{
    int result = 0;

    for (;;)
    {   char c1 = *s1;
        char c2 = *s2;

        result = c1 - c2;
        if (result)
        {
            result = toupper(c1) - toupper(c2);
            if (result)
                break;
        }
        if (!c1)
            break;
        s1++;
        s2++;
    }
    return result;
}

float Port::strtof(const char *p, char **endp)
{
    return ::strtof(p, endp);
}

double Port::strtod(const char *p, char **endp)
{
    return ::strtod(p, endp);
}

longdouble Port::strtold(const char *p, char **endp)
{
    return ::__mingw_strtold(p, endp);
}

#endif

#if __linux__ || __APPLE__ || __FreeBSD__ || __OpenBSD__ || __NetBSD__ || __DragonFly__ || __HAIKU__

#include <math.h>
#if __FreeBSD__ && __i386__
#include <ieeefp.h>
#endif
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <wchar.h>
#include <float.h>
#include <assert.h>
#include "target.h"

#if IN_LLVM
#include "llvm/ADT/APFloat.h"
#endif

double Port::nan;
longdouble Port::ldbl_nan;
longdouble Port::snan;

static double zero = 0;
double Port::infinity = 1 / zero;
longdouble Port::ldbl_infinity = 1 / zero;

double Port::dbl_max = 1.7976931348623157e308;
double Port::dbl_min = 5e-324;
longdouble Port::ldbl_max = LDBL_MAX;

#if __i386 || __x86_64__
bool Port::yl2x_supported = true;
bool Port::yl2xp1_supported = true;
#else
bool Port::yl2x_supported = false;
bool Port::yl2xp1_supported = false;
#endif

struct PortInitializer
{
    PortInitializer();
};

static PortInitializer portinitializer;

PortInitializer::PortInitializer()
{
#if IN_LLVM

#if LDC_LLVM_VER >= 400
  const auto &IEEEdouble = llvm::APFloat::IEEEdouble();
  const auto &x87DoubleExtended = llvm::APFloat::x87DoubleExtended();
  const auto &PPCDoubleDouble = llvm::APFloat::PPCDoubleDouble();
  const auto &IEEEquad = llvm::APFloat::IEEEquad();
#else
  const auto &IEEEdouble = llvm::APFloat::IEEEdouble;
  const auto &x87DoubleExtended = llvm::APFloat::x87DoubleExtended;
  const auto &PPCDoubleDouble = llvm::APFloat::PPCDoubleDouble;
  const auto &IEEEquad = llvm::APFloat::IEEEquad;
#endif

// Derive LLVM APFloat::fltSemantics from native format
#if LDBL_MANT_DIG == 53
#define FLT_SEMANTIC IEEEdouble
#elif LDBL_MANT_DIG == 64
#define FLT_SEMANTIC x87DoubleExtended
#elif LDBL_MANT_DIG == 106
#define FLT_SEMANTIC PPCDoubleDouble
#elif LDBL_MANT_DIG == 113
#define FLT_SEMANTIC IEEEquad
#else
#error "Unsupported native floating point format"
#endif

    Port::nan = *reinterpret_cast<const double*>(llvm::APFloat::getNaN(IEEEdouble).bitcastToAPInt().getRawData());
    Port::ldbl_nan = *reinterpret_cast<const long double*>(llvm::APFloat::getNaN(FLT_SEMANTIC).bitcastToAPInt().getRawData());
    Port::snan = *reinterpret_cast<const long double*>(llvm::APFloat::getSNaN(FLT_SEMANTIC).bitcastToAPInt().getRawData());

#else
    union
    {   unsigned int ui[2];
        double d;
    } nan = {{ 0, 0x7FF80000 }};
    Port::nan = nan.d;
    assert(!signbit(Port::nan));

    union
    {   unsigned int ui[4];
        longdouble ld;
    } ldbl_nan = {{ 0, 0xC0000000, 0x7FFF, 0}};
    Port::ldbl_nan = ldbl_nan.ld;

    assert(!signbit(Port::ldbl_nan));

    union
    {   unsigned int ui[4];
        longdouble     ld;
    } snan = {{ 0, 0xA0000000, 0x7FFF, 0 }};
    Port::snan = snan.ld;
#endif

#if __FreeBSD__ && __i386__
    // LDBL_MAX comes out as infinity. Fix.
    static unsigned char x[sizeof(longdouble)] =
        { 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,0x7F };
    Port::ldbl_max = *(longdouble *)&x[0];
    // FreeBSD defaults to double precision. Switch to extended precision.
    fpsetprec(FP_PE);
#endif
}

int Port::isNan(double r)
{
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

int Port::isNan(longdouble r)
{
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

int Port::isSignallingNan(double r)
{
    /* A signalling NaN is a NaN with 0 as the most significant bit of
     * its significand, which is bit 51 of 0..63 for 64 bit doubles.
     */
    return isNan(r) && !((((unsigned char*)&r)[6]) & 8);
}

int Port::isSignallingNan(longdouble r)
{
    /* A signalling NaN is a NaN with 0 as the most significant bit of
     * its significand, which is bit 62 of 0..79 for 80 bit reals.
     */
    return isNan(r) && !((((unsigned char*)&r)[7]) & 0x40);
}

int Port::isInfinity(double r)
{
#if __APPLE__
    return fpclassify(r) == FP_INFINITE;
#elif __HAIKU__ || __FreeBSD__ || __OpenBSD__ || __NetBSD__ ||  __DragonFly__
    return isinf(r);
#else
    #undef isinf
    return std::isinf(r);
#endif
}

longdouble Port::sqrt(longdouble x)
{
    return std::sqrt(x);
}

longdouble Port::fmodl(longdouble x, longdouble y)
{
#if __FreeBSD__ && __FreeBSD_version < 800000 || __OpenBSD__ || __NetBSD__ || __DragonFly__
    return ::fmod(x, y);        // hack for now, fix later
#else
    return std::fmod(x, y);
#endif
}

int Port::fequal(longdouble x, longdouble y)
{
    /* In some cases, the REALPAD bytes get garbage in them,
     * so be sure and ignore them.
     */
    return memcmp(&x, &y, Target::realsize - Target::realpad) == 0;
}

#if __i386 || __x86_64__
void Port::yl2x_impl(longdouble* x, longdouble* y, longdouble* res)
{
    __asm__ volatile("fyl2x": "=t" (*res): "u" (*y), "0" (*x) : "st(1)" );
}

void Port::yl2xp1_impl(longdouble* x, longdouble* y, longdouble* res)
{
    __asm__ volatile("fyl2xp1": "=t" (*res): "u" (*y), "0" (*x) : "st(1)" );
}
#else
void Port::yl2x_impl(longdouble* x, longdouble* y, longdouble* res)
{
    assert(0);
}

void Port::yl2xp1_impl(longdouble* x, longdouble* y, longdouble* res)
{
    assert(0);
}
#endif

char *Port::strupr(char *s)
{
    char *t = s;

    while (*s)
    {
        *s = std::toupper(*s);
        s++;
    }

    return t;
}

int Port::memicmp(const char *s1, const char *s2, int n)
{
    int result = 0;

    for (int i = 0; i < n; i++)
    {   char c1 = s1[i];
        char c2 = s2[i];

        result = c1 - c2;
        if (result)
        {
            result = std::toupper(c1) - std::toupper(c2);
            if (result)
                break;
        }
    }
    return result;
}

int Port::stricmp(const char *s1, const char *s2)
{
    int result = 0;

    for (;;)
    {   char c1 = *s1;
        char c2 = *s2;

        result = c1 - c2;
        if (result)
        {
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
}

float Port::strtof(const char *p, char **endp)
{
    return std::strtof(p, endp);
}

double Port::strtod(const char *p, char **endp)
{
    return std::strtod(p, endp);
}

longdouble Port::strtold(const char *p, char **endp)
{
    return std::strtold(p, endp);
}

#endif

#if __sun

#define __C99FEATURES__ 1       // Needed on Solaris for NaN and more
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <wchar.h>
#include <float.h>
#include <ieeefp.h>
#include <assert.h>

double Port::nan;
longdouble Port::ldbl_nan;
longdouble Port::snan;

static double zero = 0;
double Port::infinity = 1 / zero;
longdouble Port::ldbl_infinity = 1 / zero;

double Port::dbl_max = 1.7976931348623157e308;
double Port::dbl_min = 5e-324;
longdouble Port::ldbl_max = LDBL_MAX;

#if __i386 || __x86_64__
bool Port::yl2x_supported = true;
bool Port::yl2xp1_supported = true;
#else
bool Port::yl2x_supported = false;
bool Port::yl2xp1_supported = false;
#endif

struct PortInitializer
{
    PortInitializer();
};

static PortInitializer portinitializer;

PortInitializer::PortInitializer()
{
    union
    {   unsigned int ui[2];
        double d;
    } nan = {{ 0, 0x7FF80000 }};

    Port::nan = nan.d;
    assert(!signbit(Port::nan));

    union
    {   unsigned int ui[4];
        longdouble ld;
    } ldbl_nan = {{ 0, 0xC0000000, 0x7FFF, 0}};

    Port::ldbl_nan = ldbl_nan.ld;
    assert(!signbit(Port::ldbl_nan));

    union
    {   unsigned int ui[4];
        longdouble     ld;
    } snan = {{ 0, 0xA0000000, 0x7FFF, 0 }};

    Port::snan = snan.ld;
}

int Port::isNan(double r)
{
    return isnan(r);
}

int Port::isNan(longdouble r)
{
    return isnan(r);
}

int Port::isSignallingNan(double r)
{
    /* A signalling NaN is a NaN with 0 as the most significant bit of
     * its significand, which is bit 51 of 0..63 for 64 bit doubles.
     */
    return isNan(r) && !((((unsigned char*)&r)[6]) & 8);
}

int Port::isSignallingNan(longdouble r)
{
    /* A signalling NaN is a NaN with 0 as the most significant bit of
     * its significand, which is bit 62 of 0..79 for 80 bit reals.
     */
    return isNan(r) && !((((unsigned char*)&r)[7]) & 0x40);
}

int Port::isInfinity(double r)
{
    return isinf(r);
}

longdouble Port::sqrt(longdouble x)
{
    return ::sqrtl(x);
}

longdouble Port::fmodl(longdouble x, longdouble y)
{
    return ::fmodl(x, y);
}

int Port::fequal(longdouble x, longdouble y)
{
    /* In some cases, the REALPAD bytes get garbage in them,
     * so be sure and ignore them.
     */
    return memcmp(&x, &y, 10) == 0;
}
#if __i386
#if IN_LLVM
// There seems to be an issue with register usage if compiled as PIC.
void Port::yl2x_impl(long double* x, long double* y, long double* res)
{
    __asm__ volatile("movl %0, %%eax;"    // move x, y, res to registers
                     "movl %1, %%ecx;"
                     "fldt (%%edx);"      // push *y and *x to the FPU stack
                     "fldt (%%eax);"      // "t" suffix means tbyte
                     "movl %2, %%eax;"
                     "fyl2x;"             // do operation and wait
                     "fstpt (%%eax)"      // pop result to a *res
                     :                          // output: empty
                     :"r"(x), "r"(y), "r"(res)  // input: x => %0, y => %1, res => %2
                     :"%eax", "%ecx", "%eax");  // clobbered register: eax, ecx, eax
}

void Port::yl2xp1_impl(long double* x, long double* y, long double* res)
{
    __asm__ volatile("movl %0, %%eax;"    // move x, y, res to registers
                     "movl %1, %%ecx;"
                     "fldt (%%ecx);"      // push *y and *x to the FPU stack
                     "fldt (%%eax);"      // "t" suffix means tbyte
                     "movl %2, %%eax;"
                     "fyl2xp1;"            // do operation and wait
                     "fstpt (%%eax)"      // pop result to a *res
                     :                          // output: empty
                     :"r"(x), "r"(y), "r"(res)  // input: x => %0, y => %1, res => %2
                     :"%eax", "%ecx", "%eax");  // clobbered register: eax, ecx, eax
}
#else
void Port::yl2x_impl(long double* x, long double* y, long double* res)
{
    __asm__ volatile("movl %0, %%eax;"    // move x, y, res to registers
                     "movl %1, %%ebx;"
                     "movl %2, %%ecx;"
                     "fldt (%%ebx);"      // push *y and *x to the FPU stack
                     "fldt (%%eax);"      // "t" suffix means tbyte
                     "fyl2x;"             // do operation and wait
                     "fstpt (%%ecx)"      // pop result to a *res
                     :                          // output: empty
                     :"r"(x), "r"(y), "r"(res)  // input: x => %0, y => %1, res => %2
                     :"%eax", "%ebx", "%ecx");  // clobbered register: eax, ebc, ecx
}

void Port::yl2xp1_impl(long double* x, long double* y, long double* res)
{
    __asm__ volatile("movl %0, %%eax;"    // move x, y, res to registers
                     "movl %1, %%ebx;"
                     "movl %2, %%ecx;"
                     "fldt (%%ebx);"      // push *y and *x to the FPU stack
                     "fldt (%%eax);"      // "t" suffix means tbyte
                     "fyl2xp1;"            // do operation and wait
                     "fstpt (%%ecx)"      // pop result to a *res
                     :                          // output: empty
                     :"r"(x), "r"(y), "r"(res)  // input: x => %0, y => %1, res => %2
                     :"%eax", "%ebx", "%ecx");  // clobbered register: eax, ebc, ecx
}
#endif
#elif __x86_64__
void Port::yl2x_impl(long double* x, long double* y, long double* res)
{
    __asm__ volatile("movq %0, %%rcx;"    // move x, y, res to registers
                     "movq %1, %%rdx;"
                     "movq %2, %%r8;"
                     "fldt (%%rdx);"      // push *y and *x to the FPU stack
                     "fldt (%%rcx);"      // "t" suffix means tbyte
                     "fyl2x;"             // do operation and wait
                     "fstpt (%%r8)"       // pop result to a *res
                     :                          // output: empty
                     :"r"(x), "r"(y), "r"(res)  // input: x => %0, y => %1, res => %2
                     :"%rcx", "%rdx", "%r8");   // clobbered register: rcx, rdx, r8
}

void Port::yl2xp1_impl(long double* x, long double* y, long double* res)
{
    __asm__ volatile("movq %0, %%rcx;"    // move x, y, res to registers
                     "movq %1, %%rdx;"
                     "movq %2, %%r8;"
                     "fldt (%%rdx);"      // push *y and *x to the FPU stack
                     "fldt (%%rcx);"      // "t" suffix means tbyte
                     "fyl2xp1;"            // do operation and wait
                     "fstpt (%%r8)"       // pop result to a *res
                     :                          // output: empty
                     :"r"(x), "r"(y), "r"(res)  // input: x => %0, y => %1, res => %2
                     :"%rcx", "%rdx", "%r8");   // clobbered register: rcx, rdx, r8
}
#else
void Port::yl2x_impl(longdouble* x, longdouble* y, longdouble* res)
{
    assert(0);
}

void Port::yl2xp1_impl(longdouble* x, longdouble* y, longdouble* res)
{
    assert(0);
}
#endif

char *Port::strupr(char *s)
{
    char *t = s;

    while (*s)
    {
        *s = toupper(*s);
        s++;
    }

    return t;
}

int Port::memicmp(const char *s1, const char *s2, int n)
{
    int result = 0;

    for (int i = 0; i < n; i++)
    {   char c1 = s1[i];
        char c2 = s2[i];

        result = c1 - c2;
        if (result)
        {
            result = toupper(c1) - toupper(c2);
            if (result)
                break;
        }
    }
    return result;
}

int Port::stricmp(const char *s1, const char *s2)
{
    int result = 0;

    for (;;)
    {   char c1 = *s1;
        char c2 = *s2;

        result = c1 - c2;
        if (result)
        {
            result = toupper(c1) - toupper(c2);
            if (result)
                break;
        }
        if (!c1)
            break;
        s1++;
        s2++;
    }
    return result;
}

float Port::strtof(const char *p, char **endp)
{
    return ::strtof(p, endp);
}

double Port::strtod(const char *p, char **endp)
{
    return ::strtod(p, endp);
}

longdouble Port::strtold(const char *p, char **endp)
{
    return ::strtold(p, endp);
}

#endif

// Little endian
void Port::writelongLE(unsigned value, void* buffer)
{
    unsigned char *p = (unsigned char*)buffer;
    p[3] = (unsigned char)(value >> 24);
    p[2] = (unsigned char)(value >> 16);
    p[1] = (unsigned char)(value >> 8);
    p[0] = (unsigned char)(value);
}

// Little endian
unsigned Port::readlongLE(void* buffer)
{
    unsigned char *p = (unsigned char*)buffer;
    return (((((p[3] << 8) | p[2]) << 8) | p[1]) << 8) | p[0];
}

// Big endian
void Port::writelongBE(unsigned value, void* buffer)
{
    unsigned char *p = (unsigned char*)buffer;
    p[0] = (unsigned char)(value >> 24);
    p[1] = (unsigned char)(value >> 16);
    p[2] = (unsigned char)(value >> 8);
    p[3] = (unsigned char)(value);
}

// Big endian
unsigned Port::readlongBE(void* buffer)
{
    unsigned char *p = (unsigned char*)buffer;
    return (((((p[0] << 8) | p[1]) << 8) | p[2]) << 8) | p[3];
}

// Little endian
unsigned Port::readwordLE(void *buffer)
{
    unsigned char *p = (unsigned char*)buffer;
    return (p[1] << 8) | p[0];
}

// Big endian
unsigned Port::readwordBE(void *buffer)
{
    unsigned char *p = (unsigned char*)buffer;
    return (p[0] << 8) | p[1];
}
