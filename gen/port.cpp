
// Copyright (c) 1999-2012 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com

#include "port.h"

#include <cstdlib>
#include <cstring>
#include <cctype>

/* Implements all floating point operation using llvm::APFloat.
 */

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

void ldc::port_init()
{
    Port::nan = llvm::APFloat::getNaN(llvm::APFloat::IEEEdouble).convertToDouble();
    Port::infinity = llvm::APFloat::getInf(llvm::APFloat::IEEEdouble).convertToDouble();
    Port::dbl_max = llvm::APFloat::getLargest(llvm::APFloat::IEEEdouble).convertToDouble();
    Port::dbl_min = llvm::APFloat::getSmallest(llvm::APFloat::IEEEdouble).convertToDouble();

    Port::ldbl_nan = longdouble::getNaN();
    Port::snan = longdouble::getSNaN();
    Port::ldbl_infinity = longdouble::getInf();
    Port::ldbl_max = longdouble::getLargest();
}

int Port::isNan(double r)
{
    return llvm::APFloat(r).isNaN();
}

int Port::isNan(longdouble r)
{
    return r.isNaN();
}

int Port::isSignallingNan(double r)
{
    return ldouble(r).isSignaling();
}

int Port::isSignallingNan(longdouble r)
{
    return r.isSignaling();
}

int Port::isInfinity(double r)
{
    return llvm::APFloat(r).isInfinity();
}

longdouble Port::fmodl(longdouble x, longdouble y)
{
    return longdouble::fmod(x, y);
}

longdouble Port::sqrt(longdouble x)
{
	return ldouble(x).sqrt();
}

int Port::fequal(longdouble x, longdouble y)
{
    return longdouble::fequal(x, y);
}

void Port::yl2x_impl(longdouble* x, longdouble* y, longdouble* res)
{
    assert(0);
}

void Port::yl2xp1_impl(longdouble* x, longdouble* y, longdouble* res)
{
    assert(0);
}

int Port::memicmp(const char *s1, const char *s2, int n)
{
#if HAVE_MEMICMP
    return ::memicmp(s1, s2, n);
#else
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
#endif
}

int Port::stricmp(const char *s1, const char *s2)
{
#if HAVE_STRICMP
    return ::stricmp(s1, s2);
#else
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
#endif
}

char *Port::strupr(char *s)
{
#if HAVE_STRUPR
    return ::strupr(s);
#else
    char *t = s;

    while (*s)
    {
        *s = toupper(*s);
        s++;
    }

    return t;
#endif
}

float Port::strtof(const char *p, char **endp)
{
#if HAVE_STRTOF
    return ::strtof(p, endp);
#else
    return static_cast<float>(::strtod(p, endp));
#endif
}

double Port::strtod(const char *p, char **endp)
{
    return ::strtod(p, endp);
}

longdouble Port::strtold(const char *p, char **endp)
{
    assert(!endp);
    longdouble res = longdouble::convertFromString(p);
    return res;
}