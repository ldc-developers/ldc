
/* Copyright (c) 1999-2016 by Digital Mars
 * All Rights Reserved, written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 * https://github.com/D-Programming-Language/dmd/blob/master/src/root/port.h
 */

#ifndef REAL_T_H
#define REAL_T_H

#include "longdouble.h"

#include <stdint.h>

struct TargetFP
{
    static bool yl2x_supported;
    static bool yl2xp1_supported;

    static longdouble snan;

    static longdouble sqrt(longdouble x);
    static longdouble fmodl(longdouble a, longdouble b);

    static bool fequal(longdouble a, longdouble b);
    static bool isNan(longdouble r);
    static bool isInfinity(longdouble r);

    static bool isFloat32LiteralOutOfRange(const char* literal);
    static bool isFloat64LiteralOutOfRange(const char* literal);

    static longdouble strtold(const char *p, char **endp);
    static std::size_t sprint(char *str, int fmt, longdouble x);

    static void yl2x_impl(longdouble *x, longdouble *y, longdouble *res);
    static void yl2xp1_impl(longdouble *x, longdouble *y, longdouble *res);
};

#endif
