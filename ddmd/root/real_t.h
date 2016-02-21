
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

namespace target_fp
{
    bool yl2x_supported;
    bool yl2xp1_supported;

    longdouble snan;

    longdouble sqrt(longdouble x);
    void yl2x(longdouble *x, longdouble *y, longdouble *res);
    void yl2xp1(longdouble *x, longdouble *y, longdouble *res);

    bool areBitwiseEqual(longdouble a, longdouble b);
    bool isNaN(longdouble r);
    bool isInfinity(longdouble r);

    longdouble parseTargetReal(const char *literal, bool *isOutOfRange);
    bool isFloat32LiteralOutOfRange(const char *literal);
    bool isFloat64LiteralOutOfRange(const char *literal);

    std::size_t sprint(char *str, char fmt, longdouble x);
};

#endif
