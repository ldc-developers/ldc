
/* Copyright (C) 1999-2020 by The D Language Foundation, All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/dlang/dmd/blob/master/src/dmd/root/root.h
 */

#pragma once

#if IN_LLVM
#ifndef IS_PRINTF
# ifdef __GNUC__
#  define IS_PRINTF(FMTARG) __attribute((__format__(__printf__, (FMTARG), (FMTARG)+1)))
# else
#  define IS_PRINTF(FMTARG)
# endif
#endif
#endif

#include "object.h"

#include "filename.h"

#include "file.h"

#include "outbuffer.h"

#include "array.h"
