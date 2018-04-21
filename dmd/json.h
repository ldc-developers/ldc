
/* Compiler implementation of the D programming language
 * Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/dlang/dmd/blob/master/src/json.h
 */

#ifndef DMD_JSON_H
#define DMD_JSON_H

#ifdef __DMC__
#pragma once
#endif /* __DMC__ */

#include "arraytypes.h"

struct OutBuffer;

void json_generate(OutBuffer *, Modules *);

#ifdef IN_LLVM
unsigned tryParseJsonField(const char *fieldName);
#endif

#endif /* DMD_JSON_H */

