
/* Compiler implementation of the D programming language
 * Copyright (C) 2009-2019 by The D Language Foundation, All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/dlang/dmd/blob/master/src/dmd/aliasthis.h
 */

#pragma once

#include "globals.h"
#include "dsymbol.h"

/**************************************************************/

class AliasThis : public Dsymbol
{
public:
   // alias Identifier this;
    Identifier *ident;
    Dsymbol    *sym;
    bool       isDeprecated_;

    Dsymbol *syntaxCopy(Dsymbol *);
    const char *kind() const;
    AliasThis *isAliasThis() { return this; }
    void accept(Visitor *v) { v->visit(this); }
    bool isDeprecated() const { return this->isDeprecated_; }
};
