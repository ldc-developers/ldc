/**
 * Compiler implementation of the D programming language
 * http://dlang.org
 *
 * Copyright: Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors:   Walter Bright, http://www.digitalmars.com
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:    $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/root/rootobject.d, root/_rootobject.d)
 * Documentation:  https://dlang.org/phobos/dmd_root_rootobject.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/root/rootobject.d
 */

module dmd.root.rootobject;

import core.stdc.stdio;

import dmd.root.outbuffer;

/***********************************************************
 */

enum DYNCAST : int
{
    object,
    expression,
    dsymbol,
    type,
    identifier,
    tuple,
    parameter,
    statement,
    condition,
}

/***********************************************************
 */

extern (C++) class RootObject
{
    this() nothrow pure @nogc @safe
    {
    }

    bool equals(RootObject o)
    {
        return o is this;
    }

    int compare(RootObject)
    {
        assert(0);
    }

    void print()
    {
        printf("%s %p\n", toChars(), this);
    }

    const(char)* toChars()
    {
        assert(0);
    }

    void toBuffer(OutBuffer* buf) nothrow pure @nogc @safe
    {
        assert(0);
    }

    DYNCAST dyncast() const nothrow pure @nogc @safe
    {
        return DYNCAST.object;
    }
}
