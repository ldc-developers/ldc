/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/imphint.d, _imphint.d)
 * Documentation:  https://dlang.org/phobos/dmd_imphint.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/imphint.d
 */

module dmd.imphint;

import core.stdc.string;

/******************************************
 * Looks for undefined identifier s to see
 * if it might be undefined because an import
 * was not specified.
 * Not meant to be a comprehensive list of names in each module,
 * just the most common ones.
 */
extern (C++) const(char)* importHint(const(char)* s)
{
    return hints.get(cast(string) s[0..strlen(s)], null).ptr;
}

private immutable string[string] hints;

shared static this()
{
    // in alphabetic order
    hints = [
        "cos": "std.math",
        "fabs": "std.math",
        "printf": "core.stdc.stdio",
        "sin": "std.math",
        "sqrt": "std.math",
        "writefln": "std.stdio",
        "writeln": "std.stdio",
        "__va_argsave_t": "core.vararg",
    ];
}

unittest
{
    const(char)* p;
    p = importHint("printf");
    assert(p);
    p = importHint("fabs");
    assert(p);
    p = importHint("xxxxx");
    assert(!p);
}
