// Written in the D programming language by Tomas Lindquist Olsen 2008
// Binding of llvm.c.Core values for D.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
module llvm.util;

//we need <cstring>
version(Tango) {
    import tango.stdc.string;
}
else {
    import std.c.string;
}

///
char[] from_stringz(char* p)
{
    if (p is null)
        return "";
    return p[0..strlen(p)];
}

///
char* to_stringz(char[] s)
{
    return (s~\0).ptr;
}

