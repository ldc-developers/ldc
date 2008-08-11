module strings2;

import std.string;
import std.stdio;

void main()
{
    int i = 32;
    auto str = format(i);
    writefln(str);

    long l = 123123;
    str = format(l);
    writefln(str);
}
