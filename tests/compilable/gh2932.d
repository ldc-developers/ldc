// RUN: %ldc -c -allinst %s

import std.algorithm;

extern __gshared int[] array;

void funcWithNoFrame()
{
    int local;
    // lambda is codegen'd
    pragma(msg, typeof(array.map!(e => local)));
}

void funcWithFrame()
{
    int capturedVar, local;
    int nestedFunc() { return capturedVar; }
    // lambda is codegen'd with `-allinst`
    static assert(__traits(compiles, array.map!(e => local)));
}
