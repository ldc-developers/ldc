// REQUIRES: Windows
// REQUIRES: internal_lld

// RUN: %ldc -link-internally -run %s

void main()
{
    import std.stdio;
    writeln("Hello world");
}
