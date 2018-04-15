// REQUIRES: Windows

// RUN: %ldc -link-internally -run %s

void main()
{
    import std.stdio;
    writeln("Hello world");
}
