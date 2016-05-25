// Test for Github issue 1519

// Check that .mangleof strings do not contain any char 0x01.
// LDC may prepend 0x01 to prevent LLVM from modifying the symbol name, but it should not appear in user code.

// RUN: %ldc -c %s

extern (C) void fooC()
{
}

extern (C++) void fooCpp()
{
}

extern (D) void fooD()
{
}

void aliasTemplate(alias F)()
{
    F();
}

void main()
{
    import std.algorithm;

    static assert(all!"a != '\1'"(fooC.mangleof));
    static assert(all!"a != '\1'"(fooCpp.mangleof));
    static assert(all!"a != '\1'"(fooD.mangleof));
    static assert(all!"a != '\1'"(aliasTemplate!fooCpp.mangleof));
}
