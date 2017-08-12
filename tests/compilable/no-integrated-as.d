// RUN: %ldc -no-integrated-as %s

// We currently rely on gcc/clang; no MSVC toolchain support yet.
// UNSUPPORTED: Windows

module noIntegratedAs;

void main()
{
    import std.stdio;
    writeln("This object is assembled externally and linked to an executable.");
}
