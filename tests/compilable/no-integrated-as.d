// RUN: %ldc -no-integrated-as %s

// We currently rely on gcc/clang; no MSVC toolchain support yet.
// UNSUPPORTED: Windows

/* on Linux, need clang-13 with LLVM 13:
 * LLVM 12: .section __minfo,"aw",@progbits
 * LLVM 13: .section __minfo,"awR",@progbits,unique,1
 */
// UNSUPPORTED: Linux

module noIntegratedAs;

void main()
{
    import std.stdio;
    writeln("This object is assembled externally and linked to an executable.");
}
