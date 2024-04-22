// ImportC treats `inline` C functions as though `pragma(inline, true)` was applied to them.
// The presence of DMD-style inline assembly inhibits inlining,
// so if `pragma(inline, true)` is also present, an error should occur; but C doesn't
// require that an `inline` function is actually inlined, so we permit `inline` C functions
// to contain DMD-style inline assembly,

// REQUIRES: Windows && target_X86
// RUN: %ldc -mtriple=i686-pc-windows-msvc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK-NOT: alwaysinline
inline unsigned int hasDMDStyleAsm(unsigned int a) {
    asm {
        mov EAX, dword ptr [a];
    }
}
