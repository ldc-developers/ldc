// The presence of DMD-style inline assembly inhibits inlining,
// so if `pragma(inline, true)` is also present, some kind of error should occur.

// REQUIRES: target_X86
// RUN: not %ldc -mtriple=i686-pc-windows-msvc -c %s 2>&1 | FileCheck %s

module dmd_style_asm_in_inline_d_function;

// CHECK: `dmd_style_asm_in_inline_d_function.hasDMDStyleAsm` cannot be `pragma(inline, true)` as it contains DMD-style inline assembly
pragma(inline, true)
uint hasDMDStyleAsm(uint a)
{
    asm
    {
        mov EAX, dword ptr [a];
    }
}
