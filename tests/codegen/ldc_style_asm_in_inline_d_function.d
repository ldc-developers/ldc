// The presence of non-DMD-style inline assembly does not inhibit inlining,
// so `pragma(inline, true)` should work for functions containing non-DMD-style inline assembly.

// REQUIRES: target_X86
// RUN: %ldc -mtriple=i686-pc-windows-msvc -output-ll -of=%t.ll -c %s && FileCheck %s < %t.ll

import ldc.llvmasm;

// CHECK: alwaysinline
// CHECK-NEXT: {{define.+hasLDCStyleAsm}}
pragma(inline, true)
uint hasLDCStyleAsm(uint a)
{
    return __asm!uint("add 7, $0", "=r,0,~{flags}", a);
}
