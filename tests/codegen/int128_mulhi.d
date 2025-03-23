// REQUIRES: target_X86

// RUN: %ldc -output-s -mtriple=x86_64-linux-gnu -O -of=%t.s %s && FileCheck %s < %t.s

import core.int128;

// CHECK: _D12int128_mulhi5mulhiFmmZm:
ulong mulhi(ulong a, ulong b)
{
    // CHECK-NEXT: .cfi_startproc
    // CHECK-NEXT: movq	%rsi, %rax
    // CHECK-NEXT: mulq	%rdi
    // CHECK-NEXT: movq	%rdx, %rax
    // CHECK-NEXT: retq
    return mul(Cent(a), Cent(b)).hi;
}
