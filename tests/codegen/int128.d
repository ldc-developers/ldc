// REQUIRES: target_X86

// RUN: %ldc -output-s -mtriple=x86_64-linux-gnu -O -of=%t.s %s && FileCheck %s < %t.s

import core.int128;

// CHECK: _D6int1285mulhiFmmZm:
ulong mulhi(ulong a, ulong b)
{
    // CHECK-NEXT: .cfi_startproc
    // CHECK-NEXT: movq	%rsi, %rax
    // CHECK-NEXT: mulq	%rdi
    // CHECK-NEXT: movq	%rdx, %rax
    // CHECK-NEXT: retq

    return mul(Cent(a), Cent(b)).hi;
}

// CHECK: _D6int12810mul_divmodFmmmJmZm:
ulong mul_divmod(ulong a, ulong b, ulong c, out ulong modulus)
{
    // CHECK-NEXT: .cfi_startproc
    // CHECK-NEXT: movq	%rdx, %r8
    // CHECK-NEXT: movq	%rsi, %rax
    // CHECK-NEXT: mulq	%rdi
    // CHECK-NEXT: movq	$0, (%rcx)
    // CHECK-NEXT: #APP
    // CHECK-NEXT: divq	%r8
    // CHECK-NEXT: #NO_APP
    // CHECK-NEXT: movq	%rdx, (%rcx)
    // CHECK-NEXT: retq

    const product128 = mul(Cent(a), Cent(b));
    return udivmod(product128, c, modulus);
}
