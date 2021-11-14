// Makes sure the frame pointer isn't enforced for GCC-style asm.

// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-linux-gnu -mattr=avx -O -output-s -of=%t.s %s
// RUN: FileCheck %s < %t.s

alias byte32 = __vector(byte[32]);

// CHECK: _D13asm_gcc_no_fp3xorFNhG32gQgZQj:
byte32 xor(byte32 a, byte32 b)
{
    // CHECK-NEXT: .cfi_startproc
    byte32 r;
    // CHECK-NEXT: #APP
    // CHECK-NEXT: vxorps %ymm0, %ymm0, %ymm1
    // CHECK-NEXT: #NO_APP
    asm { "vxorps %0, %1, %2" : "=v" (r) : "v" (a), "v" (b); }
    // CHECK-NEXT: retq
    return r;
}
