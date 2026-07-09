// Make sure array ops are auto-vectorized with enabled optimizations.

// REQUIRES: target_X86
// RUN: %ldc -mtriple=x86_64-linux-gnu -O -mattr=+avx512vl -ffast-math -output-s -of=%t.s %s && FileCheck %s < %t.s

// CHECK: _D6gh499113kernel_staticFKG16fKxG16fKxQgKxQkZv:
void kernel_static(ref float[16] o, const ref float[16] a, const ref float[16] b, const ref float[16] c) {
    // CHECK-NEXT: .cfi_startproc
    // CHECK-NEXT: vmovups	(%rsi), %zmm0
    // CHECK-NEXT: vmovups	(%rdx), %zmm1
    // CHECK-NEXT: vfmadd213ps	(%rcx), %zmm0, %zmm1
    // CHECK-NEXT: vmovups	%zmm1, (%rdi)
    // CHECK-NEXT: vzeroupper
    // CHECK-NEXT: retq
    o[] = a[] * b[] + c[];
}

// CHECK: _D6gh499114kernel_dynamicFAfxAfxQdxQgZv:
void kernel_dynamic(float[] o, const float[] a, const float[] b, const float[] c) {
    // CHECK: vfmadd213ps
    o[] = a[] * b[] + c[];
    // CHECK: .size	_D6gh499114kernel_dynamicFAfxAfxQdxQgZv
}
