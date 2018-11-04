// REQUIRES: target_AArch64

// RUN: %ldc -mtriple=aarch64-unknown-linux -output-s -of=%t.s %s
// RUN: FileCheck %s < %t.s

// CHECK-NOT: .ctors
// CHECK-NOT: .dtors
// CHECK: .section  .init_array
// CHECK: .section  .fini_array

// No code needed to generate asm for a module.
