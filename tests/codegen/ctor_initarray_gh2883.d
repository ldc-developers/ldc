// REQUIRES: target_AArch64

// RUN: %ldc -mtriple=aarch64-unknown-linux -output-s -of=%t.s %s
// RUN: FileCheck %s < %t.s

// CHECK-NOT: .ctors
// CHECK-NOT: .dtors
// CHECK: .section  .init_array
// CHECK: .section  .fini_array

pragma(crt_constructor)
void ctor() {}

pragma(crt_destructor)
void dtor() {}
