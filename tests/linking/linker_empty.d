// Check that the user can override LDC passing `-fuse-ld` to `cc`.

// REQUIRES: target_X86

// RUN: %ldc --gcc=echo --mtriple=x86_64-linux --linker= %s | FileCheck %s

// CHECK-NOT: -fuse-ld
// CHECK: linker_empty
// CHECK-NOT: -fuse-ld
