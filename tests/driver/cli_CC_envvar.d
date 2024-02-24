// UNSUPPORTED: Windows

// RUN: not env CC='cc --linker-arg' %ldc -vv %s | FileCheck %s

// CHECK: Linking with:
// CHECK-NEXT: --linker-arg
