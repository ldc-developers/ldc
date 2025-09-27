// RUN: env PATH= CC= not %ldc -mtriple aarch64-unknown-linux-gnut64 -v -defaultlib= %s 2>&1 | FileCheck %s

// CHECK-DAG: aarch64-unknown-linux-gnut64-gcc
// CHECK-DAG: aarch64-unknown-linux-gnut64-clang

// UNSUPPORTED: Windows

extern(C) void main () {}
