// REQUIRES: target_RISCV

// RUN: %ldc %s -mtriple=riscv64-unknown-linux --gcc=echo > %t && FileCheck %s < %t
// CHECK: -mabi=lp64d -march=rv64gc

void main() {}
