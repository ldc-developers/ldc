// REQUIRES: target_RISCV

// RUN: %ldc %s -mtriple=riscv64-unknown-elf -mattr=+m,+a,+d,+f,+c -mabi=lp64d --gcc=echo > %t && FileCheck %s < %t
// CHECK: -mabi=lp64d -march=rv64gc

void main() {}
