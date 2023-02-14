// REQUIRES: target_RISCV

// RUN: %ldc %s -mtriple=riscv64-unknown-elf -mattr=+m,-m --gcc=echo > %t && FileCheck %s < %t
// CHECK: -mabi=lp64 -march=rv64i_zicsr_zifencei

void main() {}
