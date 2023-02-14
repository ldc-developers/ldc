// REQUIRES: target_RISCV

// RUN: %ldc %s -mtriple=riscv32-unknown-elf -mattr=+c,+a,-a,+a,+m --gcc=echo > %t && FileCheck %s < %t
// CHECK: -mabi=ilp32 -march=rv32imac_zicsr_zifencei

void main() {}
