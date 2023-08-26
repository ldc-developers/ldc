// REQUIRES: target_RISCV

// RUN: %ldc %s -mtriple=riscv64-unknown-linux -mattr=+m,+a,+f,-d --gcc=echo > %t && FileCheck %s < %t
// CHECK: -mabi=lp64f -march=rv64imaf_zicsr_zifencei

void main() {}
