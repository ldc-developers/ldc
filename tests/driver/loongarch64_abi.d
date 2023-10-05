// REQUIRES: target_LoongArch

// RUN: %ldc %s -mtriple=loongarch64-unknown-linux-gnu -mattr=+f,+d --gcc=echo > %t && FileCheck %s < %t
// CHECK: -mabi=lp64d

version (LoongArch64) {} else static assert(0);
// the next line checks -mattr=+f,+d
version (LoongArch_HardFloat) {} else static assert(0);
version (D_HardFloat) {} else static assert(0);

void main() {}
