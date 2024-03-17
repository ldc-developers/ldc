// REQUIRES: target_X86
// UNSUPPORTED: Windows
// RUN: %ldc -mtriple=x86_64-freebsd13 -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK: module asm ".symver __qsort_r_compat, qsort_r@FBSD_1.0"
__asm__(".symver " "__qsort_r_compat" ", " "qsort_r" "@" "FBSD_1.0");
