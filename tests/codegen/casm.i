// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK: module asm ".symver __qsort_r_compat, qsort_r@FBSD_1.0"
asm(".symver " "__qsort_r_compat" ", " "qsort_r" "@" "FBSD_1.0");
