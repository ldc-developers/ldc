// REQUIRES: target_X86
// RUN: %ldc -c %s -v -march=x86 | FileCheck %s
// RUN: %ldc -c %s -v -march x86 | FileCheck %s

// CHECK: config {{.*}}.conf (i686-
