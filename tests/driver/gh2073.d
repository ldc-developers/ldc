// RUN: %ldc -mcpu=help 2>&1 | FileCheck %s
// RUN: %ldc -mattr=help 2>&1 | FileCheck %s
// RUN: %ldc -mcpu=help -mattr=help 2>&1 | FileCheck %s

// CHECK: Available CPUs for this target:
// CHECK: Available features for this target:
// CHECK-NOT: Available CPUs for this target:
