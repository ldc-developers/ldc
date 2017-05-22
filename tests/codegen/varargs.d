// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// Make sure typesafe variadics are not lowered to LLVM variadics.
void typesafe(size_t[2] a...) {}
// CHECK: define{{.*}}typesafe
// CHECK-NOT: ...
// CHECK-SAME: {
