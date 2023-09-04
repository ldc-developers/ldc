// Makes sure static in ImportC translates to `internal` IR linkage.
// See https://github.com/ldc-developers/ldc/issues/4484.

// RUN: %ldc -output-ll %s -of=%t.ll && FileCheck %s < %t.ll

// CHECK: myPrivateGlobal = internal global i32 0
static int myPrivateGlobal;

// CHECK: define internal void {{.*}}myPrivateFunc
static void myPrivateFunc() {}
