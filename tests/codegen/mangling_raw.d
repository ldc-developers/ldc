// Tests that 'raw mangles' starting with "\1" are correctly propagated to IR.

// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK: @"\01myGlobal" = global i32
pragma(mangle, "\1myGlobal")
__gshared int myGlobal;

// CHECK: define {{.*}} @"\01myFunction"()
pragma(mangle, "\1myFunction")
void myFunction() {}
