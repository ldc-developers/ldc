// Tests that 'raw mangles' starting with "\1" are correctly propagated to IR.

// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK: @"\01my$Global" = global i32
pragma(mangle, "\1my$Global")
__gshared int myGlobal;

// CHECK: define {{.*}} @"\01my$Function"()
pragma(mangle, "\1my$Function")
void myFunction() {}
