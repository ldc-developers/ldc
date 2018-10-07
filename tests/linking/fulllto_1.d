// Test full LTO commandline flag

// REQUIRES: LTO

// RUN: %ldc %s -of=%t%obj -c -flto=full -vv | FileCheck %s
// RUN: %ldc -flto=full -run %s

// CHECK: Writing LLVM bitcode
// CHECK-NOT: Creating module summary

void main()
{
}
