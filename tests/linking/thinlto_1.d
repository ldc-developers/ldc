// Test ThinLTO commandline flag

// REQUIRES: LTO

// RUN: %ldc %s -of=%t%obj -c -flto=thin -vv | FileCheck %s
// RUN: %ldc -flto=thin -run %s

// CHECK: Writing LLVM bitcode
// CHECK: Creating module summary for ThinLTO

void main()
{
}
