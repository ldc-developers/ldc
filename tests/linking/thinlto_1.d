// Test ThinLTO commandline flag

// REQUIRES: atleast_llvm309

// RUN: %ldc %s -of=%t%obj -c -thinlto -vv | FileCheck %s

// CHECK: Writing LLVM bitcode
// CHECK: Creating module summary for ThinLTO

void main()
{
}
