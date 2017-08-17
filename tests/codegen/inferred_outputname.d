// Make sure the inferred output filename is based on the first (source or
// object) file, and independent from its module declaration.

// If it works on Windows, it will work on other platforms too, and it
// simplifies things a bit.
// REQUIRES: Windows

// 1) 2 object files compiled separately:
// RUN: %ldc -c %S/inputs/foo.d -of=%t-dir/foo%obj
// RUN: %ldc %s %t-dir/foo%obj -vv | FileCheck %s
// 2) singleObj build with external object file and 2 source files:
// RUN: %ldc %t-dir/foo%obj %s %S/inputs/attr_weak_input.d -vv | FileCheck %s

// CHECK: Linking with:
// CHECK-NEXT: '/OUT:inferred_outputname.exe'

module modulename;

void main() {}
