// Test passing of LLVM bitcode file with Linker Options set

// Linker Options are currently only set on Windows platform, so we must (cross-)compile to Windows
// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-windows -c -output-bc %S/inputs/link_bitcode_libs_input.d -of=%t.bc \
// RUN:    && %ldc -mtriple=x86_64-windows -c -singleobj -output-ll %t.bc %s -of=%t.ll \
// RUN:    && FileCheck %s < %t.ll

pragma(lib, "library_one");
pragma(lib, "library_two");

// CHECK: !llvm.linker.options = !{![[ATTR_LIB1:[0-9]+]], ![[ATTR_LIB2:[0-9]+]], ![[ATTR_LIB3:[0-9]+]], ![[ATTR_LIB4:[0-9]+]]}
// CHECK: ![[ATTR_LIB1]]{{.*}}library_one
// CHECK: ![[ATTR_LIB2]]{{.*}}library_two
// CHECK: ![[ATTR_LIB3]]{{.*}}imported_one
// CHECK: ![[ATTR_LIB4]]{{.*}}imported_two
