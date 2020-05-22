// Tests that functions are cross-module inlined when emitting multiple object
// files.

// Generate unoptimized IR for 2 source files as separate compilation units.
// RUN: %ldc -c -output-ll %s %S/inputs/inlinables.d -od=%t
// RUN: FileCheck %s < %t/inlining_gh3126.ll

import inputs.inlinables;

// no other function definitions (always_inline_chain*)
// CHECK-NOT: define

// CHECK: define {{.*}}_D15inlining_gh31263fooFZi
int foo()
{
    // CHECK-NEXT: ret i32 345
    return always_inline_chain0();
}

// CHECK-NOT: define
