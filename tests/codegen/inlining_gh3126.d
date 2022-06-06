// Tests that functions are cross-module inlined when emitting multiple object
// files.

// Generate unoptimized IR for 2 source files as separate compilation units (in both orders).
// RUN: %ldc -c -output-ll %s %S/inputs/inlinables.d -od=%t && FileCheck %s < %t/inlining_gh3126.ll
// RUN: %ldc -c -output-ll %S/inputs/inlinables.d %s -od=%t && FileCheck %s < %t/inlining_gh3126.ll

// Now test with another source file making use of inputs.inlinables instead of compiling that module directly.
// RUN: %ldc -c -output-ll -I%S %s %S/inlining_imports_pragma.d -od=%t && FileCheck %s < %t/inlining_gh3126.ll
// RUN: %ldc -c -output-ll -I%S %S/inlining_imports_pragma.d %s -od=%t && FileCheck %s < %t/inlining_gh3126.ll

import inputs.inlinables;

// no other function definitions (always_inline_chain*, call_template_foo);
// allow template_foo to be instantiated in here though
// CHECK-NOT: always_inline_chain
// CHECK-NOT: call_template_foo

// CHECK: define {{.*}}_D15inlining_gh31263fooFZi
int foo()
{
    // CHECK-NEXT: ret i32 345
    return always_inline_chain0();
}

// CHECK-NOT: always_inline_chain
// CHECK-NOT: call_template_foo

// CHECK: define {{.*}}_D15inlining_gh31263barFZi
int bar()
{
    // no calls to [call_]template_foo
    // CHECK-NOT: call {{.*}}template_foo
    return call_template_foo(123);
}

// CHECK-NOT: always_inline_chain
// CHECK-NOT: call_template_foo
