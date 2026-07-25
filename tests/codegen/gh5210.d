// Make sure `pragma(inline, true)` is ignored for unittest functions.

// RUN: %ldc -unittest -output-ll %s -of=%t.ll && FileCheck %s < %t.ll

pragma(inline, true):

// CHECK: ; Function Attrs: {{.*}}alwaysinline
// CHECK-NEXT: define {{.*}}_D6gh52103fooFZi
int foo() { return 123; }

// CHECK-NOT: alwaysinline
// CHECK: define {{.*}}_D6gh521017__unittest_L13_C1FZv
unittest {
    assert(foo() == 123);
}
