// RUN: %ldc -O3 -enable-cross-module-inlining -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import core.checkedint;

alias T = size_t;

// CHECK: define {{.*}}@{{.*}}_D6gh21313foo
T foo(T x, T y, ref bool overflow)
{
    // CHECK-NOT: and i8
    // CHECK: load i8{{.*}}, !range ![[META:[0-9]+]]
    // CHECK-NOT: and i8
    // CHECK: ret
    return mulu(x, y, overflow);
}

// CHECK: ![[META]] = {{.*}}!{i8 0, i8 2}
