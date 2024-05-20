// Test the combination of `out` arguments with in- and out-contracts.

// Github issue #1543, https://github.com/ldc-developers/ldc/issues/1543

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -run %s

module mod;

class Bar
{
    void failMe(out int some)
    in
    {
        assert(some == 0);
    }
    out
    {
        assert(some == 123);
    }
    body
    {
        some = 123;
    }
}

// Bar.failMe codegen order = function, in-contract __require function, out-contract __ensure function

// CHECK-LABEL: define {{.*}} @{{.*}}Bar6failMe
// CHECK-SAME: ptr dereferenceable(4) %some
// CHECK: store i32 0, ptr %some
// CHECK: call {{.*}} @{{.*}}Bar6failMeMFJiZ9__require
// CHECK: call {{.*}} @{{.*}}Bar6failMeMFJiZ8__ensure
// CHECK: }

// CHECK-LABEL: define {{.*}} @{{.*}}Bar6failMeMFJiZ9__require
// CHECK-SAME: ptr dereferenceable(4) %some
// CHECK-NOT: store {{.*}} %some
// CHECK: }

// CHECK-LABEL: define {{.*}} @{{.*}}Bar6failMeMFJiZ8__ensure
// CHECK-SAME: ptr dereferenceable(4) %some
// CHECK-NOT: store {{.*}} %some
// CHECK: }

void main()
{
    int some;
    auto b = new Bar;
    b.failMe(some);
    assert(some == 123);
}
