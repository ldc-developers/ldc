// Test @fastmath

// RUN: %ldc -O0 -release -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

// CHECK-LABEL: define{{.*}} @notfast
// CHECK-SAME: #[[ATTR_NOTFAST:[0-9]+]]
extern (C) double notfast(double a, double b)
{
    @fastmath
    double nested_fast(double a, double b)
    {
        return a * b;
    }

// CHECK-NOT: fmul fast
    return a * b;
}
// CHECK-LABEL: define{{.*}} @{{.*}}nested_fast
// CHECK: fmul fast


// CHECK-LABEL: define{{.*}} @fast
// CHECK-SAME: #[[ATTR_FAST:[0-9]+]]
@fastmath
extern (C) double fast(double a, double b)
{
    double c;

    double nested_slow(double a, double b)
    {
        return a * b;
    }

    // Also test new scopes when generating the IR.
    try {
// CHECK: fmul fast
        c += a * b;
    }
    catch (Throwable)
    {
// CHECK: fmul fast
        return a * b;
    }
// CHECK: fmul fast
    return c + a * b;
}
// CHECK-LABEL: define{{.*}} @{{.*}}nested_slow
// CHECK-NOT: fmul fast

// CHECK-DAG: attributes #[[ATTR_FAST]] ={{.*}} "unsafe-fp-math"="true"
// CHECK-NOT: attributes #[[ATTR_NOTFAST]] ={{.*}} "unsafe-fp-math"="true"
