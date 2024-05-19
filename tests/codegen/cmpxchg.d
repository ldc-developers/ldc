// RUN: %ldc -O -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import core.atomic;

// CHECK: define {{.*}}_D7cmpxchg3fooFiZb
bool foo(int cmp)
{
    static shared int g;
    // CHECK-NOT: ret
    // CHECK:      [[FOO1:%[0-9]]] = cmpxchg ptr
    // CHECK-NEXT: [[FOO2:%[0-9]]] = extractvalue { i32, i1 } [[FOO1]], 1
    // CHECK-NEXT: ret i1 [[FOO2]]
    return cas(&g, cmp, 456);
}

// CHECK: define {{.*}}_D7cmpxchg3barFdZd
double bar(double cmp)
{
    static shared double g;
    // CHECK-NOT: ret
    // CHECK:      [[BAR1:%[0-9]]] = bitcast double %cmp_arg to i64
    // CHECK-NEXT: [[BAR2:%[0-9]]] = cmpxchg weak ptr
    casWeak(&g, &cmp, 456.0);
    // CHECK-NEXT: [[BAR3:%[0-9]]] = extractvalue { i64, i1 } [[BAR2]], 0
    // CHECK-NEXT: [[BAR4:%[0-9]]] = bitcast i64 [[BAR3]] to double
    // CHECK-NEXT: ret double [[BAR4]]
    return cmp;
}
