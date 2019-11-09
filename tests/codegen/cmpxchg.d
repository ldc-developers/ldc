// RUN: %ldc -O -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import core.atomic;

// CHECK: define {{.*}}_D7cmpxchg3fooFiZb
bool foo(int cmp)
{
    static shared int g;
    // CHECK-NEXT: %1 = cmpxchg i32*
    // CHECK-NEXT: %2 = extractvalue { i32, i1 } %1, 1
    // CHECK-NEXT: ret i1 %2
    return cas(&g, cmp, 456);
}

// CHECK: define {{.*}}_D7cmpxchg3barFdZd
double bar(double cmp)
{
    static shared double g;
    // CHECK-NEXT: %1 = bitcast double %cmp_arg to i64
    // CHECK-NEXT: %2 = cmpxchg weak i64*
    casWeak(&g, &cmp, 456.0);
    // CHECK-NEXT: %3 = extractvalue { i64, i1 } %2, 0
    // CHECK-NEXT: %4 = bitcast i64 %3 to double
    // CHECK-NEXT: ret double %4
    return cmp;
}
