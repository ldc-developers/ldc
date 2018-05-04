// Test ldc.attributes.restrict

// RUN: %ldc -O3 -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;


extern (C) int testReturnConstant( int * p1, int * p2) @restrictAll
// CHECK-LABEL: define{{.*}}@{{.*}}testReturnConstant
// CHECK: noalias
// CHECK: ret i32 42
{
    *p1 = 42;
    *p2 = 2;
    return *p1; //has to return 42 when p1 does not alias p2
}

extern (C) int testWithoutRestrict( int * p1, int * p2)
// CHECK-LABEL: define{{.*}}@{{.*}}testWithoutRestrict
// CHECK-NOT: noalias
// CHECK: ret i32 %
{
    *p1 = 42;
    *p2 = 2;
    return *p1; //here the constant return is not possible
}

extern (C) void testDoubleApplication( int * p1, int * p2) @(restrictAll, restrictAll)
// CHECK-LABEL: define{{.*}}@{{.*}}testDoubleApplication
// CHECK: noalias
{
    *p1 = 1;
    *p2 = 2;
}

@restrictAll
{
    // CHECK-LABEL: define{{.*}}@{{.*}}testNoPointers
    // CHECK-NOT: noalias
    // CHECK: ret
    extern (C) int testNoPointers(int a, int b) @restrictAll
    {
        return a + b;
    }
}
