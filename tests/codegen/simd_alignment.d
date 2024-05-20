// RUN: %ldc -c -output-ll -O3 -of=%t.ll %s && FileCheck %s < %t.ll

import core.simd;

struct S17237
{
    bool a;
    struct
    {
        bool b;
        int8 c;
    }
}

int4 globalIntFour;
// CHECK-DAG: globalIntFour{{.*}} = {{.*}} align 16
S17237 globalStruct;
// CHECK-DAG: @{{.*}}globalStruct{{.*}}S17237{{\"?}} = {{.*}} zeroinitializer{{(, comdat)?}}, align 32

// CHECK-LABEL: define <8 x i32> @foo(
extern(C) int8 foo(S17237* s)
{
    // CHECK: %[[GEP:[0-9]]] = getelementptr {{.*}}S17237, ptr %s_arg
    // CHECK: = load <8 x i32>, ptr %[[GEP]], align 32
    return s.c;
}
