// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

void foo()
{
// CHECK: %v1 = alloca <2 x i8>, align 2
// CHECK-SAME: size/byte = 2]
// CHECK: %v2 = alloca <4 x i8>, align 4
// CHECK-SAME: size/byte = 4]
// CHECK-COUNT-2: store i8 0, i8*
// CHECK-COUNT-4: store i8 0, i8*
    __vector(void[2]) v1;
    __vector(void[4]) v2;
}
