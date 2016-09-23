// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

struct A0
{
    ubyte[0] zerolen;
}
// CHECK-DAG: %{{.*}}.A0 = type { [1 x i8] }

struct uint_0_uint
{
    uint a = 111;
    ubyte[0] zerolen;
    uint c = 333;
}
// CHECK-DAG: %{{.*}}.uint_0_uint = type { i32, i32 }

// No tests for codegen with e.g. uint_0_uint yet, because codegen could be much improved.
// I think codegen should be the same as for
//     struct uint_uint
//     {
//         uint a = 111;
//         uint c = 333;
//     }

// CHECK-LABEL: define{{.*}}fooA0{{.*}} {
auto fooA0()
{
    return A0();
    // Intentionally a regexp to not match "sret"
    // CHECK: {{ ret }}
}

// CHECK-LABEL: define{{.*}}foo_uint_0_uint
auto foo_uint_0_uint()
{
    return uint_0_uint();
    // Intentionally a regexp to not match "sret"
    // CHECK: {{ ret }}
}
