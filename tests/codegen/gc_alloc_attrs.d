// RUN: %ldc -O0 -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK-DAG: declare noalias ptr @_d_allocclass(
// CHECK-DAG: declare noalias ptr @_d_allocmemoryT(
// CHECK-DAG: declare noalias ptr @_d_allocmemory({{i32|i64}}){{.*}} #[[ALLOC:[0-9]+]]

// CHECK-DAG: attributes #[[ALLOC]] = {{.*}}allocsize(0)

class C { int x; }

C makeClass()
{
    return new C();
}

int* makeInt()
{
    return new int;
}

int delegate() makeClosure(int x)
{
    return () => x;
}
