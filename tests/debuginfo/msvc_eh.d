// Checks that !dbg is being attached to MSVC EH/cleanup runtime calls.
// REQUIRES: Windows
// RUN: %ldc -g -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

struct WithDtor
{
    int z;
    ~this() { z = -1; }
}

void throwSome() 
{
    throw new Exception("!");
}

// CHECK: define {{.*}} @{{.*}}foo_msvc
// CHECK-SAME: !dbg
void foo_msvc()
{
    try 
    {
        WithDtor swd_1;
        swd_1.z = 24;
        throwSome();
    } 
    catch(Throwable t) 
    {
        WithDtor swd_2 = { 48 };
    }
    // CHECK-DAG: call {{.*}}@_d_eh_enter_catch{{.*}} !dbg
    // CHECK-DAG: call {{.*}}@_d_enter_cleanup{{.*}} !dbg
    // CHECK-DAG: call {{.*}}@_d_leave_cleanup{{.*}} !dbg
}
