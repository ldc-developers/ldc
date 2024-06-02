// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK-LABEL: define{{.*}} @{{.*}}3foo
void[] foo()
{
    // CHECK-NEXT: %.gc_mem = call {{.*}} @_d_newarrayT
    // CHECK-NEXT: %.ptr = extractvalue {{.*}} %.gc_mem, 1
    // CHECK-NEXT: %1 = insertvalue {{.*}} { i{{32|64}} 3, ptr undef }, ptr %.ptr, 1
    // CHECK-NEXT: ret {{.*}} %1
    return new void[3];
}
