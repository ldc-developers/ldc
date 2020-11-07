// https://github.com/ldc-developers/ldc/issues/3562

// RUN: %ldc -O -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

struct AlignedInt
{
    align(64) int i;
}

// CHECK: define {{.*}}7testRef{{.*}} align 64
int testRef(ref AlignedInt p)
{
    // CHECK:      %2 = load i32, i32* %1, align 64
    // CHECK-NEXT: ret i32 %2
    return p.i;
}

// CHECK: define {{.*}}11testPointer{{.*}} align 64
int testPointer(AlignedInt* p)
{
    // CHECK:      %2 = load i32, i32* %1, align 64
    // CHECK-NEXT: ret i32 %2
    return p.i;
}
