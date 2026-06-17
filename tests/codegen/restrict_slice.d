// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

// Two restrict slices → one pairwise assume with separate_storage
// CHECK-LABEL: define {{.*}}@{{.*}}twoSlices
// CHECK: getelementptr inbounds {{.*}} i32 0, i32 1
// CHECK: [[A:%.restrict.ptr[0-9]*]] = load ptr, ptr
// CHECK: getelementptr inbounds {{.*}} i32 0, i32 1
// CHECK: [[B:%.restrict.ptr[0-9]*]] = load ptr, ptr
// CHECK: separate_storage{{.*}}ptr [[A]], ptr [[B]]
void twoSlices(@restrict int[] a, @restrict int[] b) {
    a[0] = 1;
    b[0] = 2;
}

// Three slices → three pairwise assumes
// CHECK-LABEL: define {{.*}}@{{.*}}threeSlices
// CHECK: separate_storage
void threeSlices(@restrict int[] a, @restrict int[] b, @restrict int[] c) { }

// Single restrict slice → no assume generated
// CHECK-LABEL: define {{.*}}@{{.*}}singleSlice
// CHECK-NOT: separate_storage
void singleSlice(@restrict int[] a) { }

// One restrict slice + one non-restrict slice → no assume
// CHECK-LABEL: define {{.*}}@{{.*}}oneRestrictOneNormal
// CHECK-NOT: separate_storage
void oneRestrictOneNormal(@restrict int[] a, int[] b) { }

// Mix: restrict pointer (noalias attr) + restrict slices (separate_storage)
// CHECK-LABEL: define {{.*}}@{{.*}}mixed
// CHECK-NOT: noalias %a_arg
// CHECK-NOT: noalias %b_arg
// CHECK: ptr noalias %p_arg
// CHECK-NOT: noalias %a_arg
// CHECK-NOT: noalias %b_arg
// CHECK: separate_storage
void mixed(@restrict int[] a, @restrict int[] b, @restrict int* p) {
    a[0] = *p;
    b[0] = *p;
}
