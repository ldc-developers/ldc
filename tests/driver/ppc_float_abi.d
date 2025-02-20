// REQUIRES: target_PowerPC

// RUN: %ldc -c -output-ll -of=%t.ll %s -mtriple=powerpc64le-linux-gnu -real-precision=quad && FileCheck %s --check-prefix=CHECK-GNU-IEEE < %t.ll
// RUN: %ldc -c -output-ll -of=%t.ll %s -mtriple=powerpc64le-linux-gnu -real-precision=doubledouble && FileCheck %s --check-prefix=CHECK-IBM-LDBL < %t.ll
// RUN: %ldc -c -output-ll -of=%t.ll %s -mtriple=powerpc64le-linux-musl && FileCheck %s --check-prefix=CHECK-MUSL < %t.ll

// CHECK-GNU-IEEE-LABEL: @_Z13test_functionu9__ieee128
// CHECK-IBM-LDBL-LABEL: @_Z13test_functiong
// CHECK-MUSL-LABEL: @_Z13test_functione
extern (C++) bool test_function(real arg) {
    // CHECK-GNU-IEEE: fcmp ogt fp128 {{.*}}, 0xL00000000000000000000000000000000
    // CHECK-IBM-LDBL: fcmp ogt ppc_fp128 {{.*}}, 0xM00000000000000000000000000000000
    // CHECK-MUSL: fcmp ogt double {{.*}}, 0.000000e+00
    return arg > 0.0;
}

// CHECK-GNU-IEEE: !{i32 1, !"float-abi", !"ieeequad"}
// CHECK-IBM-LDBL: !{i32 1, !"float-abi", !"doubledouble"}
