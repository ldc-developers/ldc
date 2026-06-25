// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import core.simd;
import ldc.intrinsics;

// CHECK-LABEL: define {{.*}}@{{.*}}test_fptrunc
// CHECK: fptrunc <2 x double> {{.*}} to <2 x float>
float2 test_fptrunc(double2 v) {
    return llvm_convertvector!(float2)(v);
}

// CHECK-LABEL: define {{.*}}@{{.*}}test_fpext
// CHECK: fpext <2 x float> {{.*}} to <2 x double>
double2 test_fpext(float2 v) {
    return llvm_convertvector!(double2)(v);
}

// CHECK-LABEL: define {{.*}}@{{.*}}test_sext
// CHECK: sext <4 x i16> {{.*}} to <4 x i32>
int4 test_sext(short4 v) {
    return llvm_convertvector!(int4)(v);
}

// CHECK-LABEL: define {{.*}}@{{.*}}test_zext
// CHECK: zext <4 x i16> {{.*}} to <4 x i32>
uint4 test_zext(ushort4 v) {
    return llvm_convertvector!(uint4)(v);
}

// CHECK-LABEL: define {{.*}}@{{.*}}test_trunc
// CHECK: trunc <4 x i64> {{.*}} to <4 x i32>
int4 test_trunc(long4 v) {
    return llvm_convertvector!(int4)(v);
}

// CHECK-LABEL: define {{.*}}@{{.*}}test_sitofp
// CHECK: sitofp <4 x i32> {{.*}} to <4 x float>
float4 test_sitofp(int4 v) {
    return llvm_convertvector!(float4)(v);
}

// CHECK-LABEL: define {{.*}}@{{.*}}test_uitofp
// CHECK: uitofp <4 x i32> {{.*}} to <4 x float>
float4 test_uitofp(uint4 v) {
    return llvm_convertvector!(float4)(v);
}

// CHECK-LABEL: define {{.*}}@{{.*}}test_fptosi
// CHECK: fptosi <4 x float> {{.*}} to <4 x i32>
int4 test_fptosi(float4 v) {
    return llvm_convertvector!(int4)(v);
}

// CHECK-LABEL: define {{.*}}@{{.*}}test_fptoui
// CHECK: fptoui <4 x float> {{.*}} to <4 x i32>
uint4 test_fptoui(float4 v) {
    return llvm_convertvector!(uint4)(v);
}

// CHECK-LABEL: define {{.*}}@{{.*}}test_same_width_int
// Same-width int-to-int should be a no-op (no conversion instruction, same type)
// CHECK-NOT: {{(sext|zext|trunc|sitofp|uitofp|fptosi|fptoui|fptrunc|fpext)}}
int4 test_same_width_int(int4 v) {
    return llvm_convertvector!(int4)(v);
}
