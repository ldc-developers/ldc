// Tests @target attribute for x86

// REQUIRES: target_X86

// RUN: %ldc -O -c -mcpu=i386 -mtriple=i386-linux-gnu -output-ll -of=%t.ll %s && FileCheck %s --check-prefix LLVM < %t.ll
// RUN: %ldc -O -c -mcpu=i386 -mtriple=i386-linux-gnu -output-s -of=%t.s %s && FileCheck %s  --check-prefix ASM < %t.s

import ldc.attributes;

// LLVM-LABEL: define{{.*}} void @{{.*}}foo
// ASM-LABEL: _D15attr_target_x863fooFPfQcfZv:
void foo(float *A, float* B, float K) {
    for (int i = 0; i < 128; ++i)
        A[i] *= B[i] + K;
// ASM-NOT: addps
}

// LLVM-LABEL: define{{.*}} void @{{.*}}foo_sse
// LLVM-SAME: #[[SSE:[0-9]+]]
// ASM-LABEL: _D15attr_target_x867foo_sseFPfQcfZv:
@(target("sse"))
void foo_sse(float *A, float* B, float K) {
    for (int i = 0; i < 128; ++i)
        A[i] *= B[i] + K;
// ASM: addps
}

// `sse` should take precedence over `no-sse` since it appears later.
// LLVM-LABEL: define{{.*}} void @{{.*}}bar_sse
// LLVM-SAME: #[[SSE2:[0-9]+]]
// ASM-LABEL: _D15attr_target_x867bar_sseFPfQcfZv:
@(target("no-sse\n  , \tsse  "))
void bar_sse(float *A, float* B, float K) {
    for (int i = 0; i < 128; ++i)
        A[i] *= B[i] + K;
// ASM: addps
}


// Same reason as above, except the other way around
// LLVM-LABEL: define{{.*}} void @{{.*}}bar_nosse
// LLVM-SAME: #[[NOSSE:[0-9]+]]
// ASM-LABEL: _D15attr_target_x869bar_nosseFPfQcfZv:
@(target("sse"))
@(target("  no-sse"))
void bar_nosse(float *A, float* B, float K) {
    for (int i = 0; i < 128; ++i)
        A[i] *= B[i] + K;
// ASM-NOT: addps
}

// LLVM-LABEL: define{{.*}} void @{{.*}}haswell
// LLVM-SAME: #[[HSW:[0-9]+]]
// ASM-LABEL: _D15attr_target_x867haswellFPfQcfZv:
@(target("arch=haswell "))
void haswell(float *A, float* B, float K) {
    for (int i = 0; i < 128; ++i)
        A[i] *= B[i] + K;
// ASM: vaddps
}


// LLVM-DAG: attributes #[[SSE]] = {{.*}} "target-features"="+sse"
// LLVM-DAG: attributes #[[SSE2]] = {{.*}} "target-features"="-sse,+sse"
// LLVM-DAG: attributes #[[NOSSE]] = {{.*}} "target-features"="+sse,-sse"
// LLVM-DAG: attributes #[[HSW]] = {{.*}} "target-cpu"="haswell"
