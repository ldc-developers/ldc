// Tests @target attribute for x86

// REQUIRES: atleast_llvm307

// RUN: %ldc -O -c -mcpu=i386 -mtriple i386-linux-gnu -output-ll -of=%t.ll %s && FileCheck %s --check-prefix LLVM < %t.ll
// RUN: %ldc -O -c -mcpu=i386 -mtriple i386-linux-gnu -output-s -of=%t.s %s && FileCheck %s  --check-prefix ASM < %t.s

import ldc.attributes;

// LLVM-LABEL: define{{.*}} void @{{.*}}foo
// ASM-LABEL: _D15attr_target_x863fooFPfPffZv:
void foo(float *A, float* B, float K) {
    for (int i = 0; i < 128; ++i)
        A[i] *= B[i] + K;
// ASM-NOT: addps
}

// LLVM-LABEL: define{{.*}} void @{{.*}}foo_sse
// LLVM-SAME: #[[SSE:[0-9]+]]
// ASM-LABEL: _D15attr_target_x867foo_sseFPfPffZv:
@(target("sse"))
void foo_sse(float *A, float* B, float K) {
    for (int i = 0; i < 128; ++i)
        A[i] *= B[i] + K;
// ASM: addps
}

// Make sure that no-sse overrides sse (attribute sorting). Also tests multiple @target attribs.
// LLVM-LABEL: define{{.*}} void @{{.*}}foo_nosse
// LLVM-SAME: #[[NOSSE:[0-9]+]]
// ASM-LABEL: _D15attr_target_x869foo_nosseFPfPffZv:
@(target("no-sse\n  , \tsse  "))
void foo_nosse(float *A, float* B, float K) {
    for (int i = 0; i < 128; ++i)
        A[i] *= B[i] + K;
// ASM-NOT: addps
}
// LLVM-LABEL: define{{.*}} void @{{.*}}bar_nosse
// LLVM-SAME: #[[NOSSE]]
// ASM-LABEL: _D15attr_target_x869bar_nosseFPfPffZv:
@(target("sse"))
@(target("  no-sse"))
void bar_nosse(float *A, float* B, float K) {
    for (int i = 0; i < 128; ++i)
        A[i] *= B[i] + K;
// ASM-NOT: addps
}

// LLVM-LABEL: define{{.*}} void @{{.*}}haswell
// LLVM-SAME: #[[HSW:[0-9]+]]
// ASM-LABEL: _D15attr_target_x867haswellFPfPffZv:
@(target("arch=haswell "))
void haswell(float *A, float* B, float K) {
    for (int i = 0; i < 128; ++i)
        A[i] *= B[i] + K;
// ASM: vaddps
}


// LLVM-DAG: attributes #[[SSE]] = {{.*}} "target-features"="+sse"
// LLVM-DAG: attributes #[[NOSSE]] = {{.*}} "target-features"="+sse,-sse"
// LLVM-DAG: attributes #[[HSW]] = {{.*}} "target-cpu"="haswell"
