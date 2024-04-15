// Test that @target attribute overrides command line -mattr

// REQUIRES: target_X86

// RUN: %ldc -O -c -mattr=sse -mcpu=i386 -mtriple=i386-linux-gnu -output-ll -of=%t.sse.ll %s && FileCheck %s --check-prefix LLVM-SSE < %t.sse.ll
// RUN: %ldc -O -c -mattr=-sse -mcpu=i386 -mtriple=i386-linux-gnu -output-ll -of=%t.nosse.ll %s && FileCheck %s --check-prefix LLVM-NOSSE < %t.nosse.ll

import ldc.attributes;

// LLVM-SSE-LABEL: define{{.*}} void @{{.*}}foo_nosse
// LLVM-SSE-SAME: #[[NOSSE:[0-9]+]]
// LLVM-NOSSE-LABEL: define{{.*}} void @{{.*}}foo_nosse
// LLVM-NOSSE-SAME: #[[NOSSE:[0-9]+]]
@target("no-sse")
void foo_nosse(float *A, float *B, float K) {
    for (int i = 0; i < 128; ++ i)
	A[i] *= B[i] + K;
}

// LLVM-SSE-LABEL: define{{.*}} void @{{.*}}foo_sse
// LLVM-SSE-SAME: #[[SSE:[0-9]+]]
// LLVM-NOSSE-LABEL: define{{.*}} void @{{.*}}foo_sse
// LLVM-NOSSE-SAME: #[[SSE:[0-9]+]]
@target("sse")
void foo_sse(float *A, float *B, float K) {
    for (int i = 0; i < 128; ++ i)
	A[i] *= B[i] + K;
}

// The -mattr feature should come before the @target one in the "target-features" below.

// LLVM-SSE-DAG: attributes #[[SSE]] = {{.*}} "target-features"="+sse,+sse"
// LLVM-SSE-DAG: attributes #[[NOSSE]] = {{.*}} "target-features"="+sse,-sse"
// LLVM-NOSSE-DAG: attributes #[[SSE]] = {{.*}} "target-features"="-sse,+sse"
// LLVM-NOSSE-DAG: attributes #[[NOSSE]] = {{.*}} "target-features"="-sse,-sse"
