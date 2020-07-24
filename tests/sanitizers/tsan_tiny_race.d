// Test that ThreadSanitizer+LDC works on a very basic testcase.
// Note that -betterC is used, to avoid relying on druntime for this test.

// REQUIRES: TSan

// RUN: %ldc -betterC -g -fsanitize=thread %s -of=%t%exe
// RUN: %deflake 10 %t%exe | FileCheck %s

// CHECK: WARNING: ThreadSanitizer: data race

import core.sys.posix.pthread;

shared int global;

extern(C)
void *thread1(void *x) {
  barrier_wait(&barrier);
// CHECK-DAG: thread1{{.*}}[[@LINE+1]]
  global = 42;
  return x;
}

extern(C)
int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, null, &thread1, null);
// CHECK-DAG: main{{.*}}[[@LINE+1]]
  global = 43;
  barrier_wait(&barrier);
  pthread_join(t, null);
  return global;
}

//----------------------------------------------------------------------------
// Code to facilitate thread synchronization to make this test deterministic.
// See LLVM: compiler-rt/test/tsan/test.h

// TSan-invisible barrier.
// Tests use it to establish necessary execution order in a way that does not
// interfere with tsan (does not establish synchronization between threads).
alias invisible_barrier_t = __c_ulonglong;
import core.stdc.config;
alias __c_unsigned = uint;
// Default instance of the barrier, but a test can declare more manually.
__gshared invisible_barrier_t barrier;

extern (C) {
  // These functions reside inside the tsan library.
  void __tsan_testonly_barrier_init(invisible_barrier_t *barrier, __c_unsigned count);
  void __tsan_testonly_barrier_wait(invisible_barrier_t *barrier);
}

void barrier_init(invisible_barrier_t *barrier, __c_unsigned count) {
  __tsan_testonly_barrier_init(barrier, count);
}

void barrier_wait(invisible_barrier_t *barrier) {
  __tsan_testonly_barrier_wait(barrier);
}
//----------------------------------------------------------------------------
