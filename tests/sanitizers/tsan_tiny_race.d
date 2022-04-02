// Test that ThreadSanitizer+LDC works on a very basic testcase.

// REQUIRES: TSan
// REQUIRES: atleast_llvm800

// RUN: %ldc -g -fsanitize=thread %s -of=%t%exe
// RUN: %deflake 20 %t%exe | FileCheck %s

// CHECK: WARNING: ThreadSanitizer: data race

import core.thread;

shared int global;

void thread1() {
  barrier_wait(&barrier);
// CHECK-DAG: 7thread1{{.*}}[[@LINE+1]]
  global = 42;
}

int main() {
  barrier_init(&barrier, 2);
  auto t = new Thread(&thread1);
  t.start();
// CHECK-DAG: {{(_Dmain|D main).*}}[[@LINE+1]]
  global = 43;
  barrier_wait(&barrier);
  t.join();
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
