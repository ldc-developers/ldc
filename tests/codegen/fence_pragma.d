// RUN: %ldc %s -c -output-ll -of=%t.ll && FileCheck %s < %t.ll

import ldc.intrinsics;

void fun0 () {
  llvm_memory_fence(DefaultOrdering, SynchronizationScope.CrossThread);
  // CHECK: fence seq_cst
  llvm_memory_fence(DefaultOrdering, SynchronizationScope.SingleThread);
  // CHECK: fence syncscope("singlethread") seq_cst
}
