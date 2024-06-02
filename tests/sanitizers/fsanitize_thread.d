// Test basic ThreadSanitizer functionality

// RUN: %ldc -c -output-ll -g -fsanitize=thread -of=%t.ll %s && FileCheck %s < %t.ll

// LLVM 14+: no more __tsan_func_{entry,exit} calls
// XFAIL: *

// CHECK: ; Function Attrs:{{.*}} sanitize_thread
// CHECK-NEXT: define {{.*}}D16fsanitize_thread3foo
int foo(int i) {
    // CHECK: call {{.*}}_tsan_func_entry
    return i + 1;
    // CHECK: ret i
}
