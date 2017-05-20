// Test basic ThreadSanitizer functionality

// RUN: %ldc -c -output-ll -g -fsanitize=thread -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK-LABEL: define i32 {{.*}}D16fsanitize_thread3foo
int foo(int i) {
    // CHECK: call {{.*}}_tsan_func_entry
    return i + 1;
    // CHECK: ret i
}
