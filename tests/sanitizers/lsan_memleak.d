// Test leak detection with LSan

// REQUIRES: LSan

// UNSUPPORTED: Windows, FreeBSD

// RUN: %ldc -g -fsanitize=address %s -of=%t_asan%exe
// RUN: not %env_asan_opts=detect_leaks=true %t_asan%exe 2>&1 | FileCheck %s
// RUN: %ldc -g -fsanitize=leak %s -of=%t%exe
// RUN: not %t%exe 2>&1 | FileCheck %s

import core.stdc.stdlib;

// CHECK: ERROR: LeakSanitizer: detected memory leaks

void* foo()
{
    // CHECK: Direct leak of 17 byte(s) in 1 object(s) allocated from:
    // CHECK: #{{.*}} in {{.*}}lsan_memleak.d:[[@LINE+1]]
    return malloc(17);
}

void main()
{
    foo();
}
