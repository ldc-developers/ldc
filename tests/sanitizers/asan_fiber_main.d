// AddressSanitizer: Test stack overflow detection inside a fiber of an array on main's stack.

// REQUIRES: ASan, RTSupportsSanitizers

// Test without and with fake stack enabled.

// Note on debug lineinfo: on macOS the executable contains a link back to the
// object files for debug info. Therefore the order of text execution is important,
// i.e. we should finish all testing on one compiled executable before recompiling
// with different conditional compilation settings (because it will overwrite the
// object files from the previous compilation).

// RUN: %ldc -g -fsanitize=address %s -of=%t1%exe  &&  not %t1%exe  2>&1 | FileCheck %s
// RUN: env %env_asan_opts=detect_stack_use_after_return=true not %t1%exe  2>&1 | FileCheck %s --check-prefix=FAKESTACK

// RUN: %ldc -g -fsanitize=address %s -of=%t22%exe -d-version=BAD_AFTER_YIELD  &&  not %t22%exe 2>&1 | FileCheck %s
// RUN: env %env_asan_opts=detect_stack_use_after_return=true not %t22%exe 2>&1 | FileCheck %s --check-prefix=FAKESTACK

import core.thread;

void foo(int* arr)
{
    version (BAD_AFTER_YIELD)
        Fiber.yield();

    // CHECK: stack-buffer-overflow
    // CHECK: WRITE of size 4
    // CHECK-NEXT: #0 {{.*}} in {{.*foo.*}} {{.*}}asan_fiber_main.d:[[@LINE+1]]
    arr[10] = 1; // out-of-bounds write
}

// Without fake stack, ASan only keeps track of the current stack and thus reports
// the bad memory location as a "wild pointer".
// But with fake stack enabled we get something much better:
// FAKESTACK: Address {{.*}} is located in stack of
// FAKESTACK: #0 {{.*}} in {{.*main.*}} {{.*}}asan_fiber_main.d:[[@LINE+1]]
void main()
{
    int[10] a;
    int b;

    // Use an extra variable instead of passing `&a[0]` directly to `foo`.
    // This is to keep `a` on the stack: `ptr` may be heap allocated because
    // it is used in the lambda (delegate).
    int* ptr = &a[0];
    auto fib = new Fiber(() => foo(ptr));
    fib.call();
    version (BAD_AFTER_YIELD)
        fib.call();
}
