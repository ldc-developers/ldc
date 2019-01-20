// AddressSanitizer: Test stack overflow detection of an array on a fiber's local stack.

// REQUIRES: ASan, RTSupportsSanitizers

// RUN: %ldc -g -fsanitize=address %s -of=%t%exe                             &&  not %t%exe 2>&1 | FileCheck %s
// RUN: %ldc -g -fsanitize=address %s -of=%t%exe -d-version=BAD_AFTER_YIELD  &&  not %t%exe 2>&1 | FileCheck %s

import core.thread;

// Note: the ordering of `foo` and `prefoo` is intentional to ease FileCheck checking line numbers,
// because of the order in which ASan reports the stack buffer overflow.

void foo(int* ptr)
{
    version (BAD_AFTER_YIELD)
        Fiber.yield();

    // CHECK: stack-buffer-overflow
    // CHECK: WRITE of size 4
    // CHECK-NEXT: #0 {{.*}} in {{.*foo.*}} {{.*}}asan_fiber.d:[[@LINE+1]]
    ptr[10] = 1;

}

// CHECK-NOT: wild pointer
// CHECK: Address {{.*}} is located in stack of
// CHECK-NEXT: #0 {{.*}} in {{.*prefoo.*}} {{.*}}asan_fiber.d:[[@LINE+1]]
void prefoo()
{
    int[10] a;
    foo(&a[0]);
}

void main()
{
    auto fib = new Fiber(&prefoo);
    fib.call();
    version (BAD_AFTER_YIELD)
        fib.call();
}
