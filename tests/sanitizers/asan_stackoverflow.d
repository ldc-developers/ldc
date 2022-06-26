// Test stack overflow detection with AddressSanitizer

// REQUIRES: ASan

// RUN: %ldc -g -fsanitize=address %s -of=%t%exe
// RUN: not %t%exe 2>&1 | FileCheck %s

void foo(int* arr)
{
    // CHECK: stack-buffer-overflow
    // CHECK: WRITE of size 4
    // CHECK-NEXT: #0 {{.*}} in {{.*foo.*}} {{.*}}asan_stackoverflow.d:[[@LINE+1]]
    arr[10] = 1;
}

// CHECK: Address {{.*}} is located in stack of
// CHECK-NEXT: #0 {{.*}} in {{.*main.*}} {{.*}}asan_stackoverflow.d:[[@LINE+1]]
void main()
{
    // Test for the name of the variable that is overflown.
    // CHECK: 'aiaiaiaiaiaiai'{{.*}} <== {{.*}} overflows this variable
    int[10] aiaiaiaiaiaiai;
    int b;
    foo(&aiaiaiaiaiaiai[0]);
}
