// Test unitialized data access with MemorySanitizer

// REQUIRES: MSan

// RUN: %ldc -g -fsanitize=memory -run %s
// RUN: %ldc -g -fsanitize=memory -d-version=BUG %s -of=%t%exe
// RUN: not %t%exe 2>&1 | FileCheck %s

// CHECK: MemorySanitizer: use-of-uninitialized-value
// TODO: line information is not shown correctly for this test.

int main()
{
    version (BUG)
        int x = void;
    else
        int x;

    int* p = &x;
    return *p;
}
