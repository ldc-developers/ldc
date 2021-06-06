// Test unitialized data access with MemorySanitizer

// REQUIRES: MSan, atleast_llvm800

// RUN: %ldc -g -fsanitize=memory -run %s
// RUN: %ldc -g -fsanitize=memory -fsanitize-memory-track-origins=2 -d-version=BUG %s -of=%t%exe
// RUN: not %t%exe 2>&1 | FileCheck %s

// CHECK: MemorySanitizer: use-of-uninitialized-value

// CHECK: Uninitialized value was created by an allocation of {{.*}} in the stack frame of function '_Dmain'
// CHECK-NEXT: #0 {{.*}} in _Dmain {{.*}}msan_uninitialized.d:[[@LINE+1]]
int main()
{
    version (BUG)
        int x = void;
    else
        int x;

    int* p = &x;
    return *p;
}
