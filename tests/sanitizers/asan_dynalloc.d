// Test dynamic allocated memory read overflow detection with AddressSanitizer

// REQUIRES: ASan

// RUN: %ldc -g -fsanitize=address -boundscheck=off %s -of=%t%exe
// RUN: %ldc -g -fsanitize=address -boundscheck=off %s -of=%t.malloc%exe -d-version=MALLOC
// FIXME, GC allocations are not ASan ready yet. R UN: not %t%exe 2>&1 | FileCheck %s
// RUN:  not %t.malloc%exe 2>&1 | FileCheck %s --check-prefix=MALLOC

import core.stdc.stdlib;
import ldc.asan;

int getX()
{
    version (MALLOC)
    {
        byte* x = cast(byte*) malloc(5 * (byte).sizeof);
        // MALLOC: heap-buffer-overflow
        // MALLOC: READ of size 1
        // MALLOC-NEXT: #0 {{.*}} in {{.*getX.*}} {{.*}}asan_dynalloc.d:[[@LINE+1]]
        return x[5];
    }
    else
    {
        byte[] x = new byte[5];
        // CHECK: READ of size 1
        // CHECK-NEXT: #0 {{.*}} in {{.*getX.*}} {{.*}}asan_dynalloc.d:[[@LINE+1]]
        return x[5];
    }
}

int main()
{
    auto x = getX();
    return 0;
}
