// Test the AddressSanitizer (un)poison interface

// REQUIRES: ASan

// RUN: %ldc -g -fsanitize=address -boundscheck=off %s -of=%t%exe
// RUN: not %t%exe 2>&1 | FileCheck %s

import ldc.asan;

int getX()
{
    int[] x = new int[5];
    assert(__asan_address_is_poisoned(&x[1]) == 0);

    __asan_poison_memory_region(x.ptr, 5*int.sizeof);
    assert(__asan_address_is_poisoned(&x[1]) == 1);

    __asan_unpoison_memory_region(x.ptr, 5*int.sizeof);
    assert(__asan_address_is_poisoned(&x[1]) == 0);
    int a = x[1];

    __asan_poison_memory_region(x.ptr, 5*int.sizeof);

    // CHECK: use-after-poison
    // CHECK: READ of size 4
    // CHECK-NEXT: #0 {{.*}} in {{.*getX.*}} {{.*}}asan_interface.d:[[@LINE+1]]
    return a + x[1];
}

int main()
{
    auto x = getX();
    return 0;
}
