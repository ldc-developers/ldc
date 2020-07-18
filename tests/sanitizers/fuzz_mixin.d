// Test the basic fuzz target mixin for a D-signature fuzz target.

// REQUIRES: Fuzzer
// UNSUPPORTED: Windows

// RUN: %ldc -g -fsanitize=fuzzer %s -of=%t%exe
// RUN: not %t%exe 2>&1 | FileCheck %s

int FuzzMe(in ubyte[] data)
{
    if ((data.length >= 3) && data[0] == 'F' && data[1] == 'U' && data[2] == 'Z')
    {
        // Testing this assertion also tests that druntime is initialized.
        // CHECK: fuzz_mixin.d([[@LINE+1]]): Assertion failure
        assert(false);
        // CHECK: ERROR: libFuzzer: deadly signal
    }

    return 0;
}

import ldc.libfuzzer;
mixin DefineTestOneInput!FuzzMe;
