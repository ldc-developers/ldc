// Test basic fuzz test crash

// REQUIRES: Fuzzer
// UNSUPPORTED: Windows

// RUN: %ldc -g -fsanitize=fuzzer %s -of=%t%exe
// RUN: not %t%exe 2> %t.out
// RUN: FileCheck %s < %t.out

void FuzzMe(const(ubyte*) data, size_t size)
{
    if ((size >= 3) && data[0] == 'F' && data[1] == 'U' && data[2] == 'Z')
    {
// CHECK: fuzz_basic.d([[@LINE+1]]): Assertion failure
        assert(false);
// CHECK: ERROR: libFuzzer: deadly signal
    }
}

extern (C) int LLVMFuzzerTestOneInput(const(ubyte*) data, size_t size)
{
    // D runtime must be initialized, but only once.
    static bool init = false;
    if (!init)
    {
        import core.runtime : rt_init;

        rt_init();
        init = true;
    }

    auto a = new int; // Test that GC works

    FuzzMe(data, size);
    return 0;
}
