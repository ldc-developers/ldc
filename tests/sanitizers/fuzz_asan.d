// Test Fuzz+ASan functionality

// REQUIRES: Fuzzer, ASan

// See https://github.com/ldc-developers/ldc/issues/2222 for %disable_fp_elim
// RUN: %ldc -g -fsanitize=address,fuzzer %disable_fp_elim %s -of=%t%exe
// RUN: not %t%exe 2>&1 | FileCheck %s

bool FuzzMe(ubyte* data, size_t dataSize)
{
    return dataSize >= 3 &&
           data[0] == 'F' &&
           data[1] == 'U' &&
           data[2] == 'Z' &&
    // CHECK: stack-buffer-overflow
    // CHECK-NEXT: READ of size 1
    // CHECK-NEXT: #0 {{.*}} in {{.*fuzz_asan6FuzzMe.*}} {{.*}}fuzz_asan.d:
    // FIXME, debug line info is wrong (Github issue #2090). Once fixed, add [[@LINE+1]]
           data[dataSize] == 'Z'; // :‑<
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

    ubyte[3] stackdata;
    if (data)
    {
        for (auto i = 0; (i < size) && (i < stackdata.length); ++i)
            stackdata[i] = data[i];
    }
    // CHECK-NEXT: #1 {{.*}} in LLVMFuzzerTestOneInput {{.*}}fuzz_asan.d:[[@LINE+1]]
    FuzzMe(&stackdata[0], size);

    return 0;
}
