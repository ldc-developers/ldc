// Test linking C++ stdlib (or not) with -fsanitize=fuzzer

// REQUIRES: Fuzzer

// RUN: %ldc -v -fsanitize=fuzzer %s | FileCheck %s
// RUN: not %ldc -v -fsanitize=fuzzer -link-no-cpp %s | FileCheck %s --check-prefix=NOCPP

// CHECK: libFuzzer.a
// CHECK-SAME: -l{{(c|stdc)}}++

// NOCPP-NOT: -l{{(c|stdc)}}++

extern (C) int LLVMFuzzerTestOneInput(const(ubyte*) data, size_t size)
{
    return 0;
}
