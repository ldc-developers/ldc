// Test linking C++ stdlib (or not) with -fsanitize=fuzzer

// REQUIRES: atleast_llvm500
// REQUIRES: Fuzzer

// RUN: %ldc -v -fsanitize=fuzzer %s | FileCheck %s
// RUN: not %ldc -v -fsanitize=fuzzer -link-no-cpp %s | FileCheck %s --check-prefix=NOCPP

// "libFuzzer.a" before LLVM 6.0, "lib(ldc|clang)_rt.fuzzer.*.a" since LLVM 6.0
// CHECK: {{(libFuzzer\.a|_rt\.fuzzer.*\.a)}}
// CHECK-SAME: -l{{(c|stdc)}}++

// NOCPP-NOT: -l{{(c|stdc)}}++

extern (C) int LLVMFuzzerTestOneInput(const(ubyte*) data, size_t size)
{
    return 0;
}
