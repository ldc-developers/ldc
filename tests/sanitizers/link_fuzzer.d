// Test linking C++ stdlib (or not) with -fsanitize=fuzzer

// REQUIRES: atleast_llvm500
// REQUIRES: Fuzzer

// RUN: %ldc -v -fsanitize=fuzzer %s | FileCheck %s
// "libFuzzer.a" before LLVM 6.0, "lib(ldc|clang)_rt.fuzzer.*.a" since LLVM 6.0
// CHECK: {{(libFuzzer\.a|_rt\.fuzzer.*\.a)}}
// CHECK-SAME: -l{{(std)?}}c++

// RUN: %ldc -v -fsanitize=fuzzer -link-no-cpp %s > %t_nocpp.log || true
// RUN: FileCheck %s --check-prefix=NOCPP < %t_nocpp.log
// NOCPP-NOT: -l{{(std)?}}c++

extern (C) int LLVMFuzzerTestOneInput(const(ubyte*) data, size_t size)
{
    return 0;
}
