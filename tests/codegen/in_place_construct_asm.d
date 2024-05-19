// Tests in-place construction of structs returned by inline assembly (issue #1823).

// Target Win64 for simplicity (e.g., 4x32-bit struct not returned in memory for non-Windows x64).
// REQUIRES: target_X86
// RUN: %ldc -mtriple=x86_64-pc-windows-msvc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.llvmasm;

// CHECK-LABEL: define{{.*}} @{{.*}}_D22in_place_construct_asm14inlineAssemblyFkkZv
void inlineAssembly(uint eax, uint ecx)
{
    // CHECK: store %"ldc.llvmasm.__asmtuple_t!(uint, uint, uint, uint).__asmtuple_t" %3, ptr %r
    auto r = __asmtuple!(uint, uint, uint, uint) ("cpuid",
        "={eax},={ebx},={ecx},={edx},{eax},{ecx}", eax, ecx);
}
