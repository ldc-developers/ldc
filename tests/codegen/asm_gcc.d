// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-linux-gnu -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

void cpuid()
{
    uint max_extended_cpuid;
    // CHECK: %1 = call i32 asm sideeffect "cpuid", "={eax},{eax},~{ebx},~{ecx},~{edx}"(i32 -2147483648)
    // CHECK: store i32 %1, i32* %max_extended_cpuid
    asm { "cpuid" : "=eax" max_extended_cpuid : "eax" 0x8000_0000 : "ebx", "ecx", "edx"; }
}
