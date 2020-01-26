// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-linux-gnu -output-s -of=%t.s %s
// RUN: FileCheck %s < %t.s

// CHECK: _D19asm_data_directives14_rdrand32_stepFPkZi:
int _rdrand32_step(uint* r)
{
    int ret;
    asm
    {
        // CHECK: movl -12(%rbp), %eax
        mov EAX, ret;
        // CHECK-NEXT: .byte 15
        // CHECK-NEXT: .byte 199
        // CHECK-NEXT: .byte 240
        db 0x0F, 0xC7, 0xF0; // rdrand EAX
        // CHECK-NEXT: movl %eax, -12(%rbp)
        mov ret, EAX;
    }
    if (ret != 0)
    {
        *r = ret;
        return 1;
    }
    return 0;
}
