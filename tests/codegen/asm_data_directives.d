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

// CHECK: _D19asm_data_directives3fooFZv:
void foo()
{
    asm
    {
        // CHECK:      .byte 1
        // CHECK-NEXT: .byte 128
        db 1, 0x80;
        // CHECK-NEXT: .short 2
        // CHECK-NEXT: .short 256
        ds 2, 0x100;
        // CHECK-NEXT: .long 3
        // CHECK-NEXT: .long 65536
        di 3, 0x10000;
        // CHECK-NEXT: .quad 4
        // CHECK-NEXT: .quad 4294967296
        dl 4, 0x100000000;

        // CHECK-NEXT: .long 1065353216
        // CHECK-NEXT: .long 1069547520
        df 1.0f, 1.5f;
        // CHECK-NEXT: .quad 4607182418800017408
        // CHECK-NEXT: .quad 4609434218613702656
        dd 1.0, 1.5;
        // CHECK-NEXT: .quad -9223372036854775808
        // CHECK-NEXT: .short 16383
        // CHECK-NEXT: .quad -4611686018427387904
        // CHECK-NEXT: .short 16383
        de 1.0L, 1.5L;
    }
}
