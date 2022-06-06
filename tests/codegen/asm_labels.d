// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-linux-gnu -output-s -of=%t.s %s
// RUN: FileCheck %s < %t.s

// CHECK: _D10asm_labels3fooFiZv:
void foo(int a)
{
    asm
    {
        // CHECK: jmp .L_D10asm_labels3fooFiZv_label
        jmp label;
        // CHECK-NEXT: .L_D10asm_labels3fooFiZv_label:
    label:
        ret;
    }
}

// CHECK: _D10asm_labels3fooFkZv:
void foo(uint a)
{
    asm
    {
        // CHECK: jmp .L_D10asm_labels3fooFkZv_label
        jmp label;
        // CHECK-NEXT: .L_D10asm_labels3fooFkZv_label:
    label:
        ret;
    }
}
