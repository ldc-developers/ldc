// REQUIRES: target_X86
// RUN: %ldc -mtriple=x86_64-pc-linux-gnu -O -output-s -of=%t.s %s && FileCheck %s < %t.s

import ldc.attributes;
import ldc.llvmasm;

// Without -O, LLVM (6) dumps the 2 params to stack for Linux x64 (but doesn't for Win64).
// Clang is no different though.

// CHECK: withParams:
extern(C) int withParams(int param1, void* param2) @naked
{
    // CHECK-NEXT: .cfi_startproc
    // CHECK-NEXT: #APP
    // CHECK-NEXT: xorl %eax, %eax
    // CHECK-NEXT: #NO_APP
    // CHECK-NEXT: retq
    return __asm!int("xor %eax, %eax", "={eax}");
}

// CHECK: _D10attr_naked8noReturnFZv:
void noReturn() @naked
{
    // CHECK-NEXT: .cfi_startproc
    // CHECK-NEXT: #APP
    // CHECK-NEXT: jmpq *%rax
    // CHECK-NEXT: #NO_APP
    __asm("jmp *%rax", "");
    // CHECK-NOT: retq
    // CHECK:     .cfi_endproc
}
