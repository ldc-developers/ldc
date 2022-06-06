// Test that the "asm" statement is not allowed for DCompute code.

// "asm" is only allowed for X86, so we must explicitly target X86 in this test.
// REQUIRES: target_X86

// RUN: not %ldc -mtriple=x86_64-linux-gnu -o- %s 2>&1 | FileCheck %s

@compute(CompileFor.deviceOnly) module tests.semaintic.dcompute;
import ldc.dcompute;

void func()
{
    //CHECK: dcompute_asm.d([[@LINE+1]]): Error: asm not allowed in `@compute` code
    asm {ret;}
}
