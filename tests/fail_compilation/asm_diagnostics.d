// Try to compile a single object file and check inline asm errors from
// 2 source files.

// REQUIRES: target_X86

// RUN: not %ldc -mtriple=x86_64-linux-gnu %s %S/inputs/asm_diagnostics2.d 2> %t.stderr
// RUN: FileCheck %s < %t.stderr

void foo()
{
    import asm_diagnostics2;
    barTemplate();

    asm { "nope\nnoper\nmovq %0, %%eax" : : "b" (123L); }
}

// CHECK: inputs{{.}}asm_diagnostics2.d(6):1:2: error: invalid instruction mnemonic 'hello'
// CHECK: inputs{{.}}asm_diagnostics2.d(6):2:19: error: invalid register name

// CHECK: asm_diagnostics.d(14):1:2: error: invalid instruction mnemonic 'nope'
// CHECK: asm_diagnostics.d(14):2:1: error: invalid instruction mnemonic 'noper'
// CHECK: asm_diagnostics.d(14):3:12: error: invalid operand for instruction
