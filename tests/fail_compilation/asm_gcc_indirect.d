// REQUIRES: target_X86

// RUN: not %ldc -mtriple=x86_64-linux-gnu %s 2> %t.stderr
// RUN: FileCheck %s < %t.stderr

void indirectInput(int x)
{
    asm
    {
        "movl %%eax, %0"
        :
        : "m" (&x)
        : "eax";
    }
}

// CHECK: {{.}}asm_gcc_indirect.d(12): Error: indirect `"m"` input operands require an lvalue, but `& x` is an rvalue
