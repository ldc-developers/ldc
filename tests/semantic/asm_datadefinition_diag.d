// Tests diagnostics of using data definition directives in inline asm.
// Note: this test should be removed once we _do_ support them.

// REQUIRES: target_X86

// RUN: not %ldc -mtriple=x86_64-linux-gnu -c %s 2>&1 | FileCheck %s

void foo()
{
    asm @nogc nothrow
    {
       mov EAX, EDX;
       // CHECK: ([[@LINE+1]]): Error: Unsupported data definition directive inside inline asm.
       de 2.34L, 3.14L;
       mov EAX, EDX;
    }
}
