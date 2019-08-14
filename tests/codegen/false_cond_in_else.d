// Make sure that the `else if` block is not generated if it has a constant 0 value.

// RUN: %ldc -O0 -output-ll -of=%t.s %s && FileCheck %s < %t.s

// extern(C) to avoid name mangling.
extern(C) void foo()
{
    // CHECK: foo()
    // CHECK: else:
    // CHECK-NEXT: br label
    if (1)
    {
        int a = 1;
    }
    else if(0)
    {
        int b = 2;
    }
}
