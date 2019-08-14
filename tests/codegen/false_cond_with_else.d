// Make sure that the `else`/`else if` _will_ be generated if the `if` condition has a constant 0 value.

// RUN: %ldc -O0 -output-ll -of=%t.s %s && FileCheck %s < %t.s

// extern(C) to avoid name mangling.
extern(C) void foo()
{
    // CHECK: foo()
    // CHECK: else:
    if (0)
    {
        int a = 1;
    }
    else if(1)
    {
        int b = 2;
    }
}
